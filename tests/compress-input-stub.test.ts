import assert from "node:assert/strict"
import "./lab/persistence-env"
import test from "node:test"
import type { Message } from "@opencode/ai/schema/messages"
import { project } from "../lib/v2/messages"
import { createSessionState, type CompressionBlock, type SessionState } from "../lib/state"
import type { WithParts } from "../lib/state"
import { prune, syncCompressionBlocks, buildPriorityMap } from "../lib/messages"
import type { PluginConfig } from "../lib/config"
import { Logger } from "../lib/logger"
import { assignMessageRefs } from "../lib/message-ids"
import { syncToolCache } from "../lib/state/tool-cache"
import { allocateBlockId, allocateRunId, applyCompressionState } from "../lib/compress/state"
import {
    formatCompressInputStub,
    isCompressInputStub,
    stubCompressCallInputs,
    stubbedCompressInput,
} from "../lib/compress/input-stub"
import { validateArgs as validateRangeArgs } from "../lib/compress/range-utils"
import { validateArgs as validateMessageArgs } from "../lib/compress/message-utils"
import { normalizeRangeArgs } from "../lib/compress/range-utils"
import { countAllMessageTokens, countTokens, getCurrentTokenUsage } from "../lib/token-utils"

// Shapes follow a real V2 session (Qwen, range mode, one range per call): the call message is
// [reasoning, text, tool-call compress], the result is the fixed "Compressed N messages" line,
// and later calls put `content` before `topic`. All text here is synthetic filler.

const session = {
    id: "ses_stub",
    agent: "build",
    model: { id: "test", providerID: "test" },
} as Parameters<typeof project>[2]
const logger = new Logger(false)

function makeConfig(mode: "range" | "message" = "range"): PluginConfig {
    return {
        compress: { mode, permission: "allow", protectUserMessages: false },
        turnProtection: { enabled: false, turns: 4 },
    } as PluginConfig
}

const WORDS = ["alpha", "parser", "config", "token", "branch", "module", "socket", "render"]

function filler(length: number, sentinel: string): string {
    let text = ""
    let index = 0
    while (text.length < length) {
        text += `${WORDS[index % WORDS.length]}${index} `
        index++
    }
    return `${text}${sentinel}`
}

function entries(messages: Message[]) {
    return messages
        .filter((message) => message.id)
        .map((message) => ({
            id: message.id,
            type: message.role,
            time: { created: 1 },
            agent: "build",
            model: session.model,
            content: [],
            text: "",
        })) as Parameters<typeof project>[1]
}

function user(id: string, text: string): Message {
    return { id, role: "user", content: [{ type: "text", text }] } as Message
}

function assistant(id: string, text: string): Message {
    return { id, role: "assistant", content: [{ type: "text", text }] } as Message
}

function compressCall(id: string, callId: string, input: unknown, count = 32): Message[] {
    return [
        {
            id,
            role: "assistant",
            content: [
                {
                    type: "reasoning",
                    text: "The older exploration is finished; compress it.",
                    encrypted: "signed-reasoning",
                    providerMetadata: { test: { signature: "sig" } },
                },
                { type: "text", text: "Compressing the finished exploration." },
                { type: "tool-call", id: callId, name: "compress", input },
            ],
        },
        {
            role: "tool",
            content: [
                {
                    type: "tool-result",
                    id: callId,
                    name: "compress",
                    result: {
                        type: "text",
                        value: `Compressed ${count} messages into [Compressed conversation section].`,
                    },
                },
            ],
        },
    ] as Message[]
}

function applyBlock(
    state: SessionState,
    options: {
        startId: string
        endId: string
        anchor: string
        messageIds: string[]
        compressMessageId: string
        compressCallId?: string
        summary: string
        consumed?: number[]
        mode?: "range" | "message"
    },
): number {
    const blockId = allocateBlockId(state)
    applyCompressionState(
        state,
        {
            topic: `topic ${blockId}`,
            batchTopic: `topic ${blockId}`,
            startId: options.startId,
            endId: options.endId,
            mode: options.mode ?? "range",
            runId: allocateRunId(state),
            compressMessageId: options.compressMessageId,
            compressCallId: options.compressCallId,
            summaryTokens: 0,
        },
        {
            startReference: {} as any,
            endReference: {} as any,
            messageIds: options.messageIds,
            messageTokenById: new Map(),
            toolIds: [],
            requiredBlockIds: options.consumed ?? [],
        },
        options.anchor,
        blockId,
        options.summary,
        options.consumed ?? [],
    )
    return blockId
}

function rangeInput(summary: string, contentFirst = false) {
    const content = [{ startId: "@1@", endId: "@2@", summary }]
    return contentFirst ? { content, topic: "Exploration" } : { topic: "Exploration", content }
}

function basicNative(summary: string, contentFirst = false): Message[] {
    return [
        user("msg_u1", "Explore the parser"),
        assistant("msg_a1", "The parser has three stages."),
        user("msg_u2", "Now clean up"),
        ...compressCall("msg_c1", "call_c1", rangeInput(summary, contentFirst)),
        user("msg_u3", "Continue"),
    ]
}

function basicRequest(
    native: Message[],
    configure?: (state: SessionState) => void,
): { state: SessionState; view: ReturnType<typeof project>; restored: Message[] } {
    const state = createSessionState("compact")
    applyBlock(state, {
        startId: "@1@",
        endId: "@2@",
        anchor: "msg_u1",
        messageIds: ["msg_u1", "msg_a1"],
        compressMessageId: "msg_c1",
        compressCallId: "call_c1",
        summary: `[Compressed conversation section]\n${(native[3] as any).content[2].input.content[0].summary}`,
    })
    configure?.(state)
    const view = project(native, entries(native), session)
    prune(state, logger, makeConfig(), view.messages, view.summaryBase)
    return { state, view, restored: view.restore() }
}

function callOf(messages: Message[], id = "msg_c1") {
    const message = messages.find((item) => item.id === id) as any
    return { message, call: message?.content.find((part: any) => part.type === "tool-call") }
}

function occurrences(text: string, needle: string): number {
    return text.split(needle).length - 1
}

test("V2 request: a completed compress call's summary is replaced by a stub naming its block", () => {
    for (const contentFirst of [false, true]) {
        const summary = filler(3000, "SENTINEL_basic")
        const native = basicNative(summary, contentFirst)
        const before = structuredClone(native)
        const { restored } = basicRequest(native)

        const { message, call } = callOf(restored)
        const nativeCall = callOf(native)
        assert.equal(
            call.input.content[0].summary,
            "[DCP: summary stored in compressed block @b1@]",
        )
        assert.equal(call.input.content[0].startId, "@1@")
        assert.equal(call.input.content[0].endId, "@2@")
        assert.equal(call.input.topic, "Exploration")
        assert.deepEqual(Object.keys(call.input), Object.keys(nativeCall.call.input))
        assert.deepEqual(
            Object.keys(call.input.content[0]),
            Object.keys(nativeCall.call.input.content[0]),
        )
        // Reasoning and the tool result go out exactly as the host sent them.
        assert.deepEqual(message.content[0], nativeCall.message.content[0])
        const result = restored.find((item) => item.role === "tool")
        assert.deepEqual(
            result,
            native.find((item) => item.role === "tool"),
        )
        // The host's own messages are never edited.
        assert.deepEqual(native, before)
        assert.equal(occurrences(JSON.stringify(restored), "SENTINEL_basic"), 1)
    }
})

test("V1 shape: the stub uses the xml block ref and leaves the cached raw input alone", async () => {
    const summary = filler(800, "SENTINEL_v1")
    const input = { topic: "Exploration", content: [{ startId: "m0001", endId: "m0002", summary }] }
    const messages = [
        {
            info: { id: "msg_c1", role: "assistant", sessionID: "s", time: { created: 1 } },
            parts: [
                {
                    id: "p1",
                    type: "tool",
                    tool: "compress",
                    callID: "call_c1",
                    messageID: "msg_c1",
                    sessionID: "s",
                    state: { status: "completed", input, output: "Compressed 2 messages." },
                },
            ],
        },
    ] as unknown as WithParts[]
    const state = createSessionState("xml")
    await syncToolCache(state, makeConfig(), logger, messages)
    applyBlock(state, {
        startId: "m0001",
        endId: "m0002",
        anchor: "msg_u1",
        messageIds: ["msg_u1"],
        compressMessageId: "msg_c1",
        compressCallId: "call_c1",
        summary,
    })

    stubCompressCallInputs(state, messages)

    const stub = (messages[0]!.parts[0] as any).state.input.content[0].summary
    assert.equal(stub, "[DCP: summary stored in compressed block b1]")
    assert.doesNotMatch(stub, /<dcp|@/)
    assert.equal(input.content[0].summary, summary)
    assert.equal(state.toolParameters.get("call_c1")?.parameters, input)
})

test("stubbing is deterministic and idempotent", () => {
    const summary = filler(1500, "SENTINEL_det")
    const first = basicRequest(basicNative(summary))
    const second = basicRequest(basicNative(summary))
    assert.equal(JSON.stringify(first.restored), JSON.stringify(second.restored))

    stubCompressCallInputs(first.state, first.view.messages)
    assert.equal(JSON.stringify(first.view.restore()), JSON.stringify(first.restored))
})

function v1Call(
    callId: string,
    input: unknown,
    status: "completed" | "error" | "running" | "pending" = "completed",
): any {
    return {
        id: `prt_${callId}`,
        type: "tool",
        tool: "compress",
        callID: callId,
        messageID: "msg_c",
        sessionID: "s",
        state: { status, input },
    }
}

function v1Message(...parts: any[]): WithParts[] {
    return [
        {
            info: { id: "msg_c", role: "assistant", sessionID: "s", time: { created: 1 } },
            parts,
        },
    ] as unknown as WithParts[]
}

function block(
    blockId: number,
    fields: Partial<CompressionBlock> & { startId: string },
): CompressionBlock {
    return {
        blockId,
        runId: 1,
        active: true,
        deactivatedByUser: false,
        compressedTokens: 0,
        summaryTokens: 0,
        durationMs: 0,
        mode: "range",
        topic: "t",
        endId: fields.startId,
        anchorMessageId: "msg_anchor",
        compressMessageId: "msg_c",
        compressCallId: "call_1",
        includedBlockIds: [],
        consumedBlockIds: [],
        parentBlockIds: [],
        directMessageIds: [],
        directToolIds: [],
        effectiveMessageIds: [],
        effectiveToolIds: [],
        createdAt: blockId,
        summary: "stored",
        ...fields,
    }
}

function withBlocks(...blocks: CompressionBlock[]): SessionState {
    const state = createSessionState("compact")
    for (const item of blocks) state.prune.messages.blocksById.set(item.blockId, item)
    return state
}

function summaries(messages: WithParts[], index = 0): unknown[] {
    const input = (messages[0]!.parts[index] as any).state.input
    return input.content.map((entry: any) => entry.summary)
}

test("several ranges in one call each get their own block, matched by trimmed boundaries", () => {
    const messages = v1Message(
        v1Call("call_1", {
            topic: "two",
            content: [
                { startId: "  @3@ ", endId: "@4@", summary: "first" },
                { startId: "@6@", endId: "@9@", summary: "second" },
            ],
        }),
    )
    const state = withBlocks(
        block(5, { startId: "@6@", endId: "@9@" }),
        block(4, { startId: "@3@", endId: "@4@" }),
    )
    stubCompressCallInputs(state, messages)
    assert.deepEqual(summaries(messages), [
        formatCompressInputStub(4, "compact"),
        formatCompressInputStub(5, "compact"),
    ])
    assert.equal((messages[0]!.parts[0] as any).state.input.content[0].startId, "  @3@ ")
})

test("message mode stubs matched entries and leaves skipped ones verbatim", () => {
    const messages = v1Message(
        v1Call("call_1", {
            topic: "batch",
            content: [
                { messageId: "@3@", topic: "a", summary: "sum a" },
                { messageId: " @5@", topic: "b", summary: "sum b" },
                { messageId: "@7@", topic: "c", summary: "skipped one" },
            ],
        }),
    )
    const state = withBlocks(
        block(1, { mode: "message", startId: "@3@" }),
        block(2, { mode: "message", startId: "@5@" }),
    )
    stubCompressCallInputs(state, messages)
    assert.deepEqual(summaries(messages), [
        formatCompressInputStub(1, "compact"),
        formatCompressInputStub(2, "compact"),
        "skipped one",
    ])

    const duplicate = v1Message(
        v1Call("call_1", {
            topic: "dup",
            content: [
                { messageId: "@3@", topic: "a", summary: "kept" },
                { messageId: "@3@", topic: "a", summary: "duplicate" },
            ],
        }),
    )
    stubCompressCallInputs(withBlocks(block(1, { mode: "message", startId: "@3@" })), duplicate)
    assert.deepEqual(summaries(duplicate), [formatCompressInputStub(1, "compact"), "duplicate"])
})

test("only finished compress calls with a block are stubbed", () => {
    const input = () => ({ topic: "t", content: [{ startId: "@3@", endId: "@4@", summary: "S" }] })

    const errored = v1Message(v1Call("call_1", input(), "error"))
    stubCompressCallInputs(withBlocks(), errored)
    assert.deepEqual(summaries(errored), ["S"])
    stubCompressCallInputs(withBlocks(block(9, { startId: "@1@", endId: "@2@" })), errored)
    assert.deepEqual(summaries(errored), ["S"])

    for (const status of ["running", "pending"] as const) {
        const live = v1Message(v1Call("call_1", input(), status))
        stubCompressCallInputs(withBlocks(block(1, { startId: "@3@", endId: "@4@" })), live)
        assert.deepEqual(summaries(live), ["S"])
    }

    // The block was applied but finishing the call threw: its summary is still in the block.
    const appliedThenFailed = v1Message(v1Call("call_1", input(), "error"))
    stubCompressCallInputs(
        withBlocks(block(1, { startId: "@3@", endId: "@4@" })),
        appliedThenFailed,
    )
    assert.deepEqual(summaries(appliedThenFailed), [formatCompressInputStub(1, "compact")])
})

test("the call message bytes do not depend on block liveness", () => {
    const summary = filler(2000, "SENTINEL_live")
    const callBytes = (configure?: (state: SessionState) => void) =>
        JSON.stringify(callOf(basicRequest(basicNative(summary), configure).restored).message)

    const active = callBytes()
    assert.match(active, /@b1@/)
    assert.equal(
        callBytes((state) => {
            const item = state.prune.messages.blocksById.get(1)!
            item.active = false
            item.deactivatedByBlockId = 7
            state.prune.messages.activeBlockIds.delete(1)
            state.prune.messages.activeByAnchorMessageId.delete("msg_u1")
            state.prune.messages.byMessageId.forEach((entry) => (entry.activeBlockIds = []))
        }),
        active,
    )
    const toggled = (deactivated: boolean) =>
        callBytes((state) => {
            state.prune.messages.blocksById.get(1)!.deactivatedByUser = deactivated
            const native = basicNative(summary)
            syncCompressionBlocks(state, logger, project(native, entries(native), session).messages)
        })
    assert.equal(toggled(true), active)
    assert.equal(toggled(false), active)
    assert.equal(
        callBytes((state) => {
            state.prune.messages.blocksById.get(1)!.anchorMessageId = "msg_gone"
        }),
        active,
    )

    // Decompress then recompress on one live state never changes the call message.
    const native = basicNative(summary)
    const { state } = basicRequest(native)
    const seen = new Set<string>()
    for (const deactivated of [true, false, true, false]) {
        state.prune.messages.blocksById.get(1)!.deactivatedByUser = deactivated
        syncCompressionBlocks(state, logger, project(native, entries(native), session).messages)
        const view = project(native, entries(native), session)
        prune(state, logger, makeConfig(), view.messages, view.summaryBase)
        seen.add(JSON.stringify(callOf(view.restore()).message))
    }
    assert.deepEqual([...seen], [active])
})

test("without a block in state the compress input stays raw", () => {
    const summary = filler(500, "SENTINEL_none")
    const native = basicNative(summary)
    const view = project(native, entries(native), session)
    prune(createSessionState("compact"), logger, makeConfig(), view.messages, view.summaryBase)
    assert.deepEqual(view.restore(), native)
})

test("call id fallbacks only apply when the compress part is unambiguous", () => {
    const input = (startId: string) => ({
        topic: "t",
        content: [{ startId, endId: startId, summary: `S${startId}` }],
    })

    const legacy = v1Message(v1Call("call_1", input("@3@")))
    stubCompressCallInputs(
        withBlocks(block(1, { startId: "@3@", compressCallId: undefined })),
        legacy,
    )
    assert.deepEqual(summaries(legacy), [formatCompressInputStub(1, "compact")])

    const ambiguous = v1Message(v1Call("call_1", input("@3@")), v1Call("call_2", input("@3@")))
    stubCompressCallInputs(
        withBlocks(block(1, { startId: "@3@", compressCallId: undefined })),
        ambiguous,
    )
    assert.deepEqual(summaries(ambiguous, 0), ["S@3@"])
    assert.deepEqual(summaries(ambiguous, 1), ["S@3@"])

    const mismatch = v1Message(v1Call("call_host", input("@3@")))
    stubCompressCallInputs(
        withBlocks(block(1, { startId: "@3@", compressCallId: "ctx" })),
        mismatch,
    )
    assert.deepEqual(summaries(mismatch), [formatCompressInputStub(1, "compact")])

    const otherMessage = v1Message(v1Call("call_1", input("@3@")))
    stubCompressCallInputs(
        withBlocks(block(1, { startId: "@3@", compressMessageId: "msg_other" })),
        otherMessage,
    )
    assert.deepEqual(summaries(otherMessage), ["S@3@"])

    const two = v1Message(v1Call("call_1", input("@3@")), v1Call("call_2", input("@5@")))
    stubCompressCallInputs(
        withBlocks(
            block(1, { startId: "@3@", compressCallId: "call_1" }),
            block(2, { startId: "@5@", compressCallId: "call_2" }),
        ),
        two,
    )
    assert.deepEqual(summaries(two, 0), [formatCompressInputStub(1, "compact")])
    assert.deepEqual(summaries(two, 1), [formatCompressInputStub(2, "compact")])
})

test("content sent as a JSON string stays a string with stubbed summaries", () => {
    const blocks = [block(1, { startId: "@3@", endId: "@4@" }), block(2, { startId: "@6@" })]
    const asArray = {
        topic: "t",
        content: JSON.stringify([
            { startId: "@3@", endId: "@4@", summary: "one" },
            { startId: "@6@", endId: "@6@", summary: "two" },
        ]),
    }
    const next = stubbedCompressInput(asArray, blocks, "compact") as any
    assert.equal(typeof next.content, "string")
    const normalized = normalizeRangeArgs(next)
    assert.ok(normalized.content.every((entry) => isCompressInputStub(entry.summary)))

    const single = {
        topic: "t",
        content: JSON.stringify({ startId: "@3@", endId: "@4@", summary: "one" }),
    }
    const singleNext = stubbedCompressInput(single, blocks, "compact") as any
    assert.equal(
        singleNext.content,
        JSON.stringify({
            startId: "@3@",
            endId: "@4@",
            summary: formatCompressInputStub(1, "compact"),
        }),
    )

    assert.equal(
        stubbedCompressInput({ topic: "t", content: "[not json" }, blocks, "compact"),
        undefined,
    )
    assert.equal(
        stubbedCompressInput({ topic: "t", content: "plain" }, blocks, "compact"),
        undefined,
    )
})

test("validateArgs rejects a summary that is only the stub", () => {
    const stubs = [
        formatCompressInputStub(3, "compact"),
        formatCompressInputStub(3, "xml"),
        `  ${formatCompressInputStub(3, "compact")}\n`,
    ]
    for (const summary of stubs) {
        assert.throws(
            () =>
                validateRangeArgs({
                    topic: "t",
                    content: [{ startId: "@1@", endId: "@2@", summary }],
                }),
            /placeholder DCP leaves in earlier compress calls/,
        )
        assert.throws(
            () =>
                validateMessageArgs({
                    topic: "t",
                    content: [{ messageId: "@1@", topic: "t", summary }],
                }),
            /placeholder DCP leaves in earlier compress calls/,
        )
    }
    const mentions = "Picked up where @b3@ left off and fixed the parser."
    validateRangeArgs({
        topic: "t",
        content: [{ startId: "@1@", endId: "@2@", summary: mentions }],
    })
    validateMessageArgs({
        topic: "t",
        content: [{ messageId: "@1@", topic: "t", summary: mentions }],
    })
})

test("token accounting on the request view reflects the stub", async () => {
    const summary = filler(6000, "SENTINEL_tokens")
    const summaryTokens = await countTokens(summary)
    const stubbed = basicRequest(basicNative(summary))

    const rawState = createSessionState("compact")
    const rawView = structuredClone(stubbed.view.messages)
    const rawCall = rawView.find((message) => message.info.id === "msg_c1")!
    ;(rawCall.parts.find((part) => part.type === "tool") as any).state.input = rangeInput(summary)

    const withStub = await getCurrentTokenUsage(stubbed.state, stubbed.view.messages)
    const withoutStub = await getCurrentTokenUsage(rawState, rawView)
    assert.ok(withoutStub - withStub >= 0.9 * summaryTokens, `${withoutStub} - ${withStub}`)

    const native = basicNative(summary)
    const view = project(native, entries(native), session)
    const state = createSessionState("compact")
    assignMessageRefs(state, view.messages)
    applyBlock(state, {
        startId: "@1@",
        endId: "@2@",
        anchor: "msg_u1",
        messageIds: ["msg_u1", "msg_a1"],
        compressMessageId: "msg_c1",
        compressCallId: "call_c1",
        summary,
    })
    const config = makeConfig("message")
    prune(state, logger, config, view.messages, view.summaryBase)
    const priorities = await buildPriorityMap(config, state, view.messages)
    assert.ok((priorities.get("msg_c1")?.tokenCount ?? Infinity) < 0.1 * summaryTokens)
})

// Regression modeled on the session whose post-compress floor kept rising: 21 sequential
// range compressions, two of them starting at the previous call message, and one that redoes
// the previous range plus one round and so consumes that block. Summary sizes are scaled
// (1/10) from the real calls.
const SUMMARY_SIZES = [
    2730, 2204, 2185, 1929, 430, 920, 610, 780, 505, 860, 470, 720, 655, 900, 540, 830, 495, 760,
    600, 880, 450,
]
const BACK_TO_BACK = new Set([6, 11])
const CONSUMES_PREVIOUS = 16

test("p2-09 B regression: each summary is sent once and stubbing never breaks the cache", async () => {
    const native: Message[] = []
    const state = createSessionState("compact")
    const summaries = new Map<number, string>()
    const blockByRound = new Map<number, number>()
    const requests: Array<{ restored: Message[]; view: ReturnType<typeof project> }> = []
    let previousCall: string | undefined
    let previousStart: string | undefined

    const request = () => {
        const view = project(native, entries(native), session)
        assignMessageRefs(state, view.messages)
        syncCompressionBlocks(state, logger, view.messages)
        prune(state, logger, makeConfig(), view.messages, view.summaryBase)
        const restored = view.restore()
        requests.push({ restored, view })
        return restored
    }
    const firstDiff = (before: Message[], after: Message[]) => {
        let index = 0
        while (
            index < before.length &&
            index < after.length &&
            JSON.stringify(before[index]) === JSON.stringify(after[index])
        )
            index++
        return index
    }
    const assertCallsStubbed = (restored: Message[]) => {
        for (const message of restored) {
            for (const part of (message.content ?? []) as any[]) {
                if (part.type !== "tool-call" || part.name !== "compress") continue
                const round = Number(part.id.replace("call_", ""))
                assert.equal(
                    part.input.content[0].summary,
                    formatCompressInputStub(blockByRound.get(round)!, "compact"),
                )
            }
        }
    }

    for (let round = 1; round <= SUMMARY_SIZES.length; round++) {
        const before = requests.at(-1)?.restored ?? []
        native.push(user(`msg_u${round}`, filler(400, `question ${round}`)))
        native.push(assistant(`msg_a${round}`, filler(600, `answer ${round}`)))
        const appended = request()
        assert.equal(firstDiff(before, appended), before.length, `round ${round}: pure append`)
        assertCallsStubbed(appended)

        const consume = round === CONSUMES_PREVIOUS
        const start = consume
            ? previousStart!
            : BACK_TO_BACK.has(round) && previousCall
              ? previousCall
              : `msg_u${round}`
        const ids = native.map((message) => message.id).filter(Boolean) as string[]
        const covered = ids.slice(ids.indexOf(start), ids.indexOf(`msg_a${round}`) + 1)
        const ownText = filler(SUMMARY_SIZES[round - 1]!, `SENTINEL_${round}_END`)
        const consumed = consume ? [blockByRound.get(round - 1)!] : []
        // Range mode expands a consumed block into the new summary.
        const stored = consume ? `${ownText}\n${summaries.get(round - 1)}` : ownText
        summaries.set(round, stored)

        native.push(
            ...compressCall(`msg_c${round}`, `call_${round}`, {
                ...(round >= 18 ? {} : { topic: `round ${round}` }),
                content: [{ startId: "@x@", endId: "@y@", summary: ownText }],
                ...(round >= 18 ? { topic: `round ${round}` } : {}),
            }),
        )
        const refs = (id: string) => state.messageIds.byRawId.get(id)!
        const call = native.at(-2) as any
        call.content[2].input.content[0].startId = refs(start)
        call.content[2].input.content[0].endId = refs(`msg_a${round}`)
        blockByRound.set(
            round,
            applyBlock(state, {
                startId: refs(start),
                endId: refs(`msg_a${round}`),
                anchor: start,
                messageIds: covered,
                compressMessageId: `msg_c${round}`,
                compressCallId: `call_${round}`,
                summary: stored,
                consumed,
            }),
        )
        const compressed = request()
        const anchorIndex = compressed.findIndex((message) =>
            JSON.stringify(message).includes(`SENTINEL_${round}_END`),
        )
        assert.ok(anchorIndex >= 0)
        const diff = firstDiff(appended, compressed)
        assert.ok(
            diff <= anchorIndex,
            `round ${round}: change at ${diff} after anchor ${anchorIndex}`,
        )
        assertCallsStubbed(compressed)

        previousCall = `msg_c${round}`
        previousStart = start
    }

    const { restored, view } = requests.at(-1)!
    const serialized = JSON.stringify(restored)
    for (let round = 1; round <= SUMMARY_SIZES.length; round++) {
        assert.equal(occurrences(serialized, `SENTINEL_${round}_END`), 1, `SENTINEL_${round}_END`)
    }

    // The floor without stubs is higher by the summaries of every call still in the request.
    const unstubbed = structuredClone(view.messages)
    let liveSummaryTokens = 0
    let liveCalls = 0
    for (const message of unstubbed) {
        for (const part of message.parts as any[]) {
            if (part.type !== "tool" || part.tool !== "compress") continue
            const original = callOf(native, message.info.id).call.input
            part.state.input = original
            liveSummaryTokens += await countTokens(original.content[0].summary)
            liveCalls++
        }
    }
    assert.equal(liveCalls, SUMMARY_SIZES.length - BACK_TO_BACK.size - 1)
    let floor = 0
    let unstubbedFloor = 0
    for (const message of view.messages) floor += await countAllMessageTokens(message)
    for (const message of unstubbed) unstubbedFloor += await countAllMessageTokens(message)
    const stubTokens = await countTokens(formatCompressInputStub(21, "compact"))
    assert.ok(
        unstubbedFloor - floor >= liveSummaryTokens - liveCalls * stubTokens - liveCalls * 2,
        `${unstubbedFloor} - ${floor} vs ${liveSummaryTokens}`,
    )
})
