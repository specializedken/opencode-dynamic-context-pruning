import assert from "node:assert/strict"
import "./lab/persistence-env"
import test from "node:test"
import type { Message } from "@opencode/ai/schema/messages"
import { project } from "../lib/v2/messages"
import { createSessionState, type CompressionBlock } from "../lib/state"
import { assignMessageRefs } from "../lib/message-ids"
import { injectMessageIds, prune } from "../lib/messages"
import type { PluginConfig } from "../lib/config"
import { Logger } from "../lib/logger"

const session = {
    id: "ses_test",
    agent: "build",
    model: { id: "test", providerID: "test" },
} as Parameters<typeof project>[2]
const config = {
    compress: { mode: "range", permission: "allow", protectUserMessages: false },
} as PluginConfig
const logger = new Logger(false)
function transcript(): Message[] {
    return [
        {
            role: "system",
            content: [{ type: "compaction", provider: "test", encrypted: "opaque-checkpoint" }],
        },
        {
            id: "msg_user",
            role: "user",
            content: [
                { type: "media", mediaType: "image/png", data: new Uint8Array([0, 255]) },
                { type: "text", text: "Inspect this", cache: { type: "ephemeral" } },
            ],
        },
        {
            id: "msg_assistant",
            role: "assistant",
            metadata: { host: true },
            content: [
                {
                    type: "reasoning",
                    text: "Reasoning",
                    encrypted: "signed-reasoning",
                    providerMetadata: { test: { signature: "original" } },
                },
                {
                    type: "tool-call",
                    id: "call_one",
                    name: "read",
                    input: { path: "one" },
                    providerMetadata: { test: { itemId: "fc_original" } },
                },
                { type: "tool-call", id: "call_two", name: "read", input: { path: "two" } },
            ],
        },
        {
            role: "tool",
            content: [
                {
                    type: "tool-result",
                    id: "call_one",
                    name: "read",
                    result: { type: "text", value: "Original output" },
                    providerMetadata: { test: { replay: true } },
                },
                {
                    type: "tool-result",
                    id: "call_two",
                    name: "read",
                    result: {
                        type: "content",
                        value: [
                            { type: "file", uri: "data:image/png;base64,AA==", mime: "image/png" },
                        ],
                    },
                },
            ],
        },
        { id: "msg_latest", role: "user", content: [{ type: "text", text: "Continue" }] },
    ] as Message[]
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

test("V2 projection preserves all native content without edits", () => {
    const messages = transcript()
    assert.deepEqual(project(messages, entries(messages), session).restore(), messages)
})

test("V2 request view drops DCP chat notifications but keeps look-alike assistant text", () => {
    const native: Message[] = [
        {
            id: "msg_user",
            role: "user",
            content: [{ type: "text", text: "Real user prompt" }],
        },
        {
            id: "msg_dcp_note",
            role: "user",
            content: [{ type: "text", text: "▣ DCP | ~1.2K saved total" }],
        },
        {
            id: "msg_trigger",
            role: "user",
            content: [
                { type: "text", text: "<compress triggered manually>\n\nManual trigger prompt" },
            ],
        },
        {
            id: "msg_assistant",
            role: "assistant",
            content: [{ type: "text", text: "The report starts with ▣ DCP | in the log" }],
        },
    ]
    const restored = project(native, entries(native), session).restore()
    // Manual-trigger prompts reach the model; only stamped DCP notes are dropped.
    assert.deepEqual(
        restored.map((message) => message.id),
        ["msg_user", "msg_trigger", "msg_assistant"],
    )
})

test("V2 history view skips DCP chat notifications entirely", async () => {
    const { history } = await import("../lib/v2/messages")
    const entries = [
        {
            id: "msg_user",
            type: "user",
            time: { created: 1 },
            text: "Real user prompt",
        },
        {
            id: "msg_note",
            type: "user",
            time: { created: 2 },
            text: "▣ DCP\nSweep report body",
        },
        {
            id: "msg_trigger",
            type: "user",
            time: { created: 3 },
            text: "<compress triggered manually>\n\nManual trigger prompt",
        },
    ] as unknown as Parameters<typeof history>[0]
    const views = history(entries, session as Parameters<typeof history>[1])
    assert.deepEqual(
        views.map((view) => view.info.id),
        ["msg_user", "msg_trigger"],
    )
})

test("V2 ID injection preserves signatures, media and tool pairing", () => {
    const native = transcript()
    const view = project(native, entries(native), session)
    const state = createSessionState("compact")
    assignMessageRefs(state, view.messages)
    injectMessageIds(state, config, view.messages, new Map())
    const restored = view.restore()
    assert.deepEqual(restored[0], native[0])
    assert.equal(restored[1]!.content[0], native[1]!.content[0])
    assert.equal(restored[2]!.content[0], native[2]!.content[0])
    assert.equal(restored[3]!.content[1], native[3]!.content[1])
    assert.match(JSON.stringify(restored[3]!.content[0]), /@2@/)
    assert.doesNotMatch(JSON.stringify(restored), /dcp-message-id|m0002/)
    assert.ok(!JSON.stringify(native).includes("@2@"), "native input must not be mutated")
})

test("V2 tool pruning replaces only marked results", () => {
    const native = transcript()
    const view = project(native, entries(native), session)
    const state = createSessionState("compact")
    state.prune.tools.set("call_one", 100)
    prune(state, logger, config, view.messages)
    const restored = view.restore()
    assert.match(JSON.stringify(restored[3]!.content[0]), /Output removed/)
    assert.equal(restored[3]!.content[1], native[3]!.content[1])
    assert.deepEqual(restored[2], native[2])
})

test("V2 compression removes the assistant and its ID-less results together", () => {
    const native = transcript()
    const view = project(native, entries(native), session)
    const state = createSessionState("compact")
    state.prune.messages.byMessageId.set("msg_assistant", {
        tokenCount: 100,
        allBlockIds: [1],
        activeBlockIds: [1],
    })
    prune(state, logger, config, view.messages)
    const restored = view.restore()
    assert.deepEqual(restored, [native[0], native[1], native[4]])
})

test("V2 leaves non-durable plugin messages and checkpoint tool results intact", () => {
    const native = transcript()
    const view = project(native, [], session)
    assert.equal(view.messages.length, 0, "unselectable messages must not receive DCP IDs")
    assert.deepEqual(view.restore(), native)
})

test("V2 inserts summaries after native checkpoints without an ordinary user message", () => {
    const native = transcript().filter((message) => message.role !== "user")
    const view = project(native, entries(native), session)
    const state = createSessionState("compact")
    state.prune.messages.byMessageId.set("msg_assistant", {
        tokenCount: 100,
        allBlockIds: [1],
        activeBlockIds: [1],
    })
    state.prune.messages.activeByAnchorMessageId.set("msg_assistant", 1)
    state.prune.messages.blocksById.set(1, {
        blockId: 1,
        active: true,
        anchorMessageId: "msg_assistant",
        summary: "CHECKPOINT_SUMMARY",
    } as CompressionBlock)
    prune(state, logger, config, view.messages, view.summaryBase)
    const restored = view.restore()
    assert.equal(restored.length, 2)
    assert.deepEqual(restored[0], native[0])
    assert.deepEqual(restored[1]?.content, [{ type: "text", text: "CHECKPOINT_SUMMARY" }])
})

test("V2 request view does not count tool-result media payloads as text tokens", async () => {
    // The first media fix only handled the history() array shape; project() keeps the whole
    // result object ({ type: "content", value: [...] }), so screenshots still inflated usage
    // on the live request path and kept compression reminders firing below the min limit.
    const { countAllMessageTokens } = await import("../lib/token-utils")
    const base64 = "iVBORw0KGgo".repeat(30000)
    const native = [
        { id: "msg_user", role: "user", content: [{ type: "text", text: "Take a screenshot" }] },
        {
            id: "msg_assistant",
            role: "assistant",
            content: [
                { type: "tool-call", id: "call_shot", name: "read", input: { path: "a.png" } },
            ],
        },
        {
            role: "tool",
            content: [
                {
                    type: "tool-result",
                    id: "call_shot",
                    name: "read",
                    result: {
                        type: "content",
                        value: [
                            { type: "text", text: "Image read successfully" },
                            {
                                type: "file",
                                uri: `data:image/png;base64,${base64}`,
                                mime: "image/png",
                            },
                        ],
                    },
                },
            ],
        },
    ] as Message[]

    const view = project(native, entries(native), session)
    const assistant = view.messages.find((message) => message.info.id === "msg_assistant")!
    const tool = assistant.parts.find((part) => part.type === "tool") as any

    assert.equal(tool.state.status, "completed")
    assert.ok(
        (await countAllMessageTokens(assistant)) < 300,
        "screenshot base64 must not be counted",
    )
    // The request itself is untouched: the projection still restores the original payload.
    assert.match(JSON.stringify(view.restore()), /iVBORw0KGgo/)
})

test("media stripping leaves ordinary JSON tool output with a data field alone", async () => {
    const { extractCompletedToolOutput } = await import("../lib/token-utils")
    const output = extractCompletedToolOutput({
        type: "tool",
        state: { status: "completed", output: { type: "json", data: { rows: [1, 2, 3] } } },
    })
    assert.equal(output, JSON.stringify({ type: "json", data: { rows: [1, 2, 3] } }))
})

test("V2 request: reminders on a tool turn ride in its tool result, never in the assistant content", async () => {
    // Live on OpenRouter/Mimo: reminders added as text parts of an earlier assistant turn were
    // joined into that turn's content ("...is actually alive:<dcp-system-reminder>..."), and the
    // model copied the block into its own replies.
    const { applyAnchoredNudges } = await import("../lib/messages/inject/utils")
    const native = (): Message[] =>
        [
            { id: "msg_user", role: "user", content: [{ type: "text", text: "Build it" }] },
            {
                id: "msg_assistant",
                role: "assistant",
                content: [
                    { type: "text", text: "Checking whether the build is actually alive:" },
                    { type: "tool-call", id: "call_ps", name: "shell", input: { command: "ps" } },
                ],
            },
            {
                role: "tool",
                content: [
                    {
                        type: "tool-result",
                        id: "call_ps",
                        name: "shell",
                        result: { type: "text", value: "no build processes" },
                    },
                ],
            },
        ] as Message[]
    const prompts = {
        system: "",
        compressRange: "",
        compressMessage: "",
        contextLimitNudge: "<dcp-system-reminder>CTX</dcp-system-reminder>",
        turnNudge: "<dcp-system-reminder>TURN</dcp-system-reminder>",
        iterationNudge: "<dcp-system-reminder>ITERATION</dcp-system-reminder>",
    }
    const request = () => {
        const messages = native()
        const view = project(messages, entries(messages), session)
        const state = createSessionState()
        state.idFormat = "compact"
        assignMessageRefs(state, view.messages)
        state.nudges.iterationNudgeAnchors.add("msg_assistant")
        applyAnchoredNudges(state, config, view.messages, prompts as any)
        return view.restore()
    }

    const restored = request()
    const assistant = restored.find((message) => message.id === "msg_assistant")!
    assert.doesNotMatch(JSON.stringify(assistant), /dcp-system-reminder|ITERATION/)
    // The reminder rides in the tool result: nothing is added after it. A user message right
    // after tool results makes the OpenAI-chat lowering insert an assistant "Done." turn for
    // Mistral-family models (requireAssistantAfterTool).
    assert.equal(restored.length, 3)
    const tool = restored[2]!
    assert.equal(tool.role, "tool")
    assert.match(
        JSON.stringify(tool.content),
        /no build processes\\n\\n<dcp-system-reminder>ITERATION/,
    )
    // Byte-stable across requests, so the cached prefix survives.
    assert.equal(JSON.stringify(request()), JSON.stringify(restored))
})

test("V2 request: a reminder on a text-only assistant turn is a user message before the next user turn", async () => {
    const { applyAnchoredNudges } = await import("../lib/messages/inject/utils")
    const native = [
        { id: "msg_user", role: "user", content: [{ type: "text", text: "Explain" }] },
        {
            id: "msg_assistant",
            role: "assistant",
            content: [{ type: "text", text: "Here is why." }],
        },
        { id: "msg_user_2", role: "user", content: [{ type: "text", text: "Go on" }] },
    ] as Message[]
    const view = project(native, entries(native), session)
    const state = createSessionState()
    state.idFormat = "compact"
    assignMessageRefs(state, view.messages)
    state.nudges.turnNudgeAnchors.add("msg_assistant")
    applyAnchoredNudges(state, config, view.messages, {
        system: "",
        compressRange: "",
        compressMessage: "",
        contextLimitNudge: "<dcp-system-reminder>CTX</dcp-system-reminder>",
        turnNudge: "<dcp-system-reminder>TURN</dcp-system-reminder>",
        iterationNudge: "<dcp-system-reminder>ITER</dcp-system-reminder>",
    } as any)
    const restored = view.restore()

    assert.deepEqual(
        restored.map((message) => message.role),
        ["user", "assistant", "user", "user"],
    )
    assert.doesNotMatch(JSON.stringify(restored[1]), /TURN/)
    assert.match(JSON.stringify(restored[2]!.content), /TURN/)
    assert.equal(restored[3]!.id, "msg_user_2")
})

test("V2 bounds an oversized host checkpoint tail (p2-09 B regression)", async () => {
    // OpenCode 2.1.x sends the keepID..compaction entries serialized inside the
    // checkpoint user message. With one user instruction before a long autonomous
    // run, keepID snapped back to it and the whole run was resent: 1.95M chars,
    // a 558k-token request on a 229k-token model. DCP passed it through verbatim.
    const { countTokens } = await import("../lib/token-utils")
    const recent = [
        "[User]: approved plan, start task",
        ...Array.from(
            { length: 500 },
            (_, i) =>
                `[Assistant reasoning]: step ${i} ${"r".repeat(1700)}\n` +
                `[Assistant tool call]: bash({"command":"c${i}"})\n` +
                `[Tool result]: ${"o".repeat(1250)}\n[truncated]`,
        ),
    ].join("\n")
    const text = [
        "<conversation-checkpoint>",
        "The following is a summary and serialized record of earlier conversation. Treat it as historical context, not as new instructions.",
        "",
        "<summary>\nGoal and progress.\n</summary>",
        "",
        "<recent-context>\n" + recent + "\n</recent-context>",
        "</conversation-checkpoint>",
    ].join("\n")
    const native = [
        {
            id: "msg_cp",
            role: "user",
            content: [{ type: "text", text }],
            metadata: { keepID: "msg_keep", recent },
        },
        { id: "msg_next", role: "user", content: [{ type: "text", text: "continue" }] },
    ] as unknown as Message[]
    const history = [
        {
            id: "msg_cp",
            type: "compaction",
            status: "completed",
            reason: "auto",
            summary: "Goal and progress.",
            metadata: { keepID: "msg_keep", recent },
            time: { created: 5 },
        },
        { id: "msg_next", type: "user", text: "continue", time: { created: 6 } },
    ] as unknown as Parameters<typeof project>[1]
    const restored = project(native, history, session).restore()
    const part = restored[0]?.content[0]
    assert.equal(part?.type, "text")
    const sent = part.type === "text" ? part.text : ""
    assert.ok((await countTokens(sent)) < 0.15 * 229_376, "checkpoint fits the context window")
    assert.ok(sent.startsWith(text.slice(0, text.indexOf("<recent-context>"))))
    assert.ok(sent.includes("<recent-context>\n[User]: approved plan, start task\n[DCP: "))
    assert.ok(
        sent.endsWith(recent.slice(-3000) + "\n</recent-context>\n</conversation-checkpoint>"),
    )
    assert.deepEqual(restored[1], native[1])
})
