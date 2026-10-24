import assert from "node:assert/strict"
import "../lab/persistence-env"
import { test } from "node:test"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import type { Message } from "@opencode/ai/schema/messages"
import { setup } from "../../lib/v2/index"
import { COMPRESSED_BLOCK_HEADER } from "../../lib/compress/state"
import { stateDir } from "../lab/persistence-env"

/**
 * Fake-host integration test for the real V2 `setup()` seam — the layer no
 * other test exercises. It captures the host-registered handlers from a mocked
 * plugin context, fires the session "context" hook the way the host does,
 * invokes the compress tool the way the host does, and fires the hook
 * again. The assertion is that the message array the host would send to the
 * model actually SHRINKS and contains the compressed block.
 *
 * It also covers the restart/reload edge: a second setup() instance (simulating
 * a service restart) must restore blocks from disk and keep compressing, and a
 * corrupt state file must never be overwritten by a write-through save.
 */

const PAD_A = "MARKER-A deep implementation detail worth compressing. ".repeat(30)
const PAD_B = "MARKER-B later analysis that must survive compression. ".repeat(30)

type AnyRecord = Record<string, any>

interface Harness {
    contextHook: (event: AnyRecord) => Promise<void>
    toolAdd: AnyRecord
    permissionHook: ((evaluation: AnyRecord) => void) | undefined
    modelLimits: (list: any[]) => void
}

async function boot(
    entries: () => any[] = entriesFixture,
    host: <T>(value: T) => T = (value) => value,
): Promise<Harness> {
    // Every setup() call builds fresh closures and state maps — the same
    // lifecycle a plugin reload after a service restart goes through.
    let contextHook: ((event: AnyRecord) => Promise<void>) | undefined
    let toolAdd: AnyRecord | undefined
    let permissionHook: ((evaluation: AnyRecord) => void) | undefined
    let modelEditor: ((editor: { list: () => any[] }) => void) | undefined

    const agentPermissions: any[] = []
    const sessionRow = (sessionID: string) => ({
        id: sessionID,
        agent: "build",
        model: { id: "test", providerID: "test" },
        permissions: [],
    })

    const ctx = {
        location: { directory: "/tmp/dcp-v2-setup-test" },
        options: {},
        app: {},
        client: {},
        agent: {
            get: async ({ agentID }: any) => ({
                data: host({ id: agentID, permissions: agentPermissions }),
            }),
        },
        session: {
            get: async ({ sessionID }: any) => host(sessionRow(sessionID)),
            context: async () => host(entries()),
            hook: async (kind: string, cb: (event: AnyRecord) => Promise<void>) => {
                if (kind === "context") contextHook = cb
                return { dispose: async () => {} }
            },
            prompt: async () => {},
        },
        model: {
            transform: async (cb: (editor: { list: () => any[] }) => void) => {
                modelEditor = cb
                return { dispose: async () => {} }
            },
        },
        tool: {
            transform: async (cb: (editor: { add: (def: AnyRecord) => void }) => void) => {
                cb({ add: (def: AnyRecord) => (toolAdd = def) })
                return { dispose: async () => {} }
            },
        },
        command: {
            transform: async () => ({ dispose: async () => {} }),
        },
        rpc: { register: async () => {} },
        permission: {
            hook: async (name: string, cb: (evaluation: AnyRecord) => void) => {
                if (name === "evaluate") permissionHook = cb
                return { dispose: async () => {} }
            },
        },
    }

    await setup(ctx as any)
    assert.ok(contextHook, "setup must register the session context hook")
    assert.ok(toolAdd, "setup must register the compress tool")

    return {
        contextHook: contextHook!,
        toolAdd: toolAdd!,
        permissionHook,
        modelLimits: (list) => modelEditor?.({ list: () => list }),
    }
}

function entriesFixture(): any[] {
    return [
        {
            id: "msg-1",
            type: "user",
            time: { created: 1 },
            text: "Please investigate the bug.",
            content: [],
        },
        {
            id: "msg-2",
            type: "assistant",
            time: { created: 2 },
            agent: "build",
            model: { id: "test", providerID: "test" },
            content: [],
        },
        {
            id: "msg-3",
            type: "user",
            time: { created: 3 },
            text: "What did you find?",
            content: [],
        },
        {
            id: "msg-4",
            type: "assistant",
            time: { created: 4 },
            agent: "build",
            model: { id: "test", providerID: "test" },
            content: [],
        },
    ]
}

function nativeTranscript(): Message[] {
    return [
        {
            id: "msg-1",
            role: "user",
            content: [{ type: "text", text: "Please investigate the bug." }],
        },
        { id: "msg-2", role: "assistant", content: [{ type: "text", text: PAD_A }] },
        { id: "msg-3", role: "user", content: [{ type: "text", text: "What did you find?" }] },
        { id: "msg-4", role: "assistant", content: [{ type: "text", text: PAD_B }] },
    ] as Message[]
}

function hostEvent(sessionID: string) {
    return {
        sessionID,
        agent: "build",
        model: { providerID: "test", id: "test" },
        system: [{ type: "text", text: "You are a test agent." }],
        messages: nativeTranscript(),
        tools: {} as AnyRecord,
    }
}

test("v2 setup: hook exposes compress on an empty tool set and the compress tool call shrinks the next request", async () => {
    const sessionID = `ses_v2setup_${Date.now()}`
    const harness = await boot()
    harness.modelLimits([{ providerID: "test", id: "test", limit: { context: 200000 } }])

    // Request N: empty tool set — exactly the "host dropped the tool" case.
    const eventN = hostEvent(sessionID)
    await harness.contextHook(eventN)

    // Self-heal: the tool must now be in the request the model sees.
    const exposed = eventN.tools["compress"]
    assert.ok(exposed, "compress must be (re-)added to event.tools")
    assert.equal(typeof exposed.description, "string")
    assert.equal(exposed.input?.type, "object")
    assert.ok(JSON.stringify(exposed.input).includes("startId"), "schema must describe range args")

    // Host permission evaluation: unmatched compress action must be forced to
    // allow (v2 default is "ask"), other actions untouched.
    assert.ok(harness.permissionHook, "setup must register the permission evaluate hook")
    const evaluation: AnyRecord = {
        sessionID,
        action: "compress",
        resources: ["compress"],
        effect: "ask",
    }
    harness.permissionHook!(evaluation)
    assert.equal(evaluation.effect, "allow")
    const other: AnyRecord = { sessionID, action: "bash", resources: ["*"], effect: "ask" }
    harness.permissionHook!(other)
    assert.equal(other.effect, "ask")

    // The model calls the tool between requests.
    const result = await harness.toolAdd.execute(
        {
            topic: "V2 setup e2e",
            content: [
                {
                    startId: "@1@",
                    endId: "@2@",
                    summary: "The padded opening investigation and its key findings.",
                },
            ],
        },
        { sessionID, agent: "build", messageID: "msg-tool", id: "call_1", progress: () => {} },
    )
    assert.match(String(result.content), /Compressed 2 messages/)

    // The block persisted to disk — the artifact that survives restarts.
    const stateFile = join(stateDir, `${sessionID}.json`)
    assert.ok(existsSync(stateFile), "state file must exist after compression")
    const persisted = JSON.parse(readFileSync(stateFile, "utf-8"))
    assert.equal(Object.keys(persisted.prune.messages.blocksById).length, 1)

    // Request N+1: fresh host event with the pristine transcript — the
    // transform must replace the compressed range with the summary block.
    const eventN1 = hostEvent(sessionID)
    await harness.contextHook(eventN1)
    const restoredJson = JSON.stringify(eventN1.messages)
    assert.ok(
        eventN1.messages.length < eventN.messages.length,
        `expected fewer messages, got ${eventN.messages.length} -> ${eventN1.messages.length}`,
    )
    assert.ok(restoredJson.includes(COMPRESSED_BLOCK_HEADER), "summary block must be present")
    assert.match(restoredJson, /padded opening investigation/, "summary text must be present")
    assert.ok(!restoredJson.includes("MARKER-A"), "compressed range content must be gone")
    assert.ok(restoredJson.includes("MARKER-B"), "content after the range must survive")
})

test("v2 setup: system prompt injection keeps every event.system entry a well-formed SystemPart (#640)", async () => {
    const harness = await boot()
    harness.modelLimits([{ providerID: "test", id: "test", limit: { context: 200000 } }])

    // Populated system: DCP must append to the last part's text, never coerce
    // the object to "[object Object]" and never replace extra part fields.
    const populated: AnyRecord = hostEvent(`ses_v2system_${Date.now()}`)
    populated.system = [
        { type: "text", text: "First part." },
        { type: "text", text: "You are a test agent.", cache: { type: "ephemeral" } },
    ]
    await harness.contextHook(populated)
    assert.equal(populated.system.length, 2)
    for (const part of populated.system) {
        assert.equal(typeof part, "object", "system entries must be SystemPart objects")
        assert.equal(part.type, "text")
        assert.equal(typeof part.text, "string")
        assert.ok(!part.text.includes("[object Object]"))
    }
    assert.equal(populated.system[0].text, "First part.")
    assert.ok(populated.system[1].text.startsWith("You are a test agent.\n\n"))
    assert.ok(populated.system[1].text.length > "You are a test agent.\n\n".length)
    assert.deepEqual(populated.system[1].cache, { type: "ephemeral" })

    // Empty system: the injected prompt must arrive as a SystemPart, not a bare string.
    const empty: AnyRecord = hostEvent(`ses_v2system_empty_${Date.now()}`)
    empty.system = []
    await harness.contextHook(empty)
    assert.equal(empty.system.length, 1)
    assert.equal(typeof empty.system[0], "object", "injected prompt must not be a bare string")
    assert.equal(empty.system[0].type, "text")
    assert.ok(empty.system[0].text.length > 0)
})

test("v2 setup: compression blocks survive a plugin reload (service restart)", async () => {
    const sessionID = `ses_v2reload_${Date.now()}`
    const first = await boot()
    first.modelLimits([{ providerID: "test", id: "test", limit: { context: 200000 } }])

    await first.contextHook(hostEvent(sessionID))
    const result = await first.toolAdd.execute(
        {
            topic: "Reload e2e",
            content: [{ startId: "@1@", endId: "@2@", summary: "Compressed opening context." }],
        },
        { sessionID, agent: "build", messageID: "msg-tool", id: "call_1", progress: () => {} },
    )
    assert.match(String(result.content), /Compressed 2 messages/)

    // Simulate the service restarting: a brand-new setup() instance, new state
    // maps, blocks restored from disk.
    const second = await boot()
    second.modelLimits([{ providerID: "test", id: "test", limit: { context: 200000 } }])
    const event = hostEvent(sessionID)
    await second.contextHook(event)
    const restoredJson = JSON.stringify(event.messages)
    assert.ok(restoredJson.includes(COMPRESSED_BLOCK_HEADER), "block must survive the reload")
    assert.ok(!restoredJson.includes("MARKER-A"), "range must stay compressed after reload")
    assert.ok(restoredJson.includes("MARKER-B"), "tail content must survive the reload")
})

test("v2 setup: a corrupt state file is never overwritten by a write-through save", async () => {
    const sessionID = `ses_v2corrupt_${Date.now()}`
    const stateFile = join(stateDir, `${sessionID}.json`)
    const garbage = "{ this is not valid json"

    // Seed the corrupt file BEFORE the first load, like a crashed write or an
    // old foreign-format file would.
    mkdirSync(stateDir, { recursive: true })
    writeFileSync(stateFile, garbage, "utf-8")

    const harness = await boot()
    harness.modelLimits([{ providerID: "test", id: "test", limit: { context: 200000 } }])
    await harness.contextHook(hostEvent(sessionID))
    await harness.toolAdd.execute(
        {
            topic: "Corrupt guard e2e",
            content: [
                { startId: "@1@", endId: "@2@", summary: "Should not persist over garbage." },
            ],
        },
        { sessionID, agent: "build", messageID: "msg-tool", id: "call_1", progress: () => {} },
    )

    assert.equal(
        readFileSync(stateFile, "utf-8"),
        garbage,
        "write-through saves must not destroy the unreadable file",
    )
})

function deepFreeze<T>(value: T): T {
    if (value && typeof value === "object" && !Object.isFrozen(value)) {
        Object.freeze(value)
        for (const child of Object.values(value)) deepFreeze(child)
    }
    return value
}

test("v2 setup: compress and the context hook never write to frozen host objects (#654)", async () => {
    // OpenCode V2 hands plugins readonly messages, parts and tool args; under
    // Bun (strict mode) any in-place assignment throws "Attempted to assign to
    // readonly property". Freeze everything the host owns, including a nested
    // question-tool input that pruning rewrites, and run a full cycle.
    const sessionID = `ses_v2frozen_${Date.now()}`
    const question = () => ({
        questions: [{ question: "Which approach?", options: [{ label: "A" }, { label: "B" }] }],
    })
    const entries = () => [
        ...entriesFixture().slice(0, 3),
        {
            id: "msg-4",
            type: "assistant",
            time: { created: 4 },
            agent: "build",
            model: { id: "test", providerID: "test" },
            content: [
                { type: "text", text: PAD_B },
                {
                    type: "tool",
                    id: "call_q",
                    name: "question",
                    state: {
                        status: "completed",
                        input: question(),
                        content: [{ type: "text", text: "User picked A" }],
                        metadata: { answers: [["A"]] },
                    },
                },
            ],
        },
    ]
    const event = () => ({
        ...hostEvent(sessionID),
        messages: deepFreeze([
            ...nativeTranscript().slice(0, 3),
            {
                id: "msg-4",
                role: "assistant",
                content: [
                    { type: "text", text: PAD_B },
                    { type: "tool-call", id: "call_q", name: "question", input: question() },
                ],
            },
            {
                role: "tool",
                content: [
                    {
                        type: "tool-result",
                        id: "call_q",
                        name: "question",
                        result: { type: "text", value: "User picked A" },
                    },
                ],
            },
        ]) as unknown as Message[],
    })

    const harness = await boot(entries, deepFreeze)
    harness.modelLimits([{ providerID: "test", id: "test", limit: { context: 200000 } }])

    await harness.contextHook(event())
    const result = await harness.toolAdd.execute(
        deepFreeze({
            topic: "Frozen host e2e",
            content: [{ startId: "@1@", endId: "@2@", summary: "Frozen opening summary." }],
        }),
        { sessionID, agent: "build", messageID: "msg-tool", id: "call_1", progress: () => {} },
    )
    assert.match(String(result.content), /Compressed 2 messages/)

    const next = event()
    await harness.contextHook(next)
    const json = JSON.stringify(next.messages)
    assert.ok(json.includes("Frozen opening summary."), "summary block must be present")
    assert.ok(!json.includes("MARKER-A"), "compressed range content must be gone")
    assert.ok(json.includes("MARKER-B"), "tail content must survive")
})
