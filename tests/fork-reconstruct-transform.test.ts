import assert from "node:assert/strict"
import test from "node:test"
import { mkdirSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { PluginConfig } from "../lib/config"
import type { WithParts } from "../lib/state"

const dataHome = join(tmpdir(), `opencode-dcp-fork-transform-${process.pid}`)
process.env.XDG_DATA_HOME = dataHome
process.env.XDG_CONFIG_HOME = join(dataHome, "config")
mkdirSync(dataHome, { recursive: true })

// Persistence resolves its storage directory at import time, so import after the env is set.
const { createChatMessageTransformHandler } = await import("../lib/hooks")
const { Logger } = await import("../lib/logger")
const { createSessionState } = await import("../lib/state")

const config = {
    enabled: true,
    debug: false,
    pruneNotification: "off",
    pruneNotificationType: "chat",
    commands: { enabled: true, protectedTools: [] },
    manualMode: { enabled: false, automaticStrategies: true },
    turnProtection: { enabled: false, turns: 4 },
    experimental: { allowSubAgents: false, customPrompts: false },
    protectedFilePatterns: [],
    compress: {
        mode: "range",
        permission: "allow",
        showCompression: false,
        maxContextLimit: 150000,
        minContextLimit: 50000,
        nudgeFrequency: 5,
        iterationNudgeThreshold: 15,
        nudgeForce: "soft",
        protectedTools: [],
        protectTags: false,
        protectUserMessages: false,
        summaryBuffer: true,
    },
    strategies: {
        deduplication: { enabled: true, protectedTools: [] },
        purgeErrors: { enabled: true, turns: 4, protectedTools: [] },
    },
} as unknown as PluginConfig

function message(
    sessionID: string,
    id: string,
    role: "user" | "assistant",
    created: number,
    parts: any[],
): WithParts {
    return {
        info: {
            id,
            role,
            sessionID,
            agent: "assistant",
            time: { created },
            ...(role === "user" ? { model: { providerID: "p", modelID: "m" } } : {}),
        } as WithParts["info"],
        parts: parts.map((part, i) => ({ id: `${id}-p${i}`, messageID: id, sessionID, ...part })),
    }
}

function forkedSession(sessionID: string): WithParts[] {
    return [
        message(sessionID, "u1", "user", 1, [{ type: "text", text: "first question" }]),
        message(sessionID, "a1", "assistant", 2, [{ type: "text", text: "first answer" }]),
        message(sessionID, "u2", "user", 3, [{ type: "text", text: "second question" }]),
        message(sessionID, "a2", "assistant", 4, [
            { type: "text", text: "second answer" },
            {
                type: "tool",
                callID: "call-compress",
                tool: "compress",
                state: {
                    status: "completed",
                    input: {
                        topic: "Greeting",
                        content: [{ startId: "m0001", endId: "m0002", summary: "SUMMARY-ONE" }],
                    },
                    output: "Compressed.",
                },
            },
        ]),
        message(sessionID, "u3", "user", 5, [{ type: "text", text: "third question" }]),
    ]
}

const runtimePrompts = Object.fromEntries(
    [
        "system",
        "compressRange",
        "compressMessage",
        "contextLimitNudge",
        "turnNudge",
        "iterationNudge",
        "manualExtension",
        "subagentExtension",
    ].map((key) => [key, `<dcp-system-reminder>${key}</dcp-system-reminder>`]),
)

function buildHandler(state = createSessionState()) {
    const handler = createChatMessageTransformHandler(
        { session: { get: async () => ({}) } } as any,
        state,
        new Logger(false),
        config,
        { reload() {}, getRuntimePrompts: () => runtimePrompts } as any,
        { global: undefined, agents: {} },
    )
    return { handler, state }
}

test("transform rebuilds compression from a forked session's history before pruning", async () => {
    const { handler, state } = buildHandler()
    const output = { messages: forkedSession("ses_fork_transform_1") }

    await handler({}, output)

    assert.equal(state.needsReconstruction, false)
    assert.equal(state.prune.messages.blocksById.size, 1)
    assert.equal(state.prune.messages.activeBlockIds.size, 1)

    const serialized = JSON.stringify(output.messages)
    assert.match(serialized, /SUMMARY-ONE/)
    assert.doesNotMatch(serialized, /first question/)
    assert.doesNotMatch(serialized, /first answer/)
    assert.match(serialized, /third question/)
})

test("transform leaves reconstruction off when history has no compress calls", async () => {
    const { handler, state } = buildHandler()
    const sessionID = "ses_fork_transform_2"
    const output = {
        messages: [
            message(sessionID, "u1", "user", 1, [{ type: "text", text: "hello" }]),
            message(sessionID, "a1", "assistant", 2, [{ type: "text", text: "hi" }]),
        ],
    }

    await handler({}, output)

    assert.equal(state.needsReconstruction, false)
    assert.equal(state.prune.messages.blocksById.size, 0)
})

test("message aliases are not leaked for messages that prune removed", async () => {
    const { handler, state } = buildHandler()
    const output = { messages: forkedSession("ses_fork_transform_3") }

    await handler({}, output)

    // Only messages still in the array (summary anchor, a2's tail, u3) may hold refs
    // beyond the one-time reconstruction pass; a second transform must not add more.
    const aliasCount = state.messageIds.byRawId.size
    await handler({}, { messages: forkedSession("ses_fork_transform_3") })
    assert.equal(state.messageIds.byRawId.size, aliasCount)
})
