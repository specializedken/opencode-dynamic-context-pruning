import assert from "node:assert/strict"
import test from "node:test"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { PluginConfig } from "../lib/config"
import { createCommandExecuteHandler, createSystemPromptHandler } from "../lib/hooks"
import { Logger } from "../lib/logger"
import { PromptStore } from "../lib/prompts/store"
import {
    createSessionState,
    loadManualModeSetting,
    saveSessionState,
    type WithParts,
} from "../lib/state"

// /dcp-compress puts the session in "compress-pending" for one request. That must not be
// mistaken for manual mode: rendering the manual-mode system extension changed the system
// prompt (a full prompt-cache miss) and told the model to stop after compressing.

const MANUAL_EXTENSION = /Manual mode is enabled/

function buildConfig(manualMode: boolean): PluginConfig {
    return {
        enabled: true,
        debug: false,
        pruneNotification: "off",
        pruneNotificationType: "chat",
        commands: { enabled: true, protectedTools: [] },
        manualMode: { enabled: manualMode, automaticStrategies: true },
        turnProtection: { enabled: false, turns: 4 },
        experimental: { allowSubAgents: false, customPrompts: false },
        protectedFilePatterns: [],
        compress: {
            mode: "range",
            permission: "allow",
            showCompression: false,
            summaryBuffer: true,
            maxContextLimit: 150000,
            minContextLimit: 50000,
            nudgeFrequency: 5,
            iterationNudgeThreshold: 15,
            nudgeForce: "soft",
            protectedTools: [],
            protectTags: false,
            protectUserMessages: false,
        },
        strategies: {
            deduplication: { enabled: true, protectedTools: [] },
            purgeErrors: { enabled: true, turns: 4, protectedTools: [] },
        },
    } as unknown as PluginConfig
}

function message(sessionID: string, id: string, role: "user" | "assistant", text: string) {
    return {
        info: {
            id,
            role,
            sessionID,
            agent: "build",
            time: { created: 1 },
            model: { providerID: "p", modelID: "m" },
        },
        parts: [{ id: `${id}-p`, messageID: id, sessionID, type: "text", text }],
    } as unknown as WithParts
}

async function triggerCompress(manualMode: boolean, sessionID: string) {
    const logger = new Logger(false)
    const state = createSessionState()
    const config = buildConfig(manualMode)
    const prompts = new PromptStore(logger, mkdtempSync(join(tmpdir(), "dcp-manual-")), false)
    const messages = [
        message(sessionID, "u1", "user", "first"),
        message(sessionID, "a1", "assistant", "answer"),
    ]
    const client = {
        session: {
            get: async () => ({ data: {} }),
            messages: async () => ({ data: messages }),
        },
    }
    const system = createSystemPromptHandler(state, logger, config, prompts)
    const render = async () => {
        const output = { system: ["You are a test agent."] }
        await system({ sessionID, model: { limit: { context: 200000 } } }, output)
        return output.system.join("\n")
    }
    const execute = createCommandExecuteHandler(client, state, logger, config, "/tmp", {
        global: undefined,
        agents: {},
    })

    // Initialise the session through the command path, then capture the steady-state prompt.
    const before = await (async () => {
        const output = { parts: [] as any[] }
        await execute({ command: "dcp", sessionID, arguments: "stats" }, output)
        return render()
    })()
    await execute({ command: "dcp-compress", sessionID, arguments: "" }, { parts: [] })
    assert.equal(state.manualMode, "compress-pending")
    const during = await render()
    return { state, logger, before, during }
}

test("/dcp-compress outside manual mode keeps the system prompt byte-identical", async () => {
    const { before, during } = await triggerCompress(false, `ses_trigger_auto_${Date.now()}`)

    assert.doesNotMatch(during, MANUAL_EXTENSION)
    assert.equal(during, before)
})

test("/dcp-compress in manual mode keeps the manual-mode extension", async () => {
    const { before, during } = await triggerCompress(true, `ses_trigger_manual_${Date.now()}`)

    assert.match(before, MANUAL_EXTENSION)
    assert.match(during, MANUAL_EXTENSION)
})

test("a state save during a pending trigger persists the real manual mode", async () => {
    const sessionID = `ses_trigger_persist_${Date.now()}`
    const { state, logger } = await triggerCompress(false, sessionID)

    // V2 writes block-liveness changes through on the trigger request itself.
    await saveSessionState(state, logger)

    assert.equal(await loadManualModeSetting(sessionID, logger), false)
})
