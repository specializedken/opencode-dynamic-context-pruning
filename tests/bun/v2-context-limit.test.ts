import assert from "node:assert/strict"
import "../lab/persistence-env"
import { test } from "node:test"
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { setup } from "../../lib/v2/index"

/**
 * Percent thresholds must evaluate against the context limit the host serves
 * for the request's model (#658). A plugin `model.transform` callback only sees
 * the registry at its own position in the host's transform chain; the host
 * merges `providers.*.models.*.limit` from opencode.json in a transform of its
 * own, so the editor DCP receives can still carry the 200000 fallback.
 */

type AnyRecord = Record<string, any>

const directory = mkdtempSync(join(tmpdir(), "dcp-v2-context-limit-"))
mkdirSync(join(directory, ".opencode"), { recursive: true })
writeFileSync(
    join(directory, ".opencode", "dcp.jsonc"),
    JSON.stringify({ compress: { maxContextLimit: "75%", minContextLimit: "60%" } }),
)

async function boot(served: () => any[]) {
    let contextHook: ((event: AnyRecord) => Promise<void>) | undefined
    let modelEditor: ((editor: { list: () => any[] }) => void) | undefined
    const entries = (modelID: string) => [
        { id: "msg-1", type: "user", time: { created: 1 }, text: "Start.", content: [] },
        {
            id: "msg-2",
            type: "assistant",
            time: { created: 2 },
            agent: "build",
            model: { id: modelID, providerID: "local" },
            tokens: { input: 160000, output: 100, reasoning: 0, cache: { read: 0, write: 0 } },
            content: [],
        },
        { id: "msg-3", type: "user", time: { created: 3 }, text: "Continue.", content: [] },
    ]
    let currentModel = "big"
    const ctx = {
        location: { directory },
        options: {},
        app: {},
        client: {},
        agent: { get: async ({ agentID }: any) => ({ data: { id: agentID, permissions: [] } }) },
        session: {
            get: async ({ sessionID }: any) => ({
                id: sessionID,
                agent: "build",
                model: { id: currentModel, providerID: "local" },
                permissions: [],
            }),
            context: async () => entries(currentModel),
            hook: async (kind: string, cb: (event: AnyRecord) => Promise<void>) => {
                if (kind === "context") contextHook = cb
                return { dispose: async () => {} }
            },
            prompt: async () => {},
        },
        model: {
            list: async () => ({ location: {}, data: served() }),
            transform: async (cb: (editor: { list: () => any[] }) => void) => {
                modelEditor = cb
                return { dispose: async () => {} }
            },
        },
        tool: { transform: async () => ({ dispose: async () => {} }) },
        command: { transform: async () => ({ dispose: async () => {} }) },
        rpc: { register: async () => {} },
        permission: { hook: async () => ({ dispose: async () => {} }) },
    }
    await setup(ctx as any)
    assert.ok(contextHook)
    return {
        transform: (list: any[]) => modelEditor?.({ list: () => list }),
        request: async (sessionID: string, modelID: string) => {
            currentModel = modelID
            const event = {
                sessionID,
                agent: "build",
                model: { providerID: "local", id: modelID },
                system: [{ type: "text", text: "You are a test agent." }],
                messages: [
                    { id: "msg-1", role: "user", content: [{ type: "text", text: "Start." }] },
                    { id: "msg-2", role: "assistant", content: [{ type: "text", text: "Ok." }] },
                    { id: "msg-3", role: "user", content: [{ type: "text", text: "Continue." }] },
                ],
                tools: {} as AnyRecord,
            }
            await contextHook!(event)
            return JSON.stringify(event.messages)
        },
    }
}

const MAX_NUDGE = "MAX CONTEXT LIMIT REACHED"

test("v2 setup: percent limits use the served registry limit, not a pre-merge transform snapshot", async () => {
    const served = [
        { providerID: "local", id: "big", limit: { context: 1000000, output: 32768 } },
        { providerID: "local", id: "small", limit: { context: 200000, output: 32000 } },
    ]
    const harness = await boot(() => served)
    // The editor DCP sees has not had the opencode.json limit merged yet.
    harness.transform([
        { providerID: "local", id: "big", limit: { context: 200000, output: 32000 } },
        { providerID: "local", id: "small", limit: { context: 200000, output: 32000 } },
    ])

    // 160k used: over 75% of 200k (150k), far below 75% of 1M (750k).
    const big = await harness.request(`ses_limit_big_${Date.now()}`, "big")
    assert.ok(!big.includes(MAX_NUDGE), "1M model must not get a max-limit nudge at 160k")

    // Switching the same session's model re-resolves against the new model.
    const sessionID = `ses_limit_switch_${Date.now()}`
    assert.ok(!(await harness.request(sessionID, "big")).includes(MAX_NUDGE))
    assert.ok(
        (await harness.request(sessionID, "small")).includes(MAX_NUDGE),
        "200k model must get the max-limit nudge at 160k",
    )

    // A registry rebuild that changes the served limit is picked up.
    served[1] = { providerID: "local", id: "small", limit: { context: 229376, output: 32000 } }
    harness.transform(served)
    assert.ok(
        !(await harness.request(`ses_limit_rebuild_${Date.now()}`, "small")).includes(MAX_NUDGE),
        "75% of 229376 (172032) is above 160k",
    )
})
