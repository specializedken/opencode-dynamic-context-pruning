import assert from "node:assert/strict"
import test from "node:test"
import { stripHallucinations } from "../lib/messages/utils"
import type { WithParts } from "../lib/state"

// Issue #657: hallucination stripping must only clean what DCP injected or the
// model echoed into its own text, never legitimate file contents or user input.

const REMINDER_FILE = "AAA\n\n<dcp-system-reminder>\nHELLO-INSIDE\n</dcp-system-reminder>\n\nZZZ"

function conversation(toolOutput: string): WithParts[] {
    return [
        {
            info: { id: "msg_user", role: "user", sessionID: "ses_657", time: { created: 1 } },
            parts: [{ type: "text", text: `Please review:\n${REMINDER_FILE}` }],
        },
        {
            info: {
                id: "msg_assistant",
                role: "assistant",
                sessionID: "ses_657",
                time: { created: 2 },
            },
            parts: [
                {
                    type: "text",
                    text: "Reading it.\n<dcp-system-reminder>echoed</dcp-system-reminder>",
                },
                {
                    type: "tool",
                    tool: "read",
                    callID: "call_read",
                    state: { status: "completed", input: {}, output: toolOutput },
                },
            ],
        },
    ] as unknown as WithParts[]
}

const userText = (messages: WithParts[]) => (messages[0]!.parts[0] as any).text as string
const assistantText = (messages: WithParts[]) => (messages[1]!.parts[0] as any).text as string
const toolOutput = (messages: WithParts[]) => (messages[1]!.parts[1] as any).state.output as string

test("stripHallucinations keeps <dcp-*> blocks in tool output and user text (issue #657)", () => {
    const messages = conversation(REMINDER_FILE)
    stripHallucinations(messages)
    assert.equal(toolOutput(messages), REMINDER_FILE)
    assert.equal(userText(messages), `Please review:\n${REMINDER_FILE}`)
    assert.equal(assistantText(messages), "Reading it.\n")
})

test("stripHallucinations keeps <dcp-*> blocks in tool output in compact format (issue #657)", () => {
    const messages = conversation(REMINDER_FILE)
    stripHallucinations(messages, "compact")
    assert.equal(toolOutput(messages), REMINDER_FILE)
    assert.equal(userText(messages), `Please review:\n${REMINDER_FILE}`)
    assert.equal(assistantText(messages), "Reading it.\n")
})

test("stripHallucinations still removes DCP's own injected ID tag from tool output", () => {
    const messages = conversation(
        `${REMINDER_FILE}\n<dcp-message-id priority="low">m0007</dcp-message-id>`,
    )
    stripHallucinations(messages)
    assert.equal(toolOutput(messages), REMINDER_FILE)
})
