import assert from "node:assert/strict"
import test from "node:test"
import { appendProtectedTools } from "../lib/compress/protected-content"
import { buildSearchContext, resolveSelection } from "../lib/compress/search"
import { createSessionState, type WithParts } from "../lib/state"
import { extractCompletedToolOutput } from "../lib/token-utils"

const BASE64 = "iVBORw0KGgo".repeat(20000)

function toolPart(output: unknown) {
    return {
        type: "tool",
        tool: "read",
        callID: "call-shot",
        state: { status: "completed", input: { filePath: "/tmp/shot.png" }, output },
    }
}

test("json results are counted exactly as sent, even when they contain media-looking objects", () => {
    // Providers send non-content results as text, so nothing inside them may be dropped.
    const output = {
        type: "json",
        value: {
            type: "image",
            data: "A".repeat(5000),
            mime: "image/png",
            files: [{ type: "file", url: "u" }],
        },
    }
    assert.equal(extractCompletedToolOutput(toolPart(output)), JSON.stringify(output))
})

test("only media parts of content results lose their payload", () => {
    const output = {
        type: "content",
        value: [
            { type: "text", text: "Image read successfully" },
            {
                type: "file",
                uri: `data:image/png;base64,${BASE64}`,
                mime: "image/png",
                name: "shot.png",
            },
        ],
    }
    const text = extractCompletedToolOutput(toolPart(output)) ?? ""
    assert.doesNotMatch(text, /iVBORw0KGgo/)
    assert.match(text, /Image read successfully/)
    assert.match(text, /"mime":"image\/png"/)
})

test("protected tool outputs pasted into a compress summary leave media payloads out", async () => {
    // The summary is sent to the provider as text, so a pasted data URI would really cost tokens.
    const message = {
        info: { id: "msg-shot", role: "assistant", sessionID: "s", time: { created: 1 } },
        parts: [
            {
                id: "p1",
                messageID: "msg-shot",
                sessionID: "s",
                ...toolPart([
                    { type: "text", text: "Image read successfully" },
                    { type: "file", uri: `data:image/png;base64,${BASE64}`, mime: "image/png" },
                ]),
            },
        ],
    } as unknown as WithParts
    const state = createSessionState()
    const context = buildSearchContext(state, [message])
    const selection = await resolveSelection(
        context,
        { kind: "message", rawIndex: 0, messageId: "msg-shot" },
        { kind: "message", rawIndex: 0, messageId: "msg-shot" },
    )

    const summary = await appendProtectedTools(
        {} as any,
        state,
        false,
        "summary",
        selection,
        context,
        ["read"],
    )

    assert.match(summary, /Image read successfully/)
    assert.doesNotMatch(summary, /iVBORw0KGgo/)
    assert.ok(summary.length < 2000)
})
