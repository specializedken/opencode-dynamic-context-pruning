import { SessionState, WithParts } from "./state"
import { AssistantMessage, UserMessage } from "@opencode-ai/sdk/v2"
import { Logger } from "./logger"
import { tokenCounter } from "./token-counter"
import { getLastUserMessage } from "./messages/query"

export async function getCurrentTokenUsage(
    state: SessionState,
    messages: WithParts[],
): Promise<number> {
    let foundTokens = false
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        if (msg.info.role !== "assistant") {
            continue
        }

        const assistantInfo = msg.info as AssistantMessage
        if ((assistantInfo.tokens?.output || 0) <= 0) {
            continue
        }
        foundTokens = true
        foundTokens = true

        if (
            state.lastCompaction > 0 &&
            (msg.info.time.created < state.lastCompaction ||
                (msg.info.summary === true && msg.info.time.created === state.lastCompaction))
        ) {
            return 0
        }

        const input = assistantInfo.tokens?.input || 0
        const output = assistantInfo.tokens?.output || 0
        const reasoning = assistantInfo.tokens?.reasoning || 0
        const cacheRead = assistantInfo.tokens?.cache?.read || 0
        const cacheWrite = assistantInfo.tokens?.cache?.write || 0
        const reportedUsage = input + output + reasoning + cacheRead + cacheWrite
        return Math.max(reportedUsage, await estimateCurrentMessageTokenUsage(state, messages))
    }

    if (foundTokens) {
        return 0
    }

    // No assistant message carries token accounting (some v2 storage shapes
    // omit `tokens`). Without a number the nudge thresholds never trip and
    // DCP never asks the model to compress. Fall back to a content-based
    // estimate; it is a lower bound, which is the safe direction for nudging.
    return estimateCurrentMessageTokenUsage(state, messages)
}

async function estimateCurrentMessageTokenUsage(
    state: SessionState,
    messages: WithParts[],
): Promise<number> {
    // Counting the whole window is the expensive part of a request, so reuse the
    // result while the window is unchanged.
    const lastMessage = messages[messages.length - 1]
    const key = `${messages.length}:${lastMessage?.info.id ?? ""}:${lastMessage?.parts?.length ?? 0}:${state.lastCompaction}`
    if (state.tokenUsageEstimate?.key === key) {
        return state.tokenUsageEstimate.value
    }

    let total = 0

    for (const msg of messages) {
        if (
            state.lastCompaction > 0 &&
            (msg.info.time.created < state.lastCompaction ||
                (msg.info.summary === true && msg.info.time.created === state.lastCompaction))
        ) {
            continue
        }

        total += await countAllMessageTokens(msg)
    }

    state.tokenUsageEstimate = { key, value: total }
    return total
}

export function getCurrentParams(
    state: SessionState,
    messages: WithParts[],
    logger: Logger,
): {
    providerId: string | undefined
    modelId: string | undefined
    agent: string | undefined
    variant: string | undefined
} {
    const userMsg = getLastUserMessage(messages)
    if (!userMsg) {
        logger.debug("No user message found when determining current params")
        return {
            providerId: undefined,
            modelId: undefined,
            agent: undefined,
            variant: undefined,
        }
    }
    const userInfo = userMsg.info as UserMessage
    const agent: string = userInfo.agent
    const providerId: string | undefined = userInfo.model.providerID
    const modelId: string | undefined = userInfo.model.modelID
    const variant: string | undefined = userInfo.model.variant

    return { providerId, modelId, agent, variant }
}

/** Exact count off the host event loop; rejects with TokenCountError instead of estimating. */
export async function countTokens(text: string): Promise<number> {
    if (!text) return 0
    return tokenCounter.count(text)
}

export async function estimateTokensBatch(texts: string[]): Promise<number> {
    if (texts.length === 0) return 0
    return countTokens(texts.join(" "))
}

export const COMPACTED_TOOL_OUTPUT_PLACEHOLDER = "[Old tool result content cleared]"

function stringifyToolContent(value: unknown): string {
    return typeof value === "string" ? value : JSON.stringify(value)
}

const MEDIA_PAYLOAD_KEYS = new Set(["uri", "url", "data"])
const MEDIA_PART_TYPES = new Set(["file", "media", "image"])

function withoutPayload(item: unknown): unknown {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
        return item
    }
    const record = item as Record<string, unknown>
    const isMedia =
        MEDIA_PART_TYPES.has(record.type as string) ||
        typeof record.mime === "string" ||
        typeof record.mediaType === "string"
    if (!isMedia) {
        return item
    }
    return Object.fromEntries(
        Object.entries(record).filter(([key]) => !MEDIA_PAYLOAD_KEYS.has(key)),
    )
}

// V2 tool results can carry media as data URIs in exactly two shapes: a content array
// (session history) and { type: "content", value: [...] } (request view). Providers send those
// parts as media and bill an image at a small fixed cost, so tokenizing the base64 as text
// inflated usage by hundreds of thousands of tokens. Strip payloads from media parts in those
// two places only: json/text/error results are sent as text and must be counted as they are.
function withoutMediaPayloads(value: unknown): unknown {
    if (Array.isArray(value)) {
        return value.map(withoutPayload)
    }
    const record = value as { type?: unknown; value?: unknown } | null
    if (
        record &&
        typeof record === "object" &&
        record.type === "content" &&
        Array.isArray(record.value)
    ) {
        return { ...record, value: record.value.map(withoutPayload) }
    }
    return value
}

/** Text form of a tool output as the model receives it, with media payloads left out. */
export function stringifyToolOutput(output: unknown): string {
    return stringifyToolContent(withoutMediaPayloads(output))
}

export function extractCompletedToolOutput(part: any): string | undefined {
    if (
        part?.type !== "tool" ||
        part.state?.status !== "completed" ||
        part.state?.output === undefined
    ) {
        return undefined
    }

    if (part.state?.time?.compacted) {
        return COMPACTED_TOOL_OUTPUT_PLACEHOLDER
    }

    return stringifyToolOutput(part.state.output)
}

export function extractToolContent(part: any): string[] {
    const contents: string[] = []

    if (part?.type !== "tool") {
        return contents
    }

    if (part.state?.input !== undefined) {
        contents.push(stringifyToolContent(part.state.input))
    }

    const completedOutput = extractCompletedToolOutput(part)
    if (completedOutput !== undefined) {
        contents.push(completedOutput)
    } else if (part.state?.status === "error" && part.state?.error) {
        contents.push(stringifyToolContent(part.state.error))
    }

    return contents
}

export async function countToolTokens(part: any): Promise<number> {
    const contents = extractToolContent(part)
    return estimateTokensBatch(contents)
}

export function getTotalToolTokens(state: SessionState, toolIds: string[]): number {
    let total = 0
    for (const id of toolIds) {
        const entry = state.toolParameters.get(id)
        total += entry?.tokenCount ?? 0
    }
    return total
}

export async function countMessageTextTokens(msg: WithParts): Promise<number> {
    const texts: string[] = []
    const parts = Array.isArray(msg.parts) ? msg.parts : []
    for (const part of parts) {
        if (part.type === "text") {
            texts.push(part.text)
        }
    }
    if (texts.length === 0) return 0
    return estimateTokensBatch(texts)
}

export async function countAllMessageTokens(msg: WithParts): Promise<number> {
    const parts = Array.isArray(msg.parts) ? msg.parts : []
    const texts: string[] = []
    for (const part of parts) {
        if (part.type === "text") {
            texts.push(part.text)
        } else {
            texts.push(...extractToolContent(part))
        }
    }
    if (texts.length === 0) return 0
    return estimateTokensBatch(texts)
}
