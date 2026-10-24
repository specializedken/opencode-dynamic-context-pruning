import type { SessionState, WithParts } from "../../state"
import type { PluginConfig } from "../../config"
import { isMessageCompacted } from "../../state/utils"
import { isIgnoredUserMessage, isProtectedUserMessage } from "../../messages/query"
import { formatBlockRef } from "../../message-ids"
import { isToolProtected } from "../../protected-patterns"

function formatTokenEstimate(tokens: number): string {
    if (tokens >= 1000) {
        return `~${(tokens / 1000).toFixed(1)}k`
    }
    return `~${tokens}`
}

/**
 * Guidance for compressed blocks: which blocks are active, the placeholder
 * syntax for ranges that span them, and the fact that spanning a block never
 * shrinks it.
 */
export function buildCompressedBlockGuidance(state: SessionState): string {
    const blocksById = state.prune.messages.blocksById
    // activeBlockIds are persisted and restored from disk; intersect with
    // blocksById so a block missing from disk (corrupt/older state file) is
    // not listed with a bogus "~0 tokens" cost.
    const activeIds = Array.from(state.prune.messages.activeBlockIds)
        .filter((id) => Number.isInteger(id) && id > 0 && blocksById.has(id))
        .sort((a, b) => a - b)
    const refs = activeIds.map((id) => formatBlockRef(id, state.idFormat))
    const blockCount = refs.length
    const blockList = blockCount > 0 ? refs.join(", ") : "none"
    const placeholder = state.idFormat === "compact" ? "@b1@" : "(bN)"

    const lines = [
        "Compressed block context:",
        `- Active compressed blocks in this session: ${blockCount} (${blockList})`,
    ]

    if (blockCount === 0) {
        lines.push(
            `- If your selected compression range includes any listed block, include each required placeholder exactly once in the summary using \`${placeholder}\`.`,
        )
        return lines.join("\n")
    }

    // Block summaries cannot be shrunk by compressing over them: every block inside a new
    // range is carried into the new summary verbatim (placeholders are expanded, missing
    // ones re-appended). Listing their token cost and suggesting merges led models to
    // restart ranges at the first block, which discards the cached prompt prefix for no
    // saving, so only state how blocks behave.
    lines.push(
        "- Prefer compressing closed raw messages after the latest block: their text and tool outputs leave context entirely.",
    )
    lines.push(
        "- Including an earlier block in a range does not shrink it: its summary is carried into the new summary verbatim. A range that starts at an old block rewrites everything after it and discards the cached prompt, so only include blocks a range naturally spans.",
    )
    lines.push(
        `- If your selected compression range includes any listed block, include each required placeholder exactly once in the summary using \`${placeholder}\`.`,
    )
    return lines.join("\n")
}

export function renderMessagePriorityGuidance(priorityLabel: string, refs: string[]): string {
    const refList = refs.length > 0 ? refs.join(", ") : "none"

    return [
        "Message priority context:",
        "- Higher-priority older messages consume more context and should be compressed right away if it is safe to do so.",
        `- ${priorityLabel}-priority message IDs before this point: ${refList}`,
    ].join("\n")
}

export function appendGuidanceToDcpTag(nudgeText: string, guidance: string): string {
    if (!guidance.trim()) {
        return nudgeText
    }

    const closeTag = "</dcp-system-reminder>"
    const closeTagIndex = nudgeText.lastIndexOf(closeTag)

    if (closeTagIndex === -1) {
        return nudgeText
    }

    const beforeClose = nudgeText.slice(0, closeTagIndex).trimEnd()
    const afterClose = nudgeText.slice(closeTagIndex)
    return `${beforeClose}\n\n${guidance}\n${afterClose}`
}

const SYNTHETIC_SUMMARY_PREFIX = "msg_dcp_summary"
const MAX_REPORTED_MESSAGES = 8

function estimateMessageTokens(
    state: SessionState,
    config: PluginConfig,
    message: WithParts,
): { tokens: number; toolOutputs: number } {
    let chars = 0
    let toolOutputs = 0
    for (const part of message.parts ?? []) {
        if (part.type === "text" || part.type === "reasoning") {
            chars += (part.text ?? "").length
            continue
        }
        if (part.type === "tool") {
            const toolState = part.state as { input?: unknown; output?: unknown } | undefined
            chars += JSON.stringify(toolState?.input ?? {}).length
            // Only outputs that compression would actually remove count: outputs DCP already
            // pruned are replaced by a short placeholder, and protected tool outputs are
            // appended to the summary anyway.
            const pruned =
                state.prune.tools.has(part.callID) &&
                part.tool !== "question" &&
                part.tool !== "edit" &&
                part.tool !== "write"
            const protectedTool = isToolProtected(
                part.tool,
                toolState?.input,
                config.compress?.protectedTools ?? [],
                config.protectedFilePatterns ?? [],
                part.state && "metadata" in part.state
                    ? (part.state.metadata as Record<string, unknown>)
                    : undefined,
            )
            if (!pruned && !protectedTool && typeof toolState?.output === "string") {
                toolOutputs += 1
                chars += toolState.output.length
            }
            continue
        }
        if (part.type === "file") {
            // V1 file parts carry their payload in `url`. V2's command-time history() view does
            // not include media payloads, so V2 media is not counted here.
            chars += (part.url ?? "").length
        }
    }
    return { tokens: Math.round(chars / 4), toolOutputs }
}

/**
 * Size visibility for uncompressed messages. The model sees this content but
 * has no cost model for it, so manual compress triggers fixate on recent text
 * while hundreds of KB of old tool outputs ride along in every request.
 * Naming the top offenders (with their message refs) makes them targetable.
 */
export function buildMessageOccupancyGuidance(
    state: SessionState,
    config: PluginConfig,
    messages: WithParts[],
): string {
    const sized: Array<{
        ref: string
        role: string
        tokens: number
        toolOutputs: number
        protectedMsg: boolean
    }> = []

    for (const message of messages) {
        if (isIgnoredUserMessage(message)) continue
        if (message.info.id.startsWith(SYNTHETIC_SUMMARY_PREFIX)) continue
        if (isMessageCompacted(state, message)) continue
        const ref = state.messageIds.byRawId.get(message.info.id)
        if (!ref) continue
        const { tokens, toolOutputs } = estimateMessageTokens(state, config, message)
        if (tokens <= 0) continue
        sized.push({
            ref,
            role: message.info.role,
            tokens,
            toolOutputs,
            protectedMsg: message.info.role === "user" && isProtectedUserMessage(config, message),
        })
    }

    if (sized.length === 0) {
        return ""
    }

    sized.sort((a, b) => b.tokens - a.tokens)
    const lines = [
        "Largest uncompressed messages. Anything here that is no longer needed is a DROP target: a compress range over it removes its text and tool outputs from context entirely:",
    ]
    for (const entry of sized.slice(0, MAX_REPORTED_MESSAGES)) {
        const tools =
            entry.toolOutputs > 0
                ? `, ${entry.toolOutputs} tool output${entry.toolOutputs === 1 ? "" : "s"}`
                : ""
        const flag = entry.protectedMsg ? ", protected" : ""
        lines.push(
            `- ${entry.ref} (${entry.role}, ${formatTokenEstimate(entry.tokens)} tokens${tools}${flag})`,
        )
    }
    return lines.join("\n")
}
