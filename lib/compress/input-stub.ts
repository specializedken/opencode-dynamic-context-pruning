import type { CompressionBlock, SessionState, WithParts } from "../state"
import { formatBlockRef, parseBoundaryId, type IdFormat } from "../message-ids"

/**
 * A completed compress call keeps its full summary in its tool input, and the same text is
 * rendered again as the compressed block. Once a block exists, the call's copy is replaced
 * with a short pointer to the block, so each summary is sent once.
 *
 * A call is stubbed only while its block's summary is actually in the request: rendered at
 * its anchor, or nested inside a rendered block that consumed it. Otherwise (decompressed,
 * or an anchor dropped by native compaction while the call message survived) the raw input
 * is the only copy of the summary left, so it stays verbatim.
 *
 * Cache stability: whether a block's summary is rendered only changes together with the
 * request at or before that block's anchor (the range start, or the consuming block's
 * anchor, which comes earlier still), and the anchor always precedes the call message. So
 * a stub never appears or disappears without an earlier change in the same request, and
 * between such changes it depends only on the block id and the host's ID format.
 *
 * The stub must not contain `(bN)` or `{block_N}`: those are block placeholders that range
 * mode expands.
 */
export const COMPRESS_INPUT_STUB_PREFIX = "[DCP: summary stored in compressed block "

const COMPRESS_INPUT_STUB_REGEX =
    /^\s*\[DCP: summary stored in compressed block (?:@b[1-9]\d*@|b[1-9]\d*)\]\s*$/i

export function formatCompressInputStub(blockId: number, format: IdFormat): string {
    return `${COMPRESS_INPUT_STUB_PREFIX}${formatBlockRef(blockId, format)}]`
}

export function isCompressInputStub(value: unknown): boolean {
    return typeof value === "string" && COMPRESS_INPUT_STUB_REGEX.test(value)
}

type Entry = Record<string, unknown>

function isPlainObject(value: unknown): value is Entry {
    return value !== null && typeof value === "object" && !Array.isArray(value)
}

function trimmed(value: unknown): string | undefined {
    return typeof value === "string" ? value.trim() : undefined
}

function entryMatches(block: CompressionBlock, entry: Entry, format: IdFormat): boolean {
    if (block.mode === "message") {
        const messageId = trimmed(entry.messageId)
        if (messageId === undefined) return false
        // Message mode stores the normalized ref (e.g. "M0003" -> "m0003").
        const ref = parseBoundaryId(messageId, format)?.ref ?? messageId
        return block.startId === ref || block.startId === messageId
    }
    return trimmed(entry.startId) === block.startId && trimmed(entry.endId) === block.endId
}

function stubEntries(
    entries: unknown[],
    candidates: CompressionBlock[],
    format: IdFormat,
): { entries: unknown[]; changed: boolean } {
    const sorted = [...candidates].sort((a, b) => a.blockId - b.blockId)
    const used = new Set<number>()
    let changed = false
    const next = entries.map((entry) => {
        if (!isPlainObject(entry) || typeof entry.summary !== "string") return entry
        if (isCompressInputStub(entry.summary)) return entry
        const block = sorted.find(
            (candidate) => !used.has(candidate.blockId) && entryMatches(candidate, entry, format),
        )
        if (!block) return entry
        used.add(block.blockId)
        changed = true
        // Spreading keeps the original key order, so the serialized input stays stable.
        return { ...entry, summary: formatCompressInputStub(block.blockId, format) }
    })
    return { entries: next, changed }
}

/**
 * Returns a copy of a compress tool input whose summaries are replaced by stubs for the
 * given blocks, or undefined when nothing would change. Never mutates `input`.
 */
export function stubbedCompressInput(
    input: unknown,
    candidates: CompressionBlock[],
    format: IdFormat,
): unknown | undefined {
    if (!isPlainObject(input) || candidates.length === 0) return undefined
    const content = input.content

    if (Array.isArray(content)) {
        const result = stubEntries(content, candidates, format)
        return result.changed ? { ...input, content: result.entries } : undefined
    }

    if (typeof content === "string") {
        // Some models send content as a JSON string (see coerceContentArray). Keep it a string.
        const text = content.trim()
        if (!text.startsWith("[") && !text.startsWith("{")) return undefined
        let parsed: unknown
        try {
            parsed = JSON.parse(text)
        } catch {
            return undefined
        }
        if (Array.isArray(parsed)) {
            const result = stubEntries(parsed, candidates, format)
            return result.changed
                ? { ...input, content: JSON.stringify(result.entries) }
                : undefined
        }
        if (isPlainObject(parsed)) {
            const result = stubEntries([parsed], candidates, format)
            return result.changed
                ? { ...input, content: JSON.stringify(result.entries[0]) }
                : undefined
        }
    }

    if (isPlainObject(content)) {
        // A single entry object is accepted too (see coerceContentArray). Keep it an object.
        const result = stubEntries([content], candidates, format)
        return result.changed ? { ...input, content: result.entries[0] } : undefined
    }

    return undefined
}

/**
 * The blocks whose summary text is in the request: the ones rendered at their anchor, plus
 * every block they consumed (range mode nests a consumed summary in its consumer's text),
 * transitively.
 */
export function withConsumedBlocks(
    state: SessionState,
    renderedBlockIds: Iterable<number>,
): Set<number> {
    const blocksById = state.prune.messages.blocksById
    const result = new Set<number>()
    const pending = [...renderedBlockIds]
    while (pending.length > 0) {
        const blockId = pending.pop()!
        if (result.has(blockId)) continue
        result.add(blockId)
        const consumed = blocksById.get(blockId)?.consumedBlockIds
        if (Array.isArray(consumed)) pending.push(...consumed)
    }
    return result
}

/**
 * Replaces the summary payload of every compress call whose block summary is in this
 * request (`summaryBlockIds`, see withConsumedBlocks) with a stub.
 * Runs on the per-request message copy only; history and persisted state are untouched.
 */
export function stubCompressCallInputs(
    state: SessionState,
    messages: WithParts[],
    summaryBlockIds: ReadonlySet<number>,
): void {
    const blocksById = state.prune.messages.blocksById
    if (blocksById.size === 0 || summaryBlockIds.size === 0) return

    const byMessage = new Map<string, CompressionBlock[]>()
    for (const block of blocksById.values()) {
        if (!block || typeof block.compressMessageId !== "string") continue
        if (!summaryBlockIds.has(block.blockId)) continue
        const list = byMessage.get(block.compressMessageId)
        if (list) list.push(block)
        else byMessage.set(block.compressMessageId, [block])
    }
    if (byMessage.size === 0) return

    const format = state.idFormat
    for (const message of messages) {
        if (message.info.role !== "assistant") continue
        const blocks = byMessage.get(message.info.id)
        if (!blocks) continue

        const parts = Array.isArray(message.parts) ? message.parts : []
        const compressParts = parts.filter(
            (part: any) =>
                part.type === "tool" &&
                part.tool === "compress" &&
                (part.state?.status === "completed" || part.state?.status === "error"),
        ) as any[]
        if (compressParts.length === 0) continue

        const callIds = new Set(compressParts.map((part) => part.callID))
        for (const part of compressParts) {
            let candidates = blocks.filter((block) => block.compressCallId === part.callID)
            if (candidates.length === 0 && compressParts.length === 1) {
                // Reconstructed or legacy blocks may lack a call id, or carry one that does not
                // match the host part id. With a single compress part there is no ambiguity.
                candidates = blocks.filter(
                    (block) =>
                        block.compressCallId === undefined || !callIds.has(block.compressCallId),
                )
            }
            if (candidates.length === 0) continue

            const next = stubbedCompressInput(part.state.input, candidates, format)
            if (next === undefined) continue
            // Replace the state object instead of editing the input in place: the tool cache
            // and the V1 host still hold the original. On V2, restore() writes this projected
            // `state.input` back onto the native tool-call part.
            part.state = { ...part.state, input: next }
        }
    }
}
