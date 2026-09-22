import type { SessionState, WithParts } from "../state"
import type { Logger } from "../logger"

function sortBlocksByCreation(
    a: { createdAt: number; blockId: number },
    b: { createdAt: number; blockId: number },
): number {
    const createdAtDiff = a.createdAt - b.createdAt
    if (createdAtDiff !== 0) {
        return createdAtDiff
    }
    return a.blockId - b.blockId
}

/**
 * Recompute block liveness against the current message window.
 *
 * Returns true when any block's liveness changed, so callers can persist the new
 * state immediately instead of relying on the next incidental save.
 */
export const syncCompressionBlocks = (
    state: SessionState,
    logger: Logger,
    messages: WithParts[],
): boolean => {
    const messagesState = state.prune.messages
    if (!messagesState?.blocksById?.size) {
        return false
    }
    const wasActive = new Map(
        Array.from(messagesState.blocksById.values(), (block) => [block.blockId, block.active]),
    )

    const messageIds = new Set(messages.map((msg) => msg.info.id))
    const previousActiveBlockIds = new Set<number>(
        Array.from(messagesState.blocksById.values())
            .filter((block) => block.active)
            .map((block) => block.blockId),
    )

    messagesState.activeBlockIds.clear()
    messagesState.activeByAnchorMessageId.clear()

    const now = Date.now()
    const missingOriginBlockIds: number[] = []
    const orderedBlocks = Array.from(messagesState.blocksById.values()).sort(sortBlocksByCreation)

    for (const block of orderedBlocks) {
        if (block.deactivatedByUser) {
            block.active = false
            if (block.deactivatedAt === undefined) {
                block.deactivatedAt = now
            }
            block.deactivatedByBlockId = undefined
            continue
        }

        const hasOriginMessage =
            typeof block.compressMessageId === "string" &&
            block.compressMessageId.length > 0 &&
            messageIds.has(block.compressMessageId)

        if (!hasOriginMessage) {
            // The compressing assistant message can be missing after a restart (it may never
            // be persisted, or a native compaction dropped it). The summary is still valid as
            // long as its anchor is in the window, so keep the block live and fall through to
            // the normal path, which also retires blocks it consumed. Without the anchor the
            // summary could not be rendered, so the block must not hide its range.
            const hasAnchorMessage =
                typeof block.anchorMessageId === "string" &&
                block.anchorMessageId.length > 0 &&
                messageIds.has(block.anchorMessageId)

            if (!hasAnchorMessage) {
                block.active = false
                block.deactivatedAt = now
                block.deactivatedByBlockId = undefined
                missingOriginBlockIds.push(block.blockId)
                continue
            }
        }

        for (const consumedBlockId of block.consumedBlockIds) {
            if (!messagesState.activeBlockIds.has(consumedBlockId)) {
                continue
            }

            const consumedBlock = messagesState.blocksById.get(consumedBlockId)
            if (consumedBlock) {
                consumedBlock.active = false
                consumedBlock.deactivatedAt = now
                consumedBlock.deactivatedByBlockId = block.blockId

                const mappedBlockId = messagesState.activeByAnchorMessageId.get(
                    consumedBlock.anchorMessageId,
                )
                if (mappedBlockId === consumedBlock.blockId) {
                    messagesState.activeByAnchorMessageId.delete(consumedBlock.anchorMessageId)
                }
            }

            messagesState.activeBlockIds.delete(consumedBlockId)
        }

        block.active = true
        block.deactivatedAt = undefined
        block.deactivatedByBlockId = undefined
        messagesState.activeBlockIds.add(block.blockId)
        if (messageIds.has(block.anchorMessageId)) {
            messagesState.activeByAnchorMessageId.set(block.anchorMessageId, block.blockId)
        }
    }

    for (const entry of messagesState.byMessageId.values()) {
        const allBlockIds = Array.isArray(entry.allBlockIds)
            ? [...new Set(entry.allBlockIds.filter((id) => Number.isInteger(id) && id > 0))]
            : []

        entry.allBlockIds = allBlockIds
        entry.activeBlockIds = allBlockIds.filter((id) => messagesState.activeBlockIds.has(id))
    }

    const nextActiveBlockIds = messagesState.activeBlockIds
    let deactivatedCount = 0
    let reactivatedCount = 0

    for (const blockId of previousActiveBlockIds) {
        if (!nextActiveBlockIds.has(blockId)) {
            deactivatedCount++
        }
    }
    for (const blockId of nextActiveBlockIds) {
        if (!previousActiveBlockIds.has(blockId)) {
            reactivatedCount++
        }
    }

    if (missingOriginBlockIds.length > 0 || deactivatedCount > 0 || reactivatedCount > 0) {
        logger.info("Synced compress block state", {
            missingOriginCount: missingOriginBlockIds.length,
            deactivatedCount,
            reactivatedCount,
        })
    }

    return Array.from(messagesState.blocksById.values()).some(
        (block) => wasActive.get(block.blockId) !== block.active,
    )
}
