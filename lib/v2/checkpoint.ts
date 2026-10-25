import type { Message } from "@opencode/ai/schema/messages"
import type { Plugin } from "@opencode/plugin"

type History = Awaited<ReturnType<Plugin.Context["session"]["context"]>>

// OpenCode 2.1.x (checked against 2.1.3) does not send a raw message tail after a
// compaction: session.context() starts at the latest completed compaction, and the
// entries from its keepID up to the compaction are serialized into one string
// (metadata.recent, entries joined with "\n") that the host sends inside the
// checkpoint user message, in <recent-context>. The host bounds that tail only by
// snapping keepID back to a user entry, so an autonomous run with a single user
// message is kept whole (p2-09 B: 1.95M chars, a 558k-token request on a 229k
// model, after which the session could not be compacted again).
//
// These are constants on purpose: the output must not depend on the model limit or
// config, so the checkpoint bytes stay identical on every later request (prompt
// cache), across restarts and when the model list is unavailable.
/** Tails up to this size are left byte-identical (2x the host's intended keep). */
export const RECENT_TRIGGER_CHARS = 120_000
/** Bounded tail size: the host's own keep of 15000 tokens at its 4 chars/token. */
export const RECENT_MAX_CHARS = 60_000
const PIN_MAX_CHARS = 20_000
const MARKER_RESERVE = 200
const ENTRY_MARKER_RESERVE = 100
// Entry starts of the host serializer. A tool call is never separated from its
// "[Tool result]" / "[Tool error]" lines, nor a user entry from its attachments.
const ENTRY_START =
    /\n(?=\[(?:User\]: |Assistant(?: reasoning| tool call)?\]: |Synthetic context\]: |Shell\]: |Skill activated: ))/g

function isLowSurrogate(code: number): boolean {
    return code >= 0xdc00 && code <= 0xdfff
}

function isHighSurrogate(code: number): boolean {
    return code >= 0xd800 && code <= 0xdbff
}

/**
 * Deterministically bounds the host's serialized recent context: keeps the leading
 * user instruction (the keepID entry) and the newest entries that fit, cut at entry
 * boundaries, with a plain-text marker for the omitted middle. When the newest
 * entry alone is over the budget, it is kept with its middle cut out instead.
 */
export function boundRecent(recent: string): string {
    if (recent.length <= RECENT_TRIGGER_CHARS) return recent
    const starts = [0]
    for (const match of recent.matchAll(ENTRY_START)) starts.push(match.index + 1)

    let pinned = ""
    let pinEnd = 0
    if (recent.startsWith("[User]: ") || recent.startsWith("[Skill activated: ")) {
        const end = starts.find((start) => start > 0 && recent.startsWith("[Assistant", start))
        // Without any assistant entry there is nothing to keep after a pin.
        if (end !== undefined) {
            pinEnd = end
            pinned = recent.slice(0, end - 1)
            if (pinned.length > PIN_MAX_CHARS) {
                const at = isHighSurrogate(pinned.charCodeAt(PIN_MAX_CHARS - 1))
                    ? PIN_MAX_CHARS - 1
                    : PIN_MAX_CHARS
                pinned = pinned.slice(0, at) + "\n[truncated]"
            }
        }
    }

    const budget = RECENT_MAX_CHARS - pinned.length - MARKER_RESERVE
    const head = pinned ? pinned + "\n" : ""
    const cut = starts.find((start) => start >= pinEnd && recent.length - start <= budget)
    if (cut !== undefined) {
        const omitted = starts.filter((start) => start >= pinEnd && start < cut).length
        return head + omissionMarker(omitted) + "\n" + recent.slice(cut)
    }
    // The newest entry alone exceeds the budget (e.g. a write() whose JSON input is
    // one long line). Keep it, truncated in its middle: its start names the call and
    // its end holds the "[Tool result]" lines, so the call stays with its result.
    const last = starts[starts.length - 1]!
    const entry = recent.slice(last)
    const room = budget - ENTRY_MARKER_RESERVE
    const front = safeEnd(entry, Math.floor(room / 2))
    const back = safeStart(entry, entry.length - (room - front))
    const omitted = starts.filter((start) => start >= pinEnd && start < last).length
    return (
        head +
        omissionMarker(omitted) +
        "\n" +
        entry.slice(0, front) +
        `\n[DCP: ${back - front} characters of this entry were omitted to fit the context window.]\n` +
        entry.slice(back)
    )
}

// No claim that the summary covers the omitted entries: after an overflow
// compaction the host summarizes only the history before keepID.
function omissionMarker(omitted: number): string {
    return `[DCP: ${omitted} earlier entries of the recent context were omitted to fit the context window. Re-read files or re-run commands if their details are needed.]`
}

/** Largest end <= at that does not split a surrogate pair. */
function safeEnd(text: string, at: number): number {
    return at > 0 && isHighSurrogate(text.charCodeAt(at - 1)) ? at - 1 : at
}

/** Smallest start >= at that does not split a surrogate pair. */
function safeStart(text: string, at: number): number {
    return isLowSurrogate(text.charCodeAt(at)) ? at + 1 : at
}

function recentCandidates(entry: History[number], message: Message): string[] {
    const loose = entry as { metadata?: Record<string, unknown>; recent?: unknown }
    return [loose.metadata?.recent, loose.recent, message.metadata?.recent].filter(
        (value): value is string =>
            typeof value === "string" && value.length > RECENT_TRIGGER_CHARS,
    )
}

function occurrences(text: string, needle: string): number[] {
    const found: number[] = []
    for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + 1)) {
        found.push(at)
        if (found.length > 1) break
    }
    return found
}

/**
 * Returns `native` with each oversized host checkpoint's <recent-context> bounded by
 * boundRecent(). Returns the same array when nothing changed; never mutates inputs.
 * Any mismatch with the expected host format leaves the message untouched.
 */
export function boundCheckpoints(native: Message[], entries: History): Message[] {
    const byID = new Map(entries.map((entry) => [entry.id, entry]))
    let output: Message[] | undefined
    for (const [index, message] of native.entries()) {
        if (message.role !== "user" || !message.id) continue
        const entry = byID.get(message.id)
        if (!entry || entry.type !== "compaction" || entry.status !== "completed") continue
        const loose = entry as {
            native?: { messages?: unknown }
            providerContext?: { messages?: unknown }
        }
        // Provider-native checkpoints carry no serialized tail.
        if (loose.native?.messages !== undefined || loose.providerContext?.messages !== undefined)
            continue
        for (const recent of recentCandidates(entry, message)) {
            const needle = "<recent-context>\n" + recent + "\n</recent-context>"
            const hits = message.content.flatMap((part) =>
                part.type === "text"
                    ? occurrences(part.text, needle).map((at) => ({ part, at }))
                    : [],
            )
            if (hits.length !== 1) continue
            const { part, at } = hits[0]!
            if (part.type !== "text") continue
            // slice, not String.replace: the tail may contain "$&"-style patterns.
            const text =
                part.text.slice(0, at) +
                "<recent-context>\n" +
                boundRecent(recent) +
                "\n</recent-context>" +
                part.text.slice(at + needle.length)
            output ??= [...native]
            output[index] = {
                ...message,
                content: message.content.map((item) => (item === part ? { ...part, text } : item)),
            } as Message
            break
        }
    }
    return output ?? native
}

export interface ModelLimit {
    context: number
    output: number
}

// Mirrors of OpenCode 2.1.3's output sizing (SessionModelRequest.prepare computes
// maxTokens before the plugin hook): output cap min(limit.output || 32000, 256000),
// floor 1024, and an input estimate of round(chars / 4) per part with a 15% margin.
const DEFAULT_OUTPUT_CAP = 32_000
const MAX_OUTPUT_CAP = 256_000
const ESTIMATE_MARGIN = 0.15

function estimate(text: string): number {
    return Math.max(0, Math.round(text.length / 4))
}

/** The host's chars/4 estimate of a request (non-text parts at their JSON size). */
export function estimateRequestTokens(
    system: ReadonlyArray<{ text: string }>,
    tools: Record<string, { description: string; input: unknown }>,
    messages: ReadonlyArray<Message>,
): number {
    let total = 0
    for (const part of system) total += estimate(part.text)
    for (const [name, definition] of Object.entries(tools))
        total += estimate(name + definition.description + (JSON.stringify(definition.input) ?? ""))
    for (const message of messages)
        for (const part of message.content) {
            if (part.type === "text" || part.type === "reasoning") total += estimate(part.text)
            else if (part.type === "tool-call")
                total += estimate(part.name + (JSON.stringify(part.input) ?? ""))
            else total += estimate(JSON.stringify(part) ?? "")
        }
    return total
}

/**
 * Whether the host sized this request from measured usage: an assistant entry of
 * this provider, without error, that reports input tokens. The host's estimate then
 * covers only entries after it, so the checkpoint size did not enter it.
 */
export function hasMeasuredUsage(entries: History, providerID: string): boolean {
    return entries.some((entry) => {
        const loose = entry as {
            type?: string
            error?: unknown
            model?: { providerID?: string }
            tokens?: { input?: number; cache?: { read?: number; write?: number } }
        }
        const tokens = loose.tokens
        return (
            loose.type === "assistant" &&
            loose.model?.providerID === providerID &&
            !loose.error &&
            tokens !== undefined &&
            (tokens.input ?? 0) + (tokens.cache?.read ?? 0) + (tokens.cache?.write ?? 0) > 0
        )
    })
}

/**
 * maxTokens for a request of `estimated` input tokens, by the host's own formula;
 * never below `current` (the host's value), so it can only undo the shrink caused
 * by the host estimating the unbounded checkpoint.
 */
export function outputBudget(current: number, limit: ModelLimit, estimated: number): number {
    if (!(limit.context > 0)) return current
    const cap = Math.min(limit.output > 0 ? limit.output : DEFAULT_OUTPUT_CAP, MAX_OUTPUT_CAP)
    const fit = limit.context - Math.ceil(estimated * (1 + ESTIMATE_MARGIN))
    return Math.max(current, Math.min(cap, fit))
}
