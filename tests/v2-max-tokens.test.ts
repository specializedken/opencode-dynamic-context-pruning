import assert from "node:assert/strict"
import test from "node:test"
import { hasMeasuredUsage, outputBudget } from "../lib/v2/checkpoint"

// The setup-level test (the real context hook) is tests/bun/v2-max-tokens.test.ts.
const LIMIT = { context: 229_376, output: 0 }
const MODEL = { providerID: "llamacpp", id: "qwen" }

test("outputBudget mirrors the host formula and never lowers the host value", () => {
    assert.equal(outputBudget(1024, LIMIT, 15_000), 32_000)
    assert.equal(outputBudget(1024, { context: 229_376, output: 8192 }, 15_000), 8192)
    assert.equal(outputBudget(1024, LIMIT, 190_000), 229_376 - Math.ceil(190_000 * 1.15))
    assert.equal(outputBudget(4096, LIMIT, 300_000), 4096)
    assert.equal(outputBudget(2048, { context: 0, output: 0 }, 1), 2048)
    assert.equal(outputBudget(1024, { context: 2_000_000, output: 500_000 }, 1), 256_000)
})

test("measured usage follows the host rule (same provider, no error, input tokens)", () => {
    const base = {
        type: "assistant",
        model: MODEL,
        tokens: { input: 1, cache: { read: 0, write: 0 } },
    }
    const entries = (entry: unknown) => [entry] as unknown as Parameters<typeof hasMeasuredUsage>[0]
    assert.equal(hasMeasuredUsage(entries(base), MODEL.providerID), true)
    assert.equal(
        hasMeasuredUsage(entries({ ...base, error: { type: "x" } }), MODEL.providerID),
        false,
    )
    assert.equal(hasMeasuredUsage(entries({ ...base, tokens: undefined }), MODEL.providerID), false)
    assert.equal(
        hasMeasuredUsage(
            entries({ ...base, tokens: { input: 0, cache: { read: 0, write: 0 } } }),
            MODEL.providerID,
        ),
        false,
    )
    assert.equal(hasMeasuredUsage(entries(base), "openrouter"), false)
    assert.equal(hasMeasuredUsage(entries({ ...base, type: "user" }), MODEL.providerID), false)
})
