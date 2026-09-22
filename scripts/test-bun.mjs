#!/usr/bin/env node
// Runs tests that need the Bun runtime (OpenTUI rendering, the V2 setup seam that loads
// lib/config.ts and its jsonc-parser ESM import). Uses `bun` when installed, otherwise the
// OpenCode binary, which embeds Bun and acts as the Bun CLI with BUN_BE_BUN=1.
import { spawnSync } from "node:child_process"
import { mkdtempSync, readdirSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

const files = readdirSync("tests/bun")
    .filter((name) => /\.test\.tsx?$/.test(name))
    .map((name) => join("tests/bun", name))

const has = (cmd) => spawnSync(cmd, ["--version"], { stdio: "ignore" }).status === 0
const runner = has("bun") ? "bun" : has("opencode") ? "opencode" : undefined
if (!runner) {
    console.log("[dcp] neither bun nor opencode found: Bun tests skipped")
    process.exit(0)
}

const root = mkdtempSync(join(tmpdir(), "dcp-bun-tests-"))
const result = spawnSync(runner, ["test", ...files], {
    stdio: "inherit",
    env: {
        ...process.env,
        BUN_BE_BUN: "1",
        XDG_DATA_HOME: join(root, "data"),
        XDG_CONFIG_HOME: join(root, "config"),
    },
})
process.exit(result.status ?? 1)
