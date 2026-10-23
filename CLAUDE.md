# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`@tarquinen/opencode-dcp` — an OpenCode plugin ("Dynamic Context Pruning") that shrinks the LLM context by (a) exposing a `compress` tool the model calls to replace stale conversation spans with summaries, and (b) automatic strategies (deduplication, purge-errors). Session history on disk is never modified; pruning happens on the message array right before each provider request.

Upstream is `Tarquinen/opencode-dynamic-context-pruning` (the README/package.json also reference the `Opencode-DCP` org, which redirects to it). Upstream development has slowed (author moved to "Sleev"), so open PRs are largely unmerged. This checkout is a working fork: `origin` still points at upstream until the fork is created. Branches upstream: `master` (release), `dev`.

## Commands

Node + npm (no bun needed, despite `scripts/README.md`). Install needs `--legacy-peer-deps` (OpenTUI peer versions differ between OpenCode V1 and V2).

```sh
npm ci --legacy-peer-deps
npm run build            # tsup (bundles index.ts -> dist/index.js) + tsc --emitDeclarationOnly
npm test                 # node --import tsx --test tests/*.test.ts  + tests/logger workspace tests
npm run typecheck        # tsc --noEmit + logger workspace
npm run format:check     # prettier (4-space indent, see .prettierrc); `npm run format` to fix
npm run check:package    # build + scripts/verify-package.mjs
npm run dcp -- --system  # preview prompts (scripts/print.ts)
```

Single test file: `node --import tsx --test tests/compress-range.test.ts`. Filter by name: add `--test-name-pattern="..."`.

CI (`.github/workflows/pr-checks.yml`) runs format:check → typecheck → build → test → `npm audit --audit-level=high`, on Node 20, for PRs to `master`/`dev`. Run all of these before proposing a PR; CONTRIBUTING asks for no AI-slop and tests where applicable.

Heavier verification (Docker required): `npm run sandbox` (V2) / `-- --v1` launches a real OpenCode with the local build and a request logger; `node scripts/lab.mjs` runs the containerized integration lab against a mock provider. See CONTRIBUTING.md.

## Architecture

### Two hosts, one core

The package supports OpenCode **V1** (hooks-based `Plugin` in `index.ts`, TUI panel in `tui.tsx`) and **V2** (`setup()` in `lib/v2/index.ts`, `lib/v2/tui.tsx`, RPC in `lib/v2/rpc.ts`). Both default-export `{ id, setup, server|tui }`. Shared logic lives in `lib/`; V2 adapts the host's native message model to the V1-shaped `WithParts` (`lib/v2/messages.ts` `history()`/`project()`) so the pruning code stays common. Changing shared behavior means checking both hosts. The ID formats differ: V1 uses XML tags (`<dcp-message-id>`), V2 uses compact `@4@` / `@b1@` markers (`lib/message-ids.ts`); custom prompts must match the host.

`server.js` just re-exports `dist/index.js`; `tui.tsx` is shipped as source (not bundled).

### The per-request pipeline (`lib/hooks.ts`)

`createChatMessageTransformHandler` (hook `experimental.chat.messages.transform`) is the heart. On every LLM request it, in order: `checkSession` (init/restore state) → `stripHallucinations` → `assignMessageRefs` → `syncCompressionBlocks` → `syncToolCache` → `buildToolIdList` → `prune` (applies existing compression blocks + dedup/purge-errors) → inject sub-agent results → `buildPriorityMap` → `injectCompressNudges` → `injectMessageIds` → `applyPendingManualTrigger` → `stripStaleMetadata`. Order matters; several open issues are ordering/leak bugs in this chain. The V2 setup re-implements the same sequence against V2 hooks.

Other hooks: `experimental.chat.system.transform` (injects the DCP system prompt; skips sub-agents unless `experimental.allowSubAgents`, and skips internal title/summary agents by system-prompt signature), `experimental.text.complete` (strips leaked DCP tags from model output), `command.execute.before` (`/dcp`, `/dcp-compress`), `event`, and the `config` hook (registers the `compress` tool permission, `/dcp-compress` command, `primary_tools`).

### Compression (`lib/compress/`)

`range.ts` / `message.ts` implement the two `compress` tool modes (`compress.mode`), sharing `pipeline.ts` (`prepareSession`, notifications, persistence). Model-supplied boundary IDs are resolved in `search.ts` / `range-utils.ts` / `message-utils.ts`; results become `CompressionBlock`s (`lib/state/types.ts`) that nest (range mode keeps earlier summaries inside later ones via `@bN@` placeholders). `protected-content.ts` appends protected tool outputs / file patterns / `<protect>` tags into summaries.

### State (`lib/state/`)

`SessionState` is one object per plugin instance, re-pointed at the active session by `checkSession`/`ensureSessionInitialized`; persisted per session to a JSON file (`persistence.ts`) under the OpenCode data dir. Because it is a single mutable object shared across sessions, concurrent-session bugs (e.g. issue #650) live here. Token counting (`lib/token-utils.ts`, `@anthropic-ai/tokenizer` WASM) is also on this hot path.

### Config & prompts

`lib/config.ts` (1000+ lines): merges global `~/.config/opencode/dcp.jsonc` → `$OPENCODE_CONFIG_DIR` → project `.opencode/dcp.jsonc`, validates, and auto-creates defaults. `dcp.schema.json` must be kept in sync with it. Percent limits (`"75%"`) resolve against `state.modelContextLimit`, cached from the system-transform hook. `lib/prompts/` holds the model-facing prompt text (system, nudges, compress tool descriptions); user overrides are gated by `experimental.customPrompts`.

### Other

`lib/commands/` implements `/dcp` subcommands (context, stats, sweep, manual, decompress, recompress). `lib/tui/` is the Solid/OpenTUI panel for V1; V2's lives in `lib/v2/`. `lib/host-permissions.ts` + `lib/compress-permission.ts` reconcile DCP's `compress.permission` with OpenCode's own permission config. `tests/logger/` is a dev-only npm workspace (request capture relay), excluded from the published package.

## Local environment (this user)

- OpenCode **v2.1.2** with config at `~/.config/opencode/opencode.json` (V2 format: `plugins`, `providers`, `permissions`), plugin installed as `@tarquinen/opencode-dcp@latest` (auto-updates from npm — a locally built fork must be referenced by absolute path instead, see CONTRIBUTING "Local Installation").
- Models run in parallel: local llama.cpp (`llamacpp/unsloth/Qwen3.8-27B-GGUF-MTP`, openai-compatible, 229k context, set via `providers.*.models.*.limit`) and OpenRouter models (kimi-k2.5, qwen, devstral, nemotron). `subagent_depth: 1`, native `compaction` enabled.
- `~/.config/opencode/dcp.jsonc` sets `compress.maxContextLimit: "75%"`, `minContextLimit: "25%"`.
- Prioritize V2-path issues; V1-only issues (e.g. #650) are low priority.
