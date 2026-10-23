// Persistence resolves its storage directory when it is first imported, so tests must never
// reach the real OpenCode data directory. Point XDG dirs at a throwaway location up front.
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

const root = mkdtempSync(join(tmpdir(), "dcp-tests-"))
process.env.XDG_DATA_HOME = join(root, "data")
process.env.XDG_CONFIG_HOME = join(root, "config")
