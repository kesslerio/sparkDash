#!/usr/bin/env node
// Builds the dev-only LLM layout preview into <out-dir>. Use a fresh, empty
// directory from `mktemp -d`; the guard refuses anything else.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { assertSafePreviewOutDir } from "./llm-layout-preview-guard.mjs";

const outArg = process.argv[2];
if (!outArg) {
  console.error("usage: node scripts/build-llm-layout-preview.mjs <fresh-empty-out-dir>");
  process.exit(2);
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = assertSafePreviewOutDir(outArg, root);
await build({
  root,
  configFile: path.join(root, "vite.config.ts"),
  logLevel: "warn",
  build: {
    outDir,
    emptyOutDir: false,
    rollupOptions: {
      input: path.join(root, "llm-layout-preview.html"),
    },
  },
});
