import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { assertSafePreviewOutDir } from "./llm-layout-preview-guard.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("accepts a fresh empty directory", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llm-layout-preview-guard."));
  assert.equal(assertSafePreviewOutDir(dir, root), dir);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refuses a non-empty directory without touching its files", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "llm-layout-preview-guard."));
  fs.writeFileSync(path.join(dir, "keep.txt"), "user file");
  assert.throws(() => assertSafePreviewOutDir(dir, root), /non-empty/);
  assert.equal(fs.readFileSync(path.join(dir, "keep.txt"), "utf8"), "user file");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refuses the repository root and anything inside it", () => {
  assert.throws(() => assertSafePreviewOutDir(root, root), /inside the repository/);
  assert.throws(() => assertSafePreviewOutDir(path.join(root, "dist"), root), /inside the repository/);
});

test("refuses the filesystem root", () => {
  assert.throws(() => assertSafePreviewOutDir(path.parse(root).root, root), /refusing/);
});
