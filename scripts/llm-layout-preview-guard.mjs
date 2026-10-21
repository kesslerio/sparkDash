import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/** Returns the resolved output directory, or throws before anything is written. */
export function assertSafePreviewOutDir(outDir, root) {
  const resolved = path.resolve(outDir);
  const repo = path.resolve(root);
  if (resolved === path.parse(resolved).root || resolved === os.homedir()) {
    throw new Error(`refusing to build the preview into ${resolved}`);
  }
  if (resolved === repo || resolved.startsWith(repo + path.sep)) {
    throw new Error(`refusing to build the preview inside the repository: ${resolved}`);
  }
  if (fs.existsSync(resolved)) {
    if (!fs.statSync(resolved).isDirectory()) {
      throw new Error(`output path is not a directory: ${resolved}`);
    }
    if (fs.readdirSync(resolved).length > 0) {
      throw new Error(
        `refusing non-empty output directory: ${resolved}; use a fresh directory from mktemp -d`,
      );
    }
  }
  return resolved;
}
