/**
 * DecodeBench / Showcase must forward llmTarget.apiKey into fetch,
 * not only the shared streaming helper.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import os from "node:os";
import path from "node:path";
import { DecodeBenchManager } from "../DecodeBench.js";
import { ShowcaseManager } from "../ShowcaseManager.js";
import { llmTarget } from "../LlmStreaming.js";

function sseOkResponse() {
  return new Response("data: [DONE]\n\n", {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function waitFor(pred, timeoutMs = 2000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (pred()) return;
    await new Promise((r) => setTimeout(r, 10));
  }
  throw new Error("timed out waiting for fetch");
}

test("DecodeBenchManager.start sends Bearer on chat completions", async () => {
  const seen = [];
  const orig = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    seen.push({ url: String(url), headers: opts?.headers || {} });
    return sseOkResponse();
  };
  const historyPath = path.join(
    os.tmpdir(),
    `sparkdash-bench-auth-${Date.now()}.json`
  );
  const mgr = new DecodeBenchManager(historyPath);
  try {
    const job = mgr.start({
      sparkId: "wiring-bench",
      target: llmTarget("127.0.0.1", 4000, "test-kalliope-key"),
      port: 4000,
      modelId: "qwen3.6:35b-a3b",
      concurrencies: [1],
      maxTokens: 64,
    });
    assert.equal(job._target, undefined);
    assert.equal(job.apiKey, undefined);
    await waitFor(() => seen.some((s) => String(s.url).includes("/v1/chat/completions")));
    const chat = seen.find((s) => String(s.url).includes("/v1/chat/completions"));
    assert.equal(chat.headers.Authorization, "Bearer test-kalliope-key");
    mgr.cancel("wiring-bench", job.benchId);
  } finally {
    globalThis.fetch = orig;
  }
});

test("ShowcaseManager.start sends Bearer on completions and /metrics", async () => {
  const seen = [];
  const orig = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    seen.push({ url: String(url), headers: opts?.headers || {} });
    if (String(url).endsWith("/metrics")) {
      return new Response("vllm:generation_tokens_total 1\n", {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      });
    }
    return sseOkResponse();
  };
  const historyPath = path.join(
    os.tmpdir(),
    `sparkdash-showcase-auth-${Date.now()}.json`
  );
  const mgr = new ShowcaseManager(historyPath);
  try {
    const started = mgr.start({
      sparkId: "wiring-showcase",
      target: llmTarget("127.0.0.1", 4000, "test-kalliope-key"),
      port: 4000,
      modelId: "qwen3.6:35b-a3b",
      prompts: ["hi"],
      maxTokens: 64,
    });
    await waitFor(() =>
      seen.some((s) => String(s.url).includes("/v1/chat/completions"))
    );
    const chat = seen.find((s) => String(s.url).includes("/v1/chat/completions"));
    assert.equal(chat.headers.Authorization, "Bearer test-kalliope-key");
    const metrics = seen.find((s) => String(s.url).endsWith("/metrics"));
    if (metrics) {
      assert.equal(metrics.headers.Authorization, "Bearer test-kalliope-key");
    }
    mgr.cancel("wiring-showcase", started.sessionId);
  } finally {
    globalThis.fetch = orig;
  }
});
