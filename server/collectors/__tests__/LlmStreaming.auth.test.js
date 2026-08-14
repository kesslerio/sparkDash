/**
 * Decode/showcase streaming must send the same Bearer key as LlmProbe.
 * Without it, authenticated LiteLLM (mama :4000) returns HTTP 401 in ~20ms
 * and sparkDash reports 0 tok/s / 0 streams.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { runStreamingRequest } from "../LlmStreaming.js";

function sseOkResponse() {
  return new Response("data: [DONE]\n\n", {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

test("runStreamingRequest sends Authorization Bearer when apiKey is set", async () => {
  const seen = [];
  const orig = globalThis.fetch;
  globalThis.fetch = async (_url, opts) => {
    seen.push(opts?.headers || {});
    return sseOkResponse();
  };
  try {
    await runStreamingRequest(
      "http://example.invalid/v1/chat/completions",
      { model: "qwen3.6:35b-a3b", messages: [{ role: "user", content: "hi" }] },
      AbortSignal.timeout(2000),
      { apiKey: "test-kalliope-key" }
    );
    assert.equal(seen.length, 1);
    assert.equal(seen[0].Authorization, "Bearer test-kalliope-key");
    assert.equal(seen[0]["Content-Type"], "application/json");
  } finally {
    globalThis.fetch = orig;
  }
});

test("runStreamingRequest omits Authorization when apiKey is absent (john-style open vLLM)", async () => {
  const seen = [];
  const orig = globalThis.fetch;
  globalThis.fetch = async (_url, opts) => {
    seen.push(opts?.headers || {});
    return sseOkResponse();
  };
  try {
    await runStreamingRequest(
      "http://example.invalid/v1/chat/completions",
      { model: "dspark", messages: [{ role: "user", content: "hi" }] },
      AbortSignal.timeout(2000),
      {}
    );
    assert.equal(seen.length, 1);
    assert.equal(seen[0].Authorization, undefined);
  } finally {
    globalThis.fetch = orig;
  }
});
