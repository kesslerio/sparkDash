/**
 * Unit tests for oMLX (Apple Silicon MLX server) detection and /api/status telemetry.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { LlmProbe } from "../LlmProbe.js";

const MODELS = {
  object: "list",
  data: [
    { id: "GLM-5.3-Flash-oQ4e-mtp", object: "model", owned_by: "omlx", max_model_len: 1048576 },
    { id: "Vontra--GLM-5.3-Flash-MLX-4bit-MTP", object: "model", owned_by: "omlx", max_model_len: 1048576 },
  ],
};

const status = (overrides = {}) => ({
  status: "ok",
  version: "0.7.0rc1",
  default_model: "GLM-5.3-Flash-oQ4e-mtp",
  loaded_models: ["GLM-5.3-Flash-oQ4e-mtp"],
  total_requests: 289,
  active_requests: 1,
  waiting_requests: 0,
  total_prompt_tokens: 51_216_751,
  total_completion_tokens: 44_711,
  total_cached_tokens: 50_845_521,
  cache_efficiency: 99.3,
  avg_prefill_tps: 697.8,
  avg_generation_tps: 57.3,
  model_memory_used: 195_080_732_375,
  model_memory_max: 249_108_103_168,
  ...overrides,
});

function omlxProbe(getStatus, models = MODELS) {
  const probe = new LlmProbe({ lanIp: "100.96.225.114" }, 8000);
  const requested = [];
  probe._fetch = async (url) => {
    const u = String(url);
    requested.push(u);
    if (u.endsWith("/v1/models")) return { ok: true, status: 200, json: async () => models };
    if (u.endsWith("/api/status")) {
      const body = getStatus();
      if (body == null) return { ok: false, status: 404, json: async () => ({ detail: "Not Found" }) };
      return { ok: true, status: 200, json: async () => body };
    }
    return { ok: false, status: 404, json: async () => ({}), text: async () => "" };
  };
  return { probe, requested };
}

test("owned_by omlx is classified as the omlx backend", async () => {
  const { probe } = omlxProbe(() => status());
  await probe._detectServerType();
  assert.equal(probe.serverIsOpenAI, true);
  assert.equal(probe.backendType, "omlx");
});

test("other owned_by values keep their classification", async () => {
  const probe = new LlmProbe({ lanIp: "10.0.0.1" }, 8888);
  probe._probeIsDs4 = async () => false;
  probe._probeIsSglang = async () => false;
  assert.equal(await probe._classifyOpenAIBackend("ds4.c"), "ds4");
  assert.equal(await probe._classifyOpenAIBackend("sglang"), "sglang");
  assert.equal(await probe._classifyOpenAIBackend("vllm"), "vllm");
  assert.equal(await probe._classifyOpenAIBackend(undefined), "vllm");
});

test("two /api/status samples give live rates, cache split and workload state", async () => {
  let sample = status();
  const { probe, requested } = omlxProbe(() => sample);
  await probe._detectServerType();
  requested.length = 0;
  const first = await probe.probe();
  assert.equal(first.backend, "omlx");
  assert.equal(probe.generationTps, 0, "first sample only seeds the baseline");
  assert.equal(probe.requestsRunning, 1);
  assert.equal(probe.requestsWaiting, 0);
  assert.equal(probe.status, "active");
  assert.equal(probe.prefixCacheHitRate, 0.993);
  assert.equal(probe.gpuMemoryUtilization, 0.7831);

  probe.lastProbeTime = Date.now() - 2000;
  sample = status({
    total_prompt_tokens: 51_216_751 + 20_000,
    total_cached_tokens: 50_845_521 + 18_000,
    total_completion_tokens: 44_711 + 120,
  });
  await probe.probe();
  assert.ok(probe.generationTps >= 55 && probe.generationTps <= 65, `decode ${probe.generationTps}`);
  assert.ok(probe.prefillTps >= 900 && probe.prefillTps <= 1100, `prefill ${probe.prefillTps}`);
  assert.ok(probe.cachedPrefillTps >= 8000 && probe.cachedPrefillTps <= 10000);
  assert.equal(requested.some((u) => u.endsWith("/metrics") || u.endsWith("/slots")), false);
});

test("a counter reset after a server restart never yields negative rates", async () => {
  let sample = status();
  const { probe } = omlxProbe(() => sample);
  await probe._detectServerType();
  await probe.probe();
  probe.lastProbeTime = Date.now() - 2000;
  sample = status({ total_prompt_tokens: 10, total_cached_tokens: 0, total_completion_tokens: 5 });
  await probe.probe();
  assert.equal(probe.generationTps, 0);
  assert.equal(probe.prefillTps, 0);
});

test("idle server reads as idle", async () => {
  const { probe } = omlxProbe(() => status({ active_requests: 0 }));
  await probe._detectServerType();
  await probe.probe();
  assert.equal(probe.status, "idle");
});

test("a missing /api/status degrades to unavailable telemetry, not idle", async () => {
  const { probe } = omlxProbe(() => null, { object: "list", data: [MODELS.data[0]] });
  await probe._detectServerType();
  const snap = await probe.probe();
  assert.equal(snap.backend, "omlx");
  assert.equal(probe.status, "unavailable");
  assert.equal(probe.statusReason, "metrics_unavailable");
});

test("without /api/status a multi-model catalog stays ambiguous rather than guessing", async () => {
  const { probe } = omlxProbe(() => null);
  await probe._detectServerType();
  await probe.probe();
  assert.equal(probe.status, "ambiguous_model");
});

test("the loaded model is selected from the full oMLX catalog", async () => {
  const { probe } = omlxProbe(() => status());
  await probe._detectServerType();
  const snap = await probe.probe();
  assert.equal(probe.modelId, "GLM-5.3-Flash-oQ4e-mtp");
  assert.equal(probe.contextLength, 1048576);
  assert.equal(snap.backend, "omlx");
});
