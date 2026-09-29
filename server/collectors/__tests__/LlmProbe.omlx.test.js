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
  const probe = new LlmProbe({ lanIp: "100.64.0.10" }, 8000);
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

test("while busy the live tiles use oMLX rolling averages; totals and cache come from counters", async () => {
  const { probe, requested } = omlxProbe(() => status());
  await probe._detectServerType();
  requested.length = 0;
  const snap = await probe.probe();
  assert.equal(snap.backend, "omlx");
  assert.equal(probe.generationTps, 57.3);
  assert.equal(probe.prefillTps, 697.8);
  assert.equal(probe.cachedPrefillTps, null);
  assert.equal(probe.totalOutputTokens, 44_711);
  assert.equal(probe.totalPromptTokens, 51_216_751);
  assert.equal(probe.requestsRunning, 1);
  assert.equal(probe.requestsWaiting, 0);
  assert.equal(probe.status, "active");
  assert.equal(probe.prefixCacheHitRate, 0.993);
  assert.equal(probe.gpuMemoryUtilization, 0.7831);
  assert.equal(snap.generationTps, 57.3);
  assert.equal(requested.some((u) => u.endsWith("/metrics") || u.endsWith("/slots")), false);
});

test("counters that only move at request completion never produce spikes", async () => {
  let sample = status();
  const { probe } = omlxProbe(() => sample);
  await probe._detectServerType();
  await probe.probe();
  probe.lastProbeTime = Date.now() - 2000;
  sample = status({ active_requests: 0, total_completion_tokens: 44_711 + 5000 });
  await probe.probe();
  assert.equal(probe.generationTps, 0);
  assert.equal(probe.prefillTps, 0);
  assert.equal(probe.totalOutputTokens, 49_711);
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

// Shapes from a live oMLX main build (GET /admin/api/activity → active_models).
const generating = (id, tokens, elapsed, tps) => ({
  request_id: id,
  elapsed_seconds: elapsed,
  generated_tokens: tokens,
  tokens_per_second: tps,
  last_activity_age_seconds: 0.1,
  prompt_tokens: 15022,
  max_tokens: 600,
});
const prefilling = (id, speed) => ({
  request_id: id,
  processed: 7360,
  total: 15021,
  speed,
  eta: 6.8,
  elapsed: 6.9,
  phase: "prefill",
  detail: null,
});
const activity = ({ gen = [], pre = [] } = {}) => ({
  active_models: {
    models: [{ id: "GLM-5.3-Flash-oQ4e-mtp", generating: gen, prefilling: pre, waiting: [] }],
    total_active_requests: gen.length + pre.length,
    total_waiting_requests: 0,
  },
});

/** oMLX probe with an API key and a scripted admin API. */
function keyedOmlxProbe({ getStatus = () => status(), getActivity, login = () => true } = {}) {
  const probe = new LlmProbe({ lanIp: "100.64.0.10", llmApiKeys: { 8000: "test-admin-key" } }, 8000);
  const calls = [];
  let session = 0;
  let validSession = null;
  probe._fetch = async (url, init = {}) => {
    const u = String(url);
    calls.push({ url: u, init });
    const json = (status, body) => ({ ok: status < 400, status, json: async () => body, headers: { getSetCookie: () => [] } });
    if (u.endsWith("/v1/models")) return json(200, MODELS);
    if (u.endsWith("/api/status")) return json(200, getStatus());
    if (u.endsWith("/admin/api/login")) {
      assert.equal(init.method, "POST");
      assert.deepEqual(JSON.parse(init.body), { api_key: "test-admin-key", remember: true });
      if (!login()) return json(401, { detail: "Invalid API key" });
      validSession = `s${++session}`;
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
        headers: { getSetCookie: () => [`omlx_admin_session=${validSession}; HttpOnly; Path=/; SameSite=lax`] },
      };
    }
    if (u.endsWith("/admin/api/activity")) {
      if (init.headers?.Cookie !== `omlx_admin_session=${validSession}`) return json(401, { detail: "Admin authentication required" });
      return json(200, getActivity());
    }
    return json(404, {});
  };
  return { probe, calls, expireSession: () => (validSession = null) };
}

test("with an API key the live tiles come from the admin activity feed", async () => {
  let act = activity({ gen: [generating("a", 162, 6.74, 24.05)], pre: [prefilling("b", 1130.5)] });
  const { probe, calls } = keyedOmlxProbe({ getActivity: () => act });
  await probe._detectServerType();
  await probe.probe();
  // First sight of a request uses its own running average; prefill uses the tracker speed.
  assert.equal(probe.generationTps, 24.05);
  assert.equal(probe.prefillTps, 1130.5);
  // Next poll: decode is the token delta over elapsed time, summed across requests.
  act = activity({ gen: [generating("a", 188, 7.8, 24.1), generating("c", 16, 0.98, 16.3)] });
  await probe.probe();
  assert.equal(probe.generationTps, Math.round((26 / (7.8 - 6.74) + 16.3) * 100) / 100);
  assert.equal(probe.prefillTps, 0);
  assert.equal(calls.filter((c) => c.url.endsWith("/admin/api/login")).length, 1);
});

test("an expired admin session logs in again", async () => {
  const { probe, calls, expireSession } = keyedOmlxProbe({
    getActivity: () => activity({ gen: [generating("a", 46, 1.41, 32.6)] }),
  });
  await probe._detectServerType();
  await probe.probe();
  expireSession();
  await probe.probe();
  assert.equal(probe.generationTps, 32.6);
  assert.equal(calls.filter((c) => c.url.endsWith("/admin/api/login")).length, 2);
});

test("a rejected admin login falls back to the rolling averages and backs off", async () => {
  const { probe, calls } = keyedOmlxProbe({ login: () => false, getActivity: () => activity() });
  await probe._detectServerType();
  await probe.probe();
  assert.equal(probe.generationTps, 57.3);
  assert.equal(probe.prefillTps, 697.8);
  await probe.probe();
  assert.equal(calls.filter((c) => c.url.endsWith("/admin/api/login")).length, 1);
});

test("the activity feed is not read while oMLX is idle or without a key", async () => {
  const idle = keyedOmlxProbe({ getStatus: () => status({ active_requests: 0 }), getActivity: () => activity() });
  await idle.probe._detectServerType();
  await idle.probe.probe();
  assert.equal(idle.calls.some((c) => c.url.includes("/admin/")), false);
  assert.equal(idle.probe.status, "idle");

  const { probe, requested } = omlxProbe(() => status());
  await probe._detectServerType();
  await probe.probe();
  assert.equal(requested.some((u) => u.includes("/admin/")), false);
  assert.equal(probe.generationTps, 57.3);
});
