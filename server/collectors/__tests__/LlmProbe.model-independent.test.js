import { test } from "node:test";
import { strict as assert } from "node:assert";
import { LlmProbe, selectActiveModel } from "../LlmProbe.js";

function jsonRes(data, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

function textRes(text, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => text,
    json: async () => ({}),
  };
}

test("selectActiveModel accepts alternate ids and selects exactly one loaded entry", () => {
  const result = selectActiveModel([
    { id: "old-model", status: "unloaded" },
    { id: "org/new-model", status: { value: "loaded" } },
  ]);
  assert.equal(result.modelId, "org/new-model");
  assert.deepEqual(result.models, ["old-model", "org/new-model"]);
  assert.equal(result.reason, null);
});

test("selectActiveModel refuses multiple active, multiple unqualified, and all inactive entries", () => {
  assert.equal(
    selectActiveModel([
      { id: "model-a", status: "active" },
      { id: "model-b", status: "ready" },
    ]).reason,
    "ambiguous_model"
  );
  assert.equal(
    selectActiveModel([{ id: "model-a" }, { id: "model-b" }]).reason,
    "ambiguous_model"
  );
  assert.equal(
    selectActiveModel([{ id: "model-a", status: "unloaded" }]).reason,
    "no_active_model"
  );
});

test("ambiguous model discovery does not fetch metrics or guess a benchmark model", async () => {
  const probe = new LlmProbe({ lanIp: "10.0.0.1" }, 8888);
  probe.serverIsOpenAI = true;
  probe.backendType = "vllm";
  probe.authOpen = true;
  probe._lastDetectAt = Date.now();
  const paths = [];
  probe._fetch = async (url) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    paths.push(path);
    if (path === "/v1/models") {
      return jsonRes({ data: [{ id: "router-a" }, { id: "router-b" }] });
    }
    assert.fail(`unexpected fetch ${path}`);
  };
  const snapshot = await probe.probe();
  assert.equal(snapshot.status, "ambiguous_model");
  assert.equal(snapshot.available, false);
  assert.equal(snapshot.modelId, null);
  assert.equal(snapshot.benchmarkModel, null);
  assert.deepEqual(snapshot.models, ["router-a", "router-b"]);
  assert.deepEqual(paths, ["/v1/models"]);
});

test("model labels scope Prometheus counters and relay mapping keeps API-key port", async () => {
  const probe = new LlmProbe(
    { lanIp: "10.0.0.1", llmApiKeys: { "4000": "request-key" } },
    4000,
    { telemetryPort: 9341 }
  );
  probe.serverIsOpenAI = true;
  probe.backendType = "vllm";
  probe.authOpen = true;
  probe.lastProbeTime = Date.now() - 2000;
  probe.lastTokenCounts = { input: 100, output: 50 };
  probe._fetch = async (url) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    if (path === "/v1/models") {
      return jsonRes({
        data: [
          { id: "old-model", status: "unloaded" },
          { id: "new-model", status: "loaded", max_model_len: 1000000 },
        ],
      });
    }
    if (path === "/metrics") {
      return textRes(
        'vllm:prompt_tokens_total{model_name="old-model"} 900\n' +
          'vllm:prompt_tokens_total{model_name="new-model"} 120\n' +
          'vllm:generation_tokens_total{model_name="old-model"} 900\n' +
          'vllm:generation_tokens_total{model_name="new-model"} 90\n' +
          'vllm:num_requests_running{model_name="new-model"} 1\n'
      );
    }
    assert.fail(`unexpected fetch ${path}`);
  };
  const snapshot = await probe.probe();
  assert.equal(probe.port, 4000);
  assert.equal(probe.telemetryPort, 9341);
  assert.equal(probe._apiKey(), "request-key");
  assert.equal(snapshot.modelId, "new-model");
  assert.deepEqual(snapshot.models, ["old-model", "new-model"]);
  assert.equal(snapshot.status, "active");
  assert.equal(snapshot.totalPromptTokens, 120);
  assert.equal(snapshot.totalOutputTokens, 90);
  assert.ok(snapshot.generationTps > 15 && snapshot.generationTps < 25);
});

test("model-labeled metrics with no matching series are unknown, not idle", async () => {
  const probe = new LlmProbe({ lanIp: "10.0.0.1" }, 8888);
  probe.serverIsOpenAI = true;
  probe.backendType = "vllm";
  probe.authOpen = true;
  probe._lastDetectAt = Date.now();
  probe._fetch = async (url) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    if (path === "/v1/models") return jsonRes({ data: [{ id: "active-model", status: "loaded" }] });
    if (path === "/metrics") return textRes('vllm:num_requests_running{model_name="other-model"} 0\n');
    assert.fail(`unexpected fetch ${path}`);
  };
  const snapshot = await probe.probe();
  assert.equal(snapshot.status, "unknown");
  assert.equal(snapshot.statusReason, "metric_model_mismatch");
  assert.equal(snapshot.generationTps, null);
  assert.equal(snapshot.available, false);
});

test("counter resets reseed without producing a negative or stale rate", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.modelId = "model";
  probe.lastTokenCounts = { input: 1000, output: 900 };
  probe._applyVllmMetrics(
    'vllm:prompt_tokens_total{model_name="model"} 10\n' +
      'vllm:generation_tokens_total{model_name="model"} 5\n' +
      'vllm:num_requests_running{model_name="model"} 0\n',
    2
  );
  assert.equal(probe.generationTps, 0);
  assert.equal(probe.prefillTps, 0);
  assert.deepEqual(probe.lastTokenCounts, { input: 10, output: 5 });
});

test("missing vLLM counter series cannot reuse a prior live rate", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.modelId = "model";
  probe._rateBaselineReady = true;
  probe.generationTps = 42;
  probe.prefillTps = 17;
  probe._applyVllmMetrics(
    'vllm:num_requests_running{model_name="model"} 0\n' +
      'vllm:num_requests_waiting{model_name="model"} 0\n',
    2
  );
  assert.equal(probe.generationTps, 0);
  assert.equal(probe.prefillTps, 0);
});

test("malformed Prometheus labels do not cause parser backtracking", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.modelId = "model";
  const malformed = `vllm:num_requests_running{model_name="${"\\".repeat(64)}} 0\n`;
  assert.equal(probe._getPromMetric(malformed, "vllm:num_requests_running"), null);
});

test("unlabeled metrics without workload series remain unknown, not model mismatch", () => {
  const probe = new LlmProbe({ lanIp: "10.0.0.1" }, 8888);
  probe.modelId = "active-model";
  probe.metricsAvailable = true;
  probe._metricsModelSeen = false;
  probe._metricsModelMatched = false;
  probe._getPromMetric("process_cpu_seconds_total 12\n", "process_cpu_seconds_total");
  probe._noteFreshTelemetry(Date.now());
  assert.equal(probe.status, "unknown");
  assert.equal(probe.statusReason, "workload_state_unavailable");
});

test("stale telemetry clears live rates and reports an explicit reason", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.status = "active";
  probe.statusReason = null;
  probe.metricsAvailable = false;
  probe.lastObservedAt = Date.now() - 31_000;
  probe.generationTps = 42;
  const snapshot = probe._getSnapshot();
  assert.equal(snapshot.status, "stale");
  assert.equal(snapshot.statusReason, "telemetry_stale");
  assert.equal(snapshot.generationTps, null);
  assert.equal(snapshot.available, false);
});

test("missing Prometheus info labels stay unknown instead of becoming zero", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.modelId = "model";
  const body = 'vllm:cache_config_info{model_name="model"} 1\n';
  assert.equal(
    probe._getPromInfoNumber(body, "vllm:cache_config_info", "kv_cache_size_tokens"),
    null
  );
  assert.equal(
    probe._getPromInfoNumber(
      'vllm:cache_config_info{model_name="model",kv_cache_size_tokens="123"} 1\n',
      "vllm:cache_config_info",
      "kv_cache_size_tokens"
    ),
    123
  );
});
