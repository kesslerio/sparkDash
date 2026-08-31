/**
 * LlmProbe — probes an LLM server on port 8888, auto-detects backend,
 * computes live tokens/sec (generation + prefill).
 *
 * Ported from legacy `probeLlamaServerType` and `_getLlamaMetricsFor`.
 */
import { LLM_PROBE_TIMEOUT_MS } from "../config.js";
import { classifyHostScope } from "../validate.js";
import { llmProbeHost } from "./llmHost.js";

const FAIL_RESET_THRESHOLD = 3;
const REDETECT_INTERVAL_MS = 60_000;
const TELEMETRY_STALE_MS = 30_000;
const ACTIVE_MODEL_STATUSES = new Set(["active", "loaded", "ready", "running", "serving"]);
const INACTIVE_MODEL_STATUSES = new Set([
  "idle",
  "loading",
  "starting",
  "stopped",
  "stopping",
  "unloaded",
  "exited",
]);
/**
 * SGLang's last_gen_throughput is a sticky gauge (holds last decode rate when
 * idle). Only treat it as live after we observe a change between polls, and
 * expire back to 0 if it stops changing.
 */
const SGLANG_STICKY_TPS_LIVE_MS = 6_000;

/**
 * Prefer a short model id when the server returns a Hugging Face hub cache path.
 * e.g. /root/.cache/huggingface/hub/models--org--Name/snapshots/<hash>
 *   → org/Name
 * @param {unknown} id
 * @returns {string | null}
 */
export function normalizeModelId(id) {
  if (id == null) return null;
  const s = String(id).trim();
  if (!s) return null;

  const hub = s.match(/(?:^|\/)models--([^/]+?)(?:\/snapshots\/[^/]+)?\/?$/);
  if (hub) return hub[1].replace(/--/g, "/");

  const mid = s.match(/models--([^/]+)\/snapshots\//);
  if (mid) return mid[1].replace(/--/g, "/");

  return s;
}

/** True when `id` looks like a Hugging Face hub cache directory (models--org--name). */
export function isHfHubCachePath(id) {
  if (id == null) return false;
  return /(?:^|\/)models--[^/]+/.test(String(id));
}

/**
 * Set modelId (always normalized) and modelPath (omit HF hub cache paths —
 * they duplicate the short id and clutter the LLM panel).
 * @param {unknown} raw
 */
function applyModelRef(probe, raw) {
  if (raw == null || raw === "") return;
  const s = String(raw);
  const nextId = normalizeModelId(s);
  if (probe.modelId && nextId && probe.modelId !== nextId) probe._resetRateBaselines();
  probe.modelId = nextId;
  probe.modelPath = isHfHubCachePath(s) ? null : s;
}

function modelStatusValue(model) {
  if (!model || typeof model !== "object") return null;
  let value = model.status ?? model.state ?? model.model_status;
  if (value && typeof value === "object") value = value.value;
  if (typeof value === "string") return value.trim().toLowerCase() || null;
  if (model.active === true || model.loaded === true || model.ready === true) return "active";
  if (model.loaded === false || model.active === false) return "unloaded";
  return null;
}

/**
 * Select the one model that the telemetry source says is resident/usable.
 * A single unqualified model is the normal OpenAI-compatible server shape;
 * multiple unqualified router entries are intentionally ambiguous.
 * @param {unknown} entries
 * @returns {{ model: Record<string, unknown> | null, modelId: string | null, models: string[], reason: string | null }}
 */
export function selectActiveModel(entries) {
  const rows = Array.isArray(entries) ? entries : [];
  const valid = rows.filter(
    (row) => row && typeof row === "object" && normalizeModelId(row.id)
  );
  const catalog = [];
  const seen = new Set();
  for (const row of valid) {
    const id = normalizeModelId(row.id);
    if (id && !seen.has(id)) {
      seen.add(id);
      catalog.push(id);
    }
  }
  if (!valid.length) {
    return { model: null, modelId: null, models: catalog, reason: "no_active_model" };
  }
  const active = valid.filter((row) => ACTIVE_MODEL_STATUSES.has(modelStatusValue(row)));
  if (active.length > 1) {
    return { model: null, modelId: null, models: catalog, reason: "ambiguous_model" };
  }
  if (active.length === 1) {
    return {
      model: active[0],
      modelId: normalizeModelId(active[0].id),
      models: catalog,
      reason: null,
    };
  }
  if (valid.length === 1 && modelStatusValue(valid[0]) == null) {
    return {
      model: valid[0],
      modelId: normalizeModelId(valid[0].id),
      models: catalog,
      reason: null,
    };
  }
  if (valid.every((row) => INACTIVE_MODEL_STATUSES.has(modelStatusValue(row)))) {
    return { model: null, modelId: null, models: catalog, reason: "no_active_model" };
  }
  return { model: null, modelId: null, models: catalog, reason: "ambiguous_model" };
}

/** Parse the small label grammar used by Prometheus exposition. */
function parsePromLabels(raw) {
  const labels = {};
  if (typeof raw !== "string") return labels;
  // Keep the alternatives disjoint: a backslash is either an escape prefix
  // or a non-escape character, never both. This prevents malformed labels
  // from causing catastrophic backtracking on the Node event loop.
  const re = /([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"/g;
  let match;
  while ((match = re.exec(raw)) !== null) {
    labels[match[1]] = match[2].replace(/\\([\\"nrt])/g, (_all, escaped) => {
      if (escaped === "n") return "\n";
      if (escaped === "r") return "\r";
      if (escaped === "t") return "\t";
      return escaped;
    });
  }
  return labels;
}

function promLabelValue(raw, key) {
  return parsePromLabels(raw)[key] ?? null;
}

export class LlmProbe {
  constructor(spark, port = 8888, options = {}) {
    this.spark = spark;
    this.port = Number.isInteger(Number(port)) ? Number(port) : 8888;
    const configuredTelemetryPort = Number(options?.telemetryPort);
    this.telemetryPort =
      Number.isInteger(configuredTelemetryPort) && configuredTelemetryPort >= 1 && configuredTelemetryPort <= 65535
        ? configuredTelemetryPort
        : this.port;
    this.telemetrySource = this.telemetryPort === this.port ? "direct" : "relay";
    this.baseUrl = `http://${llmProbeHost(spark)}:${this.telemetryPort}`;

    // State
    this.backendType = null; // 'vllm' | 'llama.cpp' | 'sglang' | 'ds4' | null
    this.serverIsOpenAI = null; // true = OpenAI-compatible
    /** Whether /v1/models (or /slots) answered without credentials. null = unknown. */
    this.authOpen = null;
    this.stepId = 0;
    this.modelId = null;
    this.modelPath = null;
    this.benchmarkModel = null;
    this.models = [];
    this.contextLength = null;
    this.gpuMemoryUtilization = null;
    this.slotsActive = 0;
    this.slotsTotal = 0;
    this.generationTps = 0;
    this.prefillTps = 0;
    /** Live cached-prefill tok/s when the backend splits kinds (ds4 / llama.cpp / sglang). null otherwise. */
    this.cachedPrefillTps = null;
    /** Live uncached/computed prefill tok/s when split is available. null otherwise. */
    this.uncachedPrefillTps = null;
    this.error = null;
    /** Fresh telemetry state; unavailable is never rendered as idle. */
    this.status = "unavailable";
    this.statusReason = "not_observed";
    this.lastObservedAt = null;
    this.metricsAvailable = false;
    this._metricsModelMatched = false;
    this._metricsModelSeen = false;

    // Per-slot rate tracking (for llama.cpp native path)
    this.slotState = new Map();
    this.lastTokenCounts = { input: 0, output: 0 };
    /** Previous prefill counters by kind; null until first labeled sample. */
    this.lastPrefillKinds = null;
    /** Previous vLLM TTFT histogram `_sum` (seconds). null until first sample. */
    this.lastTtftSum = null;
    /** Previous `vllm:iteration_tokens_total_sum` (engine-step tokens). */
    this.lastIterSum = null;
    this.lastProbeTime = 0;
    this._rateBaselineReady = false;

    // Cumulative total output tokens (generation) as reported by the LLM server
    this.totalOutputTokens = 0;
    // Cumulative process/runtime prompt tokens, when reported by the backend.
    this.totalPromptTokens = null;
    // Cumulative completed requests, when reported by the backend.
    this.completedRequestsTotal = null;
    this._promptCounterObserved = false;
    this._outputCounterObserved = false;
    this._completedCounterObserved = false;

    // vLLM inference metrics from /metrics (null when not vLLM / missing series)
    // Metric names follow stock vLLM Prometheus exposition (versions may differ).
    this.kvCacheUsage = null; // 0–1 fraction
    /** Engine-reported total KV cache token pool. */
    this.kvCacheCapacityTokens = null;
    /** Engine-reported theoretical max concurrency at configured model length. */
    this.kvCacheMaxConcurrency = null;
    this.requestsRunning = null;
    this.requestsWaiting = null;
    this.ttftP95Seconds = null;
    this.preemptionsTotal = null; // cumulative counter
    /** Prefix cache hit rate 0–1 (hits/queries). */
    this.prefixCacheHitRate = null;
    /** End-to-end request latency p95 (seconds). */
    this.e2eP95Seconds = null;
    /** Inter-token latency p95 (seconds). */
    this.itlP95Seconds = null;
    /** Speculative/MTP acceptance rate 0–1 (accepted/drafted). */
    this.mtpAcceptanceRate = null;

    this._consecutiveFailures = 0;
    this._lastDetectAt = 0;
    /** @type {{ value: number, liveUntil: number } | null} */
    this._sglangStickyTps = null;
  }

  /**
   * Prefill is a short burst; decode then runs with Δprompt=0.
   * Keep the last real prefill rate while the engine is still in-flight, then 0.
   * @param {number} rate
   * @param {boolean} inflight
   */
  _setPrefillTps(rate, inflight) {
    const rounded = Math.max(0, Math.round(rate * 100) / 100);
    if (rounded > 0) this.prefillTps = rounded;
    else if (!inflight) this.prefillTps = 0;
  }

  /**
   * Live cached vs computed prefill tok/s from cumulative counters.
   * First sample seeds the baseline (0 tok/s). Missing either side clears the split.
   * @param {number|null|undefined} cachedCount
   * @param {number|null|undefined} computedCount
   * @param {number} dtSec
   */
  _setPrefillSplitRates(cachedCount, computedCount, dtSec) {
    if (cachedCount == null || computedCount == null || !Number.isFinite(cachedCount) || !Number.isFinite(computedCount)) {
      this.cachedPrefillTps = null;
      this.uncachedPrefillTps = null;
      this.lastPrefillKinds = null;
      return;
    }
    const total = cachedCount + computedCount;
    this.prefixCacheHitRate =
      total > 0 ? Math.round((cachedCount / total) * 10000) / 10000 : null;
    if (this.lastPrefillKinds == null) {
      this.lastPrefillKinds = { cached: cachedCount, computed: computedCount };
      this.cachedPrefillTps = 0;
      this.uncachedPrefillTps = 0;
      return;
    }
    if (dtSec > 0 && dtSec < 10) {
      const dCached = cachedCount - this.lastPrefillKinds.cached;
      const dComputed = computedCount - this.lastPrefillKinds.computed;
      this.cachedPrefillTps = Math.max(0, Math.round((dCached / dtSec) * 100) / 100);
      this.uncachedPrefillTps = Math.max(0, Math.round((dComputed / dtSec) * 100) / 100);
    }
    this.lastPrefillKinds = { cached: cachedCount, computed: computedCount };
  }

  /** Update probe port (and host from spark). Resets detection when the target changes. */
  setPort(port) {
    const next = Number(port);
    const wasDirect = this.telemetryPort === this.port;
    const prevUrl = this.baseUrl;
    if (Number.isInteger(next) && next >= 1 && next <= 65535) {
      this.port = next;
      if (wasDirect) this.telemetryPort = next;
    }
    this.telemetrySource = this.telemetryPort === this.port ? "direct" : "relay";
    this.baseUrl = `http://${llmProbeHost(this.spark)}:${this.telemetryPort}`;
    if (this.baseUrl !== prevUrl) {
      this._resetDetection();
      this._lastDetectAt = 0;
      this._consecutiveFailures = 0;
    }
  }

  /** Update the telemetry/model-discovery port while preserving request-port API keys. */
  setTelemetryPort(port) {
    const next = Number(port);
    if (!Number.isInteger(next) || next < 1 || next > 65535) return;
    const prevUrl = this.baseUrl;
    this.telemetryPort = next;
    this.telemetrySource = this.telemetryPort === this.port ? "direct" : "relay";
    this.baseUrl = `http://${llmProbeHost(this.spark)}:${this.telemetryPort}`;
    if (this.baseUrl !== prevUrl) {
      this._resetDetection();
      this._lastDetectAt = 0;
      this._consecutiveFailures = 0;
    }
  }

  /** Probe the LLM server and return a snapshot. */
  async probe() {
    try {
      const shouldDetect =
        this.serverIsOpenAI === null ||
        Date.now() - this._lastDetectAt > REDETECT_INTERVAL_MS;

      if (shouldDetect) {
        await this._detectServerType();
        this._lastDetectAt = Date.now();
      }

      if (this.serverIsOpenAI === false) {
        const snap = await this._probeLlamaCpp();
        this._noteSuccess();
        return snap;
      } else if (this.serverIsOpenAI === true) {
        const snap = await this._probeOpenAICompatible();
        this._noteSuccess();
        return snap;
      } else {
        this._noteFailure("LLM server not reachable");
        return this._defaultLlm();
      }
    } catch (err) {
      this._noteFailure(err.message);
      return this._defaultLlm();
    }
  }

  _noteSuccess() {
    this._consecutiveFailures = 0;
    this.error = null;
  }

  _noteFailure(message) {
    this.error = message;
    this.metricsAvailable = false;
    this._metricsModelMatched = false;
    this._metricsModelSeen = false;
    this.statusReason = message || "telemetry_unavailable";
    this.status = this.lastObservedAt != null ? "stale" : "unavailable";
    this._consecutiveFailures += 1;
    if (this._consecutiveFailures >= FAIL_RESET_THRESHOLD) {
      this._resetDetection();
    }
  }

  _resetDetection() {
    this.serverIsOpenAI = null;
    this.backendType = null;
    this.authOpen = null;
    this.modelId = null;
    this.modelPath = null;
    this.benchmarkModel = null;
    this.models = [];
    this.generationTps = 0;
    this.prefillTps = 0;
    this.cachedPrefillTps = null;
    this.uncachedPrefillTps = null;
    this.status = "unavailable";
    this.statusReason = "not_observed";
    this.lastObservedAt = null;
    this.metricsAvailable = false;
    this._metricsModelMatched = false;
    this._metricsModelSeen = false;
    this.contextLength = null;
    this.gpuMemoryUtilization = null;
    this.slotsActive = 0;
    this.slotsTotal = 0;
    this.totalOutputTokens = 0;
    this.totalPromptTokens = null;
    this.completedRequestsTotal = null;
    this._promptCounterObserved = false;
    this._outputCounterObserved = false;
    this._completedCounterObserved = false;
    this.kvCacheUsage = null;
    this.kvCacheCapacityTokens = null;
    this.kvCacheMaxConcurrency = null;
    this.requestsRunning = null;
    this.requestsWaiting = null;
    this.ttftP95Seconds = null;
    this.preemptionsTotal = null;
    this.prefixCacheHitRate = null;
    this.e2eP95Seconds = null;
    this.itlP95Seconds = null;
    this.mtpAcceptanceRate = null;
    this.slotState.clear();
    this.lastTokenCounts = { input: 0, output: 0 };
    this.lastPrefillKinds = null;
    this.lastTtftSum = null;
    this.lastIterSum = null;
    this._sglangStickyTps = null;
    this._rateBaselineReady = false;
  }

  /** Clear all counter-delta state after a model swap or source change. */
  _resetRateBaselines() {
    this.generationTps = 0;
    this.prefillTps = 0;
    this.cachedPrefillTps = null;
    this.uncachedPrefillTps = null;
    this.slotState.clear();
    this.lastTokenCounts = { input: 0, output: 0 };
    this.lastPrefillKinds = null;
    this.lastTtftSum = null;
    this.lastIterSum = null;
    this._sglangStickyTps = null;
    this.totalPromptTokens = null;
    this.totalOutputTokens = 0;
    this.completedRequestsTotal = null;
    this._promptCounterObserved = false;
    this._outputCounterObserved = false;
    this._completedCounterObserved = false;
    this._rateBaselineReady = false;
  }

  /** Apply the model catalog and select exactly one resident/usable model. */
  _acceptModelCatalog(models) {
    const selection = selectActiveModel(models);
    const previous = this.modelId;
    this.models = selection.models;
    if (
      selection.modelId &&
      previous &&
      normalizeModelId(previous) !== normalizeModelId(selection.modelId)
    ) {
      this._resetRateBaselines();
      this.statusReason = "model_changed";
    }

    if (selection.modelId && selection.model) {
      this.modelId = selection.modelId;
      this.benchmarkModel = selection.modelId;
      this.modelPath = isHfHubCachePath(selection.model.id) ? null : String(selection.model.id);
      this.contextLength =
        selection.model.max_model_len ??
        selection.model.context_length ??
        this.contextLength;
      this.statusReason = null;
      return selection;
    }

    this.modelId = null;
    this.modelPath = selection.models.join(", ") || null;
    this.benchmarkModel = null;
    this.contextLength = null;
    this.metricsAvailable = false;
    this._metricsModelMatched = false;
    this._metricsModelSeen = false;
    this.status = selection.reason === "ambiguous_model" ? "ambiguous_model" : "unavailable";
    this.statusReason = selection.reason;
    this._resetRateBaselines();
    return selection;
  }

  /** Mark a successful fresh metrics/model observation and derive active vs idle. */
  _noteFreshTelemetry(observedAt = Date.now()) {
    this.metricsAvailable = true;
    this.lastObservedAt = observedAt;
    this._setWorkloadStatus();
  }

  _setWorkloadStatus() {
    if (!this.metricsAvailable) return;
    if (this._metricsModelSeen && this._metricsModelMatched === false) {
      this.status = "unknown";
      this.statusReason = "metric_model_mismatch";
      return;
    }
    const running =
      (this.requestsRunning != null && this.requestsRunning > 0) ||
      this.slotsActive > 0 ||
      this.generationTps > 0 ||
      this.prefillTps > 0;
    const waiting = this.requestsWaiting != null && this.requestsWaiting > 0;
    if (running || waiting) {
      this.status = "active";
      this.statusReason = null;
      return;
    }
    // A fresh metrics response is only evidence of idleness when it contains
    // an explicit zero-work signal. Counters alone (or an otherwise empty
    // metrics body) must not be rendered as a proven zero rate.
    const hasWorkloadSignal =
      this.requestsRunning != null ||
      this.requestsWaiting != null ||
      this.slotsTotal > 0;
    if (hasWorkloadSignal && (this.requestsRunning == null || this.requestsRunning === 0) &&
        (this.requestsWaiting == null || this.requestsWaiting === 0) &&
        this.slotsActive === 0) {
      this.status = "idle";
      this.statusReason = null;
      return;
    }
    this.status = "unknown";
    this.statusReason = "workload_state_unavailable";
  }

  /** Note auth from an HTTP status on an unauthenticated probe request. */
  _noteAuthStatus(status) {
    if (status >= 200 && status < 300) {
      this.authOpen = true;
      return "ok";
    }
    if (status === 401 || status === 403) {
      this.authOpen = false;
      return "auth";
    }
    return "other";
  }

  // ─── Server type detection ───────────────────────────────
  async _detectServerType() {
    // Skip the llama.cpp /slots probe once we've positively identified an
    // OpenAI-compatible backend. vLLM / sglang / ds4-server have no /slots,
    // so re-probing it on every re-detect cycle just spams 404s in the
    // backend's access log (#15). Still probe /slots on first contact, when
    // the type is unknown, or when the backend was previously llama.cpp.
    if (
      this.backendType !== "vllm" &&
      this.backendType !== "sglang" &&
      this.backendType !== "ds4"
    ) {
      const slotUrl = `${this.baseUrl}/slots`;
      try {
        const slotRes = await this._fetch(slotUrl);
        const auth = this._noteAuthStatus(slotRes.status);
        if (auth === "ok") {
          const slots = await slotRes.json();
          if (Array.isArray(slots)) {
            this.serverIsOpenAI = false;
            this.backendType = "llama.cpp";
            return;
          }
        } else if (auth === "auth") {
          // Authenticated llama.cpp — treat as protected OpenAI-style for posture
          this.serverIsOpenAI = false;
          this.backendType = "llama.cpp";
          return;
        }
      } catch {}
    }

    // Try OpenAI-compatible (vLLM, SGLang, or ds4-server)
    try {
      const modelRes = await this._fetch(`${this.baseUrl}/v1/models`);
      const auth = this._noteAuthStatus(modelRes.status);
      if (auth === "ok" || auth === "auth") {
        this.serverIsOpenAI = true;
        let owned = null;
        if (auth === "ok") {
          try {
            const modelsData = await modelRes.json();
            owned = modelsData?.data?.[0]?.owned_by;
          } catch {
            /* body optional for detection */
          }
        }
        this.backendType = await this._classifyOpenAIBackend(owned);
        return;
      }
    } catch {}

    this.serverIsOpenAI = null;
    this.backendType = null;
  }

  /**
   * Classify an OpenAI-compatible server: ds4-server, SGLang, or vLLM (default).
   * @param {unknown} ownedBy
   * @returns {Promise<"ds4" | "sglang" | "vllm">}
   */
  async _classifyOpenAIBackend(ownedBy) {
    if (typeof ownedBy === "string") {
      if (/ds4/i.test(ownedBy)) return "ds4";
      if (/sglang/i.test(ownedBy)) return "sglang";
    }
    if (await this._probeIsDs4()) return "ds4";
    if (await this._probeIsSglang()) return "sglang";
    return "vllm";
  }

  /** True when SGLang native server-info endpoints respond. */
  async _probeIsSglang() {
    for (const path of ["/get_server_info", "/server_info"]) {
      try {
        const res = await this._fetch(`${this.baseUrl}${path}`);
        if (!res.ok) continue;
        const data = await res.json().catch(() => null);
        if (data && typeof data === "object" && !Array.isArray(data)) return true;
      } catch {
        /* try next */
      }
    }
    return false;
  }

  /** True when Prometheus /metrics exposes ds4-server series (ds4-on-spark). */
  async _probeIsDs4() {
    try {
      const res = await this._fetch(`${this.baseUrl}/metrics`);
      if (!res.ok) return false;
      const txt = await res.text();
      return LlmProbe._metricsLookLikeDs4(txt);
    } catch {
      return false;
    }
  }

  /** @param {string} body */
  static _metricsLookLikeDs4(body) {
    return /(?:^|\n)ds4_tokens_decoded_total(?:\{|\s)/m.test(String(body || ""));
  }

  // ─── OpenAI-compatible path (vLLM/sglang/ds4) ────────────
  async _probeOpenAICompatible() {
    const now = Date.now();
    const dtSec = (now - this.lastProbeTime) / 1000;
    this.lastProbeTime = now;
    this.metricsAvailable = false;
    this._metricsModelMatched = false;
    this._metricsModelSeen = false;

    // Model info from /v1/models — 401/403 means protected; other failure = down
    let modelsOk = false;
    let owned = null;
    try {
      const modelsRes = await this._fetch(`${this.baseUrl}/v1/models`);
      const auth = this._noteAuthStatus(modelsRes.status);
      if (auth === "auth") {
        this.status = "unavailable";
        this.statusReason = "model_catalog_auth_required";
        this.models = [];
        this.modelId = null;
        this.benchmarkModel = null;
        return this._getSnapshot();
      }
      if (auth === "ok") {
        modelsOk = true;
        const modelsData = await modelsRes.json();
        const models = Array.isArray(modelsData?.data) ? modelsData.data : [];
        const selection = this._acceptModelCatalog(models);
        if (selection.model) owned = selection.model.owned_by;
        if (selection.reason) return this._getSnapshot();
      }
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : "OpenAI-compatible model catalog invalid");
    }

    if (!modelsOk) {
      throw new Error("OpenAI-compatible /v1/models unreachable");
    }

    // Self-heal backend from owned_by before branching (cheap, no extra HTTP)
    if (typeof owned === "string") {
      if (/ds4/i.test(owned)) this.backendType = "ds4";
      else if (/sglang/i.test(owned) && this.backendType !== "ds4") {
        this.backendType = "sglang";
      }
    }

    // SGLang: native info endpoints. Skip on known vLLM/ds4 to avoid 404 spam.
    if (this.backendType === "sglang" || this.backendType == null) {
      try {
        const sgRes = await this._fetch(`${this.baseUrl}/get_server_info`);
        if (sgRes.ok) {
          this.backendType = "sglang";
          this._metricsModelMatched = true;
          const sgData = await sgRes.json();
          // Load before last_gen_throughput so inflight can keep a steady rate live.
          await this._probeSglangLoad();
          this._applySglangServerInfo(sgData, dtSec);
          // Engine tile: Active vs Sleeping. SGLang has no live sleep gauge
          // (sleep_on_idle is a launch flag, not current state). A reachable
          // server with weights resident is Active / ready.
          if (this.gpuMemoryUtilization == null) this.gpuMemoryUtilization = 1;
          this._noteFreshTelemetry(now);
        }
      } catch {}
    }

    if (this.backendType === "sglang") {
      // Prometheus is optional (--enable-metrics). Do not mix those counters
      // into lastTokenCounts when /get_server_info already produced live rates.
      try {
        const metricsRes = await this._fetch(`${this.baseUrl}/metrics`);
        if (metricsRes.ok) {
          const txt = await metricsRes.text();
          const idle = this.generationTps === 0 && this.prefillTps === 0;
          if (idle) this._applySglangMetrics(txt, dtSec);
          else this._applySglangPrefillSplit(txt, dtSec);
          this._noteFreshTelemetry(Date.now());
        }
      } catch {
        /* metrics optional */
      }
      await this._enrichSglangModelInfo();
      return this._getSnapshot();
    }

    // Single /metrics fetch: ds4-server or vLLM Prometheus exposition
    try {
      const metricsRes = await this._fetch(`${this.baseUrl}/metrics`);
      if (metricsRes.ok) {
        const txt = await metricsRes.text();
        if (
          this.backendType === "ds4" ||
          LlmProbe._metricsLookLikeDs4(txt)
        ) {
          this.backendType = "ds4";
          this._applyDs4Metrics(txt, dtSec);
        } else {
          this.backendType = "vllm";
          this._applyVllmMetrics(txt, dtSec);
        }
        this._noteFreshTelemetry(Date.now());
      } else {
        this.status = "unavailable";
        this.statusReason = "metrics_unavailable";
        if (this.backendType !== "ds4") {
          this.backendType = "vllm";
        }
      }
    } catch {
      if (this.backendType !== "ds4") this.backendType = "vllm";
      this.status = "unavailable";
      this.statusReason = "metrics_unavailable";
    }

    return this._getSnapshot();
  }

  /**
   * Apply ds4-server Prometheus /metrics (Entrpi/ds4-on-spark).
   * Live tok/s from counter diffs (same as vLLM) so idle → 0. The engine's
   * `ds4_decode_tok_s` / `ds4_prefill_tok_s` gauges are ~60s windows and stay
   * non-zero long after requests finish — do not use them for the live panel.
   * @param {string} txt
   * @param {number} dtSec
   */
  _applyDs4Metrics(txt, dtSec) {
    this.generationTps = 0;
    this.prefillTps = 0;
    const decoded = this._getPromMetric(txt, "ds4_tokens_decoded_total");
    const computedPrefill = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "computed"
    );
    const prefilled =
      computedPrefill ?? this._getPromMetric(txt, "ds4_tokens_prefilled_total");
    const inflightHint = this._getPromMetric(txt, "ds4_requests_inflight");
    const inflight = inflightHint != null && inflightHint > 0;

    if (decoded != null) {
      const baselineReady =
        this._rateBaselineReady ||
        this.lastTokenCounts.input !== 0 ||
        this.lastTokenCounts.output !== 0;
      const counterReset =
        (prefilled != null && prefilled < this.lastTokenCounts.input) ||
        decoded < this.lastTokenCounts.output;
      if (prefilled != null && dtSec > 0 && dtSec < 10) {
        const deltaIn = prefilled - this.lastTokenCounts.input;
        const deltaOut = decoded - this.lastTokenCounts.output;
        if (!baselineReady || counterReset) {
          this.generationTps = 0;
          this.prefillTps = 0;
        } else {
          this.generationTps = Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
          this.prefillTps = Math.max(0, Math.round((deltaIn / dtSec) * 100) / 100);
        }
      } else if (dtSec > 0 && dtSec < 10) {
        const deltaOut = decoded - this.lastTokenCounts.output;
        this.generationTps = !baselineReady || counterReset
          ? 0
          : Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        if (!inflight && (counterReset || deltaOut <= 0)) this.prefillTps = 0;
      }
      if (prefilled != null) this.lastTokenCounts.input = prefilled;
      this.lastTokenCounts.output = decoded;
      this._rateBaselineReady = true;
      this.totalOutputTokens = decoded;
      this.totalPromptTokens = prefilled;
      this._outputCounterObserved = true;
      this._promptCounterObserved = prefilled != null;
    } else {
      // No counters — fall back to window gauges only while something is in flight
      const gaugeGen = this._getPromMetric(txt, "ds4_decode_tok_s");
      const gaugePrefill = this._getPromMetric(txt, "ds4_prefill_tok_s");
      if (inflight) {
        if (gaugeGen != null) {
          this.generationTps = Math.max(0, Math.round(gaugeGen * 100) / 100);
        }
        if (gaugePrefill != null) {
          this.prefillTps = Math.max(0, Math.round(gaugePrefill * 100) / 100);
        }
      } else {
        this.generationTps = 0;
        this.prefillTps = 0;
      }
    }

    this.completedRequestsTotal =
      this._getPromMetric(txt, "ds4_requests_completed_total") ??
      this._getPromMetric(txt, "ds4_requests_completed");
    this._completedCounterObserved = this.completedRequestsTotal != null;

    const inflightCount = this._getPromMetric(txt, "ds4_requests_inflight");
    this.requestsRunning = inflightCount;
    if (inflightCount != null) this.slotsActive = Math.round(inflightCount);

    const banksTotal = this._getPromMetric(txt, "ds4_banks_total");
    if (banksTotal != null) this.slotsTotal = Math.round(banksTotal);

    const specAccept = this._getPromMetric(txt, "ds4_spec_accept_ratio");
    this.mtpAcceptanceRate =
      specAccept != null ? Math.round(specAccept * 10000) / 10000 : null;

    // Prefix-cache hit rate + live tok/s from prefill kind labels
    const cached = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "cached"
    );
    const computed = this._getPromMetricLabeled(
      txt,
      "ds4_tokens_prefilled_total",
      "kind",
      "computed"
    );
    this._setPrefillSplitRates(cached, computed, dtSec);

    // Clear tiles that are vLLM-histogram-specific (no ds4 equivalent yet)
    this.kvCacheUsage = null;
    this.kvCacheCapacityTokens = null;
    this.kvCacheMaxConcurrency = null;
    this.requestsWaiting = null;
    this.ttftP95Seconds = null;
    this.preemptionsTotal = null;
    this.e2eP95Seconds = null;
    this.itlP95Seconds = null;
  }

  /**
   * Apply stock vLLM Prometheus /metrics (tok/s + inference tiles).
   * @param {string} txt
   * @param {number} dtSec
   */
  _applyVllmMetrics(txt, dtSec) {
    this.generationTps = 0;
    this.prefillTps = 0;
    const promptTokens = this._getVllmMetric(txt, "prompt_tokens_total");
    const genTokens = this._getVllmMetric(txt, "generation_tokens_total");
    const running = this._getVllmMetric(txt, "num_requests_running");
    const iterSum = this._getVllmMetric(txt, "iteration_tokens_total_sum");
    if (promptTokens != null && genTokens != null) {
      const baselineReady =
        this._rateBaselineReady ||
        this.lastTokenCounts.input !== 0 ||
        this.lastTokenCounts.output !== 0;
      const deltaIn = promptTokens - this.lastTokenCounts.input;
      const deltaOut = genTokens - this.lastTokenCounts.output;
      const counterReset = deltaIn < 0 || deltaOut < 0;
      this.lastTokenCounts.input = promptTokens;
      this.lastTokenCounts.output = genTokens;
      this.totalPromptTokens = promptTokens;
      this.totalOutputTokens = genTokens;
      this._promptCounterObserved = true;
      this._outputCounterObserved = true;
      const ttftSum = this._getVllmMetric(txt, "time_to_first_token_seconds_sum");
      const deltaIter =
        iterSum != null && this.lastIterSum != null ? iterSum - this.lastIterSum : 0;
      if (dtSec > 0 && dtSec < 10) {
        this.generationTps = !baselineReady || counterReset
          ? 0
          : Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
        if (!baselineReady || counterReset) this.prefillTps = 0;
        const deltaTtft =
          ttftSum != null && this.lastTtftSum != null ? ttftSum - this.lastTtftSum : 0;
        // Engine-step tokens include prefill+decode. Surplus over generation is
        // prefill, including the common case where a short/cached prefill lands
        // in the same poll as the first decode tokens.
        const prefillIter =
          !baselineReady || counterReset ? 0 : Math.max(0, deltaIter - Math.max(0, deltaOut));
        const specNoise = deltaOut > 0 && prefillIter > 0 && prefillIter < deltaOut * 0.5;
        const livePrefill =
          prefillIter > 0 && !specNoise ? prefillIter / dtSec : 0;
        const finishedPrefill =
          deltaIn > 0 && deltaTtft > 0
            ? deltaIn / deltaTtft
            : deltaIn > 0 && livePrefill <= 0
              ? deltaIn / dtSec
              : 0;
        this.prefillTps = Math.max(
          0,
          Math.round((livePrefill > 0 ? livePrefill : finishedPrefill) * 100) / 100
        );
      }
      if (ttftSum != null) this.lastTtftSum = ttftSum;
      this._rateBaselineReady = true;
    }
    if (iterSum != null) this.lastIterSum = iterSum;

    this.requestsRunning = running;
    if (running != null) this.slotsActive = Math.round(running);

    if (this.gpuMemoryUtilization == null) {
      const sleepState = this._getVllmMetric(txt, "engine_sleep_state");
      if (sleepState != null) this.gpuMemoryUtilization = sleepState;
    }

    this.requestsWaiting = this._getVllmMetric(txt, "num_requests_waiting");
    this.kvCacheUsage = this._getVllmMetric(txt, "kv_cache_usage_perc");
    this.kvCacheCapacityTokens = this._getPromInfoNumber(
      txt,
      "vllm:cache_config_info",
      "kv_cache_size_tokens"
    );
    this.kvCacheMaxConcurrency = this._getPromInfoNumber(
      txt,
      "vllm:cache_config_info",
      "kv_cache_max_concurrency"
    );
    this.preemptionsTotal = this._getVllmMetric(txt, "num_preemptions_total");
    this.completedRequestsTotal =
      this._getVllmMetric(txt, "num_requests_success_total") ??
      this._getVllmMetric(txt, "request_success_total");
    this._completedCounterObserved = this.completedRequestsTotal != null;

    const ttftHist = this._parseVllmHistogram(txt, "vllm:time_to_first_token_seconds");
    const ttftP95 = this._histogramQuantile(ttftHist.buckets, ttftHist.total, 0.95);
    this.ttftP95Seconds = ttftP95 == null ? null : Math.round(ttftP95 * 1000) / 1000;

    const e2eHist = this._parseVllmHistogram(txt, "vllm:e2e_request_latency_seconds");
    const e2eP95 = this._histogramQuantile(e2eHist.buckets, e2eHist.total, 0.95);
    this.e2eP95Seconds = e2eP95 == null ? null : Math.round(e2eP95 * 1000) / 1000;

    const itlHist = this._parseVllmHistogram(txt, "vllm:inter_token_latency_seconds");
    const itlP95 = this._histogramQuantile(itlHist.buckets, itlHist.total, 0.95);
    this.itlP95Seconds = itlP95 == null ? null : Math.round(itlP95 * 1000) / 1000;

    const prefixHits = this._getVllmMetric(txt, "prefix_cache_hits_total");
    const prefixQueries = this._getVllmMetric(txt, "prefix_cache_queries_total");
    this.prefixCacheHitRate =
      prefixHits != null && prefixQueries != null && prefixQueries > 0
        ? Math.round((prefixHits / prefixQueries) * 10000) / 10000
        : null;

    const mtpAccepted = this._getVllmMetric(txt, "spec_decode_num_accepted_tokens_total");
    const mtpDrafted = this._getVllmMetric(txt, "spec_decode_num_draft_tokens_total");
    this.mtpAcceptanceRate =
      mtpAccepted != null && mtpDrafted != null && mtpDrafted > 0
        ? Math.round((mtpAccepted / mtpDrafted) * 10000) / 10000
        : null;
  }

  /**
   * Apply SGLang /get_server_info.
   * Older builds expose total_input_tokens / total_output_tokens.
   * Current builds (metrics often off) expose sticky last_gen_throughput under
   * internal_states[i]. Only treat it as live after the value changes between
   * polls, then expire to 0 when it stops moving (idle leftover).
   * @param {Record<string, unknown>} sgData
   * @param {number} dtSec
   */
  _applySglangServerInfo(sgData, dtSec) {
    this.kvCacheCapacityTokens = null;
    this.kvCacheMaxConcurrency = null;
    // Prefer true max context (context_length / max_total_tokens). Do NOT use
    // max_total_num_tokens — that is the KV-cache pool budget across concurrent
    // sequences and is often ~2× the configured context (showed 2.1M for a 1M run).
    const explicitCtx =
      LlmProbe._positiveNumber(sgData.context_length) ??
      LlmProbe._positiveNumber(sgData.max_total_tokens);
    if (explicitCtx != null) {
      this.contextLength = explicitCtx;
    } else if (this.contextLength == null) {
      this.contextLength =
        LlmProbe._positiveNumber(sgData.max_req_input_len) ??
        LlmProbe._positiveNumber(sgData.max_total_num_tokens) ??
        null;
    }

    if (sgData.model_path && !this.modelId) {
      applyModelRef(this, sgData.model_path);
    }

    const maxRunning = Number(sgData.max_running_requests);
    if (Number.isFinite(maxRunning) && maxRunning > 0) {
      this.slotsTotal = Math.round(maxRunning);
    }

    const inTok = sgData.total_input_tokens;
    const outTok = sgData.total_output_tokens;
    if (inTok != null && outTok != null) {
      const input = Number(inTok);
      const output = Number(outTok);
      if (Number.isFinite(input) && Number.isFinite(output)) {
        const baselineReady =
          this._rateBaselineReady ||
          this.lastTokenCounts.input !== 0 ||
          this.lastTokenCounts.output !== 0;
        const deltaIn = input - this.lastTokenCounts.input;
        const deltaOut = output - this.lastTokenCounts.output;
        const counterReset = deltaIn < 0 || deltaOut < 0;
        this.lastTokenCounts.input = input;
        this.lastTokenCounts.output = output;
        this.totalPromptTokens = input;
        this.totalOutputTokens = output;
        this._promptCounterObserved = true;
        this._outputCounterObserved = true;
        if (dtSec > 0 && dtSec < 10) {
          this.generationTps = !baselineReady || counterReset
            ? 0
            : Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
          this._setPrefillTps(!baselineReady || counterReset ? 0 : deltaIn / dtSec, deltaOut > 0);
        }
        this._rateBaselineReady = true;
        return;
      }
    }

    // No cumulative counters — sticky last_gen_throughput only while it moves,
    // unless /v1/loads (or /get_load) says requests are in flight.
    const lastGen = LlmProbe._sglangLastGenThroughput(sgData);
    this.generationTps = this._sglangStickyThroughput(lastGen, this._sglangInflight());
  }

  /** True when SGLang load probe reported running or waiting requests. */
  _sglangInflight() {
    return (
      this.slotsActive > 0 ||
      (this.requestsRunning != null && this.requestsRunning > 0) ||
      (this.requestsWaiting != null && this.requestsWaiting > 0)
    );
  }

  /**
   * Prefer /v1/loads (num_running_reqs). Fall back to /get_load, where
   * num_reqs is running + waiting.
   */
  async _probeSglangLoad() {
    for (const path of ["/v1/loads", "/get_load"]) {
      try {
        const res = await this._fetch(`${this.baseUrl}${path}`);
        if (!res.ok) continue;
        const data = await res.json().catch(() => null);
        if (this._applySglangLoad(data)) return;
      } catch {
        /* try next */
      }
    }
  }

  /**
   * Apply SGLang /v1/loads or /get_load. Returns true when a row was applied.
   * @param {unknown} payload
   * @returns {boolean}
   */
  _applySglangLoad(payload) {
    const rows = LlmProbe._sglangLoadRows(payload);
    if (!rows.length) return false;
    let running = 0;
    let waiting = 0;
    let saw = false;
    for (const row of rows) {
      const wait = Number(row.num_waiting_reqs);
      const runDirect = Number(row.num_running_reqs);
      const total = Number(row.num_reqs);
      if (Number.isFinite(wait) && wait >= 0) {
        waiting += wait;
        saw = true;
      }
      if (Number.isFinite(runDirect) && runDirect >= 0) {
        running += runDirect;
        saw = true;
      } else if (Number.isFinite(total) && total >= 0) {
        const waitPart = Number.isFinite(wait) && wait >= 0 ? wait : 0;
        running += Math.max(0, total - waitPart);
        saw = true;
      }
    }
    if (!saw) return false;
    this.requestsRunning = running;
    this.requestsWaiting = waiting;
    this.slotsActive = Math.round(running);
    return true;
  }

  /**
   * @param {unknown} payload
   * @returns {Array<Record<string, unknown>>}
   */
  static _sglangLoadRows(payload) {
    if (payload == null) return [];
    if (Array.isArray(payload)) {
      return payload.filter((row) => row && typeof row === "object");
    }
    if (typeof payload !== "object") return [];
    if (Array.isArray(payload.loads)) {
      return payload.loads.filter((row) => row && typeof row === "object");
    }
    if (payload.num_reqs != null || payload.num_running_reqs != null) {
      return [payload];
    }
    return [];
  }

  /**
   * Map SGLang's sticky last_gen_throughput gauge to a live panel rate.
   * Returns 0 until the value changes between polls (avoids showing a stale
   * leftover after idle); stays live for a short window after each change.
   * When `inflight` is true (independent load signal), keep a positive rate
   * even if the gauge is not moving — a busy decode can report a constant value.
   * @param {number | null} raw
   * @param {boolean} [inflight]
   * @returns {number}
   */
  _sglangStickyThroughput(raw, inflight = false) {
    if (raw == null || !Number.isFinite(raw) || raw < 0) {
      this._sglangStickyTps = null;
      return 0;
    }
    const rounded = Math.round(raw * 100) / 100;
    const now = Date.now();
    const prev = this._sglangStickyTps;

    if (!prev) {
      // First sample after reset/start — seed only unless load says we are busy
      this._sglangStickyTps = {
        value: rounded,
        liveUntil: inflight && rounded > 0 ? now + SGLANG_STICKY_TPS_LIVE_MS : 0,
      };
      return inflight && rounded > 0 ? rounded : 0;
    }

    if (rounded !== prev.value) {
      this._sglangStickyTps = {
        value: rounded,
        liveUntil: now + SGLANG_STICKY_TPS_LIVE_MS,
      };
      return rounded;
    }

    if (prev.liveUntil > now) {
      return rounded;
    }
    if (inflight && rounded > 0) {
      return rounded;
    }
    return 0;
  }

  /**
   * @param {unknown} v
   * @returns {number | null}
   */
  static _positiveNumber(v) {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? n : null;
  }

  /**
   * Max last_gen_throughput across internal_states (or top-level).
   * @param {Record<string, unknown>} sgData
   * @returns {number | null}
   */
  static _sglangLastGenThroughput(sgData) {
    if (!sgData || typeof sgData !== "object") return null;
    const top = Number(sgData.last_gen_throughput);
    if (Number.isFinite(top) && top >= 0) return top;

    const states = sgData.internal_states;
    if (!Array.isArray(states) || !states.length) return null;
    let best = null;
    for (const st of states) {
      if (!st || typeof st !== "object") continue;
      const v = Number(st.last_gen_throughput);
      if (!Number.isFinite(v) || v < 0) continue;
      if (best == null || v > best) best = v;
    }
    return best;
  }

  /**
   * Apply SGLang Prometheus /metrics (--enable-metrics).
   * Supports both `sglang:` and `sglang_` prefixes.
   * @param {string} txt
   * @param {number} dtSec
   */
  _applySglangMetrics(txt, dtSec) {
    this.generationTps = 0;
    this.prefillTps = 0;
    const gen =
      this._getPromMetric(txt, "sglang:generation_tokens_total") ??
      this._getPromMetric(txt, "sglang_generation_tokens_total");
    const prompt =
      this._getPromMetric(txt, "sglang:prompt_tokens_total") ??
      this._getPromMetric(txt, "sglang_prompt_tokens_total");
    if (gen == null) {
      const gauge =
        this._getPromMetric(txt, "sglang:gen_throughput") ??
        this._getPromMetric(txt, "sglang_gen_throughput");
      if (gauge != null) {
        this.generationTps = Math.max(0, Math.round(gauge * 100) / 100);
      }
      return;
    }

    const baselineReady =
      this._rateBaselineReady ||
      this.lastTokenCounts.input !== 0 ||
      this.lastTokenCounts.output !== 0;
    const deltaOut = gen - this.lastTokenCounts.output;
    const deltaIn = prompt != null ? prompt - this.lastTokenCounts.input : null;
    const counterReset = deltaOut < 0 || (deltaIn != null && deltaIn < 0);
    if (dtSec > 0 && dtSec < 10) {
      this.generationTps =
        !baselineReady || counterReset
          ? 0
          : Math.max(0, Math.round((deltaOut / dtSec) * 100) / 100);
      if (prompt != null) {
        this._setPrefillTps(
          !baselineReady || counterReset ? 0 : deltaIn / dtSec,
          deltaOut > 0
        );
      } else if (deltaOut <= 0) {
        this.prefillTps = 0;
      }
    }
    if (prompt != null) this.lastTokenCounts.input = prompt;
    this.lastTokenCounts.output = gen;
    this._rateBaselineReady = true;
    if (prompt != null) this.totalPromptTokens = prompt;
    this.totalOutputTokens = gen;
    this._promptCounterObserved = prompt != null;
    this._outputCounterObserved = true;
    this.completedRequestsTotal =
      this._getPromMetric(txt, "sglang:request_success_total") ??
      this._getPromMetric(txt, "sglang_request_success_total");
    this._completedCounterObserved = this.completedRequestsTotal != null;

    const running =
      this._getPromMetric(txt, "sglang:num_running_reqs") ??
      this._getPromMetric(txt, "sglang_num_running_reqs");
    if (running != null) {
      this.requestsRunning = running;
      this.slotsActive = Math.round(running);
    }

    const cached = this._sglangCachedTokens(txt);
    if (cached != null && prompt != null) {
      this._setPrefillSplitRates(cached, prompt, dtSec);
    }
  }

  /**
   * Cache split only — does not touch generation/prefill lastTokenCounts.
   * Prefers cache_source="device" so HiCache L1/L2/L3 labels are not summed.
   */
  _applySglangPrefillSplit(txt, dtSec) {
    const prompt =
      this._getPromMetric(txt, "sglang:prompt_tokens_total") ??
      this._getPromMetric(txt, "sglang_prompt_tokens_total");
    const cached = this._sglangCachedTokens(txt);
    if (cached != null && prompt != null) {
      this._setPrefillSplitRates(cached, prompt, dtSec);
    }
  }

  _sglangCachedTokens(txt) {
    return (
      this._getPromMetricLabeled(txt, "sglang:cached_tokens_total", "cache_source", "device") ??
      this._getPromMetricLabeled(txt, "sglang_cached_tokens_total", "cache_source", "device") ??
      this._getPromMetricMax(txt, "sglang:cached_tokens_total") ??
      this._getPromMetricMax(txt, "sglang_cached_tokens_total")
    );
  }

  /** Prefer SGLang /get_model_info (or /model_info) over raw HF cache paths. */
  async _enrichSglangModelInfo() {
    for (const path of ["/get_model_info", "/model_info"]) {
      try {
        const res = await this._fetch(`${this.baseUrl}${path}`);
        if (!res.ok) continue;
        const data = await res.json();
        const raw = data?.model_path || data?.tokenizer_path;
        if (!raw) continue;
        if (!this.modelId) applyModelRef(this, raw);
        return;
      } catch {
        /* try next */
      }
    }
  }

  // ─── llama.cpp native path ────────────────────────────────
  async _probeLlamaCpp() {
    const now = Date.now();
    const dtSec = (now - this.lastProbeTime) / 1000;
    this.lastProbeTime = now;
    this.metricsAvailable = false;
    this._metricsModelMatched = false;
    this._metricsModelSeen = false;

    // Slots
    let slotsOk = false;
    try {
      const slotsRes = await this._fetch(`${this.baseUrl}/slots`);
      const auth = this._noteAuthStatus(slotsRes.status);
      if (auth === "auth") {
        this.status = "unavailable";
        this.statusReason = "metrics_auth_required";
        return this._getSnapshot();
      }
      if (auth === "ok") {
        const slots = await slotsRes.json();
        if (Array.isArray(slots)) {
          slotsOk = true;
          this._metricsModelMatched = true;
          this.slotsTotal = slots.length;
          // Some llama.cpp builds use is_processing instead of state
          this.slotsActive = slots.filter((s) => s.is_processing || (s.state && s.state !== "idle")).length;

          let totalGen = 0;
          let totalPrefill = 0;
          let totalDecoded = 0;
          let promptedSum = 0;
          let cachedSum = 0;
          let sawCache = false;

          for (const slot of slots) {
            const slotId = slot.id ?? "default";
            const decoded = this._getSlotDecoded(slot);
            const prompted = this._getSlotPrefilled(slot);
            const cached = this._getSlotCached(slot);
            totalDecoded += decoded;
            promptedSum += prompted;
            if (cached != null) {
              sawCache = true;
              cachedSum += cached;
            }
            const hasBaseline = this.slotState.has(slotId);
            const lastState = this.slotState.get(slotId) || { decoded: 0, prompted: 0 };
            const dDecoded = decoded - lastState.decoded;
            const dPrompted = prompted - lastState.prompted;
            this.slotState.set(slotId, { decoded, prompted });
            if (hasBaseline && dtSec > 0 && dtSec < 10) {
              // A restarted slot can expose lower counters. Re-seed it and
              // omit the negative delta from the live rate.
              if (dDecoded >= 0) totalGen += dDecoded / dtSec;
              if (dPrompted >= 0) totalPrefill += dPrompted / dtSec;
            }
          }

          this.totalOutputTokens = totalDecoded;
          this.totalPromptTokens = promptedSum;
          this._outputCounterObserved = true;
          this._promptCounterObserved = true;
          this.generationTps = Math.max(0, Math.round(totalGen * 100) / 100);
          this._setPrefillTps(totalPrefill, totalGen > 0);
          if (sawCache) this._setPrefillSplitRates(cachedSum, promptedSum, dtSec);
          this._noteFreshTelemetry(now);
        }
      }
    } catch {}

    if (!slotsOk) {
      throw new Error("llama.cpp /slots unreachable");
    }

    // Props (model info)
    try {
      const propsRes = await this._fetch(`${this.baseUrl}/props`);
      if (propsRes.ok) {
        const props = await propsRes.json();
        const raw = props.model_alias || props.model_path || this.modelId;
        if (props.model_path && !isHfHubCachePath(props.model_path)) {
          this.modelPath = props.model_path;
        } else if (isHfHubCachePath(props.model_path) || isHfHubCachePath(props.model_alias)) {
          this.modelPath = null;
        }
        if (raw) {
          applyModelRef(this, raw);
          this.benchmarkModel = this.modelId;
          this.models = this.modelId ? [this.modelId] : [];
        }
        // Preserve the native server's explicit model path even though the
        // alias is the user-facing model id.
        if (props.model_path && !isHfHubCachePath(props.model_path)) {
          this.modelPath = props.model_path;
        }
        this.contextLength = props.total_context_length || props.context_length || this.contextLength;
      }
    } catch {}

    this.backendType = "llama.cpp";
    return this._getSnapshot();
  }

  // ─── Metrics helpers ─────────────────────────────────────
  /**
   * Keep labeled Prometheus series scoped to the exact model selected from
   * /v1/models. Unlabeled legacy series remain usable; labeled series without
   * a selected model are rejected instead of being guessed or summed.
   */
  _promLabelsMatchModel(rawLabels) {
    const labels = parsePromLabels(rawLabels);
    const labeledModel = labels.model_name ?? labels.model;
    if (labeledModel == null) {
      // A malformed model label must not be mistaken for an unlabeled legacy
      // series. Reject it so malformed exposition cannot bypass model scoping.
      if (typeof rawLabels === "string" && /(?:^|,)\s*(?:model_name|model)\s*=/.test(rawLabels)) {
        this._metricsModelSeen = true;
        return false;
      }
      return true;
    }
    this._metricsModelSeen = true;
    // Direct helper callers and legacy native backends may not have a model
    // id yet. Live OpenAI-compatible probing never reaches metrics without a
    // selected model, so retaining this compatibility does not permit a
    // model guess on the monitored path.
    if (!this.modelId) {
      this._metricsModelMatched = true;
      return true;
    }
    const matches = normalizeModelId(labeledModel) === normalizeModelId(this.modelId);
    if (matches) this._metricsModelMatched = true;
    return matches;
  }

  /**
   * Sum all Prometheus series matching `name` (optional labels).
   * @param {string} body
   * @param {string} name Full metric name, e.g. "ds4_decode_tok_s" or "vllm:prompt_tokens_total"
   * @returns {number | null}
   */
  _getPromMetric(body, name) {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${esc}(?:\\{([^}]*)\\})?\\s+([\\d.eE+-]+)\\s*$`, "gm");
    let sum = 0;
    let found = false;
    let m;
    while ((m = re.exec(body)) !== null) {
      if (!this._promLabelsMatchModel(m[1])) continue;
      const v = parseFloat(m[2]);
      if (Number.isFinite(v)) {
        sum += v;
        found = true;
      }
    }
    return found ? sum : null;
  }

  /**
   * Max of Prometheus series matching `name` (avoids summing HiCache layers).
   * @param {string} body
   * @param {string} name
   * @returns {number | null}
   */
  _getPromMetricMax(body, name) {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${esc}(?:\\{([^}]*)\\})?\\s+([\\d.eE+-]+)\\s*$`, "gm");
    let best = null;
    let m;
    while ((m = re.exec(body)) !== null) {
      if (!this._promLabelsMatchModel(m[1])) continue;
      const v = parseFloat(m[2]);
      if (Number.isFinite(v)) best = best == null ? v : Math.max(best, v);
    }
    return best;
  }

  /**
   * Sum series of `name` whose label `labelKey` equals `labelValue`.
   * @param {string} body
   * @param {string} name
   * @param {string} labelKey
   * @param {string} labelValue
   * @returns {number | null}
   */
  _getPromMetricLabeled(body, name, labelKey, labelValue) {
    const escName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${escName}\\{([^}]*)\\}\\s+([\\d.eE+-]+)\\s*$`, "gm");
    let sum = 0;
    let found = false;
    let m;
    while ((m = re.exec(body)) !== null) {
      if (promLabelValue(m[1], labelKey) !== labelValue) continue;
      if (!this._promLabelsMatchModel(m[1])) continue;
      const v = parseFloat(m[2]);
      if (Number.isFinite(v)) {
        sum += v;
        found = true;
      }
    }
    return found ? sum : null;
  }

  _getVllmMetric(body, name) {
    return this._getPromMetric(body, `vllm:${name}`);
  }

  /** Read a numeric label from a Prometheus info metric. */
  _getPromInfoNumber(body, name, labelKey) {
    const escName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const line = new RegExp(`^${escName}\\{([^}]*)\\}\\s+[\\d.eE+-]+\\s*$`, "m").exec(body);
    if (!line) return null;
    if (!this._promLabelsMatchModel(line[1])) return null;
    const raw = promLabelValue(line[1], labelKey);
    if (raw == null) return null;
    const value = Number(raw);
    return Number.isFinite(value) && value >= 0 ? value : null;
  }

  /**
   * Parse a vLLM Prometheus histogram from /metrics text.
   * Returns { buckets: [{upper, count}], total } with cumulative counts per `le`,
   * summed across label sets. `total` is the summed `_count` series (or null).
   */
  _parseVllmHistogram(body, metricPrefix) {
    const esc = metricPrefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Bucket lines: <metricPrefix>_bucket{...le="X"...} VALUE
    const bucketRe = new RegExp(`^${esc}_bucket\\{([^}]*)\\}\\s+([\\d.eE+-]+)\\s*$`, "gm");
    const byUpper = new Map();
    let infCount = 0;
    let m;
    while ((m = bucketRe.exec(body)) !== null) {
      if (!this._promLabelsMatchModel(m[1])) continue;
      const le = promLabelValue(m[1], "le");
      if (le == null) continue;
      const count = parseFloat(m[2]);
      if (!Number.isFinite(count)) continue;
      const upper = le === "+Inf" ? Infinity : parseFloat(le);
      if (upper !== Infinity && !Number.isFinite(upper)) continue;
      if (upper === Infinity) infCount += count;
      byUpper.set(upper, (byUpper.get(upper) || 0) + count);
    }
    const total = this._getVllmMetric(body, `${metricPrefix.replace(/^vllm:/, "")}_count`);
    // Prometheus invariant: +Inf bucket count == _count. Mismatch → refuse quantile.
    if (total != null && infCount > 0 && Math.abs(infCount - total) > 1e-6) {
      return { buckets: [], total: null };
    }
    const buckets = Array.from(byUpper, ([upper, count]) => ({ upper, count }));
    buckets.sort((a, b) => a.upper - b.upper);
    return { buckets, total };
  }

  /**
   * Prometheus-style linear interpolation for a histogram quantile.
   * Returns null when empty / invalid or target is in the +Inf tail.
   */
  _histogramQuantile(buckets, total, quantile) {
    if (!buckets || !buckets.length || total == null || total <= 0) return null;
    const target = total * quantile;
    let prevUpper = 0.0;
    let prevCount = 0.0;
    for (const { upper, count } of buckets) {
      if (count >= target) {
        if (!Number.isFinite(upper)) return null;
        if (count === prevCount) return upper;
        return prevUpper + (upper - prevUpper) * ((target - prevCount) / (count - prevCount));
      }
      prevUpper = upper;
      prevCount = count;
    }
    return null;
  }

  _getSlotDecoded(slot) {
    // Some llama.cpp builds nest n_decoded inside next_token[0]
    if (slot.n_decoded != null) {
      if (Array.isArray(slot.n_decoded)) return slot.n_decoded[0] || 0;
      return slot.n_decoded || 0;
    }
    // Fallback: next_token[0].n_decoded (newer llama.cpp)
    if (Array.isArray(slot.next_token) && slot.next_token[0]?.n_decoded != null) {
      return slot.next_token[0].n_decoded;
    }
    return 0;
  }

  _getSlotPrefilled(slot) {
    if (slot?.n_prompt_tokens_processed != null) {
      const n = Number(slot.n_prompt_tokens_processed);
      if (Number.isFinite(n)) return n;
    }
    return slot?.n_prompt_tokens || 0;
  }

  /** Cached prompt tokens on a llama.cpp slot, or null when the field is absent. */
  _getSlotCached(slot) {
    if (slot == null || slot.n_prompt_tokens_cache == null) return null;
    const n = Number(slot.n_prompt_tokens_cache);
    return Number.isFinite(n) ? n : null;
  }

  /**
   * Observational exposure hint from probe target + unauthenticated reachability.
   * Does not claim process bind address (0.0.0.0 vs interface).
   */
  _buildPosture() {
    if (this.authOpen == null) return null;

    const host = llmProbeHost(this.spark);
    const scope = classifyHostScope(host);
    const keyed = Boolean(this._apiKey());
    /** @type {"open" | "protected" | "keyed"} */
    let auth;
    if (keyed) {
      // Key configured: success → keyed; 401/403 → protected (rejected)
      auth = this.authOpen === false ? "protected" : "keyed";
    } else {
      auth = this.authOpen ? "open" : "protected";
    }

    let level = "ok";
    if (auth === "open") {
      if (scope === "public") level = "danger";
      else if (scope === "local") level = "ok";
      else level = "warn"; // lan or unknown hostname
    } else if (keyed && auth === "protected") {
      level = "danger";
    }

    const scopeWords = {
      local: "loopback",
      lan: "LAN",
      public: "public",
      unknown: "unknown-host",
    };
    const shortScope = {
      local: "Local",
      lan: "LAN",
      public: "Public",
      unknown: "Host",
    };
    const label =
      auth === "protected"
        ? keyed
          ? "Bad API key"
          : "Auth required"
        : auth === "keyed"
          ? `API key · ${shortScope[scope]}`
          : `Open · ${shortScope[scope]}`;
    const detail =
      auth === "protected"
        ? keyed
          ? `Configured API key was rejected (401/403) · ${scopeWords[scope]} target (${host || "—"}).`
          : `API key required · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`
        : auth === "keyed"
          ? `Using configured API key · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`
          : `Unauthenticated · ${scopeWords[scope]} target (${host || "—"}). Based on the configured probe host, not the process bind address.`;

    return { level, auth, scope, label, detail };
  }

  _snapshotStatus() {
    if (this.status === "ambiguous_model") return "ambiguous_model";
    if (this.metricsAvailable && (this.status === "active" || this.status === "idle")) {
      return this.status;
    }
    if (
      this.lastObservedAt != null &&
      Date.now() - this.lastObservedAt > TELEMETRY_STALE_MS
    ) {
      return "stale";
    }
    return this.status === "unknown" ? "unknown" : "unavailable";
  }

  _getSnapshot() {
    const status = this._snapshotStatus();
    const metricsLive = this.metricsAvailable && (status === "active" || status === "idle");
    const liveRate = (value) => {
      if (!metricsLive || !Number.isFinite(value)) return null;
      if (status === "idle") return 0;
      return value > 0 ? value : null;
    };
    const liveCounter = (value, observed) =>
      metricsLive && observed && Number.isFinite(value) ? value : null;
    return {
      available: metricsLive,
      status,
      lastObservedAt: this.lastObservedAt,
      statusReason:
        status === "active" || status === "idle"
          ? null
          : this.statusReason || (status === "stale" ? "telemetry_stale" : "telemetry_unavailable"),
      backend: this.backendType,
      modelId: this.modelId || null,
      modelPath: this.modelPath || null,
      benchmarkModel: this.benchmarkModel || null,
      models: this.models,
      contextLength: this.contextLength,
      gpuMemoryUtilization: this.gpuMemoryUtilization,
      slotsActive: this.slotsActive,
      slotsTotal: this.slotsTotal,
      generationTps: liveRate(this.generationTps),
      prefillTps: liveRate(this.prefillTps),
      cachedPrefillTps: liveRate(this.cachedPrefillTps),
      uncachedPrefillTps: liveRate(this.uncachedPrefillTps),
      totalPromptTokens: liveCounter(this.totalPromptTokens, this._promptCounterObserved),
      totalOutputTokens: liveCounter(this.totalOutputTokens, this._outputCounterObserved),
      completedRequestsTotal: liveCounter(
        this.completedRequestsTotal,
        this._completedCounterObserved
      ),
      telemetrySource: this.telemetrySource,
      kvCacheUsage: metricsLive ? this.kvCacheUsage : null,
      kvCacheCapacityTokens: metricsLive ? this.kvCacheCapacityTokens : null,
      kvCacheMaxConcurrency: metricsLive ? this.kvCacheMaxConcurrency : null,
      requestsRunning: metricsLive ? this.requestsRunning : null,
      requestsWaiting: metricsLive ? this.requestsWaiting : null,
      ttftP95Seconds: metricsLive ? this.ttftP95Seconds : null,
      preemptionsTotal: metricsLive ? this.preemptionsTotal : null,
      prefixCacheHitRate: metricsLive ? this.prefixCacheHitRate : null,
      e2eP95Seconds: metricsLive ? this.e2eP95Seconds : null,
      itlP95Seconds: metricsLive ? this.itlP95Seconds : null,
      mtpAcceptanceRate: metricsLive ? this.mtpAcceptanceRate : null,
      posture: this._buildPosture(),
      error: this.error,
    };
  }

  _defaultLlm() {
    return this._getSnapshot();
  }

  // ─── HTTP helpers ────────────────────────────────────────
  _apiKey() {
    const keys = this.spark?.llmApiKeys;
    if (!keys || typeof keys !== "object") return null;
    const raw = keys[String(this.port)] ?? keys[this.port];
    const key = raw != null ? String(raw).trim() : "";
    return key || null;
  }

  async _fetch(url) {
    const headers = {};
    const apiKey = this._apiKey();
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    return fetch(url, { signal: AbortSignal.timeout(LLM_PROBE_TIMEOUT_MS), headers });
  }
}
