/**
 * Bounded, restart-safe LLM telemetry for incident and capacity analysis.
 *
 * One latest sample is retained per 10-second bucket for seven days. Unlike
 * LlmDaily, unavailable samples are retained so outages remain visible after
 * the inference process or dashboard restarts.
 */
import fs from "fs";
import { LLM_TELEMETRY_JSON_PATH } from "../config.js";
import { atomicWrite } from "../util/atomicWrite.js";

const BUCKET_MS = 10_000;
const MAX_HOURS = 168;
const RETENTION_MS = MAX_HOURS * 60 * 60 * 1000;
const MAX_POINTS = RETENTION_MS / BUCKET_MS;
const FLUSH_MS = 30_000;

function seriesKey(sparkId, port) {
  return `${sparkId}:${port}`;
}

function finiteOrNull(value) {
  const n = Number(value);
  return value != null && Number.isFinite(n) ? n : null;
}

function pointFrom(metrics, time) {
  return {
    t: Math.floor(time / BUCKET_MS) * BUCKET_MS,
    available: metrics?.available === true,
    backend: typeof metrics?.backend === "string" ? metrics.backend : null,
    generationTps: finiteOrNull(metrics?.generationTps),
    prefillTps: finiteOrNull(metrics?.prefillTps),
    requestsRunning: finiteOrNull(metrics?.requestsRunning),
    requestsWaiting: finiteOrNull(metrics?.requestsWaiting),
    kvCacheUsage: finiteOrNull(metrics?.kvCacheUsage),
    ttftP95Seconds: finiteOrNull(metrics?.ttftP95Seconds),
    e2eP95Seconds: finiteOrNull(metrics?.e2eP95Seconds),
    itlP95Seconds: finiteOrNull(metrics?.itlP95Seconds),
    preemptionsTotal: finiteOrNull(metrics?.preemptionsTotal),
    prefixCacheHitRate: finiteOrNull(metrics?.prefixCacheHitRate),
    mtpAcceptanceRate: finiteOrNull(metrics?.mtpAcceptanceRate),
    kvCacheCapacityTokens: finiteOrNull(metrics?.kvCacheCapacityTokens),
    kvCacheMaxConcurrency: finiteOrNull(metrics?.kvCacheMaxConcurrency),
  };
}

function validPoint(point) {
  return point && Number.isFinite(point.t) && typeof point.available === "boolean";
}

export class LlmTelemetryStore {
  /** @param {string} [filePath] */
  constructor(filePath = LLM_TELEMETRY_JSON_PATH) {
    this.filePath = filePath;
    /** @type {Record<string, Array<Record<string, unknown>>>} */
    this._series = {};
    this._dirty = false;
    this._flushTimer = null;
    this._load();
  }

  _load() {
    try {
      if (!fs.existsSync(this.filePath)) return;
      const raw = JSON.parse(fs.readFileSync(this.filePath, "utf8"));
      const series = raw?.series;
      if (!series || typeof series !== "object" || Array.isArray(series)) return;
      for (const [key, points] of Object.entries(series)) {
        if (!Array.isArray(points)) continue;
        this._series[key] = points.filter(validPoint).slice(-MAX_POINTS);
      }
    } catch {
      this._series = {};
    }
  }

  _scheduleFlush() {
    if (this._flushTimer) return;
    this._flushTimer = setTimeout(() => {
      this._flushTimer = null;
      this.flush();
    }, FLUSH_MS);
    this._flushTimer.unref?.();
  }

  flush() {
    if (!this._dirty) return;
    try {
      atomicWrite(
        this.filePath,
        JSON.stringify({ version: 1, bucketMs: BUCKET_MS, retentionHours: MAX_HOURS, series: this._series }),
        0o600
      );
      this._dirty = false;
    } catch (err) {
      console.error("[LlmTelemetry] write failed:", err.message);
    }
  }

  /** Record or replace the current ten-second bucket. */
  record(sparkId, port, metrics, now = new Date()) {
    if (!sparkId || !Number.isInteger(port) || !metrics) return;
    const time = now instanceof Date ? now.getTime() : Number(now);
    if (!Number.isFinite(time)) return;
    const key = seriesKey(sparkId, port);
    const point = pointFrom(metrics, time);
    const cutoff = point.t - RETENTION_MS + BUCKET_MS;
    const points = (this._series[key] || []).filter((p) => p.t >= cutoff);
    if (points.at(-1)?.t === point.t) points[points.length - 1] = point;
    else points.push(point);
    this._series[key] = points.slice(-MAX_POINTS);
    this._dirty = true;
    this._scheduleFlush();
  }

  getSeries(sparkId, port, opts = {}) {
    const hours = Math.min(MAX_HOURS, Math.max(1, Number(opts.hours) || 24));
    const now = opts.now instanceof Date ? opts.now.getTime() : Date.now();
    const cutoff = now - hours * 60 * 60 * 1000;
    const points = (this._series[seriesKey(sparkId, port)] || []).filter((p) => p.t >= cutoff);
    return { sparkId, port, bucketSeconds: BUCKET_MS / 1000, retentionHours: MAX_HOURS, points };
  }
}

export const llmTelemetry = new LlmTelemetryStore();
