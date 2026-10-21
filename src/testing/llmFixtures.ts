import type { ConversationRow, LlmDailyDay, LlmMetrics, SparkSnapshot } from "../api/types";
import type { LlmTokenSeriesTotals } from "../api/llmTokenTypes";
import { ingestSnapshots } from "../hooks/metricsStore";
import { makeSpark } from "./fixtures";

export const LLM_PREVIEW_SPARK_ID = "llm-layout-preview";
export const DEEPSEEK_MODEL_ID = "DeepSeek-V4.1-Flash-oQ2.5e-mtp";
export const DEEPSEEK_PORT = 8010;
export const GLM_MODEL_ID = "GLM-5.3-Flash-oQ4e-mtp";
export const GLM_PORT = 8000;

const SAMPLE_INTERVAL_MS = 150_000;
const DENSE_INTERVAL_MS = 2_000;
const SAMPLE_GENERATION = [0, 0, 42, 48, 0, 55, 58, 60.3, 57, 61, 59, 60.3];
const SAMPLE_PREFILL = [0, 120, 400, 680, 500, 240, 0, 0, 350, 700, 0, 0];

const DAILY_DECODE_MAX = [24, 29, 31, 28, 34, 42, 46, 41, 48, 51, 54, 58, 61, 60.3];
const DAILY_PREFILL_MAX = [180, 240, 220, 310, 280, 400, 460, 410, 500, 480, 620, 640, 710, 700];

/** Full LlmMetrics with every field present; overrides set only what a case needs. */
export function makeLlmMetrics(overrides: Partial<LlmMetrics> = {}): LlmMetrics {
  return {
    available: false,
    status: "unavailable",
    lastObservedAt: null,
    statusReason: null,
    backend: null,
    modelId: null,
    modelPath: null,
    benchmarkModel: null,
    contextLength: null,
    gpuMemoryUtilization: null,
    slotsActive: 0,
    slotsTotal: 0,
    generationTps: null,
    prefillTps: null,
    totalOutputTokens: null,
    error: null,
    ...overrides,
  };
}

/** Preview/test demo state: DeepSeek is running. Numbers are illustrative, not a performance claim. */
export function makeDeepSeekActive(now: number): LlmMetrics {
  return makeLlmMetrics({
    available: true,
    status: "active",
    lastObservedAt: now,
    backend: "omlx",
    modelId: DEEPSEEK_MODEL_ID,
    contextLength: 1_048_576,
    slotsActive: 1,
    generationTps: 60.3,
    prefillTps: 0,
    cachedPrefillTps: 0,
    uncachedPrefillTps: 0,
    totalOutputTokens: 904,
    totalPromptTokens: 287_215,
    completedRequestsTotal: 6,
    telemetrySource: "direct",
  });
}

export function makeDeepSeekIdle(now: number): LlmMetrics {
  return makeLlmMetrics({
    available: true,
    status: "idle",
    lastObservedAt: now,
    backend: "omlx",
    modelId: DEEPSEEK_MODEL_ID,
    contextLength: 1_048_576,
    generationTps: 0,
    prefillTps: 0,
    totalOutputTokens: 904,
    totalPromptTokens: 287_215,
    completedRequestsTotal: 6,
    telemetrySource: "direct",
  });
}

export function makeDeepSeekUnavailable(now: number): LlmMetrics {
  return makeLlmMetrics({
    available: false,
    status: "unavailable",
    lastObservedAt: now - 900_000,
    statusReason: "Telemetry source did not answer in the last 15 minutes.",
    backend: "omlx",
    modelId: DEEPSEEK_MODEL_ID,
    contextLength: 1_048_576,
  });
}

export function makeGlmIdle(now: number): LlmMetrics {
  return makeLlmMetrics({
    available: true,
    status: "idle",
    lastObservedAt: now,
    backend: "omlx",
    modelId: GLM_MODEL_ID,
    contextLength: 300_000,
    generationTps: 0,
    prefillTps: 0,
    telemetrySource: "direct",
  });
}

/** Ingest dense 2 s snapshots through the real store path, ending at `now`. Each anchor holds until the next. */
export function seedLlmHistory(sparkId: string, now: number): void {
  const span = SAMPLE_INTERVAL_MS * (SAMPLE_GENERATION.length - 1);
  for (let offset = 0; offset <= span; offset += DENSE_INTERVAL_MS) {
    const i = Math.min(Math.floor(offset / SAMPLE_INTERVAL_MS), SAMPLE_GENERATION.length - 1);
    const generation = SAMPLE_GENERATION[i];
    const prefill = SAMPLE_PREFILL[i];
    const at = now - span + offset;
    const deepseek = makeLlmMetrics({
      available: true,
      status: generation > 0 ? "active" : "idle",
      lastObservedAt: at,
      backend: "omlx",
      modelId: DEEPSEEK_MODEL_ID,
      generationTps: generation,
      prefillTps: prefill,
      cachedPrefillTps: prefill * 2,
      uncachedPrefillTps: prefill,
      ttftSeconds: generation > 0 ? 0.8 : 0,
    });
    const glm = makeGlmIdle(at);
    const base = makeSpark(sparkId, true);
    const snapshot = {
      ...base,
      llmPorts: [DEEPSEEK_PORT, GLM_PORT],
      metrics: { ...base.metrics, llm: [deepseek, glm] },
    } as SparkSnapshot;
    ingestSnapshots([snapshot], at);
  }
}

/** Stopped by design: available false, no history or totals. */
export function makeDs4Stopped(): LlmMetrics {
  return makeLlmMetrics({
    available: false,
    status: "unavailable",
    lastObservedAt: null,
    backend: "ds4",
    modelId: "deepseek-v4.1-flash",
    modelPath: null,
    contextLength: 1_048_576,
    statusReason: "Stopped intentionally to reserve memory for oMLX DeepSeek.",
  });
}

export function makeLlmDailyDays(): LlmDailyDay[] {
  return DAILY_DECODE_MAX.map((decodeMax, i) => {
    const date = new Date(Date.UTC(2026, 9, 7 + i)).toISOString().slice(0, 10);
    const prefillMax = DAILY_PREFILL_MAX[i];
    return {
      date,
      decodeMax,
      decodeAvg: Math.round(decodeMax * 0.8 * 100) / 100,
      prefillMax,
      prefillAvg: Math.round(prefillMax * 0.6 * 100) / 100,
      cachedPrefillMax: null,
      cachedPrefillAvg: null,
      uncachedPrefillMax: null,
      uncachedPrefillAvg: null,
    };
  });
}

export function makeLlmTokenSeries(sparkId: string, now: number): LlmTokenSeriesTotals[] {
  return [
    {
      sparkId,
      port: DEEPSEEK_PORT,
      updatedAt: now,
      lastModelId: DEEPSEEK_MODEL_ID,
      totals: { promptTokens: 745_000, completionTokens: 6_911, cachedTokens: 680_000 },
      models: [
        {
          modelId: DEEPSEEK_MODEL_ID,
          promptTokens: 745_000,
          completionTokens: 6_911,
          cachedTokens: 680_000,
          lastSeenAt: now,
        },
      ],
    },
  ];
}

export function makeSessionRows(now: number): ConversationRow[] {
  return [
    {
      id: "session-openclaw",
      source: "openclaw",
      handle: "main",
      badge: "generating",
      port: DEEPSEEK_PORT,
      lastUsedAt: now,
      agent: "main",
      contextUsed: 287_215,
      contextWindow: 1_048_576,
    },
    {
      id: "session-hermes",
      source: "hermes",
      handle: "research-thread-with-a-deliberately-long-handle-for-overflow-checks",
      badge: "unknown",
      port: DEEPSEEK_PORT,
      lastUsedAt: now - 8 * 60_000,
      gateway: "mac-mini",
      contextUsed: 64_000,
      contextWindow: 1_048_576,
      contextApprox: true,
    },
  ];
}
