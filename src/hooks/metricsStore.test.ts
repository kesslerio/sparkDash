import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { LlmMetrics, SparkSnapshot } from "../api/types";
import {
  _resetStore,
  getMetricHistorySamples,
  ingestSnapshots,
  useMetricsHistory,
} from "./metricsStore";
import { makeSpark } from "../testing/fixtures";

describe("metricsStore timestamp contract", () => {
  beforeEach(_resetStore);

  it.each([1_000, 2_000, 5_000])("preserves a %sms source cadence", (interval) => {
    const spark = makeSpark();
    ingestSnapshots([spark], 10_000);
    spark.metrics.gpu!.usage = 50;
    ingestSnapshots([spark], 10_000 + interval);
    expect(getMetricHistorySamples(spark.id, "gpu.usage")).toEqual([
      { at: 10_000, value: 42 },
      { at: 10_000 + interval, value: 50 },
    ]);
  });

  it("replaces duplicate frames and retains real disconnect gaps", () => {
    const spark = makeSpark();
    ingestSnapshots([spark], 1_000);
    spark.metrics.gpu!.usage = 55;
    ingestSnapshots([spark], 1_000);
    spark.metrics.gpu!.usage = 60;
    ingestSnapshots([spark], 61_000);
    expect(getMetricHistorySamples(spark.id, "gpu.usage")).toEqual([
      { at: 1_000, value: 55 },
      { at: 61_000, value: 60 },
    ]);
  });

  it("ignores out-of-order frames instead of rewinding chart time", () => {
    const spark = makeSpark();
    ingestSnapshots([spark], 5_000);
    spark.metrics.gpu!.usage = 99;
    ingestSnapshots([spark], 4_000);
    expect(getMetricHistorySamples(spark.id, "gpu.usage")).toEqual([
      { at: 5_000, value: 42 },
    ]);
  });
});

function llm(overrides: Partial<LlmMetrics> = {}): LlmMetrics {
  return {
    available: true,
    status: "active",
    lastObservedAt: 1,
    statusReason: null,
    backend: "vllm",
    modelId: "model",
    modelPath: null,
    contextLength: null,
    gpuMemoryUtilization: null,
    slotsActive: 0,
    slotsTotal: 1,
    generationTps: null,
    prefillTps: null,
    totalOutputTokens: null,
    ...overrides,
  };
}

function snapshot(metrics: LlmMetrics): SparkSnapshot {
  return {
    id: "spark-1",
    name: "Spark",
    online: true,
    uptime: null,
    disabledDevices: [],
    disabledInterfaces: [],
    llmPort: 4000,
    llmPorts: [4000],
    hardware: {} as SparkSnapshot["hardware"],
    metrics: {
      gpu: null,
      cpu: null,
      ram: null,
      storage: [],
      network: null,
      unifiedMemory: null,
      llm: [metrics],
    },
  };
}

describe("metricsStore LLM rate ingestion", () => {
  beforeEach(() => {
    _resetStore();
  });

  it("does not append null or non-finite rates, but preserves proven zero idle rates", () => {
    const { result } = renderHook(() => ({
      generation: useMetricsHistory("spark-1", "llm:4000.tps"),
      prefill: useMetricsHistory("spark-1", "llm:4000.prefill"),
    }));

    act(() => {
      ingestSnapshots([
        snapshot(llm({ generationTps: 10, prefillTps: 20 })),
      ]);
    });
    expect(result.current.generation).toEqual([10]);
    expect(result.current.prefill).toEqual([20]);

    act(() => {
      ingestSnapshots([
        snapshot(llm({ generationTps: Number.NaN, prefillTps: null })),
      ]);
    });
    expect(result.current.generation).toEqual([10]);
    expect(result.current.prefill).toEqual([20]);

    act(() => {
      ingestSnapshots([
        snapshot(llm({ generationTps: 0, prefillTps: 0 })),
      ]);
    });
    expect(result.current.generation).toEqual([10, 0]);
    expect(result.current.prefill).toEqual([20, 0]);
  });
});
