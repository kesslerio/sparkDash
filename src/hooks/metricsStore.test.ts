import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import type { LlmMetrics, SparkSnapshot } from "../api/types";
import { _resetStore, ingestSnapshots, useMetricsHistory } from "./metricsStore";

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

describe("metricsStore Mac units", () => {
  beforeEach(() => {
    _resetStore();
  });

  function macSnapshot(): SparkSnapshot {
    const base = snapshot(llm({ backend: "omlx" }));
    return {
      ...base,
      id: "mac-1",
      kind: "mac",
      metrics: {
        ...base.metrics,
        gpu: {
          temperature: 0,
          usage: 97,
          power: { draw: 84.9, limit: 0, systemDraw: 90 },
          vram: { used: 185502, total: 262144, percentage: 75, available: 64552 },
          processes: [],
        } as unknown as SparkSnapshot["metrics"]["gpu"],
        cpu: { usage: 4, temperature: 0, draw: 5.1, tdp: 0 } as unknown as SparkSnapshot["metrics"]["cpu"],
      },
    };
  }

  it("records GPU usage but no fake 0° temperature samples for Macs", () => {
    const { result } = renderHook(() => ({
      usage: useMetricsHistory("mac-1", "gpu.usage"),
      temp: useMetricsHistory("mac-1", "gpu.temp"),
      cpuTemp: useMetricsHistory("mac-1", "cpu.temp"),
    }));
    act(() => {
      ingestSnapshots([macSnapshot()]);
    });
    expect(result.current.usage.at(-1)).toBe(97);
    expect(result.current.temp).toHaveLength(0);
    expect(result.current.cpuTemp).toHaveLength(0);
  });
});
