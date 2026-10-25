import { describe, expect, it } from "vitest";
import { idleLabel, isLlmIdle } from "./llmIdle";

const NOW = 1_800_000_000_000;

describe("llm idle helpers", () => {
  it("treats only positive finite rates as serving", () => {
    expect(isLlmIdle({ generationTps: 0, prefillTps: 0 })).toBe(true);
    expect(isLlmIdle({ generationTps: Number.NaN, prefillTps: 0 })).toBe(true);
    expect(isLlmIdle({ generationTps: 0.2, prefillTps: 0 })).toBe(false);
    expect(isLlmIdle({ generationTps: 0, prefillTps: 1 })).toBe(false);
  });

  it("is not idle while a prompt is prefilling, even with both rates at 0 (#202)", () => {
    expect(isLlmIdle({ generationTps: 0, prefillTps: 0, prefillActive: true })).toBe(false);
    expect(isLlmIdle({ generationTps: 0, prefillTps: 0, prefillActive: false })).toBe(true);
  });

  it("labels the idle state", () => {
    expect(idleLabel(NOW - 30_000, NOW)).toBe("Idle · last served 30s ago");
    expect(idleLabel(NOW - 3 * 3_600_000, NOW)).toBe("Idle · last served 3h ago");
    expect(idleLabel(null, NOW)).toBe("Idle");
    expect(idleLabel(undefined, NOW)).toBe("Idle");
  });

  it("is not idle while Strata is decoding without a reported rate", () => {
    expect(isLlmIdle({ generationTps: 0, prefillTps: 0, decodeActive: true })).toBe(false);
    expect(isLlmIdle({ generationTps: 0, prefillTps: 0, decodeActive: false })).toBe(true);
  });
});
