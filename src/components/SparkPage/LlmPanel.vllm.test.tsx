import { describe, expect, it, vi } from "vitest";
import { render } from "../../testing/render";
import { makeSpark } from "../../testing/fixtures";
import { LlmPanel } from "./LlmPanel";

vi.mock("./LlmTokenTotals", () => ({ LlmTokenTotals: () => null }));
vi.mock("./LlmDailyChart", () => ({ LlmDailyChart: () => null }));
vi.mock("./LlmTrendChart", () => ({ LlmTrendChart: () => null }));

describe("vLLM prefill measurement", () => {
  it("labels poll-window throughput without claiming individual prefill speed", () => {
    const llm = { ...makeSpark().metrics.llm[0], backend: "vllm" as const,
      generationTps: 400, prefillTps: 100, uncachedPrefillTps: 100, cachedPrefillTps: 900 };
    const { container } = render(<LlmPanel llm={llm} sparkId="fixture" llmPort={8888} />);
    const tile = [...container.querySelectorAll(".sp-metric")].find(el =>
      el.querySelector(".eyebrow")?.textContent === "Prefill");
    expect(tile?.getAttribute("title")).toContain("wall-clock throughput");
    expect(tile?.getAttribute("title")).toContain("excluding cached tokens");
    expect(tile?.textContent).toContain("100");
    expect(tile?.getAttribute("title")).not.toContain("cache-served + computed");
  });
});
