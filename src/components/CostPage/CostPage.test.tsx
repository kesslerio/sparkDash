import { act } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import { render, flush } from "../../testing/render";
import { CostPage } from "./CostPage";

vi.mock("../../api/client", () => ({ fetchTokenHistory: vi.fn(async () => ({ hour: [], day: [] })) }));
vi.mock("../../api/llmTokenClient", () => ({ fetchLlmTokenTotals: vi.fn(async () => ({ series: [] })) }));
vi.mock("../../api/costClient", () => ({ fetchModelPrices: vi.fn(async () => ({
  source: "fixture", models: [{ id: "unrelated-first-model", provider: "test", mode: "chat",
    inputCostPerToken: 0.0000003, outputCostPerToken: 0.0000125,
    cacheReadCostPerToken: null, cacheWriteCostPerToken: null, maxInputTokens: 1049000, maxOutputTokens: null }],
})) }));

beforeEach(() => localStorage.clear());

it("requires an explicit hosted-model choice instead of pricing as the first catalog entry", async () => {
  const { container } = render(<CostPage />);
  await flush();
  expect(container.textContent).toContain("Choose a hosted model");
  expect(container.querySelector(".cp-toggle__price")).toBeNull();
  expect(localStorage.getItem("sparkdash.cost.model")).toBeNull();
  act(() => (container.querySelector(".cp-toggle") as HTMLButtonElement).click());
  act(() => (container.querySelector(".cp-item") as HTMLButtonElement).click());
  await flush();
  expect(localStorage.getItem("sparkdash.cost.model")).toBe("unrelated-first-model");
  expect(container.textContent).toContain("$0.3 / 1M in · $12.5 / 1M out");
});

it("uses a consistent USD locale for prices and context even with a German browser", async () => {
  const original = Number.prototype.toLocaleString;
  vi.spyOn(Number.prototype, "toLocaleString").mockImplementation(function (this: number, locales, options) {
    return original.call(this, locales ?? "de-DE", options);
  });
  localStorage.setItem("sparkdash.cost.model", "unrelated-first-model");
  const { container } = render(<CostPage />);
  await flush();
  expect(container.textContent).toContain("1,049k context");
  expect(container.textContent).toContain("$0.3 / 1M in · $12.5 / 1M out");
});
