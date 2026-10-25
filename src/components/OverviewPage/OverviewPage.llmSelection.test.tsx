/**
 * Which LLM the fleet card shows.
 *
 * A Spark can expose several LLM ports at once. The card must show the engine
 * that is doing work, not the first one that merely exists: an idle engine on
 * a low port used to win over a busy engine on a higher port, so a decoding
 * DeepSeek V4.1 stayed hidden behind an idle GLM sibling.
 */
import { describe, expect, it } from "vitest";
import type { SparkSnapshot } from "../../api/types";
import { makeSpark } from "../../testing/fixtures";
import { render } from "../../testing/render";
import { OverviewPage } from "./OverviewPage";

function llmEntry(overrides: Partial<SparkSnapshot["metrics"]["llm"][number]>) {
  return {
    backend: "vllm",
    modelId: null,
    status: "idle",
    generationTps: null,
    prefillTps: null,
    available: true,
    ...overrides,
  } as SparkSnapshot["metrics"]["llm"][number];
}

/**
 * The card sees the LLM array in the server's port order, so array order is
 * what "a lower port" means here.
 */
function sparkWithLlms(llm: SparkSnapshot["metrics"]["llm"]): SparkSnapshot {
  const base = makeSpark("fleet-node", true);
  return {
    ...base,
    metrics: { ...base.metrics, llm },
  } as SparkSnapshot;
}

describe("overview card LLM selection", () => {
  it("shows the idle serving model instead of an unloaded catalog on a lower port", () => {
    const { container } = render(
      <OverviewPage sparks={[sparkWithLlms([
        llmEntry({ backend: "omlx", modelId: null, models: ["GLM-5.3-Flash"],
          status: "unavailable", statusReason: "no_active_model", available: false }),
        llmEntry({ backend: "vllm", modelId: "DeepSeek-V4.1-Flash", status: "idle" }),
      ])]} />
    );
    expect(container.textContent).toContain("DeepSeek-V4.1-Flash");
    expect(container.textContent).not.toContain("GLM-5.3-Flash");
  });

  it("shows the active engine even when an idle one sits on a lower port", () => {
    const { container } = render(
      <OverviewPage
        sparks={[
          sparkWithLlms([
            llmEntry({ backend: "omlx", modelId: "GLM-5.3-Flash", status: "idle" }),
            llmEntry({
              backend: "vllm",
              modelId: "DeepSeek-V4.1-Flash",
              status: "active",
              generationTps: 48.2,
            }),
          ]),
        ]}
      />
    );
    // The serving engine is the card's model; the idle sibling only appears
    // as an extra "Loaded" row, never as the monitored model.
    const rows = [...container.querySelectorAll(".ov-modelrow")];
    expect(rows[0].querySelector("code")?.textContent).toBe("DeepSeek-V4.1-Flash");
    expect(rows.find((r) => r.textContent?.includes("GLM-5.3-Flash"))?.textContent).toContain("Loaded");
  });

  it("falls back to an idle engine when nothing is active", () => {
    const { container } = render(
      <OverviewPage
        sparks={[
          sparkWithLlms([
            llmEntry({ backend: "omlx", modelId: "GLM-5.3-Flash", status: "idle" }),
            llmEntry({ backend: "vllm", modelId: null, status: "unavailable" }),
          ]),
        ]}
      />
    );
    const text = container.textContent ?? "";
    expect(text).toContain("GLM-5.3-Flash");
  });
});
