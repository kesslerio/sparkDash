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
    const text = container.textContent ?? "";
    expect(text).toContain("DeepSeek-V4.1-Flash");
    expect(text).not.toContain("GLM-5.3-Flash");
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
