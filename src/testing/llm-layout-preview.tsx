/**
 * Dev-only LLM layout preview. Built from llm-layout-preview.html, never from
 * the production index. Renders the real LlmPanel against illustrative fixtures
 * and answers every request with a local fixture or a local 403; no request
 * ever reaches a network.
 */
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "../index.css";
import { LlmPanel } from "../components/SparkPage/LlmPanel";
import type { ConversationRow, LlmMetrics } from "../api/types";
import {
  DEEPSEEK_PORT,
  GLM_PORT,
  LLM_PREVIEW_SPARK_ID,
  makeDeepSeekActive,
  makeDeepSeekIdle,
  makeDeepSeekUnavailable,
  makeDs4Stopped,
  makeGlmIdle,
  makeLlmDailyDays,
  makeLlmTokenSeries,
  makeSessionRows,
  seedLlmHistory,
} from "./llmFixtures";

const DS4_PORT = 8001;
const LLM_PORTS = [DEEPSEEK_PORT, GLM_PORT, DS4_PORT];
const LONG_MODEL_ID =
  "DeepSeek-V4.1-Flash-oQ2.5e-mtp-quantized-checkpoint-with-a-deliberately-long-name-for-overflow-checks";
const RANGES = new Set(["all", "today", "7d", "14d", "30d"]);
const PREVIEW_ONLY = "Preview only — no live requests";

type PreviewState = "active" | "idle" | "unavailable";
type Theme = "dark" | "light" | "white" | "oled";

function installPreviewFetch(now: number) {
  const days = makeLlmDailyDays();
  const series = makeLlmTokenSeries(LLM_PREVIEW_SPARK_ID, now);
  const reply = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    const url = new URL(request.url, window.location.href);
    if (request.method !== "GET") return reply({ error: PREVIEW_ONLY }, 403);
    if (url.pathname === `/api/sparks/${LLM_PREVIEW_SPARK_ID}/llm/daily`) {
      const port = Number(url.searchParams.get("port"));
      return reply({
        sparkId: LLM_PREVIEW_SPARK_ID,
        port,
        days: port === DEEPSEEK_PORT ? days : [],
      });
    }
    const range = url.searchParams.get("range") ?? "all";
    if (url.pathname === "/api/llm-token-totals" && RANGES.has(range)) {
      return reply({ range, series });
    }
    return reply({ error: PREVIEW_ONLY }, 403);
  };
}

function deepseekFor(state: PreviewState, now: number, longName: boolean): LlmMetrics {
  const base =
    state === "idle"
      ? makeDeepSeekIdle(now)
      : state === "unavailable"
        ? makeDeepSeekUnavailable(now)
        : makeDeepSeekActive(now);
  return longName ? { ...base, modelId: LONG_MODEL_ID } : base;
}

function PreviewApp({ now }: { now: number }) {
  const [state, setState] = useState<PreviewState>("active");
  const [theme, setTheme] = useState<Theme>("dark");
  const [populated, setPopulated] = useState(true);
  const [narrow, setNarrow] = useState(false);
  const [longName, setLongName] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const sessions: ConversationRow[] = populated ? makeSessionRows(now) : [];
  const openHarness = () => setNote("Harness wizard is disabled in the layout preview.");
  const laneClass = narrow ? undefined : "md:col-span-2";

  return (
    <main
      style={{
        maxWidth: 1180,
        margin: "0 auto",
        padding: 24,
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr)",
        gap: 16,
      }}
    >
      <header style={{ display: "grid", gap: 8 }}>
        <h1 style={{ margin: 0, fontSize: 18 }}>LLM card layout preview</h1>
        <p role="status" style={{ margin: 0, fontSize: 13 }}>
          Design preview — illustrative telemetry, no live connection
        </p>
      </header>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, fontSize: 13, maxWidth: "100%" }}>
        <label>
          DeepSeek state{" "}
          <select value={state} onChange={(e) => setState(e.target.value as PreviewState)}>
            <option value="active">Active</option>
            <option value="idle">Idle</option>
            <option value="unavailable">Unavailable</option>
          </select>
        </label>
        <label>
          Theme{" "}
          <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)}>
            <option value="dark">Dark</option>
            <option value="light">Light</option>
            <option value="white">White</option>
            <option value="oled">OLED</option>
          </select>
        </label>
        <label>
          <input type="checkbox" checked={populated} onChange={(e) => setPopulated(e.target.checked)} />{" "}
          Sample sessions
        </label>
        <label>
          <input type="checkbox" checked={longName} onChange={(e) => setLongName(e.target.checked)} />{" "}
          Long model name
        </label>
        <label>
          <input type="checkbox" checked={narrow} onChange={(e) => setNarrow(e.target.checked)} />{" "}
          Narrow (400px, stacked)
        </label>
      </div>
      {note && <p style={{ margin: 0, fontSize: 12 }}>{note}</p>}
      <div
        className="spark-page grid grid-cols-1 md:grid-cols-2"
        style={{
          gap: "var(--density-page-gap)",
          width: narrow ? 400 : "100%",
          maxWidth: "100%",
          gridTemplateColumns: narrow ? "minmax(0, 1fr)" : undefined,
        }}
      >
        <LlmPanel
          llm={deepseekFor(state, now, longName)}
          sparkId={LLM_PREVIEW_SPARK_ID}
          sparkName="Preview unit"
          llmPort={DEEPSEEK_PORT}
          conversations={sessions}
          llmPorts={LLM_PORTS}
          onOpenHarnessWizard={openHarness}
          className={laneClass}
        />
        <LlmPanel
          llm={makeGlmIdle(now)}
          sparkId={LLM_PREVIEW_SPARK_ID}
          sparkName="Preview unit"
          llmPort={GLM_PORT}
          llmPorts={LLM_PORTS}
          onOpenHarnessWizard={openHarness}
        />
        <LlmPanel
          llm={makeDs4Stopped()}
          sparkId={LLM_PREVIEW_SPARK_ID}
          sparkName="Preview unit"
          llmPort={DS4_PORT}
          llmPorts={LLM_PORTS}
          onOpenHarnessWizard={openHarness}
        />
      </div>
    </main>
  );
}

const now = Date.now();
installPreviewFetch(now);
seedLlmHistory(LLM_PREVIEW_SPARK_ID, now);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <PreviewApp now={now} />
  </StrictMode>,
);
