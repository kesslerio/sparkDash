import { useEffect, useState } from "react";
import type { LlmMetrics, LlmTelemetryPoint } from "../../api/types";
import { fetchLlmTelemetry } from "../../api/client";
import { llmConcurrency, summarizeLlmHour, type LlmHourSummary } from "./llmHour";

const HOUR_MS = 60 * 60_000;
const REFRESH_MS = 30_000;
/** Pips past this collapse into a "+N" count so a busy engine cannot wrap the row. */
const MAX_PIPS = 16;

function formatRate(rate: number | null | undefined): string {
  return rate == null || !Number.isFinite(rate) ? "—" : rate.toFixed(0);
}

/** Last hour of ten-second telemetry for one LLM port, refreshed every 30 s. */
function useLlmHour(sparkId: string, port: number | null): LlmHourSummary | null {
  const [summary, setSummary] = useState<LlmHourSummary | null>(null);
  useEffect(() => {
    setSummary(null);
    if (port == null) return;
    let alive = true;
    const load = () =>
      fetchLlmTelemetry(sparkId, port, 1)
        .then((r) => alive && setSummary(summarizeLlmHour(r.points ?? [])))
        // A failed read leaves the hour unknown ("—"), never a fake zero.
        .catch(() => alive && setSummary(null));
    void load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [sparkId, port]);
  return summary;
}

function busyCaption(hour: LlmHourSummary | null): string {
  if (!hour) return "60m —";
  if (hour.observedBuckets === 0) return "no data";
  if (hour.busyBuckets === 0) return "idle all hour";
  return `${Math.max(1, Math.round((hour.busyBuckets * 10) / 60))}m busy`;
}

/**
 * `now` comes from the parent's render, so the window advances with each
 * 30 s telemetry refresh rather than on its own clock; that is fine-grained
 * enough for an hour-wide chart.
 */
function DecodeSparkline({ points, p50, now }: { points: LlmTelemetryPoint[]; p50: number | null; now: number }) {
  const W = 180;
  const H = 34;
  const start = now - HOUR_MS;
  const inHour = points.filter((p) => p.t >= start);
  const max = Math.max(1, ...inHour.map((p) => p.generationTps ?? 0));
  const x = (t: number) => ((t - start) / HOUR_MS) * W;
  const y = (v: number) => H - 2 - (v / max) * (H - 4);
  const path = inHour
    .map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.generationTps ?? 0).toFixed(1)}`)
    .join("");
  const idle = p50 == null;
  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      className="block shrink-0"
      role="img"
      aria-label={idle ? "Decode tok/s, last 60 minutes: idle" : `Decode tok/s, last 60 minutes, median ${formatRate(p50)}`}
    >
      <line x1={0} x2={W} y1={H - 2} y2={H - 2} stroke="var(--color-border)" />
      {path && !idle ? <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={1.25} /> : null}
      {/* A lone "M" path draws nothing, so the first busy sample gets a dot. */}
      {inHour.length === 1 && !idle ? (
        <circle cx={x(inHour[0].t)} cy={y(inHour[0].generationTps ?? 0)} r={1.5} fill="var(--color-accent)" />
      ) : null}
      {!idle ? (
        <line
          x1={0}
          x2={W}
          y1={y(p50)}
          y2={y(p50)}
          stroke="var(--color-text)"
          strokeOpacity={0.55}
          strokeDasharray="2 3"
        />
      ) : null}
    </svg>
  );
}

/** Filled = running, outlined warning = queued, faint = room up to the slot cap or the hour's peak. */
function RequestPips({ running, waiting, extent }: { running: number; waiting: number; extent: number }) {
  const total = Math.max(extent, running + waiting, 1);
  const shown = Math.min(total, MAX_PIPS);
  return (
    <span className="flex items-center gap-1" aria-hidden>
      {Array.from({ length: shown }, (_, i) => (
        <span
          key={i}
          className={`h-2.5 w-2.5 rounded-full border ${
            i < running ? "border-accent bg-accent" : i < running + waiting ? "border-warning" : "border-border"
          }`}
        />
      ))}
      {total > shown ? <span className="text-[11px] text-muted">+{total - shown}</span> : null}
    </span>
  );
}

/**
 * Overview card footer: live decode rate against the hour's shape, concurrent
 * requests, and prefill. The hour comes from the restart-safe telemetry store,
 * so medians survive a dashboard restart.
 */
export function OverviewLlmFooter({ sparkId, port, llm }: { sparkId: string; port: number | null; llm: LlmMetrics }) {
  const hour = useLlmHour(sparkId, port);
  const { running, waiting, capacity } = llmConcurrency(llm);
  const extent = capacity ?? hour?.peakRunning ?? 0;
  const requestsTitle =
    running == null
      ? "This engine does not report running requests"
      : `${running} running${waiting ? `, ${waiting} queued` : ""}${
          capacity != null ? ` of ${capacity} slots` : hour?.peakRunning != null ? `; peak ${hour.peakRunning} in the last hour` : ""
        }`;

  return (
    <div className="mt-3.5 flex flex-col gap-3 border-t border-border pt-3">
      <div className="flex items-end justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <div>
            <span className="font-tabular text-[28px] font-bold leading-none text-text-strong">
              {formatRate(llm.generationTps)}
            </span>
            <span className="text-sm font-normal text-muted"> tok/s</span>
          </div>
          <div className="font-tabular text-[11px] text-muted" title="Median decode tok/s over busy ten-second samples in the last hour">
            p50 <span className="text-text">{formatRate(hour?.decodeP50)}</span> · {busyCaption(hour)}
          </div>
        </div>
        <DecodeSparkline points={hour?.points ?? []} p50={hour?.decodeP50 ?? null} now={Date.now()} />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 text-[12px]">
        <div className="flex items-center gap-2 whitespace-nowrap" title={requestsTitle}>
          <span className="text-muted">Requests</span>
          {running == null ? (
            <span className="text-muted">—</span>
          ) : (
            <>
              <RequestPips running={running} waiting={waiting ?? 0} extent={extent} />
              <span className="font-tabular text-text">
                {running}
                {capacity != null ? <span className="text-muted">/{capacity}</span> : null}
                {waiting ? <span className="text-warning"> +{waiting}q</span> : null}
                {capacity == null && hour?.peakRunning ? <span className="text-muted"> · pk {hour.peakRunning}</span> : null}
              </span>
            </>
          )}
        </div>
        <div
          className="whitespace-nowrap font-tabular text-muted"
          title="Live prefill tok/s · median over busy ten-second samples in the last hour"
        >
          prefill <span className="font-semibold text-text">{formatRate(llm.prefillTps)}</span>
          <span> · p50 {formatRate(hour?.prefillP50)}</span>
        </div>
      </div>
    </div>
  );
}
