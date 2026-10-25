/**
 * LlmTokenTotals — per-model cumulative prompt/completion token table.
 * Local-only component: lives in its own file, rendered by LlmPanel via one
 * anchor line + one import line (see docs/LOCAL-MODS.md).
 */
import { useEffect, useState } from "react";
import { fetchLlmTokenTotals } from "../../api/llmTokenClient";
import { formatTokensCompact } from "../../shared/tokenFormat";
import { formatSince } from "../../shared/formatSince";
import type { LlmTokenRange, LlmTokenSeriesTotals } from "../../api/llmTokenTypes";
import { LEDGER_HINT, LEDGER_TITLE } from "./tokenTotalsCopy";

const POLL_MS = 60_000;

const RANGE_OPTIONS: Array<{ value: LlmTokenRange; label: string }> = [
  { value: "all", label: "All time" },
  { value: "today", label: "Last 24 hours" },
  { value: "6h", label: "Last 6 hours" },
  { value: "1h", label: "Last 1 hour" },
  { value: "7d", label: "Last 7 days" },
  { value: "14d", label: "Last 14 days" },
  { value: "30d", label: "Last month" },
];

/** Column widths shared by the header and every row so they cannot drift. */
const COL = {
  bar: "llm-token-bar",
  cached: "w-12",
  prefill: "w-12",
  generated: "w-16",
  total: "llm-token-total",
} as const;

/**
 * Composition bar for one model row: cached prompt, computed prefill and
 * generated tokens in the same proportions as the token table's columns, so
 * the cache-hit ratio is readable without doing the arithmetic.
 */
function CompositionBar({
  cached,
  prefill,
  generated,
}: {
  cached: number;
  prefill: number;
  generated: number;
}) {
  const total = cached + prefill + generated;
  if (total <= 0) return null;
  const parts = [
    { label: "cached", value: cached, color: "var(--color-muted)" },
    { label: "prefill", value: prefill, color: "var(--color-text)" },
    { label: "generated", value: generated, color: "var(--color-accent)" },
  ];
  return (
    <span
      className={`${COL.bar} inline-flex h-1.5 w-14 shrink-0 overflow-hidden rounded-full bg-surface-elevated align-middle`}
      aria-hidden
    >
      {parts.map((part) => {
        const pct = (part.value / total) * 100;
        if (pct <= 0) return null;
        return (
          <span
            key={part.label}
            className="h-full"
            style={{ width: `${pct}%`, background: part.color }}
            title={`${part.label} ${pct.toFixed(0)}%`}
          />
        );
      })}
    </span>
  );
}

export function LlmTokenTotals({ sparkId, llmPort }: { sparkId: string; llmPort: number }) {
  const [series, setSeries] = useState<LlmTokenSeriesTotals[] | null>(null);
  const [range, setRange] = useState<LlmTokenRange>("all");

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetchLlmTokenTotals(range)
        .then((res) => {
          if (!cancelled) setSeries(res.series || []);
        })
        .catch(() => {
          if (!cancelled) setSeries([]);
        });
    };
    load();
    const t = setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [sparkId, llmPort, range]);

  if (!series || series.length === 0) return null;

  const rows = series
    .filter((s) => s.sparkId === sparkId && s.port === llmPort)
    .flatMap((s) => s.models);

  if (rows.length === 0 && range === "all") return null;

  return (
    <div className="sp-section">
      <div className="sp-section__head">
        <span className="min-w-0" title={LEDGER_TITLE}>
          <span className="eyebrow block">Total tokens by model</span>
          <span className="sp-muted block">{LEDGER_HINT}</span>
        </span>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <select
            value={range}
            onChange={(e) => setRange(e.target.value as LlmTokenRange)}
            aria-label="Token totals time range"
            className="sp-select"
          >
            {RANGE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <span className="sp-tok-cols">
            <span className="inline-block w-14 text-right">Cached</span>

            <span className="inline-block w-12 text-right">Prefill</span>

            <span className="inline-block w-16 text-right">Generated</span>

            <span className={`${COL.total} inline-block w-14 text-right`}>Total</span>
          </span>
        </div>
      </div>
      <div className="sp-tok-rows">
        {rows.length === 0 ? (
          <p className="sp-muted">No tokens recorded in this period.</p>
        ) : (
          rows.map((row) => {
          const seen = formatSince(row.lastSeenAt);
          const prefillTokens = row.promptTokens - row.cachedTokens;
          const totalTokens = row.promptTokens + row.completionTokens;
          return (
            <div
              key={row.modelId}
              className="flex items-center justify-between gap-2 text-[11px]"
              title={`${row.promptTokens.toLocaleString()} prompt · ${row.cachedTokens.toLocaleString()} cached · ${prefillTokens.toLocaleString()} prefill · ${row.completionTokens.toLocaleString()} generated${seen ? ` · ${seen} ago` : ""}`}
            >
              <span
                className="min-w-0 flex-1 truncate text-text"
                title={row.modelId}
              >
                {row.modelId}
              </span>
              <span className="shrink-0 font-tabular text-muted">
                <CompositionBar
                  cached={row.cachedTokens}
                  prefill={prefillTokens}
                  generated={row.completionTokens}
                />

                <span className="inline-block w-12 text-right">
                  {row.cachedTokens > 0 ? formatTokensCompact(row.cachedTokens) : "—"}
                </span>

                <span className="inline-block w-12 text-right">
                  {formatTokensCompact(prefillTokens)}
                </span>

                <span className="inline-block w-16 text-right text-text">
                  {formatTokensCompact(row.completionTokens)}
                </span>

                <span className={`${COL.total} inline-block w-14 text-right text-text`}>
                  {formatTokensCompact(totalTokens)}
                </span>
              </span>
            </div>
          );
        })
        )}
      </div>
    </div>
  );
}
