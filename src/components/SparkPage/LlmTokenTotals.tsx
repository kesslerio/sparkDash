/**
 * LlmTokenTotals — per-model cumulative prompt/completion token table.
 * Local-only component: lives in its own file, rendered by LlmPanel via one
 * anchor line + one import line (see docs/LOCAL-MODS.md).
 */
import { useEffect, useState } from "react";
import { fetchLlmTokenTotals } from "../../api/llmTokenClient";
import { formatTokensCompact } from "../../shared/tokenFormat";
import type { LlmTokenRange, LlmTokenSeriesTotals } from "../../api/llmTokenTypes";

const POLL_MS = 60_000;

const RANGE_OPTIONS: Array<{ value: LlmTokenRange; label: string }> = [
  { value: "all", label: "All time" },
  { value: "today", label: "Today" },
  { value: "7d", label: "Last 7 days" },
  { value: "14d", label: "Last 14 days" },
  { value: "30d", label: "Last month" },
];

function age(ms: number | null): string | null {
  if (ms == null || ms <= 0) return null;
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 90) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 90) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

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
    <div className="border-t border-border pt-3 space-y-1.5">
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <span className="text-[10px] uppercase tracking-wide text-muted">
          Total tokens by model
        </span>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <select
            value={range}
            onChange={(e) => setRange(e.target.value as LlmTokenRange)}
            aria-label="Token totals time range"
            className="rounded border border-border bg-surface-elevated text-text"
            style={{ height: "20px", padding: "0 4px", fontSize: "9px", width: "auto" }}
          >
            {RANGE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <span className="shrink-0 whitespace-nowrap text-[10px] text-muted">
            <span className="inline-block w-12 text-right">Cached</span>

            <span className="inline-block w-12 text-right">Prefill</span>

            <span className="inline-block w-16 text-right">Generated</span>

            <span className={`${COL.total} inline-block w-14 text-right`}>Total</span>
          </span>
        </div>
      </div>
      <div className="space-y-1">
        {rows.length === 0 ? (
          <p className="text-[11px] text-muted">No tokens recorded in this period.</p>
        ) : (
          rows.map((row) => {
          const seen = age(row.lastSeenAt);
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
