import type { CpuMetrics, HardwareInfo, MetricGap } from "../../api/types";
import { Sparkline } from "../ui/Sparkline";
import { Panel } from "../ui/Panel";
import { CpuIcon } from "../ui/icons";
import { honestGap } from "../ui/platform";
import { useMetricsHistoryTail } from "../../hooks/metricsStore";

interface CpuPanelProps {
  cpu: CpuMetrics | null;
  hardware?: HardwareInfo | null;
  sparkId: string;
  temperatureUnit: "celsius" | "fahrenheit";
  className?: string;
  /** Apple Silicon Mac: no CPU temperature sensor, and power only with root. */
  mac?: boolean;
  /** Metrics this node declared unavailable (mac agent transport). */
  unavailable?: MetricGap[] | null;
}

function celsiusToFahrenheit(c: number): number {
  return Math.round((c * 9) / 5 + 32);
}

function MetricRow({
  label,
  spark,
  value,
  color = "var(--color-accent)",
}: {
  label: string;
  spark: React.ReactNode;
  value: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted">{label}</span>
      <div className="flex items-center gap-3">
        <span style={{ color }}>{spark}</span>
        <span className="font-tabular text-sm font-semibold text-text">{value}</span>
      </div>
    </div>
  );
}

/**
 * CPU panel — usage, temperature, and power for the SoC CPU.
 *
 * On GB10 devices (DGX Spark / GX10) the CPU and GPU share one package and
 * one power envelope, so the CPU is often the part that runs hot first —
 * this panel makes that visible at the device level. For non-Spark GPU
 * hosts it covers the discrete CPU.
 */
export function CpuPanel({
  cpu,
  hardware,
  sparkId,
  temperatureUnit,
  className,
  mac = false,
  unavailable = null,
}: CpuPanelProps) {
  const usageHistory = useMetricsHistoryTail(sparkId, "cpu.usage");
  const tempHistory = useMetricsHistoryTail(sparkId, "cpu.temp");

  const usage = cpu?.usage ?? 0;
  const temperature = cpu?.temperature ?? 0;
  const draw = cpu?.draw ?? 0;
  const tdp = cpu?.tdp ?? 0;

  const displayTemp =
    temperatureUnit === "fahrenheit" ? celsiusToFahrenheit(temperature) : temperature;
  const tempLabel = temperatureUnit === "fahrenheit" ? `${displayTemp}°F` : `${displayTemp}°C`;

  // A missing Mac temperature stays unavailable. A finite reading from the
  // privileged collector is a real value, not a platform gap.
  const hasTemperature =
    typeof cpu?.temperature === "number" && Number.isFinite(cpu.temperature) && cpu.temperature > 0;
  const tempView = honestGap(unavailable, "cpu.temperature", tempLabel, {
    nonDgx: !hasTemperature && (mac || cpu?.temperature == null),
  });
  const powerView = honestGap(unavailable, "cpu.power", `${draw}W${tdp > 0 ? ` / ${tdp}W` : ""}`, {
    nonDgx: mac && draw === 0,
  });

  // GB10 SoC bands: the CPU complex derates in the mid-80s; x86 hosts run
  // hotter before throttling, so the danger band sits higher.
  const tempColor =
    temperature > 95
      ? "var(--color-danger)"
      : temperature > 85
        ? "var(--color-warning)"
        : "var(--color-accent)";

  const model = hardware?.cpuModel;
  const cores = hardware?.cpuCores;

  return (
    <Panel
      title="CPU"
      icon={<CpuIcon />}
      className={`panel-cpu ${className ?? ""}`}
      bodyClassName="space-y-3"
    >
      <MetricRow
        label="Usage"
        color="var(--color-accent)"
        spark={<Sparkline data={usageHistory} color="var(--color-accent)" width={180} />}
        value={<span className="text-text-strong">{usage}%</span>}
      />
      <MetricRow
        label="Temperature"
        color={tempColor}
        spark={tempView.muted ? null : <Sparkline data={tempHistory} color={tempColor} width={180} />}
        value={
          <span
            className={`text-text-strong ${tempView.muted ? "font-normal text-muted" : ""}`}
            title={tempView.title}
          >
            {tempView.muted ? `unavailable: ${tempView.title}` : tempView.text}
          </span>
        }
      />
      <div className="flex justify-between text-sm">
        <span className="text-muted">CPU Power</span>
        <span
          className={`font-tabular text-sm ${powerView.muted ? "text-muted" : "text-text"}`}
          title={powerView.title}
        >
          {powerView.text}
        </span>
      </div>
      {model && (
        <div className="flex justify-between border-t border-border pt-3 text-xs">
          <span className="text-muted">Model</span>
          <span className="font-tabular text-text" title={model}>
            {model}
            {cores != null ? ` · ${cores} cores` : ""}
          </span>
        </div>
      )}
    </Panel>
  );
}
