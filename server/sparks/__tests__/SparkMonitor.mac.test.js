/**
 * SparkMonitor wiring for Apple Silicon Mac units (kind "mac").
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { SparkMonitor, createCollector } from "../SparkMonitor.js";
import { SystemCollector } from "../../collectors/SystemCollector.js";
import { MacSystemCollector } from "../../collectors/MacSystemCollector.js";

function unit(overrides = {}) {
  return {
    id: "qualitycorp",
    name: "qualitycorp",
    lanIp: "100.96.225.114",
    isLocal: false,
    ssh: { host: "100.96.225.114", user: "qualitycorp", auth: "key" },
    llmPorts: [8000],
    role: "standalone",
    llmMonitoring: true,
    disabledDevices: [],
    disabledInterfaces: [],
    kind: "mac",
    ...overrides,
  };
}

test("collector is chosen by unit kind", () => {
  assert.ok(createCollector(unit()) instanceof MacSystemCollector);
  assert.ok(!(createCollector(unit({ kind: "host" })) instanceof MacSystemCollector));
  assert.ok(createCollector(unit({ kind: "host" })) instanceof SystemCollector);
  assert.ok(!(createCollector(unit({ kind: "spark" })) instanceof MacSystemCollector));
  assert.ok(!(createCollector(unit({ kind: undefined })) instanceof MacSystemCollector));
});

test("a mac unit gets a Mac hardware summary, not the DGX Spark specs", () => {
  const monitor = new SparkMonitor(unit());
  assert.ok(monitor.collector instanceof MacSystemCollector);
  assert.equal(monitor._hardwareSummary.device, "Apple Silicon Mac");
  assert.notEqual(monitor._hardwareSummary.cpuModel, "GB10");
  monitor.stop?.();
});

test("spark and host summaries are unchanged", () => {
  const spark = new SparkMonitor(unit({ kind: "spark", isLocal: true, ssh: undefined }));
  assert.equal(spark._hardwareSummary.device, "NVIDIA DGX Spark");
  spark.stop?.();
  const host = new SparkMonitor(unit({ kind: "host" }));
  assert.equal(host._hardwareSummary.device, "Linux GPU host");
  host.stop?.();
});

test("mac uptime comes from kern.boottime", async () => {
  const monitor = new SparkMonitor(unit());
  const boot = Math.floor(Date.now() / 1000) - 3600;
  monitor.collector.readBootTime = async () => boot;
  const up = await monitor._readUptime();
  assert.ok(up >= 3599 && up <= 3601, `uptime ${up}`);
  monitor.collector.readBootTime = async () => null;
  assert.equal(await monitor._readUptime(), null);
  monitor.stop?.();
});

test("switching a unit to kind mac swaps in the Mac collector", () => {
  const monitor = new SparkMonitor(unit({ kind: "host" }));
  assert.ok(!(monitor.collector instanceof MacSystemCollector));
  monitor.updateConfig(unit({ kind: "mac" }));
  assert.ok(monitor.collector instanceof MacSystemCollector);
  monitor.updateConfig(unit({ kind: "host" }));
  assert.ok(!(monitor.collector instanceof MacSystemCollector));
  monitor.stop?.();
});
