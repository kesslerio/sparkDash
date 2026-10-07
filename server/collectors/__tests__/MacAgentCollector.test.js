import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MacAgentCollector,
  MAC_AGENT_DEFAULT_PORT,
  macAgentBaseUrl,
  isUnavailable,
} from "../MacAgentCollector.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const snapshot = JSON.parse(
  fs.readFileSync(path.join(here, "fixtures", "mac", "agent-snapshot.json"), "utf8")
);

const spark = {
  id: "macbook-m5",
  kind: "mac",
  isLocal: false,
  lanIp: "192.168.4.125",
  agent: { port: 8790 },
  ssh: { user: "kesslerio", host: "192.168.4.125", auth: "key" },
};

/** Collector over a stub agent; `mode` decides how /metrics answers. */
function collectorWith(mode, { execCalls = [] } = {}) {
  const requested = [];
  const fetchImpl = async (url) => {
    requested.push(url);
    if (mode === "down") throw new Error("ECONNREFUSED");
    if (mode === "garbage") return { ok: true, json: async () => ({ hello: "world" }) };
    if (mode === "http500") return { ok: false, status: 503, json: async () => ({}) };
    return { ok: true, json: async () => structuredClone(snapshot) };
  };
  const exec = async (_spark, cmd) => {
    execCalls.push(cmd);
    // SSH fallback answers look like the batched Mac snapshot read.
    return "sysctl hw.memsize: 274877906944\n";
  };
  const collector = new MacAgentCollector({ ...spark }, { fetchImpl, exec });
  return { collector, requested, execCalls };
}

test("agent url: configured url wins, then host/port, then the default port", () => {
  assert.equal(
    macAgentBaseUrl({ kind: "mac", agent: { url: "http://mac.tailnet.ts.net:9000/" } }),
    "http://mac.tailnet.ts.net:9000"
  );
  assert.equal(
    macAgentBaseUrl({ kind: "mac", lanIp: "10.0.0.9", agent: { port: 9100 } }),
    "http://10.0.0.9:9100"
  );
  assert.equal(
    macAgentBaseUrl({ kind: "mac", isLocal: true, agent: {} }),
    `http://127.0.0.1:${MAC_AGENT_DEFAULT_PORT}`
  );
});

test("vitals come from one agent read and keep the existing shapes", async () => {
  const { collector, requested } = collectorWith("ok");
  const [cpu, ram, um, storage, gpu] = await Promise.all([
    collector.collectCpu(),
    collector.collectRam(),
    collector.collectUnifiedMemory(),
    collector.collectStorage(),
    collector.collectGpu(),
  ]);
  assert.equal(cpu.usage, 37.5);
  assert.equal(cpu.temperature, 0, "Apple Silicon reports no CPU temperature");
  assert.equal(ram.total, 131072);
  assert.equal(ram.used, 110103);
  assert.deepEqual(ram.swap, { total: 10240, used: 8607, free: 1633 });
  assert.equal(um.gpuUsed, 14938);
  assert.equal(um.cpuUsed, 95165);
  assert.deepEqual(um.bandwidth, { current: 0, peak: 0 });
  assert.equal(storage[0].label, "/System/Volumes/Data");
  assert.equal(storage[0].total, 3811147);
  assert.equal(gpu.vram.total, 131072);
  assert.equal(gpu.vram.used, 110103);
  assert.equal(gpu.powermetricsAvailable, false, "powermetrics was refused for lack of root");
  assert.equal(isUnavailable(gpu.unavailable, "gpu.power"), true);
  assert.equal(isUnavailable(gpu.unavailable, "ane.power"), true);
  await collector.collectNetwork();
  assert.equal(
    requested.filter((url) => url.endsWith("/metrics")).length <= 2,
    true,
    "the TTL cache keeps a poll cycle to one or two agent reads"
  );
});

test("powermetrics refusal shows as a declared gap, not a zero", async () => {
  const { collector } = collectorWith("ok");
  const gpu = await collector.collectGpu();
  assert.equal(gpu.usage, 0);
  assert.equal(gpu.power.draw, 0);
  assert.equal(gpu.powermetricsAvailable, false);
  const gap = gpu.unavailable.find((row) => row.metric === "gpu.power");
  assert.match(gap.reason, /root/);
});

test("network: interface, ip, reachability probes, and counter-delta speeds", async () => {
  const { collector } = collectorWith("ok");
  const first = await collector.collectNetwork();
  assert.equal(first.primaryInterface, "en0");
  assert.equal(first.interfaces[0].ip, "192.168.4.125");
  assert.equal(first.interfaces[0].rxSpeed, 0, "no previous sample to difference against");
  assert.deepEqual(
    first.reachability.map((row) => row.ok),
    [true, true]
  );
  collector._agentCache = null;
  collector._lastNet.time -= 2000;
  snapshot.network.interfaces[0].rxBytes += 4_000_000;
  const second = await collector.collectNetwork();
  assert.ok(
    second.interfaces[0].rxSpeed >= 1_900_000 && second.interfaces[0].rxSpeed <= 2_100_000,
    `expected ~2 MB/s, got ${second.interfaces[0].rxSpeed}`
  );
});

test("runtimes: names, ports, models, and the unknown-listener row", async () => {
  const { collector } = collectorWith("ok");
  const result = await collector.collectRuntimes();
  assert.equal(result.agentOnline, true);
  assert.deepEqual(
    result.runtimes.map((row) => [row.name, row.port, row.state]),
    [
      ["mtplx-mux", 8200, "serving"],
      ["tensorfold", 8300, "serving"],
      ["other-runtime", 9099, "listening"],
    ]
  );
  assert.equal(
    result.runtimes.find((row) => row.name === "tensorfold").model,
    "qwen3.8-27b"
  );
  assert.equal(
    result.runtimes.find((row) => row.name === "other-runtime").label,
    "infer.py",
    "an uncatalogued server keeps its command basename"
  );
  assert.ok(result.unavailable.length >= 3);
});

test("uptime and hardware come from the agent too", async () => {
  const { collector } = collectorWith("ok");
  assert.equal(await collector.readBootTime(), 1790917878);
  const hw = await collector.detectHardware();
  assert.equal(hw.device, "Mac (Mac17,6)");
  assert.equal(hw.cpuModel, "Apple M5 Max");
  assert.equal(hw.cpuCores, 18);
  assert.equal(hw.totalMemoryGB, 128);
  assert.equal(await collector.pingHost(), true);
});

test("agent down: SSH answers the vitals and the gap says the agent is gone", async () => {
  for (const mode of ["down", "garbage", "http500"]) {
    const execCalls = [];
    const { collector } = collectorWith(mode, { execCalls });
    const runtimes = await collector.collectRuntimes();
    assert.equal(runtimes.agentOnline, false, mode);
    assert.deepEqual(runtimes.runtimes, [], mode);
    const gap = runtimes.unavailable.find((row) => row.metric === "agent");
    assert.match(gap.reason, /mac agent unreachable/, mode);

    // Vitals fall back to the SSH reads instead of going blank.
    await collector.collectRam();
    assert.ok(
      execCalls.some((cmd) => cmd.includes("vm_stat") || cmd.includes("sysctl")),
      `${mode}: expected an SSH fallback read`
    );
  }
});
