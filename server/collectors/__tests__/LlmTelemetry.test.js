import { test } from "node:test";
import { strict as assert } from "node:assert";
import fs from "fs";
import os from "os";
import path from "path";
import { LlmTelemetryStore } from "../LlmTelemetry.js";

function makeStore() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sparkdash-llm-telemetry-"));
  return { dir, file: path.join(dir, "telemetry.json"), store: new LlmTelemetryStore(path.join(dir, "telemetry.json")) };
}

test("LlmTelemetryStore buckets, retains unavailable samples, and persists", () => {
  const { dir, file, store } = makeStore();
  try {
    const base = new Date("2026-08-27T12:00:01.000Z");
    store.record("john", 8888, { available: true, backend: "vllm", requestsRunning: 2 }, base);
    store.record("john", 8888, { available: true, backend: "vllm", requestsRunning: 4 }, new Date(base.getTime() + 8_000));
    store.record("john", 8888, { available: false, backend: "vllm", requestsRunning: null }, new Date(base.getTime() + 11_000));

    const series = store.getSeries("john", 8888, { hours: 1, now: new Date(base.getTime() + 20_000) });
    assert.equal(series.bucketSeconds, 10);
    assert.equal(series.retentionHours, 168);
    assert.equal(series.points.length, 2);
    assert.equal(series.points[0].requestsRunning, 4);
    assert.equal(series.points[1].available, false);

    store.flush();
    assert.ok(fs.statSync(`${file}.journal`).size > 0);
    const restored = new LlmTelemetryStore(file);
    assert.deepEqual(
      restored.getSeries("john", 8888, { hours: 1, now: new Date(base.getTime() + 20_000) }),
      series
    );
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("LlmTelemetryStore compacts its recovery journal without losing samples", () => {
  const { dir, file, store } = makeStore();
  try {
    const now = new Date("2026-08-27T12:00:00.000Z");
    store.record("john", 8888, { available: true, requestsWaiting: 3 }, now);
    store.flush();
    store.compact();

    assert.equal(fs.readFileSync(`${file}.journal`, "utf8"), "");
    const restored = new LlmTelemetryStore(file);
    assert.equal(restored.getSeries("john", 8888, { hours: 1, now }).points[0].requestsWaiting, 3);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("LlmTelemetryStore prunes samples older than seven days", () => {
  const { dir, store } = makeStore();
  try {
    const now = new Date("2026-08-27T12:00:00.000Z");
    store.record("john", 8888, { available: false }, new Date(now.getTime() - 8 * 86400000));
    store.record("john", 8888, { available: true }, now);
    const series = store.getSeries("john", 8888, { hours: 168, now });
    assert.equal(series.points.length, 1);
    assert.equal(series.points[0].available, true);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
