/**
 * The MQTT bridge is driven entirely through injected dependencies, so these
 * tests exercise the real class against a stub client — no broker needed.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { ACK_TIMEOUT_MS, MAX_IN_FLIGHT, MAX_PENDING_REMOVALS, MqttPublisher } from "../MqttPublisher.js";

/**
 * Stub mqtt.js client capturing the calls the publisher makes. With `autoAck`
 * (the default) every publish completes at once; otherwise its callback waits
 * in `calls.pending` until the test calls `ack()`.
 */
function fakeClient({ autoAck = true, ...overrides } = {}) {
  const calls = { publishes: [], pending: [], ended: 0, endForce: null, handlers: {} };
  const client = {
    connected: true,
    publish(topic, payload, opts, cb) {
      calls.publishes.push({ topic, payload: String(payload), opts: opts ?? {} });
      if (autoAck) cb?.();
      else calls.pending.push(cb);
    },
    end(force) {
      calls.ended += 1;
      calls.endForce = force;
      // Like mqtt.js: closing fails every outstanding publish.
      for (const cb of calls.pending.splice(0)) cb?.(new Error("Connection closed"));
    },
    on(event, fn) {
      calls.handlers[event] = fn;
    },
    ...overrides,
  };
  return { client, calls };
}

const ENABLED = { mqttEnabled: true, mqttUrl: "mqtt://broker.local:1883", mqttUsername: "", mqttTopicPrefix: "sparkdash" };

/** Build a publisher wired to a stub client; `connect` records every attempt. */
function harness({ settings = {}, client: clientOverrides = {}, now } = {}) {
  let current = { ...ENABLED, ...settings };
  const attempts = [];
  const warnings = [];
  const { client, calls } = fakeClient(clientOverrides);
  const publisher = new MqttPublisher({
    getSettings: () => current,
    env: {},
    connect: (url, opts) => {
      attempts.push({ url, opts });
      return client;
    },
    log: { warn: (m) => warnings.push(m), error: (m) => warnings.push(m) },
    ...(now ? { now } : {}),
  });
  const h = {
    publisher,
    client,
    calls,
    attempts,
    warnings,
    set(patch) {
      current = { ...current, ...patch };
    },
    /** Fire a lifecycle event on the stub client, like mqtt.js would. */
    fire(event, arg) {
      calls.handlers[event]?.(arg);
    },
    /** Connect, let the broker accept, and forget the availability announcement. */
    ready() {
      publisher.applySettings();
      h.fire("connect");
      while (calls.pending.length) h.ack();
      calls.publishes.length = 0;
    },
    /** Complete the oldest outstanding publish (optionally with an error). */
    ack(err) {
      calls.pending.shift()?.(err);
    },
    topics(from = 0) {
      return calls.publishes.slice(from).map((p) => p.topic);
    },
  };
  return h;
}

test("disabled: no connection is attempted and nothing is published", () => {
  const h = harness({ settings: { mqttEnabled: false } });
  h.publisher.applySettings();
  h.publisher.publishEvent({ type: "spark.offline", message: "x" });
  h.publisher.publishFleet([{ id: "alpha", online: true }]);
  assert.equal(h.attempts.length, 0);
  assert.equal(h.calls.publishes.length, 0);
});

test("applySettings connects, sends credentials, and publishes dashboard availability on connect", () => {
  const h = harness();
  h.publisher.applySettings();
  assert.equal(h.attempts.length, 1);
  assert.equal(h.attempts[0].url, "mqtt://broker.local:1883");
  assert.match(h.attempts[0].opts.clientId, /^sparkdash-[0-9a-f]{8}$/);
  assert.equal(h.attempts[0].opts.reconnectPeriod, 5_000);
  // The will mirrors our own availability so a crashed dashboard reads 0 at the broker.
  assert.deepEqual(h.attempts[0].opts.will, {
    topic: "sparkdash/dashboard/status",
    payload: "0",
    qos: 0,
    retain: true,
  });
  h.fire("connect");
  assert.deepEqual(h.calls.publishes, [
    { topic: "sparkdash/dashboard/status", payload: "1", opts: { qos: 1, retain: true } },
  ]);
});

test("the broker password comes from the environment, the username from settings", () => {
  let current = { ...ENABLED, mqttUsername: "mia" };
  const attempts = [];
  const publisher = new MqttPublisher({
    getSettings: () => current,
    env: { SPARKDASH_MQTT_PASSWORD: "hunter2" },
    connect: (url, opts) => {
      attempts.push({ url, opts });
      return fakeClient().client;
    },
  });
  publisher.applySettings();
  assert.equal(attempts[0].opts.username, "mia");
  assert.equal(attempts[0].opts.password, "hunter2");
  // A settings PUT that changes nothing relevant must not reconnect…
  publisher.applySettings();
  assert.equal(attempts.length, 1);
  // …but a username change applies live, without touching the password.
  current = { ...current, mqttUsername: "" };
  publisher.applySettings();
  assert.equal(attempts.length, 2);
  assert.equal(attempts[1].opts.username, undefined);
  assert.equal(attempts[1].opts.password, "hunter2");
});

test("events stream to {prefix}/events with QoS 1 and the EventLog record verbatim", () => {
  const h = harness();
  h.publisher.applySettings();
  const event = { id: 7, ts: 1234, type: "spark.offline", severity: "warn", sparkId: "alpha", sparkName: "Alpha", message: "Alpha went offline" };
  h.publisher.publishEvent(event);
  assert.deepEqual(h.calls.publishes, [
    { topic: "sparkdash/events", payload: JSON.stringify(event), opts: { qos: 1 } },
  ]);
  // A custom prefix changes the topic, not the payload.
  h.set({ mqttTopicPrefix: "home/fleet" });
  h.publisher.publishEvent({ ...event, id: 8 });
  assert.equal(h.calls.publishes[1].topic, "home/fleet/events");
});

test("the fleet publishes retained availability plus a metrics payload per unit", () => {
  const h = harness();
  h.ready();
  h.publisher.publishFleet([
    { snapshot: { id: "alpha", online: true, uptime: 100 } }, // wrapped provenance form
    { id: "beta", online: false }, // bare snapshot
  ]);
  assert.equal(h.calls.publishes.length, 4);
  assert.deepEqual(
    h.calls.publishes.filter((p) => p.topic.startsWith("sparkdash/status/")),
    [
      { topic: "sparkdash/status/alpha", payload: "1", opts: { retain: true } },
      { topic: "sparkdash/status/beta", payload: "0", opts: { retain: true } },
    ]
  );
  // The metrics payload is the unit's Prometheus series as JSON: up + what was
  // measured. Retained, so a fresh subscriber sees the last state immediately.
  assert.equal(h.calls.publishes[1].topic, "sparkdash/metrics/alpha");
  assert.equal(h.calls.publishes[1].opts.qos, 0);
  assert.equal(h.calls.publishes[1].opts.retain, true);
  const alpha = JSON.parse(h.calls.publishes[1].payload);
  assert.equal(typeof alpha.ts, "number");
  assert.equal(alpha.up, 1);
  assert.equal(alpha.uptime_seconds, 100);
  // An offline unit says so on its metrics topic too ("sparkdash_up 0 and nothing else").
  assert.equal(h.calls.publishes[3].topic, "sparkdash/metrics/beta");
  assert.equal(h.calls.publishes[3].opts.retain, true);
  const beta = JSON.parse(h.calls.publishes[3].payload);
  assert.deepEqual(beta, { ts: beta.ts, up: 0 });
});

test("a Spark id is percent-encoded into its topics", () => {
  const h = harness();
  h.ready();
  h.publisher.publishFleet([{ id: "spark one", online: true, uptime: 5 }]);
  assert.deepEqual(
    h.calls.publishes.map((p) => p.topic),
    ["sparkdash/status/spark%20one", "sparkdash/metrics/spark%20one"]
  );
});

test("publishes are skipped while the broker is disconnected — never thrown, never queued", () => {
  const h = harness({ client: { connected: false } });
  h.publisher.applySettings();
  h.publisher.publishEvent({ type: "x", message: "y" });
  h.publisher.publishFleet([{ id: "alpha", online: true }]);
  h.fire("connect");
  assert.equal(h.calls.publishes.length, 0);
});

test("garbage entries are ignored", () => {
  const h = harness();
  h.ready();
  h.publisher.publishFleet(null);
  h.publisher.publishFleet([null, {}, { snapshot: {} }, { online: true }]);
  assert.equal(h.calls.publishes.length, 0);
});

test("a changed broker URL or prefix reopens the connection", () => {
  const h = harness();
  h.publisher.applySettings();
  h.set({ mqttUrl: "mqtts://broker.local:8883" });
  h.publisher.applySettings();
  assert.equal(h.calls.ended, 1);
  assert.equal(h.calls.endForce, true);
  assert.equal(h.attempts.length, 2);
  assert.equal(h.attempts[1].url, "mqtts://broker.local:8883");
  h.set({ mqttTopicPrefix: "home" });
  h.publisher.applySettings();
  assert.equal(h.calls.ended, 2);
  assert.equal(h.attempts.length, 3);
  assert.equal(h.attempts[2].opts.will.topic, "home/dashboard/status");
});

test("disabling tears the connection down", () => {
  const h = harness();
  h.publisher.applySettings();
  h.set({ mqttEnabled: false });
  h.publisher.applySettings();
  assert.equal(h.calls.ended, 1);
  // Late publishes after teardown are silent no-ops.
  h.publisher.publishEvent({ type: "x", message: "y" });
  assert.equal(h.calls.publishes.length, 0);
});

test("stop() marks the dashboard offline, waits for the broker's ack, then closes", async () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  const stopped = h.publisher.stop();
  assert.deepEqual(h.calls.publishes, [
    { topic: "sparkdash/dashboard/status", payload: "0", opts: { qos: 1, retain: true } },
  ]);
  // The connection must stay open until the broker has acknowledged the "0".
  await new Promise((r) => setImmediate(r));
  assert.equal(h.calls.ended, 0);
  h.ack();
  await stopped;
  assert.equal(h.calls.ended, 1);
  assert.equal(h.calls.endForce, true);
  assert.deepEqual(h.warnings, []);
});

test("stop() gives up waiting after its timeout and still closes", async () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  await h.publisher.stop({ timeoutMs: 10 });
  assert.equal(h.calls.ended, 1);
  assert.match(h.warnings.join("\n"), /not acknowledged before shutdown/);
});

test("stop() without a live connection just tears down", async () => {
  const h = harness({ client: { connected: false } });
  h.publisher.applySettings();
  await h.publisher.stop();
  assert.equal(h.calls.publishes.length, 0);
  assert.equal(h.calls.ended, 1);
});

test("nothing throws when a connect attempt fails outright", async () => {
  const publisher = new MqttPublisher({
    getSettings: () => ENABLED,
    env: {},
    log: { warn() {}, error() {} },
    connect: () => {
      throw new Error("bad url");
    },
  });
  publisher.applySettings();
  publisher.publishEvent({ type: "x", message: "y" });
  publisher.publishFleet([{ id: "alpha", online: true }]);
  publisher.removeUnit("alpha");
  await publisher.stop();
});

test("each publisher gets its own client id (PID is 1 in every container)", () => {
  const a = harness();
  const b = harness();
  a.publisher.applySettings();
  b.publisher.applySettings();
  assert.notEqual(a.attempts[0].opts.clientId, b.attempts[0].opts.clientId);
});

test("status and metrics are resent only when they change, and retried after a skipped publish", () => {
  const h = harness();
  h.publisher.applySettings();
  const at = (uptime) => [{ snapshot: { id: "alpha", online: true, uptime } }];
  const topics = (from) => h.calls.publishes.slice(from).map((p) => p.topic);
  h.publisher.publishFleet(at(1));
  let mark = h.calls.publishes.length;
  h.publisher.publishFleet(at(2)); // metrics changed, status did not
  assert.deepEqual(topics(mark), ["sparkdash/metrics/alpha"]);
  h.client.connected = false;
  h.publisher.publishFleet([{ id: "alpha", online: false }]); // skipped: broker down
  h.client.connected = true;
  mark = h.calls.publishes.length;
  h.publisher.publishFleet([{ id: "alpha", online: false }]);
  assert.deepEqual(topics(mark), ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]);
});

test("a Spark with id \"dashboard\" does not touch the dashboard's availability topic", () => {
  const h = harness();
  h.publisher.applySettings();
  h.fire("connect");
  const mark = h.calls.publishes.length;
  h.publisher.publishFleet([{ id: "dashboard", online: false }]);
  const topics = h.calls.publishes.slice(mark).map((p) => p.topic);
  assert.deepEqual(topics, ["sparkdash/status/dashboard", "sparkdash/metrics/dashboard"]);
  assert.ok(!topics.includes("sparkdash/dashboard/status"));
});

test("a reconnect resends the retained status and metrics on the next tick", () => {
  const h = harness();
  h.publisher.applySettings();
  const fleet = [{ snapshot: { id: "alpha", online: true, uptime: 1 } }];
  h.publisher.publishFleet(fleet);
  let mark = h.calls.publishes.length;
  h.publisher.publishFleet(fleet); // unchanged: nothing resent
  assert.equal(h.calls.publishes.length, mark);
  h.fire("connect"); // like a restarted broker handing the session back
  mark = h.calls.publishes.length;
  h.publisher.publishFleet(fleet);
  assert.deepEqual(
    h.calls.publishes.slice(mark).map((p) => p.topic),
    ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]
  );
});

test("unacknowledged publishes are capped; skipped state is retried once the broker catches up", () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  for (let i = 0; i < MAX_IN_FLIGHT; i += 1) h.publisher.publishEvent({ id: i });
  assert.equal(h.calls.publishes.length, MAX_IN_FLIGHT);
  // At the cap: events are dropped and retained state is skipped, not queued.
  h.publisher.publishEvent({ id: "over" });
  h.publisher.publishFleet([{ id: "alpha", online: true, uptime: 1 }]);
  assert.equal(h.calls.publishes.length, MAX_IN_FLIGHT);
  assert.match(h.warnings.join("\n"), /awaiting the broker; publish skipped/);
  // Two acks free two slots: the skipped status and metrics go out on the next tick.
  h.ack();
  h.ack();
  h.publisher.publishFleet([{ id: "alpha", online: true, uptime: 1 }]);
  assert.deepEqual(h.topics(MAX_IN_FLIGHT), ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]);
});

test("a publish that fails after hand-off is resent on the next tick", () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  const fleet = [{ id: "alpha", online: true, uptime: 1 }];
  h.publisher.publishFleet(fleet);
  const mark = h.calls.publishes.length;
  h.ack(new Error("Connection closed")); // status
  h.ack(); // metrics
  h.publisher.publishFleet(fleet);
  assert.deepEqual(h.topics(mark), ["sparkdash/status/alpha"]);
});

test("a connection that stops acknowledging is dropped and reopened", () => {
  let clock = 1_000_000;
  const h = harness({ client: { autoAck: false }, now: () => clock });
  h.ready();
  h.publisher.publishEvent({ id: 1 }); // never acknowledged
  clock += ACK_TIMEOUT_MS - 1;
  h.publisher.publishFleet([]);
  assert.equal(h.attempts.length, 1);
  clock += 1;
  h.publisher.publishFleet([]);
  assert.equal(h.calls.ended, 1);
  assert.equal(h.attempts.length, 2);
  assert.match(h.warnings.join("\n"), /has not acknowledged a publish/);
});

test("publishes resent after a reconnect get a fresh ack deadline", () => {
  let clock = 1_000_000;
  const h = harness({ client: { autoAck: false }, now: () => clock });
  h.ready();
  h.publisher.publishEvent({ id: 1 });
  // A long outage, then mqtt.js reconnects and resends the unacknowledged event.
  clock += 5 * ACK_TIMEOUT_MS;
  h.fire("connect");
  h.publisher.publishFleet([]);
  assert.equal(h.attempts.length, 1);
  assert.equal(h.calls.ended, 0);
});

test("removing a unit clears its retained topics and forgets what was sent", () => {
  const h = harness();
  h.ready();
  const fleet = [{ id: "spark one", online: true, uptime: 1 }];
  h.publisher.publishFleet(fleet);
  const mark = h.calls.publishes.length;
  h.publisher.removeUnit("spark one");
  // A zero-length retained message deletes the retained value at the broker.
  assert.deepEqual(h.calls.publishes.slice(mark), [
    { topic: "sparkdash/status/spark%20one", payload: "", opts: { qos: 1, retain: true } },
    { topic: "sparkdash/metrics/spark%20one", payload: "", opts: { qos: 1, retain: true } },
  ]);
  // Cleared once only.
  const after = h.calls.publishes.length;
  h.publisher.publishFleet([]);
  assert.equal(h.calls.publishes.length, after);
  // Re-added under the same id: its state is published again, not skipped as unchanged.
  h.publisher.publishFleet(fleet);
  assert.deepEqual(h.topics(after), ["sparkdash/status/spark%20one", "sparkdash/metrics/spark%20one"]);
});

test("a unit removed while the broker is down is cleared once it is back", () => {
  const h = harness();
  h.ready();
  h.client.connected = false;
  h.publisher.removeUnit("alpha");
  assert.equal(h.calls.publishes.length, 0);
  h.client.connected = true;
  h.fire("connect");
  assert.deepEqual(
    h.calls.publishes.filter((p) => p.payload === "").map((p) => p.topic),
    ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]
  );
});

test("a failed clear is retried, unless the unit has come back", () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  h.publisher.removeUnit("alpha");
  h.ack(new Error("Connection closed"));
  h.ack();
  let mark = h.calls.publishes.length;
  h.publisher.publishFleet([]);
  assert.deepEqual(h.topics(mark), ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]);
  // Fails again, but the unit is re-added before the next tick: publish it, never clear it.
  h.ack(new Error("Connection closed"));
  h.ack();
  mark = h.calls.publishes.length;
  h.publisher.publishFleet([{ id: "alpha", online: true, uptime: 1 }]);
  assert.ok(h.calls.publishes.slice(mark).every((p) => p.payload !== ""));
  assert.deepEqual(h.topics(mark), ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]);
});

test("the availability announcement is retried when the cap blocked it at connect", () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  for (let i = 0; i < MAX_IN_FLIGHT; i += 1) h.publisher.publishEvent({ id: i });
  h.fire("connect"); // reconnect with a full window: the "1" cannot go out yet
  assert.ok(!h.topics(MAX_IN_FLIGHT).includes("sparkdash/dashboard/status"));
  h.ack();
  h.publisher.publishFleet([]);
  assert.deepEqual(h.topics(MAX_IN_FLIGHT), ["sparkdash/dashboard/status"]);
});

/** Topics of the zero-length retained clears sent since `from`. */
const clears = (h, from = 0) => h.calls.publishes.slice(from).filter((p) => p.payload === "").map((p) => p.topic);

test("removals are not recorded while MQTT has no client", () => {
  const h = harness({ settings: { mqttEnabled: false } });
  h.publisher.applySettings();
  for (let i = 0; i < 3 * MAX_PENDING_REMOVALS; i += 1) h.publisher.removeUnit(`gone-${i}`);
  h.set({ mqttEnabled: true });
  h.publisher.applySettings();
  h.fire("connect");
  h.publisher.publishFleet([]);
  assert.deepEqual(clears(h), []);
});

test("disabling MQTT drops pending clears, including ones failed by the teardown", () => {
  const h = harness({ client: { autoAck: false } });
  h.ready();
  h.publisher.removeUnit("alpha"); // handed off, awaiting PUBACK
  h.client.connected = false;
  h.publisher.removeUnit("beta"); // queued while the broker is down
  h.set({ mqttEnabled: false });
  h.publisher.applySettings(); // end(true) fails alpha's outstanding clears
  h.set({ mqttEnabled: true });
  h.client.connected = true;
  const mark = h.calls.publishes.length;
  h.publisher.applySettings();
  h.fire("connect");
  h.publisher.publishFleet([]);
  assert.deepEqual(clears(h, mark), []);
});

test("changing the topic prefix drops clears queued for the old prefix", () => {
  const h = harness();
  h.ready();
  h.client.connected = false;
  h.publisher.removeUnit("alpha");
  h.set({ mqttTopicPrefix: "other" });
  h.client.connected = true;
  h.publisher.applySettings();
  h.fire("connect");
  h.publisher.publishFleet([]);
  assert.deepEqual(clears(h), []);
});

test("at most MAX_PENDING_REMOVALS clears are kept, oldest dropped first", () => {
  const h = harness();
  h.ready();
  h.client.connected = false;
  const extra = 5;
  for (let i = 0; i < MAX_PENDING_REMOVALS + extra; i += 1) h.publisher.removeUnit(`u${i}`);
  h.client.connected = true;
  h.fire("connect");
  const cleared = clears(h);
  assert.equal(cleared.length, 2 * MAX_PENDING_REMOVALS);
  assert.equal(cleared[0], `sparkdash/status/u${extra}`);
  assert.equal(cleared.at(-1), `sparkdash/metrics/u${MAX_PENDING_REMOVALS + extra - 1}`);
});

test("a stall reconnect keeps a clear that was still awaiting the broker", () => {
  let clock = 1_000_000;
  const h = harness({ client: { autoAck: false }, now: () => clock });
  h.ready();
  h.publisher.removeUnit("alpha");
  clock += ACK_TIMEOUT_MS;
  const mark = h.calls.publishes.length;
  // Watchdog: the teardown fails the outstanding clears, then the reconnect resends them.
  h.publisher.publishFleet([]);
  assert.equal(h.attempts.length, 2);
  assert.deepEqual(clears(h, mark), ["sparkdash/status/alpha", "sparkdash/metrics/alpha"]);
});
