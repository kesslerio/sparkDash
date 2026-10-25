import { randomBytes } from "node:crypto";
import mqtt from "mqtt";
import { unitMetricJson } from "../prometheus.js";
import { dashboardStatusTopic, unitMetricsTopic, unitStatusTopic } from "./topics.js";

const LOG_QUIET_MS = 30_000;
const DEFAULT_PREFIX = "sparkdash";
/**
 * Publishes handed to the client but not yet acknowledged (QoS 1) or written
 * to the socket (QoS 0). At the cap new publishes are skipped, so a slow or
 * stalled broker costs a bounded amount of memory instead of a growing queue.
 */
export const MAX_IN_FLIGHT = 100;
/**
 * A publish still unacknowledged this long on a live connection means the
 * broker or the socket is wedged: drop the connection and start over.
 */
export const ACK_TIMEOUT_MS = 30_000;
/** How long stop() waits for the broker to acknowledge the offline status. */
export const STOP_TIMEOUT_MS = 1_500;
/**
 * Removed units whose retained topics are still waiting to be cleared. Past
 * this the oldest are forgotten, so removals during a long outage cost a
 * bounded amount of memory.
 */
export const MAX_PENDING_REMOVALS = 100;

/**
 * Outbound MQTT bridge (opt-in, Settings → MQTT): republishes what sparkDash
 * already holds to a user-configured broker —
 *
 *   {prefix}/events            one JSON EventLog record per accepted event
 *   {prefix}/status/{sparkId}  retained "1"/"0", mirroring sparkdash_up
 *   {prefix}/metrics/{sparkId} retained JSON twin of the unit's Prometheus series
 *   {prefix}/dashboard/status  retained "1" while connected, LWT "0" otherwise
 *
 * Status and metrics are republished only when they change, and all of them
 * again after a reconnect, so a broker restart self-heals. A removed unit's
 * retained topics are cleared at the broker.
 *
 * Publishing never throws and never blocks a poll loop: while the broker is
 * unreachable the client keeps retrying on its own and publishes are skipped
 * until it is back. Best-effort — a long outage or a slow broker drops
 * messages (at most MAX_IN_FLIGHT are outstanding) instead of queueing them.
 */
export class MqttPublisher {
  /**
   * @param {{ getSettings: () => object, env?: Record<string, string | undefined>, connect?: typeof mqtt.connect, log?: Pick<Console, "warn" | "error">, now?: () => number }} deps
   */
  constructor({ getSettings, env = process.env, connect = mqtt.connect, log = console, now = Date.now } = {}) {
    this._getSettings = getSettings;
    this._env = env;
    this._connect = connect;
    this._log = log;
    this._now = now;
    /** @type {import("mqtt").MqttClient | null} */
    this._client = null;
    /** Broker URL, topic base and username the current client was built with, so settings PUTs can no-op. */
    this._url = null;
    this._base = null;
    this._user = null;
    this._lastQuietLog = 0;
    /**
     * Unique per process: PID alone is 1 in every container, and two clients
     * sharing an id kick each other off the broker in a loop.
     */
    this._clientId = `sparkdash-${randomBytes(4).toString("hex")}`;
    /** Last status / metrics payload sent per topic, so unchanged values are not resent. */
    /** @type {Map<string, string>} */
    this._sentState = new Map();
    /** Outstanding publishes: token → time handed to the client (insertion order = oldest first). */
    /** @type {Map<number, number>} */
    this._inFlight = new Map();
    this._seq = 0;
    /**
     * Removed unit ids whose retained topics still need clearing at the broker
     * (insertion order = oldest first). Only kept for the current client's
     * broker and prefix; see removeUnit().
     */
    /** @type {Set<string>} */
    this._removedUnits = new Set();
    /** Bumped whenever pending removals are dropped, so late publish failures from before are not re-queued. */
    this._removalsEpoch = 0;
  }

  /**
   * Apply current settings: connects when enabled, reopens the connection when
   * the broker URL or topic prefix changed, tears down when disabled. Safe to
   * call on every settings PUT — a no-op when nothing relevant changed.
   */
  applySettings() {
    const s = this._getSettings();
    if (!s.mqttEnabled || !s.mqttUrl) return this._teardown();
    const base = this._prefix();
    const user = s.mqttUsername || "";
    if (this._client && this._url === s.mqttUrl && this._base === base && this._user === user) return;
    // A different broker, prefix or user: removals queued for the old one no longer apply.
    if (this._client) this._teardown();
    this._url = s.mqttUrl;
    this._base = base;
    this._user = user;
    let client;
    try {
      client = this._connect(s.mqttUrl, {
        clientId: this._clientId,
        reconnectPeriod: 5_000,
        connectTimeout: 10_000,
        // Never buffer QoS 0 while offline; _publish skips then anyway.
        queueQoSZero: false,
        username: user || undefined,
        // The password never touches settings.json (it is written world-readable);
        // it comes from the SPARKDASH_MQTT_PASSWORD environment variable.
        password: this._env.SPARKDASH_MQTT_PASSWORD || undefined,
        // LWT: the broker flips our availability to 0 if we die without a clean disconnect.
        will: { topic: dashboardStatusTopic(base), payload: "0", qos: 0, retain: true },
      });
    } catch (err) {
      this._url = null;
      this._base = null;
      this._user = null;
      this._dropPendingRemovals();
      return this._quietLog(`connect failed: ${err?.message}`);
    }
    this._client = client;
    client.on("connect", () => {
      if (this._client !== client) return;
      // Fired on every (re)connect. A restarted broker may have lost the
      // retained state: forget what was sent so the next tick resends it all.
      this._sentState.clear();
      // mqtt.js resends unacknowledged QoS 1 publishes now; restart their ack clocks
      // so an outage is not mistaken for a stalled connection.
      const at = this._now();
      for (const token of this._inFlight.keys()) this._inFlight.set(token, at);
      this._announce();
      this._clearRemovedUnits();
    });
    client.on("error", (err) => this._quietLog(`broker error: ${err?.message}`));
    // "close" / "reconnect" are expected — MQTT.js retries with reconnectPeriod.
  }

  /** Stream one EventLog record. Called from the EventLog's subscriber list. */
  publishEvent(event) {
    const s = this._getSettings();
    if (!s.mqttEnabled) return;
    this._publish(`${this._prefix()}/events`, JSON.stringify(event), { qos: 1 });
  }

  /**
   * Publish the fleet: retained availability plus a metrics payload per unit,
   * from the same entries the /metrics route renders (`metricsEntries()` output
   * — bare snapshots also work, they just lack collection provenance). The
   * metrics payload is the unit's `unitMetricJson()` with a `ts` added; an
   * offline unit gets `{up: 0}` so the metrics topic shows the unit went down,
   * matching the "sparkdash_up 0 and nothing else" rule.
   * @param {Array<object | { snapshot: object, collected?: object, collectedAt?: object } | null | undefined>} entries
   */
  publishFleet(entries) {
    const s = this._getSettings();
    if (!s.mqttEnabled || !Array.isArray(entries)) return;
    // Called every broadcast tick, so this doubles as the ack watchdog.
    this._checkStall();
    const base = this._prefix();
    const units = [];
    for (const entry of entries) {
      const snap = entry && typeof entry === "object" && entry.snapshot ? entry.snapshot : entry;
      if (!snap?.id) continue;
      units.push({ entry, snap });
      // Re-added under the same id: publish its state, do not clear it.
      this._removedUnits.delete(String(snap.id));
    }
    // Retried every tick until it lands (skipped at connect when at the in-flight cap).
    this._announce();
    this._clearRemovedUnits();
    for (const { entry, snap } of units) {
      const up = snap.online === true;
      this._publishIfChanged(unitStatusTopic(base, snap.id), up ? "1" : "0", () => (up ? "1" : "0"), { retain: true });
      const json = unitMetricJson(entry);
      // Retained, so a new or restarted subscriber sees the last state
      // immediately; the availability topics say how stale it may be. The
      // change check ignores `ts`, which would otherwise differ every tick.
      this._publishIfChanged(
        unitMetricsTopic(base, snap.id),
        JSON.stringify(json),
        () => JSON.stringify({ ts: this._now(), ...json }),
        { qos: 0, retain: true }
      );
    }
  }

  /**
   * A unit was deleted: clear its retained status and metrics topics so
   * subscribers do not keep seeing a unit that no longer exists. If the broker
   * is unreachable (or refuses), the clear is retried on later ticks.
   *
   * Pending clears are bounded: none are recorded while there is no client
   * (MQTT disabled or unconfigured), they are dropped when MQTT is disabled or
   * the broker, prefix or user changes, and at most MAX_PENDING_REMOVALS are
   * kept (oldest dropped first). A stall reconnect keeps them.
   */
  removeUnit(sparkId) {
    if (sparkId == null || sparkId === "" || !this._client) return;
    const id = String(sparkId);
    const base = this._prefix();
    this._sentState.delete(unitStatusTopic(base, id));
    this._sentState.delete(unitMetricsTopic(base, id));
    this._queueRemoval(id);
    this._clearRemovedUnits();
  }

  /**
   * Graceful shutdown: publish our own availability as 0 and wait (bounded) for
   * the broker to acknowledge it before closing the connection.
   * @param {{ timeoutMs?: number }} [opts]
   * @returns {Promise<void>}
   */
  async stop({ timeoutMs = STOP_TIMEOUT_MS } = {}) {
    const client = this._client;
    if (client?.connected) {
      const topic = dashboardStatusTopic(this._base ?? this._prefix());
      const acked = await new Promise((resolve) => {
        const timer = setTimeout(() => resolve(false), timeoutMs);
        const done = (err) => {
          clearTimeout(timer);
          resolve(!err);
        };
        try {
          client.publish(topic, "0", { qos: 1, retain: true }, done);
        } catch (err) {
          done(err);
        }
      });
      if (!acked) {
        this._log.warn("[mqtt] offline status not acknowledged before shutdown; the broker's last-will message will mark the dashboard offline");
      }
    }
    this._teardown();
  }

  /** Our own availability: retained "1", resent after every reconnect. */
  _announce() {
    if (!this._base) return;
    this._publishIfChanged(dashboardStatusTopic(this._base), "1", () => "1", { qos: 1, retain: true });
  }

  /** Send an empty retained message (which deletes the retained value) to each removed unit's topics. */
  _clearRemovedUnits() {
    if (!this._removedUnits.size || !this._client?.connected) return;
    const base = this._prefix();
    const epoch = this._removalsEpoch;
    const retry = (id) => {
      if (this._removalsEpoch === epoch) this._queueRemoval(id);
    };
    for (const id of [...this._removedUnits]) {
      let handedOff = true;
      for (const topic of [unitStatusTopic(base, id), unitMetricsTopic(base, id)]) {
        const sent = this._publish(topic, "", { qos: 1, retain: true }, () => retry(id));
        if (!sent) handedOff = false;
      }
      if (handedOff) this._removedUnits.delete(id);
    }
  }

  /** Record a pending clear, forgetting the oldest beyond MAX_PENDING_REMOVALS. */
  _queueRemoval(id) {
    this._removedUnits.add(id);
    while (this._removedUnits.size > MAX_PENDING_REMOVALS) {
      const oldest = this._removedUnits.values().next().value;
      this._removedUnits.delete(oldest);
      this._quietLog(`more than ${MAX_PENDING_REMOVALS} removed units awaiting the broker; retained topics for "${oldest}" will not be cleared`);
    }
  }

  _dropPendingRemovals() {
    this._removedUnits.clear();
    this._removalsEpoch += 1;
  }

  /**
   * Publish `render()` only when `key` differs from what was last sent on the
   * topic. The cache is updated only on a real send, and forgotten again if
   * delivery fails, so a skipped or failed publish is retried on the next tick.
   */
  _publishIfChanged(topic, key, render, opts) {
    if (this._sentState.get(topic) === key) return;
    const sent = this._publish(topic, render(), opts, () => {
      if (this._sentState.get(topic) === key) this._sentState.delete(topic);
    });
    if (sent) this._sentState.set(topic, key);
  }

  /**
   * Hand one message to the client, within the in-flight cap.
   * @param {(err: Error) => void} [onLateError] called if delivery fails after this returned true
   * @returns {boolean} whether the message was handed to the client
   */
  _publish(topic, payload, opts, onLateError) {
    const client = this._client;
    // Skipping while disconnected keeps a long broker outage from queueing
    // unbounded messages in the client.
    if (!client?.connected) {
      this._quietLog("broker not connected; publish skipped");
      return false;
    }
    if (this._inFlight.size >= MAX_IN_FLIGHT) {
      this._quietLog(`${this._inFlight.size} publishes awaiting the broker; publish skipped`);
      return false;
    }
    const token = ++this._seq;
    this._inFlight.set(token, this._now());
    let returned = false;
    let failedNow = false;
    try {
      // The callback fires on PUBACK for QoS 1, once written (or drained) for
      // QoS 0, and with an error if the connection closes first.
      client.publish(topic, payload, opts, (err) => {
        this._inFlight.delete(token);
        if (!err) return;
        this._quietLog(`publish to ${topic} failed: ${err?.message}`);
        if (returned) onLateError?.(err);
        else failedNow = true;
      });
    } catch (err) {
      this._inFlight.delete(token);
      this._quietLog(`publish failed: ${err?.message}`);
      failedNow = true;
    }
    returned = true;
    return !failedNow;
  }

  /** Reconnect when the oldest outstanding publish has gone unacknowledged too long. */
  _checkStall() {
    if (!this._client?.connected || this._inFlight.size === 0) return;
    const oldest = this._inFlight.values().next().value;
    if (this._now() - oldest < ACK_TIMEOUT_MS) return;
    this._log.warn(`[mqtt] broker has not acknowledged a publish in ${ACK_TIMEOUT_MS / 1000} s; reconnecting`);
    // Same broker and prefix: the clears still apply after the reconnect.
    this._teardown({ keepPendingRemovals: true });
    this.applySettings();
  }

  /** @param {{ keepPendingRemovals?: boolean }} [opts] */
  _teardown({ keepPendingRemovals = false } = {}) {
    const client = this._client;
    this._client = null;
    this._url = null;
    this._base = null;
    this._user = null;
    this._sentState.clear();
    this._inFlight.clear();
    // Before end(): it fails every outstanding clear, which must not re-queue them.
    if (!keepPendingRemovals) this._dropPendingRemovals();
    if (!client) return;
    try {
      client.end(true); // force: don't wait on in-flight QoS acks during teardown
    } catch {
      /* ignore */
    }
  }

  _prefix() {
    const p = this._getSettings().mqttTopicPrefix;
    return typeof p === "string" && p ? p : DEFAULT_PREFIX;
  }

  /** Rate-limited warn: a down broker must not flood the log every poll tick. */
  _quietLog(message) {
    const now = this._now();
    if (now - this._lastQuietLog < LOG_QUIET_MS) return;
    this._lastQuietLog = now;
    this._log.warn(`[mqtt] ${message}`);
  }
}

export default MqttPublisher;
