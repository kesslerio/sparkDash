import test from "node:test";
import assert from "node:assert/strict";
import { SparkRegistry } from "../SparkRegistry.js";

const r = Object.create(SparkRegistry.prototype);
const n = (partial) =>
  r._normalizeConfig({ id: "u1", name: "U1", lanIp: "10.0.0.9", ...partial });

test("kind mac round-trips through normalization", () => {
  assert.equal(n({ kind: "mac" }).kind, "mac");
});

test("existing kinds are unchanged and unknown kinds default to spark", () => {
  assert.equal(n({ kind: "host" }).kind, "host");
  assert.equal(n({ kind: "spark" }).kind, "spark");
  assert.equal(n({}).kind, "spark");
  assert.equal(n({ kind: "windows" }).kind, "spark");
  assert.equal(n({ kind: "MAC" }).kind, "spark");
});
