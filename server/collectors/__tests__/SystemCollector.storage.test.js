import assert from "node:assert/strict";
import test from "node:test";

import { SystemCollector } from "../SystemCollector.js";

const DF_OUTPUT = [
  "Filesystem     Type 1B-blocks      Used Available Use% Mounted on",
  "/dev/nvme0n1p2 ext4 982819848192 813953400832 118866411520 88% /",
  "/dev/nvme0n1p1 vfat 535805952 6680576 529125376 2% /boot/efi",
].join("\n");

function remoteSpark() {
  return {
    id: "mama",
    name: "mama",
    isLocal: false,
    lanIp: "100.124.155.99",
    ssh: { host: "100.124.155.99", user: "art", auth: "key" },
  };
}

test("remote storage parses the df table that arrives with a nonzero exit", async () => {
  const calls = [];
  const exec = async (_spark, cmd, options) => {
    calls.push({ cmd, options });
    return DF_OUTPUT;
  };
  const collector = new SystemCollector(remoteSpark(), { exec });

  const disks = await collector._getRemoteStorage();

  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.allowNonZeroExit, true);
  assert.deepEqual(disks, [
    {
      device: "nvme0n1p2",
      label: "/",
      used: 776246,
      total: 937290,
      available: 113360,
      percentage: 88,
      readSpeed: 0,
      writeSpeed: 0,
      disabled: false,
    },
  ]);
});

test("remote storage still surfaces a real SSH failure", async () => {
  const exec = async () => {
    throw new Error("SSH to 100.124.155.99 failed: Permission denied");
  };
  const collector = new SystemCollector(remoteSpark(), { exec });

  const disks = await collector._getRemoteStorage();

  assert.deepEqual(disks, []);
});
