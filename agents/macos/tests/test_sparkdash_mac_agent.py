"""Unit tests for the Mac node agent (stdlib unittest, no third-party deps).

Run:  python3 -m unittest discover -s agents/macos/tests
"""

from __future__ import annotations

import importlib.util
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
FIXTURES = ROOT / "server" / "collectors" / "__tests__" / "fixtures" / "mac"

spec = importlib.util.spec_from_file_location("mac_agent", HERE / "sparkdash_mac_agent.py")
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

INVENTORY = json.loads((HERE / "runtimes.json").read_text("utf-8"))


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text("utf-8")


def proc(pid: int, argv: str, ppid: int = 1) -> dict:
    argv_list = argv.split()
    return {
        "pid": pid,
        "ppid": ppid,
        "argv": argv_list,
        "argvText": argv,
        "comm": agent._basename(argv_list[0]),
        "script": agent._script_identity(argv_list),
    }


def owned(pid: int, port: int) -> dict[int, list[dict]]:
    return {pid: [{"port": port, "address": f"*:{port}"}]}


def ok(port: int, model: str | None = None) -> dict:
    return {port: {"ok": True, "url": f"http://127.0.0.1:{port}/v1/models", "model": model}}


class InventoryDetectionTests(unittest.TestCase):
    """Detection is driven by the inventory file, not by names baked into code."""

    def setUp(self):
        self._real_health = agent.health_check

    def tearDown(self):
        agent.health_check = self._real_health

    def detect(self, procs, ports, health: dict | None = None):
        mapping = health or {}
        agent.health_check = lambda port, path, timeout: mapping.get(
            port, {"ok": False, "url": f"http://127.0.0.1:{port}{path}"}
        )
        return agent.detect_runtimes(INVENTORY, procs, ports)

    def test_tensorfold_matches_on_executable_and_port_with_the_argv_model(self):
        procs = {2302: proc(
            2302,
            "/Users/dev/.venv/bin/python /Users/dev/.venv/bin/tensorfold serve "
            "Vontra/Qwen3.8-27B-MLX-4bit --context 131072 --port 8300 --host 0.0.0.0",
        )}
        found = self.detect(procs, owned(2302, 8300), health=ok(8300, "qwen3.8-27b"))
        self.assertEqual(len(found), 1)
        row = found[0]
        self.assertEqual(row["name"], "tensorfold")
        self.assertEqual(row["port"], 8300)
        self.assertEqual(row["state"], "serving")
        # The API's own id is what clients call, so it wins over argv.
        self.assertEqual(row["model"], "qwen3.8-27b")

    def test_variant_entry_wins_over_its_catch_all_parent(self):
        procs = {10: proc(
            10,
            "/opt/homebrew/opt/splash/libexec/python/bin/python3 -u "
            "/opt/homebrew/opt/splash/libexec/server/server.py "
            "/models/incoai/Qwen3.6-35B-A3B-Splash/target "
            "--model incoai/Qwen3.6-35B-A3B-Splash --port 8100",
        )}
        found = self.detect(procs, owned(10, 8100), health=ok(8100))
        self.assertEqual([row["name"] for row in found], ["splash-35b"])
        self.assertEqual(found[0]["model"], "incoai/Qwen3.6-35B-A3B-Splash")

    def test_listener_without_a_matching_entry_is_other_runtime(self):
        # A Python server nobody catalogued keeps its own name; it is never
        # renamed into a catalogued engine.
        procs = {77: proc(77, "/usr/bin/python3 /opt/experiment/infer.py --port 9099")}
        found = self.detect(procs, owned(77, 9099))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["name"], "other-runtime")
        self.assertEqual(found[0]["label"], "infer.py")
        self.assertEqual(found[0]["port"], 9099)
        self.assertEqual(found[0]["detectedBy"], "listener")

    def test_processes_that_are_not_serving_are_skipped(self):
        procs = {
            78: proc(78, "/usr/bin/python3 /usr/local/bin/serve_helper.py"),  # no listener
            79: proc(79, "/usr/bin/python3 -c print(1)"),                     # listener, not serving
        }
        self.assertEqual(self.detect(procs, owned(79, 9100)), [])

    def test_watchdog_entry_reports_running_without_a_health_probe(self):
        procs = {90: proc(90, "/bin/zsh /Users/dev/projects/dwarfstar/ds4-serve.sh")}
        found = self.detect(procs, {})
        self.assertEqual([row["name"] for row in found], ["ds4-serve"])
        self.assertEqual(found[0]["state"], "running")
        self.assertNotIn("health", found[0])

    def test_engine_that_does_not_own_its_port_is_not_reported(self):
        # A server binary that lost its listener (still loading, or wedged) must
        # not be reported as serving on a port it does not own.
        procs = {91: proc(91, "/Users/dev/models/ds4-server --host 0.0.0.0 --port 8000")}
        self.assertEqual(self.detect(procs, owned(91, 8123)), [])

    def test_inventory_is_the_only_name_source(self):
        procs = {92: proc(92, "/opt/venv/bin/python /opt/venv/bin/omlx --port 8000")}
        found = self.detect(procs, owned(92, 8000), health=ok(8000, "Qwen3.8-27B-4bit"))
        self.assertEqual(found[0]["name"], "omlx")
        self.assertEqual(found[0]["model"], "Qwen3.8-27B-4bit")

    def test_inventory_lists_the_engines_the_owner_starts(self):
        names = {entry["name"] for entry in INVENTORY["runtimes"]}
        for expected in ("omlx", "qflash", "splash", "splash-27b", "splash-35b", "ds4",
                         "ds4-serve", "mtplx", "mtplx-27b", "mtplx-35b", "mtplx-flash",
                         "tensorfold"):
            self.assertIn(expected, names)


class ModelExtractionTests(unittest.TestCase):
    def test_flag_then_positional(self):
        flags_only = {"flags": ["--model"], "positional": False}
        self.assertEqual(
            agent.extract_model(["engine", "--model", "a/b", "--port", "1"], flags_only), "a/b"
        )
        self.assertEqual(agent.extract_model(["engine", "--model=a/b"], flags_only), "a/b")
        self.assertIsNone(agent.extract_model(["engine", "--port", "1"], flags_only))
        self.assertEqual(
            agent.extract_model(["engine", "serve", "Vontra/Qwen3.8-27B-MLX-4bit"],
                                {"flags": [], "positional": True}),
            "Vontra/Qwen3.8-27B-MLX-4bit",
        )

    def test_a_long_path_collapses_to_its_last_two_parts(self):
        self.assertEqual(agent._display_model("/models/incoai/Qwen3.6-35B-Splash/target"),
                         "Qwen3.6-35B-Splash/target")


class ParsingTests(unittest.TestCase):
    """Shared arithmetic with the SSH collector: same fixture, same numbers."""

    def test_vm_stat_and_sysctl_match_the_ssh_collector_fixture_expectations(self):
        vm = agent.parse_vm_stat(fixture("vm_stat.txt"))
        memsize = int(agent.parse_sysctl(fixture("sysctl.txt"))["hw.memsize"])
        mem = agent.unified_memory(vm, memsize)
        self.assertEqual(vm["pageSize"], 16384)
        self.assertEqual(mem["total"], 262144)
        self.assertEqual(mem["gpuUsed"], 185502)
        self.assertEqual(mem["cpuUsed"], 9802)
        self.assertEqual(mem["used"], 195304)
        self.assertEqual(mem["available"], 66840)
        self.assertEqual(mem["percentage"], 75)
        self.assertEqual(mem["oomRisk"], "medium")

    def test_an_unwired_idle_model_still_counts_as_used(self):
        vm = agent.parse_vm_stat(fixture("vm_stat-unwired.txt"))
        memsize = int(agent.parse_sysctl(fixture("sysctl.txt"))["hw.memsize"])
        mem = agent.unified_memory(vm, memsize)
        self.assertEqual(mem["gpuUsed"], 4585)
        self.assertEqual(mem["cpuUsed"], 192794)
        self.assertEqual(mem["used"], 197379)

    def test_boot_time_and_swap(self):
        sys = agent.parse_sysctl(fixture("sysctl.txt"))
        self.assertEqual(agent.parse_boot_time(sys["kern.boottime"]), 1790650919)
        self.assertEqual(agent.parse_swap_usage(sys["vm.swapusage"]),
                         {"total": 1024, "used": 78, "free": 946})

    def test_powermetrics_fixture(self):
        parsed = agent.parse_powermetrics(fixture("powermetrics-busy.txt"))
        self.assertEqual(parsed["gpuActivePct"], 100)
        self.assertEqual(parsed["gpuW"], 84.9)
        self.assertEqual(parsed["cpuW"], 5.1)
        self.assertEqual(parsed["combinedW"], 90)
        self.assertEqual(parsed["thermalPressure"], "Nominal")

    def test_netstat_link_row(self):
        self.assertEqual(agent.parse_netstat_bytes(fixture("netstat.txt")),
                         {"name": "en1", "rxBytes": 392966387598, "txBytes": 7952987339})

    def test_df_prefers_the_data_volume(self):
        original = agent.run
        agent.run = lambda cmd, timeout=1: fixture("df.txt")
        try:
            disks = agent.disk_stats()
        finally:
            agent.run = original
        self.assertEqual(len(disks), 1)
        self.assertEqual(disks[0]["label"], "/System/Volumes/Data")
        self.assertEqual(disks[0]["total"], 948534)

    def test_iostat_reads_the_second_sample(self):
        text = (
            "              disk0               cpu    load average\n"
            "KB/t  tps MB/s  us sy id   1m   5m  15m\n"
            "19.71 776 14.9  18 27 55  61.44 64.76 69.57\n"
            " 9.62 555  5.2  45 55  0  61.44 64.76\n"
        )
        original = agent.run
        agent.run = lambda cmd, timeout=1: text
        try:
            sample = agent.cpu_sample_iostat()
        finally:
            agent.run = original
        self.assertEqual(sample["usagePercent"], 100.0)
        self.assertEqual(sample["source"], "iostat")
        self.assertIsNone(sample["perCorePercent"])

    def test_iostat_without_a_us_sy_id_header_is_no_sample(self):
        original = agent.run
        agent.run = lambda cmd, timeout=1: "iostat: unsupported\n"
        try:
            self.assertIsNone(agent.cpu_sample_iostat())
        finally:
            agent.run = original


class AvailabilityTests(unittest.TestCase):
    """Anything macOS only hands to root is declared unavailable, never zeroed."""

    def test_gpu_signals_without_root_are_declared_unavailable(self):
        block, unavailable = agent.gpu_stats(allow_powermetrics=False)
        self.assertEqual(block["powermetrics"], "requires-root")
        self.assertIsNone(block["activePercent"])
        self.assertIsNone(block["anePowerW"])
        self.assertEqual(
            {row["metric"] for row in unavailable},
            {"gpu.utilization", "gpu.power", "ane.power"},
        )
        self.assertTrue(all("root" in row["reason"] for row in unavailable))

    def test_powermetrics_that_returns_nothing_is_still_unavailable(self):
        original = agent.run
        agent.run = lambda cmd, timeout=1: ""
        try:
            block, unavailable = agent.gpu_stats(allow_powermetrics=True)
        finally:
            agent.run = original
        self.assertEqual(block["powermetrics"], "failed")
        self.assertEqual(len(unavailable), 3)

    def test_network_without_a_default_route_is_declared_unavailable(self):
        original = agent.default_interface
        agent.default_interface = lambda: None
        try:
            network, notes = agent.network_stats()
        finally:
            agent.default_interface = original
        self.assertIsNone(network)
        self.assertEqual(notes, ["no default route"])

    def test_snapshot_reports_what_it_could_not_read(self):
        inventory = {"defaults": {}, "runtimes": []}
        outputs = {
            "sysctl": fixture("sysctl.txt"),
            "vm_stat": fixture("vm_stat.txt"),
            "df": fixture("df.txt"),
            "route": "   interface: en0\n",
            "netstat": fixture("netstat.txt"),
            "ipconfig": "10.0.0.5\n",
        }
        original_run = agent.run
        original_sockets = agent.listening_sockets
        original_probe = agent.probe_reachability
        agent.run = lambda cmd, timeout=1: outputs.get(cmd[0], "")
        agent.listening_sockets = lambda: {}
        agent.probe_reachability = lambda *a: [{"target": "1.1.1.1:53", "ok": True, "rttMs": 9}]
        try:
            snap = agent.collect_snapshot(inventory, allow_powermetrics=False,
                                         cpu_probe=None, procs={}, listeners={})
        finally:
            agent.run = original_run
            agent.listening_sockets = original_sockets
            agent.probe_reachability = original_probe

        self.assertEqual(snap["schema"], agent.SCHEMA)
        self.assertEqual(snap["bootTime"], 1790650919)
        self.assertGreater(snap["uptimeSeconds"], 0)
        self.assertEqual(snap["runtimes"], [])
        self.assertEqual(snap["memory"]["total"], 262144)
        self.assertEqual(snap["disks"][0]["label"], "/System/Volumes/Data")
        self.assertEqual(snap["network"]["interfaces"][0]["ip"], "10.0.0.5")
        self.assertEqual(snap["network"]["reachability"][0]["ok"], True)
        metrics = {row["metric"] for row in snap["unavailable"]}
        self.assertEqual(snap["cpu"]["usagePercent"], None)
        self.assertIn("cpu.usage", metrics)
        self.assertIn("cpu.perCore", metrics)
        self.assertIn("gpu.power", metrics)


class ServeTests(unittest.TestCase):
    def test_metrics_and_health_endpoints(self):
        inventory = {"defaults": {}, "runtimes": []}
        cache = agent.SampleCache(inventory, allow_powermetrics=False, ttl=0)
        cache.get = lambda force=False: {"schema": agent.SCHEMA, "ok": True}
        httpd = agent.ThreadingHTTPServer(("127.0.0.1", 0), agent.make_handler(cache))
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
                self.assertEqual(json.loads(resp.read())["ok"], True)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                self.assertEqual(json.loads(resp.read())["schema"], agent.SCHEMA)
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=2)
            self.assertEqual(ctx.exception.code, 404)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_cache_serves_one_sample_inside_its_ttl(self):
        calls = []
        original = agent.collect_snapshot
        agent.collect_snapshot = lambda inventory, **kw: (calls.append(1), {"n": len(calls)})[1]
        try:
            cache = agent.SampleCache({"runtimes": []}, allow_powermetrics=False, ttl=60)
            self.assertEqual(cache.get()["n"], 1)
            self.assertEqual(cache.get()["n"], 1)
            self.assertEqual(cache.get(force=True)["n"], 2)
        finally:
            agent.collect_snapshot = original


if __name__ == "__main__":
    unittest.main()
