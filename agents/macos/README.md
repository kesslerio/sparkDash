# sparkDash Mac node agent

A small collector that turns an Apple Silicon Mac into a sparkDash node: system
vitals plus an honest picture of which local model runtime, if any, is serving.

- **stdlib only.** Runs on the system `python3`. `psutil` is used when present
  (per-core CPU detail) and never required, so there is nothing to `pip install`.
- **Nothing invented.** GPU busy-ness, GPU/CPU/ANE power and thermal pressure
  only exist behind `powermetrics`, which needs root. Without it the agent
  reports those metrics in `unavailable` with a reason, and the dashboard renders
  "unavailable" instead of a convincing zero.
- **Nothing hardcoded about the engines.** Which runtimes exist, their ports, how
  to recognise them, and how to read the served model out of argv all come from
  [`runtimes.json`](runtimes.json). A Python server that is not in the inventory
  shows up as `other-runtime` with its own command basename rather than being
  folded into an engine it merely resembles.

## Run it

```bash
python3 agents/macos/sparkdash_mac_agent.py --once --pretty   # one snapshot to stdout
python3 agents/macos/sparkdash_mac_agent.py --serve --port 8790   # HTTP: /metrics, /health
python3 agents/macos/sparkdash_mac_agent.py --once --config my-runtimes.json
```

`GET /metrics` returns the snapshot; `GET /health` is a cheap liveness answer.
Samples are shared for two seconds (`SAMPLE_TTL_S`) so a fast poll loop costs one
`iostat` + one `lsof`, not one per request.

## Install as a LaunchAgent (one line)

```bash
sed "s|/Users/PLACEHOLDER|$HOME|g; s|REPOS/sparkDash|$(pwd)|" agents/macos/ai.onyx.sparkdash-mac-agent.plist > ~/Library/LaunchAgents/ai.onyx.sparkdash-mac-agent.plist && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.onyx.sparkdash-mac-agent.plist
```

Run from the sparkDash checkout, after editing the plist's `--host`/`--port` if
the defaults (bind all interfaces, port 8790 — no credentials, so keep it on a
trusted LAN or tailnet) are wrong for you. Then add the unit in sparkDash with
**Unit type → Apple Silicon Mac** and a **Mac node agent** port of 8790.

Stop it with `launchctl bootout gui/$(id -u)/ai.onyx.sparkdash-mac-agent`.

### Unlocking GPU and ANE power

Run the agent as root and it samples `powermetrics` itself — copy the plist to
`/Library/LaunchDaemons/` (a system daemon), or run `sudo python3 … --serve` and
grant `powermetrics` NOPASSWD in `sudoers` and pass `--powermetrics`. Until then
`gpu.utilization`, `gpu.power` and `ane.power` stay declared-unavailable, which
is what the dashboard is built to show.

## Runtime inventory

`runtimes.json` is a list of entries, evaluated top to bottom, first match wins —
so a model-specific variant (`splash-35b`) is listed before its catch-all parent
(`splash`). Each entry may use:

| key           | meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `name`        | reported runtime name                                                    |
| `label`       | display label (defaults to `name`)                                       |
| `ports`       | listener ports the engine is expected to own; no match → not reported    |
| `exe`         | regex against the process command basename and, for interpreters, the script being run |
| `argvAll`     | regexes that must all match the full command line                        |
| `role`        | `server` (default) or `watchdog` (a launcher: reported as `running`)     |
| `health`      | `{path, timeoutMs}` GET on loopback that proves "serving" (default `/v1/models`) |
| `model`       | `{flags, positional}` — how to read the served model out of argv         |

Anything matched by an entry is claimed by it. Every remaining TCP listener owned
by a Python-ish process is reported once as `other-runtime`.

The shipped file is seeded from the captain's `model-start()` shell functions
(`omlx`, `qflash`, `splash` + `-27b`/`-35b`, `ds4`, `ds4-serve`, `mtplx` +
`-27b`/`-35b`/`-flash`, the `:8200` MTPLX mux, and `tensorfold`). Copy it, edit
it, and point `--config` at your own — the agent has no built-in engine names.

## Tests

```bash
python3 -m unittest discover -s agents/macos/tests
```

They reuse the SSH collector's macOS fixtures, so both transports are asserted to
produce the same unified-memory and powermetrics numbers from the same capture.
