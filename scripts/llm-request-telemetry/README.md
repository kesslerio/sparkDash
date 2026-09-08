# LLM request telemetry

Reusable ASGI observation and seven-day SQLite history for OpenAI chat requests.
This helper has no Qwen model, container, host, or recipe path hardcoded. The
management repository owns installation and its systemd unit. Other ASGI-backed
engines can load RequestTelemetry; vLLM-specific timing fields remain null when
an engine does not emit them. This is not an inference proxy.

The observer emits LLM_REQUEST_START/LLM_REQUEST metadata only. The collector
also accepts legacy QWEN_REQUEST records. Model identity comes from the request;
deployment identity comes from trusted operator configuration. Reports group by
deployment, model and hashed client so a model switch cannot mix their timings.
Missing identities in migrated history remain legacy-unknown, not relabeled.

Run `python3 request_history.py --config /etc/model-cluster/active.json`.
The JSON has `telemetry.container` and `telemetry.deployment` strings. Configure
a new deployment label when changing model, runtime or serving profile.
Run `python3 request_history.py --report-hours 24` for retained reports.
The default database is ~/.local/state/llm-telemetry/requests.sqlite3 (0600).
Only one collector should own that database. Retention is seven days.

Never record prompts, answers, headers, API keys or tool arguments. Start records
allow incomplete streams to remain visible. A raw ASGI disconnect after a terminal
finish reason is not counted as an interruption; exceptions still are. Tool-call
presence is not tool success. Application crashes may leave unfinished records.

Native vLLM time_to_first_token_ms excludes queue time. Its tokens_per_second
includes prefill and excludes queue. Decode-only throughput is 1000/mean_itl_ms;
end-to-end throughput includes observed queue, prefill and generation. First
progress includes reasoning and tool output, not necessarily visible answer text.

Tests: `python3 -m unittest discover -s scripts/llm-request-telemetry/tests -q`.
