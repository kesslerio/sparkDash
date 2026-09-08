"""ASGI request metadata only: never persist prompts, outputs, headers or keys."""
import hashlib
import json
import sys
import time
import uuid

BODY_LIMIT = 8 * 1024 * 1024
EVENT_LIMIT = 1024 * 1024
TIMINGS = ('time_to_first_token_ms', 'generation_time_ms', 'queue_time_ms',
           'mean_itl_ms', 'tokens_per_second')


def number(value):
    return value if type(value) in (int, float) and 0 <= value < 1e15 else None


def emit(marker, record):
    # Server logging configuration can suppress unrelated named loggers.
    # Emit only this observer's allowlisted metadata to the rotated container log.
    print(marker, json.dumps(record, separators=(',', ':')), file=sys.stderr, flush=True)


class RequestTelemetry:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/v1/chat/completions':
            return await self.app(scope, receive, send)
        started = time.monotonic()
        peer = str((scope.get('client') or ('unknown',))[0])
        record = {'event': 'llm_request', 'id': uuid.uuid4().hex,
                  'started_at': time.time(),
                  'client': hashlib.sha256(peer.encode()).hexdigest()[:12]}
        body, event_buffer = bytearray(), bytearray()
        body_overflow = False
        first_progress = None
        finishes = set()

        def event(data):
            nonlocal first_progress
            try:
                value = json.loads(data)
            except (ValueError, UnicodeError):
                return
            if not isinstance(value, dict):
                return
            choices = value.get('choices')
            for choice in choices if isinstance(choices, list) else []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get('delta') or choice.get('message') or {}
                if isinstance(delta, dict) and any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                    if first_progress is None:
                        first_progress = time.monotonic() - started
                    if delta.get('tool_calls'):
                        record['tool_call'] = True
                finish = choice.get('finish_reason')
                if finish in ('stop', 'length', 'tool_calls', 'content_filter'):
                    finishes.add(finish)
            usage = value.get('usage')
            if isinstance(usage, dict):
                for key in ('prompt_tokens', 'completion_tokens'):
                    record[key] = number(usage.get(key))
                details = usage.get('prompt_tokens_details') or {}
                if isinstance(details, dict):
                    record['cached_tokens'] = number(details.get('cached_tokens'))
            metrics = value.get('metrics')
            if isinstance(metrics, dict):
                record.update({key: number(metrics.get(key)) for key in TIMINGS})
            if 'error' in value:
                record['stream_error'] = True

        async def observed_receive():
            nonlocal body_overflow
            message = await receive()
            if message['type'] == 'http.disconnect':
                record['disconnected'] = True
            if message['type'] == 'http.request' and not body_overflow:
                part = message.get('body', b'')
                if len(body) + len(part) > BODY_LIMIT:
                    body.clear()
                    body_overflow = True
                    record['request_metadata_omitted'] = True
                else:
                    body.extend(part)
                if not message.get('more_body') and body:
                    try:
                        value = json.loads(body)
                        if isinstance(value, dict):
                            model = value.get('model')
                            if isinstance(model, str) and 0 < len(model) <= 256:
                                record['model'] = model
                            record['max_tokens'] = number(value.get('max_tokens'))
                            kwargs = value.get('chat_template_kwargs') or {}
                            if isinstance(kwargs, dict):
                                if type(kwargs.get('enable_thinking')) is bool:
                                    record['thinking'] = kwargs['enable_thinking']
                                effort = kwargs.get('reasoning_effort')
                                if effort in ('none', 'low', 'medium', 'high', 'xhigh'):
                                    record['reasoning_effort'] = effort
                    except (ValueError, UnicodeError):
                        pass
                    body.clear()
            return message

        async def observed_send(message):
            if message['type'] == 'http.response.start':
                record['status'] = message['status']
            elif message['type'] == 'http.response.body':
                part = message.get('body', b'')
                if len(event_buffer) + len(part) <= EVENT_LIMIT:
                    event_buffer.extend(part)
                    while b'\n\n' in event_buffer:
                        block, rest = event_buffer.split(b'\n\n', 1)
                        event_buffer[:] = rest
                        for line in block.splitlines():
                            if line.startswith(b'data: '):
                                event(line[6:])
                    if not message.get('more_body') and event_buffer:
                        event(bytes(event_buffer))
                        event_buffer.clear()
                else:
                    event_buffer.clear()
                    record['response_metadata_omitted'] = True
            await send(message)

        emit('LLM_REQUEST_START', record)
        try:
            await self.app(scope, observed_receive, observed_send)
        except BaseException:
            record['interrupted'] = True
            raise
        finally:
            record['elapsed_ms'] = round((time.monotonic() - started) * 1000, 2)
            record['first_progress_ms'] = round(first_progress * 1000, 2) if first_progress is not None else None
            record['finish_reasons'] = sorted(finishes)
            metadata = scope.get('state', {}).get('request_metadata')
            usage = getattr(metadata, 'final_usage_info', None)
            if usage is not None:
                for key in ('prompt_tokens', 'completion_tokens'):
                    record[key] = number(getattr(usage, key, None))
            itl = record.get('mean_itl_ms')
            record['decode_tokens_per_second'] = 1000 / itl if itl else None
            count = record.get('completion_tokens')
            elapsed = record['elapsed_ms']
            record['end_to_end_tokens_per_second'] = count * 1000 / elapsed if count is not None and elapsed else None
            emit('LLM_REQUEST', record)
