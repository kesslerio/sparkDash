import asyncio
import importlib.util
import json
import contextlib
import io
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('telemetry', Path(__file__).resolve().parents[1] / 'request_telemetry.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampling_cache_and_profile_metadata(self):
        with patch.dict(m.os.environ, {'VLLM_QUALIFICATION_ID': 'kv24', 'VLLM_QUALIFICATION_SHA256': 'f'*64,
                                     'VLLM_QUALIFICATION_DEFAULT_SAMPLING': '{"temperature":1,"top_k":20}'}):
            _, row = await self.exercise(b'{"temperature":0}', [b'data: {"usage":{"prompt_tokens":50,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":40}}}\n\n'])
        self.assertEqual(row['effective_temperature'], 0)
        self.assertEqual(row['effective_top_k'], 20)
        self.assertIsNone(row['effective_top_p'])
        self.assertEqual(row['new_prompt_tokens'], 10)
        self.assertEqual(row['profile'], 'kv24')
        self.assertEqual(row['http_inflight_at_start'], 1)
        self.assertEqual(row['http_inflight_at_finish'], 0)

    async def test_missing_cache_is_unknown(self):
        _, row = await self.exercise(b'{}', [b'data: {"usage":{"prompt_tokens":50,"completion_tokens":2}}\n\n'])
        self.assertIsNone(row['new_prompt_tokens'])

    async def test_disabled_top_k_and_invalid_defaults(self):
        with patch.dict(m.os.environ, {'VLLM_QUALIFICATION_DEFAULT_SAMPLING': '[]'}):
            _, row=await self.exercise(b'{"top_k":-1}', [])
        self.assertEqual(row['effective_top_k'],-1)
        self.assertIsNone(row['effective_temperature'])

    async def exercise(self, body, chunks, error=False):
        sent, logs = [], []
        async def app(scope, receive, send):
            msg = await receive()
            self.assertEqual(msg['body'], body)
            await send({'type': 'http.response.start', 'status': 200})
            for chunk in chunks:
                await send({'type': 'http.response.body', 'body': chunk, 'more_body': True})
            if error:
                raise asyncio.CancelledError()
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}
        async def send(message):
            sent.append(message)
        scope = {'type': 'http', 'path': '/v1/chat/completions', 'client': ('192.0.2.1', 1)}
        with patch.object(m, 'emit', side_effect=lambda marker, value: logs.append(dict(value))):
            if error:
                with self.assertRaises(asyncio.CancelledError):
                    await m.RequestTelemetry(app)(scope, receive, send)
            else:
                await m.RequestTelemetry(app)(scope, receive, send)
        return sent, logs[-1]

    async def test_stream_passthrough_and_metadata_without_content(self):
        payload = {'messages': [{'content': 'PRIVATE_PROMPT'}], 'max_tokens': 64, 'chat_template_kwargs': {'enable_thinking': True}}
        chunks = [b'data: {"choices":[{"delta":{"content":"PRIVATE_OUTPUT"}}]}\n', b'\ndata: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":10},"metrics":{"queue_time_ms":12,"mean_itl_ms":25}}\n\ndata: [DONE]\n\n']
        sent, row = await self.exercise(json.dumps(payload).encode(), chunks)
        self.assertEqual([x['body'] for x in sent if x['type']=='http.response.body'], chunks+[b''])
        self.assertEqual(row['queue_time_ms'], 12)
        self.assertEqual(row['decode_tokens_per_second'], 40)
        self.assertIsNotNone(row['end_to_end_tokens_per_second'])
        self.assertEqual(row['prompt_tokens'], 40)
        self.assertTrue(row['thinking'])
        self.assertNotIn('PRIVATE', json.dumps(row))
        self.assertNotIn('192.0.2.1', json.dumps(row))
        self.assertIsNotNone(row['first_progress_ms'])

    async def test_large_requests_are_not_retained_or_modified(self):
        with patch.object(m, 'BODY_LIMIT', 16):
            _, row = await self.exercise(b'x'*32, [b'data: [DONE]\n\n'])
        self.assertTrue(row['request_metadata_omitted'])

    async def test_cancel_is_recorded_and_propagated(self):
        _, row = await self.exercise(b'{}', [], True)
        self.assertTrue(row['interrupted'])
        self.assertIsNone(row['first_progress_ms'])

    async def test_nonstreaming_tool_usage(self):
        chunk = json.dumps({'choices':[{'message':{'tool_calls':[{'function':{'arguments':'PRIVATE'}}]},'finish_reason':'tool_calls'}], 'usage':{'prompt_tokens':20,'completion_tokens':5}}).encode()
        _, row = await self.exercise(b'{}', [chunk])
        self.assertTrue(row['tool_call'])
        self.assertEqual(row['finish_reasons'], ['tool_calls'])
        self.assertEqual(row['completion_tokens'], 5)

    def test_emission_survives_disabled_named_loggers(self):
        output = io.StringIO()
        previous = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stderr(output):
                m.emit('LLM_REQUEST', {'completion_tokens': 7})
        finally:
            logging.disable(previous)
        self.assertEqual(output.getvalue(), 'LLM_REQUEST {"completion_tokens":7}\n')
