import json
from unittest.mock import patch

import pytest

from swarmleak.llm.agents import (
    SYS_ORCH,
    SYS_PEER,
    SYS_WORK,
    OpenAICompatibleClient,
    response_schema,
    validate_response,
)


def test_orchestrator_requires_all_29_workers():
    system = SYS_ORCH.format(n=30, phase='track', event='new contact', workers=list(range(1, 30)))
    schema = response_schema(system)
    value = {'plan_id': 'p', 'tasks': {str(i): 'observe' for i in range(1, 30)}}
    validate_response(value, schema)
    del value['tasks']['29']
    with pytest.raises(ValueError, match='missing'):
        validate_response(value, schema)


@pytest.mark.parametrize('value', [
    {'status': 'ok', 'observations': 'x'},
    {'status': 'ok', 'observations': 'x', 'need_peer': 'false'},
    {'status': 'ok', 'observations': 'x', 'need_peer': 0},
])
def test_worker_rejects_missing_or_nonboolean_peer(value):
    with pytest.raises(ValueError):
        validate_response(value, response_schema(SYS_WORK))


def test_http_request_carries_schema_and_raw_audit(tmp_path):
    audit = tmp_path / 'calls.jsonl'
    client = OpenAICompatibleClient('http://example.test/v1', 'qwen2.5:7b',
                                   audit_path=str(audit), structured_outputs=True)
    valid = {'status': 'ok', 'observations': 'x', 'need_peer': False}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({'choices': [{'message': {'content': json.dumps(valid)},
                                          'finish_reason': 'stop'}]}).encode()
    with patch('urllib.request.urlopen', return_value=Response()) as request:
        text, latency = client(SYS_WORK)
    body = json.loads(request.call_args.args[0].data)
    assert body['response_format']['type'] == 'json_schema'
    assert body['response_format']['json_schema']['schema']['properties']['need_peer']['type'] == 'boolean'
    assert json.loads(text) == valid and latency >= 0
    assert json.loads(audit.read_text())['response_format'] == body['response_format']


def test_server_ignoring_schema_still_fails_and_preserves_raw_audit(tmp_path):
    audit = tmp_path / 'calls.jsonl'
    client = OpenAICompatibleClient('http://example.test/v1', 'qwen2.5:7b',
                                   audit_path=str(audit), structured_outputs=True)
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"summary\\":123}"},"finish_reason":"stop"}]}'
    with patch('urllib.request.urlopen', return_value=Response()):
        with pytest.raises(ValueError):
            client(SYS_PEER)
    assert audit.exists()
