import io
import json
from unittest.mock import patch
import pytest
from xrag.client import TraceClient


def test_client_sends_validated_trace():
    response = io.BytesIO(b'{"ok":true}')
    with patch('xrag.client.urlopen', return_value=response) as send:
        result = TraceClient().evaluate({'question': 'Q?', 'answer': 'A.', 'retrieved_chunks': [{'text': 'Evidence'}]})
    body = json.loads(send.call_args.args[0].data)
    assert body['trace']['retrieved_chunks'][0]['chunk_id'] == 'chunk-1'
    assert body['claim_mode'] == 'sentences'
    assert result == {'ok': True}


def test_client_rejects_invalid_trace_before_sending():
    with patch('xrag.client.urlopen') as send:
        with pytest.raises(ValueError):
            TraceClient().evaluate({'question': '', 'answer': 'A', 'retrieved_chunks': []})
        send.assert_not_called()
