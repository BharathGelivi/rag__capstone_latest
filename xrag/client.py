"""Small client for submitting externally collected RAG traces."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import TraceInput


class TraceClient:
    def __init__(self, base_url="http://127.0.0.1:8020", timeout=60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def evaluate(self, trace, claim_mode="sentences"):
        """Evaluate caller-supplied evidence; text is sent to the configured server."""
        payload = TraceInput.model_validate(trace).model_dump(mode="json")
        request = Request(self.base_url + "/evaluate", method="POST",
                          data=json.dumps({"trace": payload, "claim_mode": claim_mode}).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Trace evaluation returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("Could not reach the trace evaluation server") from exc
