"""Live API smoke check for both experiment archives and inline report viewing."""
import json
from urllib.request import urlopen
from urllib.error import HTTPError

BASE = "http://127.0.0.1:8010"

def read(path):
    with urlopen(BASE + path, timeout=30) as response:
        return json.load(response)

for source in ("active", "legacy"):
    listing = read(f"/ui/experiments?source={source}")
    assert len(listing["experiments"]) == 13
    for experiment in listing["experiments"]:
        detail = read(f"/ui/experiments/{experiment['key']}?source={source}")
        assert detail["title"] == experiment["title"]
        if experiment["status"] == "complete":
            assert detail["summary"] is not None
            if source == "active" and listing["run"] and listing["run"].get("limit") == 40:
                assert detail["summary"]["n_examples"] == 40
    print(f"{source}: 13 experiment details accessible")
report = read("/ui/reports")["reports"][0]
with urlopen(BASE + report["url"], timeout=30) as response:
    assert "text/html" in response.headers["Content-Type"]
    assert "attachment" not in response.headers.get("Content-Disposition", "")
    content = response.read().lower()
    assert b"<title" in content and b"diagnostic report" in content
print("Report HTML opens inline")
try:
    urlopen(BASE + "/ui/experiments/unknown_study", timeout=30)
    raise AssertionError("Unknown experiment was accepted")
except HTTPError as error:
    assert error.code == 404
print("Unknown experiment rejected")
