"""Exercise the real /pooling endpoint (start a CPU server separately)."""

import os

import httpx2

url = os.environ.get("GLINER_URL", "http://localhost:8000") + "/pooling"
payload = {
    "model": os.environ.get("GLINER_MODEL", "/models/current"),
    "data": {
        "text": "Alice works at Acme in Paris.",
        "schema": {"entities": {"person": "", "organization": "", "location": ""}},
        "runtime": "classic",
        "include_spans": True,
        "include_confidence": True,
    },
}
response = httpx2.post(url, json=payload, timeout=180)
response.raise_for_status()
result = response.json()
assert "data" in result, result
assert isinstance(result["data"], dict), result
assert "entities" in result["data"], result
assert isinstance(result["data"]["entities"], dict), result
usage = result["usage"]
assert usage["prompt_tokens"] > 0, result
assert usage["completion_tokens"] == 0, result
assert usage["total_tokens"] == usage["prompt_tokens"], result
entities = result["data"]["entities"]
for label, text in (("person", "Alice"), ("organization", "Acme"), ("location", "Paris")):
    assert any(item["text"] == text for item in entities.get(label, [])), (label, result)
    for item in entities[label]:
        assert isinstance(item["start"], int)
        assert isinstance(item["end"], int)
        assert isinstance(item["confidence"], (int, float))
