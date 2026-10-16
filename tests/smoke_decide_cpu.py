"""Exercise Decide classification against a separately started CPU /pooling server."""

import os

import httpx2

url = os.environ.get("GLINER_URL", "http://localhost:8000") + "/pooling"
payload = {
    "model": os.environ.get("GLINER_MODEL", "/models/current"),
    "data": {
        "text": "Please cancel my subscription.",
        "schema": {
            "classifications": [
                {"task": "intent", "labels": ["cancel", "refund", "technical_support", "other"]},
            ],
        },
        "threshold": 0,
    },
}
response = httpx2.post(url, json=payload, timeout=180)
response.raise_for_status()
result = response.json()
assert result["data"]["intent"] == "cancel", result
