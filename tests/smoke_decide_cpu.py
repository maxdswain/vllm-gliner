"""Exercise Decide on CPU through both /pooling and vLLM's /v1/systemone."""

import os

import httpx2

base_url = os.environ.get("GLINER_URL", "http://localhost:8000")
model = os.environ.get("GLINER_MODEL", "/models/current")
payload = {
    "model": model,
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
response = httpx2.post(base_url + "/pooling", json=payload, timeout=180)
response.raise_for_status()
result = response.json()
assert result["data"]["intent"] == "cancel", result

response = httpx2.post(
    base_url + "/v1/systemone",
    json={
        "model": model,
        "state": "Please cancel my subscription.",
        "questions": {
            "intent": {
                "type": "choice",
                "criteria": {
                    "cancel": None,
                    "refund": None,
                    "technical_support": None,
                    "other": None,
                },
            },
            "cancellation": {
                "type": "noul",
                "instructions": "Does the customer want to cancel?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["low", "medium", "high"],
            },
        },
    },
    timeout=180,
)
response.raise_for_status()
result = response.json()
answers = result["answers"]
assert answers["intent"]["choice"] == "cancel", result
assert answers["intent"]["probabilities"]["cancel"] > 0.5, result
assert set(answers) == {"intent", "cancellation", "urgency"}, result
assert 0 <= answers["cancellation"]["noul"] <= 1, result
assert 0 <= answers["urgency"]["score"] <= 2, result
assert result["diagnostics"]["intent"]["label_mass"] == 1.0, result
assert result["usage"]["input_tokens"] > 0, result
