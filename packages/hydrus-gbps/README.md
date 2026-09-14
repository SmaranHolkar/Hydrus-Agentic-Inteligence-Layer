# ⚡ Hydrus-GBPS: Grounded Belief Path Search SDK

> **Sub-15ms Grounding & Hallucination Verification for LLMs, RAG & Agents**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)

**Hydrus-GBPS** is a fast, plug-and-play verification engine that evaluates whether LLM responses are faithfully grounded in retrieved context without requiring slow and expensive LLM judge calls.

---

## 🚀 Quickstart (3 Lines of Code)

```python
from hydrus_gbps import GBPSVerifier

verifier = GBPSVerifier()

result = verifier.verify(
    query="What is the recommended pediatric dosage for Amoxicillin?",
    context="Amoxicillin pediatric dosage is 20 to 40 mg/kg/day in divided doses every 8 hours.",
    response="Amoxicillin pediatric dosage is 20 to 40 mg/kg/day in divided doses."
)

print(result.is_grounded)      # True
print(result.grounding_score)  # 0.96
print(result.flagged_claims)   # []
print(result.latency_ms)       # 2.4 ms
```

---

## 🛡️ LangChain Integration

```python
from hydrus_gbps import GBPSGuardrail

guardrail = GBPSGuardrail(threshold=0.70)
verified_output = guardrail({
    "query": query,
    "context": retrieved_docs,
    "response": llm_response
})
```

---

## 🌐 Launch Standalone REST API Microservice

```bash
uvicorn hydrus_gbps.integrations.fastapi_app:create_app --factory --port 8000
```

### POST `/v1/verify`
```json
{
  "query": "Is Python dynamically typed?",
  "context": "Python is a dynamically typed programming language.",
  "response": "Yes, Python uses dynamic typing."
}
```
