"""Telemetry — JSONL logging for model invocations."""

from __future__ import annotations

import json
from pathlib import Path

from multi_model_lib.models import QueryResult


def log_telemetry(result: QueryResult, log_dir: Path) -> None:
    """Append one JSONL entry per model invoked to agent_runs.jsonl."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "agent_runs.jsonl"

    with log_file.open("a") as f:
        for model_result in result.results:
            entry = {
                "timestamp": result.timestamp,
                "session_id": result.session_id,
                "model": model_result.name,
                "provider": model_result.provider,
                "invocation_method": model_result.invocation_method,
                "status": model_result.status,
                "elapsed_seconds": model_result.elapsed_seconds,
                "cost_usd": _estimate_cost(
                    model_result.name,
                    model_result.invocation_method,
                    len(model_result.parsed_response),
                ),
                "prompt_length": len(result.prompt),
                "response_length": len(model_result.parsed_response),
            }
            f.write(json.dumps(entry) + "\n")


def _estimate_cost(name: str, invocation_method: str, response_length: int) -> float:
    """Estimate cost based on provider pricing.

    Returns 0.0 for subscription/local models, estimated cost for Bedrock/API fallback.
    """
    # Subscription CLI models and local models are zero-cost
    if invocation_method == "cli" or invocation_method == "http":
        return 0.0

    # API fallback and Bedrock have costs
    # Rough estimates based on ~500 input tokens + response tokens
    input_tokens_est = 500
    output_tokens_est = max(response_length // 4, 100)  # ~4 chars per token

    cost_per_1m_input: dict[str, float] = {
        "claude": 3.0,      # Anthropic API
        "codex": 2.5,       # OpenAI API
        "gemini": 1.25,     # Google API
        "bedrock": 3.0,     # Default Bedrock (Sonnet)
        "perplexity": 1.0,  # Perplexity API
    }
    cost_per_1m_output: dict[str, float] = {
        "claude": 15.0,
        "codex": 10.0,
        "gemini": 5.0,
        "bedrock": 15.0,
        "perplexity": 1.0,
    }

    input_rate = cost_per_1m_input.get(name, 3.0)
    output_rate = cost_per_1m_output.get(name, 15.0)

    cost = (input_tokens_est * input_rate / 1_000_000) + (
        output_tokens_est * output_rate / 1_000_000
    )
    return round(cost, 6)
