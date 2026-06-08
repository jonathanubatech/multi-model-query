"""Unit tests for telemetry — JSONL logging and cost estimation (WALK Step 8)."""

from __future__ import annotations

import json
from pathlib import Path

from multi_model_lib.models import ModelResult, QueryResult
from multi_model_lib.telemetry import _estimate_cost, log_telemetry


def _make_query_result(results: list[ModelResult]) -> QueryResult:
    return QueryResult(
        prompt="hello world",
        timestamp="2026-06-05T00:00:00+00:00",
        session_id="mmq-20260605-000000",
        results=results,
    )


def test_log_telemetry_writes_one_line_per_model(tmp_path: Path) -> None:
    results = [
        ModelResult(
            name="claude",
            provider="anthropic-pro",
            subscription="Anthropic Pro",
            status="success",
            invocation_method="cli",
            parsed_response="four chars",
        ),
        ModelResult(
            name="ollama",
            provider="ollama-local",
            subscription="Free / Local",
            status="success",
            invocation_method="http",
            parsed_response="local response",
        ),
    ]
    log_telemetry(_make_query_result(results), tmp_path)

    log_file = tmp_path / "agent_runs.jsonl"
    assert log_file.exists()
    lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["model"] == "claude"
    assert first["session_id"] == "mmq-20260605-000000"
    assert first["status"] == "success"
    assert first["prompt_length"] == len("hello world")
    assert first["response_length"] == len("four chars")
    assert "cost_usd" in first


def test_log_telemetry_appends(tmp_path: Path) -> None:
    r = ModelResult(
        name="claude",
        provider="anthropic-pro",
        subscription="Anthropic Pro",
        status="success",
        invocation_method="cli",
        parsed_response="x",
    )
    log_telemetry(_make_query_result([r]), tmp_path)
    log_telemetry(_make_query_result([r]), tmp_path)

    log_file = tmp_path / "agent_runs.jsonl"
    lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_log_telemetry_creates_missing_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "logs"
    r = ModelResult(
        name="claude",
        provider="anthropic-pro",
        subscription="Anthropic Pro",
        status="success",
        invocation_method="cli",
    )
    log_telemetry(_make_query_result([r]), nested)
    assert (nested / "agent_runs.jsonl").exists()


def test_estimate_cost_zero_for_cli() -> None:
    assert _estimate_cost("claude", "cli", 1000) == 0.0


def test_estimate_cost_zero_for_http_local() -> None:
    assert _estimate_cost("ollama", "http", 1000) == 0.0


def test_estimate_cost_nonzero_for_api_fallback() -> None:
    cost = _estimate_cost("claude", "api-fallback", 4000)
    assert cost > 0.0


def test_estimate_cost_nonzero_for_bedrock_api() -> None:
    cost = _estimate_cost("bedrock", "api", 4000)
    assert cost > 0.0


def test_estimate_cost_uses_default_rates_for_unknown_model() -> None:
    # Unknown model name falls back to default input/output rates (3.0 / 15.0).
    cost = _estimate_cost("totally-unknown", "api", 0)
    # 500 input tokens @ 3.0/1M + 100 output tokens (floor) @ 15.0/1M
    expected = round((500 * 3.0 / 1_000_000) + (100 * 15.0 / 1_000_000), 6)
    assert cost == expected


def test_estimate_cost_output_tokens_floor() -> None:
    # Even a zero-length response should bill the 100-token output floor.
    cost = _estimate_cost("gemini", "api", 0)
    assert cost > 0.0
