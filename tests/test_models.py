"""Tests for multi_model_lib.models."""

from multi_model_lib.models import ModelResult, QueryResult


def test_model_result_defaults() -> None:
    """ModelResult has correct default values."""
    r = ModelResult(
        name="claude", provider="anthropic-pro", subscription="Anthropic Pro", status="success"
    )
    assert r.invocation_method == "cli"
    assert r.raw_output == ""
    assert r.parsed_response == ""
    assert r.elapsed_seconds == 0.0
    assert r.stderr == ""
    assert r.exit_code == -1


def test_model_result_to_dict() -> None:
    """ModelResult serializes to dict correctly."""
    r = ModelResult(
        name="codex", provider="chatgpt-pro", subscription="ChatGPT Pro", status="success"
    )
    d = r.to_dict()
    assert d["name"] == "codex"
    assert d["provider"] == "chatgpt-pro"
    assert isinstance(d, dict)


def test_model_result_to_json() -> None:
    """ModelResult serializes to JSON string."""
    r = ModelResult(
        name="ollama", provider="ollama-local", subscription="Free / Local", status="success"
    )
    j = r.to_json()
    assert '"name": "ollama"' in j
    assert '"provider": "ollama-local"' in j


def test_query_result_defaults() -> None:
    """QueryResult has correct default values."""
    q = QueryResult(
        prompt="test", timestamp="2026-05-03T12:00:00Z", session_id="mmq-20260503-120000"
    )
    assert q.results == []
    assert q.total_elapsed_seconds == 0.0
    assert q.models_available == []
    assert q.models_invoked == []
    assert q.models_unavailable == []
    assert q.models_fell_back == []


def test_query_result_with_results() -> None:
    """QueryResult tracks model results and fallbacks."""
    r1 = ModelResult(
        name="claude", provider="anthropic-pro", subscription="Anthropic Pro", status="success"
    )
    r2 = ModelResult(
        name="codex", provider="chatgpt-pro", subscription="ChatGPT Pro",
        status="fallback", invocation_method="api-fallback",
    )
    q = QueryResult(
        prompt="test",
        timestamp="2026-05-03T12:00:00Z",
        session_id="mmq-20260503-120000",
        results=[r1, r2],
        models_invoked=["claude", "codex"],
        models_fell_back=["codex"],
    )
    assert len(q.results) == 2
    assert q.models_fell_back == ["codex"]


def test_query_result_to_json() -> None:
    """QueryResult serializes to JSON including nested ModelResults."""
    r = ModelResult(
        name="bedrock", provider="aws-bedrock", subscription="AWS IAM", status="success"
    )
    q = QueryResult(
        prompt="hello",
        timestamp="2026-05-03T12:00:00Z",
        session_id="mmq-20260503-120000",
        results=[r],
    )
    j = q.to_json()
    assert '"prompt": "hello"' in j
    assert '"name": "bedrock"' in j
