"""Unit + integration tests for the async fan-out engine (WALK Steps 6 & 9).

Subprocess, Bedrock (boto3) and Ollama (httpx) calls are all mocked — these
tests never spawn a real CLI or touch the network.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_model_lib.config import MultiModelConfig
from multi_model_lib.engine import (
    _build_command,
    _invoke_cli,
    _parse_output,
    fan_out,
)
from multi_model_lib.models import ModelResult

CLI_ENTRY: dict[str, Any] = {
    "binary": "claude",
    "args_template": ["-p", "{prompt}", "--output-format", "json"],
    "provider": "anthropic-pro",
    "subscription": "Anthropic Pro",
    "parse_mode": "text",
    "api_fallback": None,
}


# ---------------------------------------------------------------------------
# _build_command
# ---------------------------------------------------------------------------
def test_build_command_substitutes_prompt() -> None:
    cmd = _build_command(CLI_ENTRY, "hello world")
    assert cmd == ["-p", "hello world", "--output-format", "json"]


def test_build_command_none_template() -> None:
    entry: dict[str, Any] = {"args_template": None}
    assert _build_command(entry, "x") == []


# ---------------------------------------------------------------------------
# _parse_output
# ---------------------------------------------------------------------------
def test_parse_output_text_mode_strips() -> None:
    assert _parse_output("codex", "  hi there \n", "text") == "hi there"


def test_parse_output_json_result_field() -> None:
    raw = '{"result": "the answer"}'
    assert _parse_output("claude", raw, "json") == "the answer"


def test_parse_output_jsonl_assistant_message() -> None:
    raw = "\n".join(
        [
            '{"type": "system", "subtype": "init"}',
            '{"type": "assistant", "message": {"content": '
            '[{"type": "text", "text": "hello from claude"}]}}',
            '{"type": "result", "result": "hello from claude"}',
        ]
    )
    assert _parse_output("claude", raw, "json") == "hello from claude"


def test_parse_output_json_falls_back_to_raw_on_garbage() -> None:
    raw = "not json at all"
    assert _parse_output("claude", raw, "json") == "not json at all"


# ---------------------------------------------------------------------------
# _invoke_cli — mocked subprocess
# ---------------------------------------------------------------------------
def _fake_proc(stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


@pytest.mark.asyncio
async def test_invoke_cli_success() -> None:
    proc = _fake_proc(b"the response", returncode=0)
    with patch(
        "multi_model_lib.engine.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        result = await _invoke_cli("claude", CLI_ENTRY, "hi", timeout=30)

    assert isinstance(result, ModelResult)
    assert result.status == "success"
    assert result.invocation_method == "cli"
    assert result.parsed_response == "the response"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_invoke_cli_nonzero_exit_is_error() -> None:
    proc = _fake_proc(b"", stderr=b"kaboom", returncode=2)
    with patch(
        "multi_model_lib.engine.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        result = await _invoke_cli("claude", CLI_ENTRY, "hi", timeout=30)

    assert result.status == "error"
    assert result.exit_code == 2
    assert "kaboom" in result.stderr


@pytest.mark.asyncio
async def test_invoke_cli_timeout_raises() -> None:
    proc = _fake_proc(b"")
    proc.communicate = AsyncMock(side_effect=TimeoutError())
    with patch(
        "multi_model_lib.engine.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(asyncio.TimeoutError):
        await _invoke_cli("claude", CLI_ENTRY, "hi", timeout=1)
    proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# fan_out — integration with mocked detect_available + _invoke_model
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fan_out_aggregates_results() -> None:
    config = MultiModelConfig()
    available = {
        "claude": CLI_ENTRY,
        "codex": {**CLI_ENTRY, "provider": "chatgpt-pro", "subscription": "ChatGPT Pro"},
    }

    async def fake_invoke(name: str, entry: dict[str, Any], prompt: str, cfg: Any) -> ModelResult:
        return ModelResult(
            name=name,
            provider=entry["provider"],
            subscription=entry["subscription"],
            status="success",
            invocation_method="cli",
            parsed_response=f"{name} says hi",
        )

    with (
        patch("multi_model_lib.engine.detect_available", return_value=available),
        patch("multi_model_lib.engine._invoke_model", new=AsyncMock(side_effect=fake_invoke)),
    ):
        result = await fan_out("hello", config)

    assert len(result.results) == 2
    names = {r.name for r in result.results}
    assert names == {"claude", "codex"}
    assert set(result.models_invoked) == {"claude", "codex"}
    assert result.session_id.startswith("mmq-")
    assert result.prompt == "hello"


@pytest.mark.asyncio
async def test_fan_out_captures_exceptions_as_error_results() -> None:
    config = MultiModelConfig()
    available = {"claude": CLI_ENTRY}

    with (
        patch("multi_model_lib.engine.detect_available", return_value=available),
        patch(
            "multi_model_lib.engine._invoke_model",
            new=AsyncMock(side_effect=RuntimeError("explode")),
        ),
    ):
        result = await fan_out("hello", config)

    assert len(result.results) == 1
    assert result.results[0].status == "error"
    assert "explode" in result.results[0].stderr


@pytest.mark.asyncio
async def test_fan_out_tracks_fallback_models() -> None:
    config = MultiModelConfig()
    available = {"claude": CLI_ENTRY}

    async def fell_back(name: str, entry: dict[str, Any], prompt: str, cfg: Any) -> ModelResult:
        return ModelResult(
            name=name,
            provider=entry["provider"],
            subscription=entry["subscription"],
            status="success",
            invocation_method="api-fallback",
            parsed_response="via api",
        )

    with (
        patch("multi_model_lib.engine.detect_available", return_value=available),
        patch("multi_model_lib.engine._invoke_model", new=AsyncMock(side_effect=fell_back)),
    ):
        result = await fan_out("hello", config)

    assert result.models_fell_back == ["claude"]


@pytest.mark.asyncio
async def test_fan_out_dedupe_providers() -> None:
    config = MultiModelConfig(dedupe_providers=True)
    # Two anthropic-based providers; dedupe should keep only the first.
    available = {
        "claude": {**CLI_ENTRY, "provider": "anthropic-pro"},
        "goose": {**CLI_ENTRY, "provider": "anthropic-acp"},
    }

    async def fake_invoke(name: str, entry: dict[str, Any], prompt: str, cfg: Any) -> ModelResult:
        return ModelResult(
            name=name,
            provider=entry["provider"],
            subscription="x",
            status="success",
            invocation_method="cli",
        )

    with (
        patch("multi_model_lib.engine.detect_available", return_value=available),
        patch("multi_model_lib.engine._invoke_model", new=AsyncMock(side_effect=fake_invoke)),
    ):
        result = await fan_out("hello", config)

    # provider_base "anthropic" seen once -> only one model invoked
    assert len(result.results) == 1
    assert result.results[0].name == "claude"
