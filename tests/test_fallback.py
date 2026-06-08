"""Unit tests for API key fallback (WALK Step 7).

All HTTP calls are mocked — no live network access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from multi_model_lib.fallback import _check_api_key, try_api_fallback

CLAUDE_ENTRY: dict[str, Any] = {
    "provider": "anthropic-pro",
    "subscription": "Anthropic Pro",
    "api_fallback": {
        "provider": "anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
    },
}

NO_FALLBACK_ENTRY: dict[str, Any] = {
    "provider": "atlassian-rovo",
    "subscription": "Atlassian Rovo",
    "api_fallback": None,
}


def test_check_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_KEY", "sk-abc")
    assert _check_api_key("SOME_KEY") == "sk-abc"


def test_check_api_key_blank_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_KEY", "   ")
    assert _check_api_key("SOME_KEY") is None


def test_check_api_key_missing_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOME_KEY", raising=False)
    assert _check_api_key("SOME_KEY") is None


@pytest.mark.asyncio
async def test_try_api_fallback_no_config_returns_none() -> None:
    result = await try_api_fallback("rovo-dev", NO_FALLBACK_ENTRY, "hi", 30)
    assert result is None


@pytest.mark.asyncio
async def test_try_api_fallback_no_key_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = await try_api_fallback("claude", CLAUDE_ENTRY, "hi", 30)
    assert result is None


@pytest.mark.asyncio
async def test_try_api_fallback_anthropic_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch(
        "multi_model_lib.fallback._invoke_anthropic_api",
        new=AsyncMock(return_value="fallback answer"),
    ):
        result = await try_api_fallback("claude", CLAUDE_ENTRY, "hi", 30)

    assert result is not None
    assert result.status == "success"
    assert result.invocation_method == "api-fallback"
    assert result.parsed_response == "fallback answer"
    assert result.name == "claude"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_try_api_fallback_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch(
        "multi_model_lib.fallback._invoke_anthropic_api",
        new=AsyncMock(side_effect=httpx.TimeoutException("boom")),
    ):
        result = await try_api_fallback("claude", CLAUDE_ENTRY, "hi", 30)

    assert result is not None
    assert result.status == "timeout"
    assert result.invocation_method == "api-fallback"


@pytest.mark.asyncio
async def test_try_api_fallback_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch(
        "multi_model_lib.fallback._invoke_anthropic_api",
        new=AsyncMock(side_effect=ValueError("bad request")),
    ):
        result = await try_api_fallback("claude", CLAUDE_ENTRY, "hi", 30)

    assert result is not None
    assert result.status == "error"
    assert "bad request" in result.stderr


@pytest.mark.asyncio
async def test_try_api_fallback_unknown_provider_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEIRD_KEY", "k")
    entry: dict[str, Any] = {
        "provider": "weird",
        "subscription": "Weird",
        "api_fallback": {
            "provider": "not-a-real-provider",
            "env_var": "WEIRD_KEY",
            "default_model": "x",
        },
    }
    result = await try_api_fallback("weird", entry, "hi", 30)
    assert result is None
