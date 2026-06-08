"""Tests for multi_model_lib.registry."""

from unittest.mock import patch

from multi_model_lib.config import MultiModelConfig
from multi_model_lib.registry import (
    MODEL_REGISTRY,
    detect_available,
    get_registry_entry,
    list_all_models,
)

EXPECTED_MODELS = ["claude", "codex", "goose", "rovo-dev", "gemini", "bedrock", "ollama"]
REQUIRED_KEYS = {"binary", "provider", "subscription", "parse_mode"}


def test_registry_has_all_models() -> None:
    """MODEL_REGISTRY contains all 7 expected entries."""
    for name in EXPECTED_MODELS:
        assert name in MODEL_REGISTRY, f"Missing registry entry: {name}"


def test_registry_entry_required_keys() -> None:
    """Each registry entry has required keys."""
    for name, entry in MODEL_REGISTRY.items():
        for key in REQUIRED_KEYS:
            assert key in entry, f"Model '{name}' missing key: {key}"


def test_list_all_models() -> None:
    """list_all_models returns all registered model names."""
    models = list_all_models()
    assert len(models) == 7
    for name in EXPECTED_MODELS:
        assert name in models


def test_get_registry_entry_found() -> None:
    """get_registry_entry returns entry for valid name."""
    entry = get_registry_entry("claude")
    assert entry is not None
    assert entry["binary"] == "claude"
    assert entry["provider"] == "anthropic-pro"


def test_get_registry_entry_not_found() -> None:
    """get_registry_entry returns None for unknown name."""
    assert get_registry_entry("nonexistent") is None


def test_detect_available_with_mocked_binaries() -> None:
    """detect_available finds models whose binaries exist."""
    config = MultiModelConfig()

    def mock_which(binary: str) -> str | None:
        available_binaries = {"claude", "codex", "goose"}
        return f"/usr/local/bin/{binary}" if binary in available_binaries else None

    with (
        patch(
            "multi_model_lib.registry._check_binary",
            side_effect=lambda b: mock_which(b) is not None,
        ),
        patch("multi_model_lib.registry._check_bedrock_access", return_value=False),
        patch("multi_model_lib.registry._check_ollama_running", return_value=False),
    ):
        available = detect_available(config)

    assert "claude" in available
    assert "codex" in available
    assert "goose" in available
    assert "bedrock" not in available
    assert "ollama" not in available
    assert "gemini" not in available


def test_detect_available_with_model_filter() -> None:
    """detect_available respects config.models filter."""
    config = MultiModelConfig(models=["claude"])

    with (
        patch("multi_model_lib.registry._check_binary", return_value=True),
        patch("multi_model_lib.registry._check_bedrock_access", return_value=True),
        patch("multi_model_lib.registry._check_ollama_running", return_value=True),
    ):
        available = detect_available(config)

    assert "claude" in available
    assert "codex" not in available
    assert len(available) == 1


def test_detect_available_bedrock_requires_iam() -> None:
    """Bedrock requires both aws binary and IAM access."""
    config = MultiModelConfig(models=["bedrock"])

    with (
        patch("multi_model_lib.registry._check_binary", return_value=True),
        patch("multi_model_lib.registry._check_bedrock_access", return_value=True),
        patch("multi_model_lib.registry._check_ollama_running", return_value=False),
    ):
        available = detect_available(config)
    assert "bedrock" in available

    with (
        patch("multi_model_lib.registry._check_binary", return_value=True),
        patch("multi_model_lib.registry._check_bedrock_access", return_value=False),
        patch("multi_model_lib.registry._check_ollama_running", return_value=False),
    ):
        available = detect_available(config)
    assert "bedrock" not in available


def test_detect_available_ollama_requires_server() -> None:
    """Ollama requires both binary and running server."""
    config = MultiModelConfig(models=["ollama"])

    with (
        patch("multi_model_lib.registry._check_binary", return_value=True),
        patch("multi_model_lib.registry._check_bedrock_access", return_value=False),
        patch("multi_model_lib.registry._check_ollama_running", return_value=True),
    ):
        available = detect_available(config)
    assert "ollama" in available

    with (
        patch("multi_model_lib.registry._check_binary", return_value=True),
        patch("multi_model_lib.registry._check_bedrock_access", return_value=False),
        patch("multi_model_lib.registry._check_ollama_running", return_value=False),
    ):
        available = detect_available(config)
    assert "ollama" not in available


def test_api_fallback_entries() -> None:
    """Models with API fallback have correct structure."""
    for name in ["claude", "codex", "gemini"]:
        entry = MODEL_REGISTRY[name]
        fb = entry["api_fallback"]
        assert fb is not None, f"{name} should have api_fallback"
        assert "env_var" in fb
        assert "endpoint" in fb
        assert "default_model" in fb

    for name in ["goose", "rovo-dev", "bedrock", "ollama"]:
        entry = MODEL_REGISTRY[name]
        assert entry["api_fallback"] is None, f"{name} should not have api_fallback"
