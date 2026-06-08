"""Tests for multi_model_lib.config."""

from pathlib import Path

from multi_model_lib.config import REPO_ROOT, MultiModelConfig


def test_config_defaults() -> None:
    """MultiModelConfig has correct default values."""
    c = MultiModelConfig()
    assert c.timeout == 120
    assert c.models is None
    assert c.dedupe_providers is False
    assert c.verbose is False
    assert c.dry_run is False
    assert c.api_key_fallback is True
    assert c.ollama_model == "qwen3:4b"
    assert c.bedrock_model == "us.anthropic.claude-sonnet-4-20250514-v1:0"
    assert c.bedrock_region == "us-east-1"


def test_config_output_dir_default() -> None:
    """Output dir defaults to REPO_ROOT/outputs/multi-model."""
    c = MultiModelConfig()
    assert c.output_dir == REPO_ROOT / "outputs" / "multi-model"
    assert isinstance(c.output_dir, Path)


def test_config_log_dir_default() -> None:
    """Log dir defaults to REPO_ROOT/logs."""
    c = MultiModelConfig()
    assert c.log_dir == REPO_ROOT / "logs"
    assert isinstance(c.log_dir, Path)


def test_config_override() -> None:
    """MultiModelConfig accepts overrides via kwargs."""
    c = MultiModelConfig(
        timeout=60,
        models=["claude", "ollama"],
        dedupe_providers=True,
        ollama_model="llama3.1:8b",
        bedrock_region="us-west-2",
    )
    assert c.timeout == 60
    assert c.models == ["claude", "ollama"]
    assert c.dedupe_providers is True
    assert c.ollama_model == "llama3.1:8b"
    assert c.bedrock_region == "us-west-2"


def test_repo_root_is_project_dir() -> None:
    """REPO_ROOT points to the project root directory."""
    assert REPO_ROOT.is_dir()
    assert (REPO_ROOT / "pyproject.toml").exists()
