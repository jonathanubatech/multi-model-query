"""API key fallback — direct API calls when subscription CLI is unavailable."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from multi_model_lib.models import ModelResult


async def try_api_fallback(
    name: str, entry: dict[str, Any], prompt: str, timeout: int
) -> ModelResult | None:
    """Attempt API key fallback for a model whose CLI is unavailable.

    Returns a ModelResult with invocation_method="api-fallback" if successful,
    or None if no API key is available.
    """
    fallback_config = entry.get("api_fallback")
    if fallback_config is None:
        return None

    api_key = _check_api_key(fallback_config["env_var"])
    if api_key is None:
        return None

    provider = fallback_config["provider"]
    default_model = fallback_config["default_model"]
    start = time.perf_counter()

    try:
        if provider == "anthropic":
            response_text = await _invoke_anthropic_api(prompt, api_key, default_model, timeout)
        elif provider == "openai":
            response_text = await _invoke_openai_api(prompt, api_key, default_model, timeout)
        elif provider == "google":
            response_text = await _invoke_google_api(prompt, api_key, default_model, timeout)
        elif provider == "perplexity":
            response_text = await _invoke_perplexity_api(prompt, api_key, timeout)
        else:
            return None

        elapsed = time.perf_counter() - start
        return ModelResult(
            name=name,
            provider=entry["provider"],
            subscription=entry["subscription"],
            status="success",
            invocation_method="api-fallback",
            raw_output=response_text,
            parsed_response=response_text,
            elapsed_seconds=round(elapsed, 3),
            exit_code=0,
        )
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - start
        return ModelResult(
            name=name,
            provider=entry["provider"],
            subscription=entry["subscription"],
            status="timeout",
            invocation_method="api-fallback",
            elapsed_seconds=round(elapsed, 3),
            stderr=f"API fallback timeout after {timeout}s",
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return ModelResult(
            name=name,
            provider=entry["provider"],
            subscription=entry["subscription"],
            status="error",
            invocation_method="api-fallback",
            elapsed_seconds=round(elapsed, 3),
            stderr=f"API fallback error: {e}",
        )


def _check_api_key(env_var: str) -> str | None:
    """Check if the API key environment variable is set."""
    key = os.environ.get(env_var, "").strip()
    return key if key else None


async def _invoke_anthropic_api(prompt: str, api_key: str, model: str, timeout: int) -> str:
    """Direct Anthropic Messages API call."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("content", [])
        if content and isinstance(content, list):
            return str(content[0].get("text", ""))
        return json.dumps(data)


async def _invoke_openai_api(prompt: str, api_key: str, model: str, timeout: int) -> str:
    """Direct OpenAI Chat Completions API call."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
        return json.dumps(data)


async def _invoke_google_api(prompt: str, api_key: str, model: str, timeout: int) -> str:
    """Direct Google Generative AI API call."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
            },
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return str(parts[0].get("text", ""))
        return json.dumps(data)


async def _invoke_perplexity_api(prompt: str, api_key: str, timeout: int) -> str:
    """Direct Perplexity API call."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", ""))
        return json.dumps(data)
