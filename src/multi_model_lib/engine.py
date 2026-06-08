"""Async fan-out engine — invoke multiple AI models in parallel."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from multi_model_lib.config import MultiModelConfig
from multi_model_lib.models import ModelResult, QueryResult
from multi_model_lib.registry import MODEL_REGISTRY, detect_available


async def fan_out(prompt: str, config: MultiModelConfig) -> QueryResult:
    """Fan out a prompt to all available models in parallel.

    Returns a QueryResult with responses from all invoked models.
    """
    start_time = time.perf_counter()
    timestamp = datetime.now(UTC).isoformat()
    session_id = f"mmq-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    available = detect_available(config)
    all_models = list(MODEL_REGISTRY.keys())
    unavailable = [m for m in all_models if m not in available]

    if config.models:
        unavailable = [m for m in config.models if m not in available]

    # Deduplicate providers if requested
    if config.dedupe_providers:
        seen_providers: set[str] = set()
        filtered: dict[str, dict[str, Any]] = {}
        for name, entry in available.items():
            provider_base = entry["provider"].split("-")[0]  # "anthropic-pro" -> "anthropic"
            if provider_base not in seen_providers:
                seen_providers.add(provider_base)
                filtered[name] = entry
        available = filtered

    # Fan out to all available models concurrently
    tasks = []
    model_names = []
    for name, entry in available.items():
        model_names.append(name)
        tasks.append(_invoke_model(name, entry, prompt, config))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    model_results: list[ModelResult] = []
    models_fell_back: list[str] = []

    for name, result in zip(model_names, results, strict=True):
        if isinstance(result, BaseException):
            model_results.append(ModelResult(
                name=name,
                provider=available[name]["provider"],
                subscription=available[name]["subscription"],
                status="error",
                stderr=str(result),
            ))
        else:
            model_results.append(result)
            if result.invocation_method == "api-fallback":
                models_fell_back.append(name)

    total_elapsed = time.perf_counter() - start_time

    return QueryResult(
        prompt=prompt,
        timestamp=timestamp,
        session_id=session_id,
        results=model_results,
        total_elapsed_seconds=round(total_elapsed, 3),
        models_available=list(available.keys()),
        models_invoked=model_names,
        models_unavailable=unavailable,
        models_fell_back=models_fell_back,
    )


async def _invoke_model(
    name: str, entry: dict[str, Any], prompt: str, config: MultiModelConfig
) -> ModelResult:
    """Invoke a single model using the appropriate method."""
    invocation_method = entry.get("invocation_method", "cli")

    if invocation_method == "boto3":
        return await _invoke_bedrock(
            prompt, config.bedrock_model, config.bedrock_region, config.timeout
        )
    elif invocation_method == "http":
        model = config.ollama_model
        return await _invoke_ollama(prompt, model, config.timeout)
    else:
        # CLI-based invocation
        try:
            return await _invoke_cli(name, entry, prompt, config.timeout)
        except (TimeoutError, FileNotFoundError) as e:
            # Try API fallback if enabled
            if config.api_key_fallback and entry.get("api_fallback"):
                from multi_model_lib.fallback import try_api_fallback

                fallback_result = await try_api_fallback(name, entry, prompt, config.timeout)
                if fallback_result is not None:
                    return fallback_result

            # Return error if no fallback
            return ModelResult(
                name=name,
                provider=entry["provider"],
                subscription=entry["subscription"],
                status="timeout" if isinstance(e, asyncio.TimeoutError) else "error",
                stderr=str(e),
            )


async def _invoke_cli(
    name: str, entry: dict[str, Any], prompt: str, timeout: int
) -> ModelResult:
    """Invoke a CLI-based model via subprocess."""
    start = time.perf_counter()
    cmd = _build_command(entry, prompt)

    proc = await asyncio.create_subprocess_exec(
        entry["binary"],
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        raise

    elapsed = time.perf_counter() - start
    raw_output = stdout.decode("utf-8", errors="replace")
    parsed = _parse_output(name, raw_output, entry["parse_mode"])

    return ModelResult(
        name=name,
        provider=entry["provider"],
        subscription=entry["subscription"],
        status="success" if proc.returncode == 0 else "error",
        invocation_method="cli",
        raw_output=raw_output,
        parsed_response=parsed,
        elapsed_seconds=round(elapsed, 3),
        stderr=stderr.decode("utf-8", errors="replace"),
        exit_code=proc.returncode or 0,
    )


async def _invoke_bedrock(
    prompt: str, model_id: str, region: str, timeout: int
) -> ModelResult:
    """Invoke AWS Bedrock via boto3."""

    start = time.perf_counter()

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _bedrock_sync_call, prompt, model_id, region),
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return ModelResult(
            name="bedrock",
            provider="aws-bedrock",
            subscription="AWS IAM (pay-per-use or committed)",
            status="success",
            invocation_method="api",
            raw_output=result,
            parsed_response=result,
            elapsed_seconds=round(elapsed, 3),
            exit_code=0,
        )
    except TimeoutError:
        elapsed = time.perf_counter() - start
        return ModelResult(
            name="bedrock",
            provider="aws-bedrock",
            subscription="AWS IAM (pay-per-use or committed)",
            status="timeout",
            invocation_method="api",
            elapsed_seconds=round(elapsed, 3),
            stderr=f"Timeout after {timeout}s",
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return ModelResult(
            name="bedrock",
            provider="aws-bedrock",
            subscription="AWS IAM (pay-per-use or committed)",
            status="error",
            invocation_method="api",
            elapsed_seconds=round(elapsed, 3),
            stderr=str(e),
        )


def _bedrock_sync_call(prompt: str, model_id: str, region: str) -> str:
    """Synchronous Bedrock invocation (runs in executor)."""
    import boto3  # type: ignore[import-untyped]

    client = boto3.client("bedrock-runtime", region_name=region)

    # Determine body format based on model provider
    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        })
    elif "amazon" in model_id:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": 4096},
        })
    else:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 4096,
        })

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    response_body = json.loads(response["body"].read())

    # Extract text based on model provider
    if "anthropic" in model_id:
        return str(response_body.get("content", [{}])[0].get("text", ""))
    elif "amazon" in model_id:
        results = response_body.get("results", [{}])
        return results[0].get("outputText", "") if results else ""
    else:
        return json.dumps(response_body)


async def _invoke_ollama(prompt: str, model: str, timeout: int) -> ModelResult:
    """Invoke Ollama via HTTP API."""
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.perf_counter() - start

            return ModelResult(
                name="ollama",
                provider="ollama-local",
                subscription="Free / Local",
                status="success",
                invocation_method="http",
                raw_output=json.dumps(data),
                parsed_response=data.get("response", ""),
                elapsed_seconds=round(elapsed, 3),
                exit_code=0,
            )
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - start
        return ModelResult(
            name="ollama",
            provider="ollama-local",
            subscription="Free / Local",
            status="timeout",
            invocation_method="http",
            elapsed_seconds=round(elapsed, 3),
            stderr=f"Timeout after {timeout}s",
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return ModelResult(
            name="ollama",
            provider="ollama-local",
            subscription="Free / Local",
            status="error",
            invocation_method="http",
            elapsed_seconds=round(elapsed, 3),
            stderr=str(e),
        )


def _build_command(entry: dict[str, Any], prompt: str) -> list[str]:
    """Build subprocess command args from registry entry."""
    args_template = entry.get("args_template")
    if args_template is None:
        return []
    return [arg.replace("{prompt}", prompt) for arg in args_template]


def _parse_output(name: str, raw: str, parse_mode: str) -> str:
    """Extract response text from model output."""
    if parse_mode == "json":
        # Claude --output-format json produces JSONL (one JSON object per line)
        # with system init, assistant messages, and result messages.
        # We want the assistant's text content.
        try:
            # Try parsing as JSON array first
            data = json.loads(raw)
            if isinstance(data, list):
                # JSONL parsed as array — find assistant message or result
                for item in reversed(data):
                    if isinstance(item, dict):
                        if item.get("type") == "assistant" and "message" in item:
                            msg = item["message"]
                            if isinstance(msg, dict) and "content" in msg:
                                content = msg["content"]
                                if isinstance(content, list):
                                    texts = [
                                        b.get("text", "")
                                        for b in content
                                        if b.get("type") == "text"
                                    ]
                                    if texts:
                                        return "\n".join(texts)
                        if item.get("type") == "result":
                            return str(item.get("result", raw))
                return raw
            elif isinstance(data, dict):
                if "result" in data:
                    return str(data["result"])
                if "content" in data:
                    content = data["content"]
                    if isinstance(content, list) and content:
                        return str(content[0].get("text", str(content[0])))
                    return str(content)
            return raw
        except json.JSONDecodeError:
            # Try JSONL (newline-delimited JSON)
            lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
            for line in reversed(lines):
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        if item.get("type") == "assistant" and "message" in item:
                            msg = item["message"]
                            if isinstance(msg, dict) and "content" in msg:
                                content = msg["content"]
                                if isinstance(content, list):
                                    texts = [
                                        b.get("text", "")
                                        for b in content
                                        if b.get("type") == "text"
                                    ]
                                    if texts:
                                        return "\n".join(texts)
                        if item.get("type") == "result":
                            return str(item.get("result", ""))
                except json.JSONDecodeError:
                    continue
            return raw.strip()
    else:
        return raw.strip()
