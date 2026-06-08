"""Multi-Model Query CLI — Fan out a prompt to multiple AI models.

This module is the installable entry point referenced by the ``mmq``
console script (``[project.scripts] mmq = "multi_model_lib.cli:main"`` in
pyproject.toml). The standalone ``scripts/multi-model-query.py`` is a thin
shim that delegates here so both invocation paths share one implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from multi_model_lib.config import MultiModelConfig
from multi_model_lib.engine import fan_out
from multi_model_lib.models import QueryResult
from multi_model_lib.telemetry import log_telemetry

console = Console()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fan out a prompt to multiple AI models and compare responses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s "What is async/await?"
  %(prog)s --models claude,ollama "Hello"
  %(prog)s --dry-run "test prompt"
  %(prog)s --ollama-model llama3.1:8b "Hello"
  %(prog)s --no-api-fallback "test"
""",
    )
    parser.add_argument("prompt", nargs="?", help="The prompt to send to all models")
    parser.add_argument("--prompt-file", help="Read prompt from a file")
    parser.add_argument("--models", help="Comma-separated list of models to query")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without executing"
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Output raw JSON to stdout"
    )
    parser.add_argument("--dedupe-providers", action="store_true", help="Skip duplicate providers")
    parser.add_argument(
        "--ollama-model", default="qwen3:4b", help="Ollama model (default: qwen3:4b)"
    )
    parser.add_argument(
        "--bedrock-model",
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="Bedrock model ID",
    )
    parser.add_argument("--no-api-fallback", action="store_true", help="Disable API key fallback")
    parser.add_argument("--timeout", type=int, default=120, help="Per-model timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", help="Override output directory")

    args = parser.parse_args()

    # Resolve prompt
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    if not prompt:
        parser.error("Provide a prompt as an argument or via --prompt-file")

    # Build config
    models_list = args.models.split(",") if args.models else None
    config = MultiModelConfig(
        timeout=args.timeout,
        models=models_list,
        dedupe_providers=args.dedupe_providers,
        verbose=args.verbose,
        dry_run=args.dry_run,
        api_key_fallback=not args.no_api_fallback,
        ollama_model=args.ollama_model,
        bedrock_model=args.bedrock_model,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    # Dry run mode
    if args.dry_run:
        _print_dry_run(config)
        return 0

    # Execute fan-out
    result = asyncio.run(fan_out(prompt, config))

    # Log telemetry
    log_telemetry(result, config.log_dir)

    # Output
    if args.json_output:
        print(result.to_json())
    else:
        _print_rich_table(result)

    # Write output files
    _write_outputs(result, config)

    return 0


def _print_dry_run(config: MultiModelConfig) -> None:
    """Print model availability and commands without executing."""
    import os

    from multi_model_lib.registry import MODEL_REGISTRY, detect_available

    available = detect_available(config)
    all_models = list(MODEL_REGISTRY.keys())

    table = Table(title="Multi-Model Query — Dry Run")
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Method")
    table.add_column("Provider")
    table.add_column("Subscription")

    for name in all_models:
        if config.models and name not in config.models:
            continue
        entry = MODEL_REGISTRY[name]
        is_available = name in available
        status = "✅ Available" if is_available else "❌ Not Available"
        method = entry.get("invocation_method", "cli")
        if not is_available and entry.get("api_fallback") and config.api_key_fallback:
            env_var = entry["api_fallback"]["env_var"]
            if os.environ.get(env_var):
                status = "🔄 Fallback (API key)"
                method = "api-fallback"

        table.add_row(name, status, method, entry["provider"], entry["subscription"])

    console.print(table)
    console.print(
        f"\n[dim]Timeout: {config.timeout}s | Ollama model: {config.ollama_model} | "
        f"Bedrock model: {config.bedrock_model}[/dim]"
    )


def _print_rich_table(result: QueryResult) -> None:
    """Rich-formatted comparison table to console."""
    table = Table(title=f"Multi-Model Query Results — {result.session_id}")
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Method")
    table.add_column("Time (s)", justify="right")
    table.add_column("Response (first 100 chars)")

    for r in result.results:
        status_style = "green" if r.status == "success" else "red"
        response_preview = (
            r.parsed_response[:100] + "..."
            if len(r.parsed_response) > 100
            else r.parsed_response
        )
        table.add_row(
            r.name,
            f"[{status_style}]{r.status}[/{status_style}]",
            r.invocation_method,
            f"{r.elapsed_seconds:.1f}",
            response_preview.replace("\n", " "),
        )

    console.print(table)
    console.print(f"\n[bold]Total time:[/bold] {result.total_elapsed_seconds:.1f}s")
    console.print(f"[bold]Models invoked:[/bold] {len(result.models_invoked)}")
    if result.models_fell_back:
        console.print(f"[yellow]Fell back to API:[/yellow] {', '.join(result.models_fell_back)}")
    if result.models_unavailable:
        console.print(f"[dim]Unavailable:[/dim] {', '.join(result.models_unavailable)}")


def _write_outputs(result: QueryResult, config: MultiModelConfig) -> None:
    """Write JSON and markdown output files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = config.output_dir / f"query-{result.session_id}.json"
    json_path.write_text(result.to_json())

    # Markdown output
    md_path = config.output_dir / f"query-{result.session_id}.md"
    md_lines = [
        f"# Multi-Model Query: {result.session_id}",
        "",
        f"**Prompt:** {result.prompt}",
        f"**Timestamp:** {result.timestamp}",
        f"**Total Time:** {result.total_elapsed_seconds:.1f}s",
        f"**Models Invoked:** {', '.join(result.models_invoked)}",
        "",
        "## Responses",
        "",
    ]
    for r in result.results:
        md_lines.extend([
            f"### {r.name} ({r.provider})",
            f"- **Status:** {r.status}",
            f"- **Method:** {r.invocation_method}",
            f"- **Time:** {r.elapsed_seconds:.1f}s",
            "",
            "```",
            r.parsed_response,
            "```",
            "",
        ])
    md_path.write_text("\n".join(md_lines))

    if not config.verbose:
        console.print(f"\n[dim]Output: {json_path}[/dim]")


if __name__ == "__main__":
    sys.exit(main())
