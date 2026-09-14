"""
ask.py
Ask ScoreChat a question from the command line.

The tool trace is printed alongside the answer because the trace is the
citation: it shows which stored analysis each claim rests on.
"""

from __future__ import annotations
import json

# These CLIs read DATABASE_URL and the provider keys straight from the
# environment, so the .env a developer already has must be loaded before any
# db.store import builds an engine from it.
from dotenv import load_dotenv

load_dotenv()

import click

from pipeline.tools import answer


@click.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--model", default=None, help="Override the configured chat model.")
@click.option("--trace/--no-trace", default=True, show_default=True,
              help="Show the tool calls the answer was built from.")
def main(question: tuple[str, ...], model: str | None, trace: bool):
    result = answer(" ".join(question), model=model)

    if trace:
        for step in result["trace"]:
            args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            click.echo(click.style(f"  → {step['tool']}({args})", fg="cyan"))
            summary = json.dumps(step["result"], ensure_ascii=False)
            click.echo(f"    {summary[:220]}{'…' if len(summary) > 220 else ''}")
        if result["trace"]:
            click.echo()

    if result["error"]:
        click.echo(click.style(result["error"], fg="yellow"))
    if result["answer"]:
        click.echo(result["answer"])


if __name__ == "__main__":
    main()
