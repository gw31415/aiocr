from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional, Sequence

import typer

from . import agents
from .config import load_config
from .models import OpenAICompatClient
from .ocr import create_engine
from .ui import UI

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["--help"]},
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"}


def _load_dotenv() -> None:
    """Load .env from CWD if present (does not override existing env vars)."""
    env_path = pathlib.Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _collect_images(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Accept image files only. Glob expansion is left to the shell."""
    images: list[pathlib.Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            images.append(p)
    return sorted(images, key=lambda x: x.name)


def _show_completion_callback(value: bool) -> None:
    if not value:
        return
    sys.stdout.write(
        '# Bash\n'
        '_aiocr_completion() {\n'
        '    local IFS=$\'\\n\'\n'
        '    local response\n'
        '    response=$(env COMP_WORDS="${COMP_WORDS[@]}" '
        'COMP_CWORD=$COMP_CWORD _AIOCR_COMPLETE=bash_complete aiocr)\n'
        '    for completion in $response; do\n'
        '        IFS="," read type value <<< "$completion"\n'
        '        if [[ $type == "dir" ]]; then '
        'COMREPLY+=("$value/"); else COMREPLY+=("$value"); fi\n'
        '    done\n'
        '    return 0\n'
        '}\n'
        'complete -F _aiocr_completion -o default aiocr\n\n'
        '# Zsh\n'
        'compdef _aiocr_completion aiocr\n'
    )
    raise typer.Exit(0)


@app.command(name="aiocr")
def main(
    paths: list[pathlib.Path] = typer.Argument(
        ..., help="Image files."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full streaming output without truncation."
    ),
    completion: bool = typer.Option(
        False,
        "--completion",
        help="Show shell completion script.",
        callback=_show_completion_callback,
        is_eager=True,
        is_flag=True,
    ),
) -> None:
    """Convert screen-capture images into a single Markdown document."""
    _load_dotenv()

    ui = UI(verbose=verbose)

    try:
        config, api_key = load_config()
    except (RuntimeError, FileNotFoundError) as exc:
        ui.error(str(exc))
        raise typer.Exit(1) from exc

    if not api_key:
        ui.error(
            f"Environment variable {config.api.api_key_env!r} is not set."
        )
        raise typer.Exit(1)

    images = _collect_images(paths)
    if not images:
        ui.error("No image files found. Specify image files directly (e.g. *.png).")
        raise typer.Exit(1)

    # ── banner ─────────────────────────────────────────────────────────
    ui.banner(
        len(images),
        config.model.primary_model,
        config.model.mode,
    )

    # ── mechanical OCR ─────────────────────────────────────────────────
    ui.section("🔍", "Mechanical OCR")
    ocr_engine = create_engine(config.ocr.engine, config.ocr.languages)
    ocr_pages: list[str] = []
    for idx, img in enumerate(images, 1):
        with ui.spinner(f"[{idx}/{len(images)}] {img.name}"):
            ocr_text = ocr_engine.recognize(img)
        ocr_pages.append(ocr_text)
        ui.ok(f"[dim][{idx}/{len(images)}][/dim] {img.name} [dim]· {len(ocr_text)} chars[/dim]")

    # ── agent pipeline ─────────────────────────────────────────────────
    client = OpenAICompatClient(
        base_url=config.api.base_url,
        api_key=api_key,
        timeout=config.api.timeout,
    )

    try:
        ui.section("📄", "Extract")
        page_summaries = agents.extract_pages(
            client, config, images, ocr_pages, ui
        )

        final_markdown = agents.run_agent_loop(
            client=client,
            config=config,
            image_paths=images,
            ocr_pages=ocr_pages,
            page_summaries=page_summaries,
            ui=ui,
        )
    except RuntimeError as exc:
        ui.error(str(exc))
        raise typer.Exit(1) from exc

    ui.done(len(final_markdown))

    sys.stdout.write(final_markdown)
    if not final_markdown.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def run(argv: Optional[Sequence[str]] = None) -> None:
    app(argv)


if __name__ == "__main__":
    run()
