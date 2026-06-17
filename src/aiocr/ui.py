"""Rich-based terminal UI for aiocr progress display (stderr only)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.theme import Theme

_MAX_VISIBLE_LINES = 25

_theme = Theme(
    {
        "aiocr.ok": "green",
        "aiocr.err": "red",
        "aiocr.dim": "dim",
        "aiocr.accent": "cyan",
    }
)


class StreamDisplay:
    """Live streaming display for AI token output.

    Used as a context manager:  with ui.stream(...) as disp:  disp.append(token)
    While active, shows a spinner + scrolling panel of the last N lines.
    On clean exit, prints a one-line completion summary.
    """

    def __init__(
        self,
        console: Console,
        icon: str,
        title: str,
        model: str = "",
        subtitle: str = "",
        verbose: bool = False,
    ):
        self._console = console
        self._icon = icon
        self._title = title
        self._model = model
        self._subtitle = subtitle
        self._verbose = verbose
        self._text = Text()
        self._live: Live | None = None

    # ── internal renderers ─────────────────────────────────────────────

    def _spinner_text(self) -> Text:
        parts: list[tuple[str, str]] = [
            (f"  {self._icon} ", "bold cyan"),
            (self._title, "bold"),
        ]
        if self._model:
            parts.append((f"  · {self._model}", "dim italic"))
        if self._subtitle:
            parts.append((f"  · {self._subtitle}", "dim"))
        return Text.assemble(*parts)

    def _render(self):
        spinner = Spinner("dots", text=self._spinner_text(), style="cyan")
        if not self._text.plain:
            return spinner

        lines = self._text.plain.split("\n")
        max_lines = len(lines) if self._verbose else _MAX_VISIBLE_LINES

        content = Text()
        if len(lines) > max_lines:
            visible = lines[-max_lines:]
            content.append(
                f"  … {len(lines) - max_lines} lines above\n\n", style="dim"
            )
        else:
            visible = lines

        for i, line in enumerate(visible):
            if i > 0:
                content.append("\n")
            content.append(line)

        panel = Panel(
            content,
            border_style="dim cyan",
            padding=(0, 1),
        )
        return Group(spinner, Text(""), panel)

    # ── context-manager protocol ───────────────────────────────────────

    def __enter__(self) -> "StreamDisplay":
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=15,
            transient=True,
        )
        self._live.__enter__()
        return self

    def append(self, token: str) -> None:
        self._text.append(token)
        if self._live:
            self._live.update(self._render())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

        if exc_type is None:
            n = len(self._text.plain)
            parts: list[tuple[str, str]] = [
                ("  ✓ ", "green"),
                (self._title, "bold"),
            ]
            if self._model:
                parts.append((f"  · {self._model}", "dim italic"))
            if self._subtitle:
                parts.append((f"  · {self._subtitle}", "dim"))
            parts.append((f"  · {n:,} chars", "dim"))
            self._console.print(Text.assemble(*parts))
        return False


class UI:
    """Top-level Rich-based stderr UI."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.console = Console(stderr=True, theme=_theme, force_terminal=True)

    def banner(self, n_images: int, model: str, mode: str) -> None:
        content = Text.assemble(
            ("📸 ", ""),
            (f"{n_images} image(s)", "bold"),
            ("     model  ", "dim"),
            (model, "cyan"),
            ("     mode  ", "dim"),
            (mode, "italic"),
        )
        self.console.print()
        self.console.print(
            Panel(
                content,
                title="[bold cyan]aiocr[/bold cyan]",
                border_style="cyan",
                padding=(0, 2),
            )
        )

    @contextmanager
    def spinner(self, label: str) -> Generator:
        with self.console.status(
            f"  {label}",
            spinner="dots",
            spinner_style="cyan",
        ):
            yield

    def ok(self, message: str) -> None:
        self.console.print(f"  [green]✓[/green] {message}")

    def section(self, icon: str, title: str) -> None:
        self.console.print(f"\n  [bold cyan]{icon} {title}[/bold cyan]")

    def info(self, message: str) -> None:
        self.console.print(f"  [dim]{message}[/dim]")

    def blank(self) -> None:
        self.console.print()

    def stream(
        self,
        icon: str,
        title: str,
        model: str = "",
        subtitle: str = "",
    ) -> StreamDisplay:
        return StreamDisplay(
            console=self.console,
            icon=icon,
            title=title,
            model=model,
            subtitle=subtitle,
            verbose=self.verbose,
        )

    def done(self, char_count: int) -> None:
        self.console.print()
        self.console.print(
            Text.assemble(
                ("  ✨ ", ""),
                ("Done", "bold green"),
                (f"  · {char_count:,} chars → stdout", "dim"),
            )
        )
        self.console.print()

    def error(self, message: str) -> None:
        self.console.print(
            f"\n  [red]✗[/red] [bold red]{message}[/bold red]\n"
        )
