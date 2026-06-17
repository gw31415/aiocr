from __future__ import annotations

import base64
import io
import json
import mimetypes
import pathlib
from typing import Any, Generator

import httpx
from PIL import Image


def _lanczos() -> int:
    """Return the Lanczos resampling filter constant for the installed Pillow."""
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    return Image.LANCZOS  # type: ignore[attr-defined]


def encode_image(
    path: pathlib.Path,
    max_width: int = 2048,
    quality: int = 85,
    max_bytes: int = 5 * 1024 * 1024,
) -> str:
    """Encode an image to a JPEG data URL, resizing/compressing if necessary."""
    img = Image.open(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), _lanczos())

    buf = io.BytesIO()
    current_quality = quality
    while True:
        buf.seek(0)
        buf.truncate()
        img.save(buf, format="JPEG", quality=current_quality, optimize=True)
        if buf.tell() <= max_bytes or current_quality <= 40:
            break
        current_quality -= 10

    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def guess_mime(path: pathlib.Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/jpeg"


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat completions client with streaming."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ── non-streaming (kept for fallback / tests) ──────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response shape: {data}") from exc

    # ── streaming ──────────────────────────────────────────────────────

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Yield content deltas from a streaming chat completion."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
                follow_redirects=True,
            ) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"API error {response.status_code}: {body}"
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):  # SSE keepalive comment
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {}).get("content")
                            if delta:
                                yield delta
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except httpx.HTTPError as exc:
            raise RuntimeError(f"API streaming failed: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ama/aiocr",
            "X-Title": "aiocr",
        }
