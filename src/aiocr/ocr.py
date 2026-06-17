from __future__ import annotations

import abc
import pathlib
from typing import Any


class OCREngine(abc.ABC):
    """Abstract mechanical OCR engine."""

    @abc.abstractmethod
    def recognize(self, path: pathlib.Path) -> str:
        ...


class RapidOCREngine(OCREngine):
    """OCR using RapidOCR (ONNX Runtime, no external binary required)."""

    def __init__(self) -> None:
        # Import lazily because the model download can be heavy.
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'rapidocr-onnxruntime is not installed. '
                'Install it with "pip install rapidocr-onnxruntime" '
                'or choose engine="pytesseract".'
            ) from exc
        self._engine: Any = RapidOCR()

    def recognize(self, path: pathlib.Path) -> str:
        result, _ = self._engine(str(path))
        if not result:
            return ""

        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                # Legacy tuple shape: (box, text, score)
                text = item[1] if len(item) > 1 else ""
            if text:
                parts.append(str(text))
        return "\n".join(parts)


class TesseractEngine(OCREngine):
    """OCR using pytesseract (requires the tesseract binary)."""

    def __init__(self, languages: list[str]) -> None:
        try:
            import pytesseract  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'pytesseract is not installed. '
                'Install it with "pip install pytesseract" '
                'or choose the "tesseract" extra.'
            ) from exc
        self._languages = languages

    def recognize(self, path: pathlib.Path) -> str:
        import pytesseract
        from PIL import Image

        lang = "+".join(self._languages) if self._languages else "eng"
        img = Image.open(path)
        return pytesseract.image_to_string(img, lang=lang)


def create_engine(engine: str, languages: list[str]) -> OCREngine:
    if engine == "rapidocr":
        return RapidOCREngine()
    if engine == "pytesseract":
        return TesseractEngine(languages)
    raise ValueError(f"Unknown OCR engine: {engine}")
