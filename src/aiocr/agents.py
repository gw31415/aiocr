from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

from .config import Config
from .models import OpenAICompatClient, encode_image

if TYPE_CHECKING:
    from .ui import UI


# ── system prompts ─────────────────────────────────────────────────────

DEFAULT_EXTRACT_SYSTEM = textwrap.dedent(
    """\
    あなたは、スクリーンショットのような画面画像を読み取り、テキストを整形するアシスタントです。
    画像のほかに機械OCRの結果が参考として与えられます。以下の点に注意してください。
    - 文字認識の誤り、改行の誤り、画面の端で途切れた文章を画像から修正してください。
    - 画像内のレイアウト（見出し、箇条書き、表、コードブロックなど）をできるだけマークダウンで再現してください。
    - 隣接ページと重複する可能性がある内容はあえて残して構いません（後工程で統合します）。
    - 出力は、修正後のページ内容のみをマークダウンで書いてください。説明文は不要です。
    """
)

VISION_EXTRACT_SYSTEM = textwrap.dedent(
    """\
    あなたは画像からテキスト・レイアウトを詳しく書き起こすアシスタントです。
    マークダウン形式で、画像に表示されている情報をできるだけ忠実に出力してください。
    機械OCRの結果を参考にしながら、誤認識を修正してください。
    出力はマークダウンのみとしてください。
    """
)

INTEGRATE_SYSTEM = textwrap.dedent(
    """\
    あなたは、複数の画面画像を1つの文書に統合するエディタです。
    機械OCR結果と各ページの書き起こしを元に、以下を行ってください。

    1. スクロールの重複部分を検出し、重複を除去して1つの連続した文章にする。
    2. ページ境界で途切れた単語・文章を正しくつなぐ。
    3. 見出し、箇条書き、表、コードブロックなどの構造を保持する。
    4. ノイズや明らかなOCRミスを修正する。
    5. 出力は統合後のマークダウン文書のみとする。余計な説明は不要。

    最終的なマークダウンを code fence なしで出力してください。
    """
)

CRITIQUE_SYSTEM = textwrap.dedent(
    """\
    あなたは、統合されたマークダウン文書を厳しく校正するレビュアーです。
    機械OCR結果と各ページの書き起こしを照合し、以下をチェックしてください。

    - ページ間の重複が残っていないか
    - スクロールで途切れた文章が正しくつながっているか
    - 欠落している単語・段落がないか
    - 誤認識された文字や数字がないか
    - マークダウンの構造（見出し・箇条書き・表）が崩れていないか

    問題があれば、簡潔に箇条書きで列挙し、修正案を示してください。
    問題がなければ、`<no_issues/>` とだけ出力してください。
    """
)

REVISE_SYSTEM = textwrap.dedent(
    """\
    あなたは、レビュー指摘を反映して文書を修正するエディタです。
    ドラフト、レビュー指摘、元のOCR・書き起こしを参考に、最終的なマークダウン文書を出力してください。
    出力は修正後のマークダウンのみとしてください。
    """
)


# ── helpers ────────────────────────────────────────────────────────────

def clean_markdown(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = re.sub(r"</?final_markdown[^>]*>", "", text).strip()
    return text


def has_no_issues(critique_text: str) -> bool:
    return bool(re.search(r"<\s*no[_\s]issues\s*/?>", critique_text, re.I))


def _stream_and_collect(
    client: OpenAICompatClient,
    model: str,
    messages: list[dict],
    config: Config,
    ui: "UI",
    icon: str,
    title: str,
    subtitle: str = "",
) -> str:
    """Stream a chat completion, display tokens live, return cleaned result."""
    with ui.stream(icon, title, model, subtitle) as display:
        response = ""
        for token in client.chat_stream(
            model=model,
            messages=messages,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            max_tokens=config.model.max_tokens,
        ):
            response += token
            display.append(token)
    return clean_markdown(response)


# ── pipeline stages ────────────────────────────────────────────────────

def extract_pages(
    client: OpenAICompatClient,
    config: Config,
    image_paths: list,
    ocr_pages: list[str],
    ui: "UI",
) -> list[str]:
    """Run the extraction agent on each page."""
    summaries: list[str] = []
    is_hybrid = config.model.mode == "hybrid"
    total = len(image_paths)
    system_prompt = (
        config.agent.extract_system
        or (VISION_EXTRACT_SYSTEM if is_hybrid else DEFAULT_EXTRACT_SYSTEM)
    )

    for idx, (path, ocr_text) in enumerate(zip(image_paths, ocr_pages), start=1):
        model = (
            config.model.vision_model if is_hybrid else config.model.primary_model
        )

        text_prefix = (
            f"第 {idx} ページの画像です。\n"
            f"[機械OCRの結果]\n{ocr_text}\n\n"
            "上記を参考に、画像の内容をマークダウンで書き起こしてください。"
        )
        if is_hybrid:
            text_prefix += "\n特に文字とレイアウトに注目してください。"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prefix},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(path)},
                    },
                ],
            },
        ]

        result = _stream_and_collect(
            client, model, messages, config, ui,
            icon="📄", title="Extract",
            subtitle=f"Page {idx}/{total}",
        )
        summaries.append(result)

    return summaries


def integrate(
    client: OpenAICompatClient,
    config: Config,
    image_paths: list,
    ocr_pages: list[str],
    page_summaries: list[str],
    ui: "UI",
) -> str:
    """Combine per-page data into a single coherent Markdown document."""
    system_prompt = config.agent.integrate_system or INTEGRATE_SYSTEM

    parts: list[str] = [
        "以下は、スクロール撮影された複数画面の機械OCR結果と、各ページの書き起こしです。",
        "これらを統合して、1つの連続したマークダウンドキュメントにしてください。",
    ]
    for idx, (ocr, summary) in enumerate(zip(ocr_pages, page_summaries), start=1):
        parts.append(f"\n--- ページ {idx} ---\n")
        parts.append(f"[機械OCR]\n{ocr}\n")
        parts.append(f"[視覚的書き起こし]\n{summary}\n")

    if config.model.mode == "multimodal":
        content: list[dict] | str = [{"type": "text", "text": "\n".join(parts)}]
        for path in image_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": encode_image(path)}}
            )
    else:
        content = "\n".join(parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    return _stream_and_collect(
        client, config.model.primary_model, messages, config, ui,
        icon="🔗", title="Integrate",
        subtitle=f"{len(page_summaries)} pages",
    )


def critique(
    client: OpenAICompatClient,
    config: Config,
    draft: str,
    ocr_pages: list[str],
    page_summaries: list[str],
    ui: "UI",
) -> str:
    """Review the draft and return a list of issues (or a no-issue marker)."""
    system_prompt = config.agent.critique_system or CRITIQUE_SYSTEM

    source_parts: list[str] = ["[統合 draft]\n" + draft]
    for idx, (ocr, summary) in enumerate(zip(ocr_pages, page_summaries), start=1):
        source_parts.append(f"\n--- ページ {idx} ---\n")
        source_parts.append(f"[機械OCR]\n{ocr}\n")
        source_parts.append(f"[書き起こし]\n{summary}\n")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(source_parts)},
    ]

    return _stream_and_collect(
        client, config.model.primary_model, messages, config, ui,
        icon="🔎", title="Critique",
    )


def revise(
    client: OpenAICompatClient,
    config: Config,
    draft: str,
    critique_text: str,
    ocr_pages: list[str],
    page_summaries: list[str],
    ui: "UI",
) -> str:
    """Apply critique to produce a revised Markdown document."""
    source_parts: list[str] = []
    for idx, (ocr, summary) in enumerate(zip(ocr_pages, page_summaries), start=1):
        source_parts.append(f"--- ページ {idx} ---\n")
        source_parts.append(f"[機械OCR]\n{ocr}\n")
        source_parts.append(f"[書き起こし]\n{summary}\n")

    user_content = (
        "[ドラフト]\n" + draft
        + "\n\n[レビュー指摘]\n" + critique_text
        + "\n\n[元資料]\n" + "\n".join(source_parts)
    )

    messages = [
        {"role": "system", "content": REVISE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    return _stream_and_collect(
        client, config.model.primary_model, messages, config, ui,
        icon="✏️", title="Revise",
    )


def run_agent_loop(
    client: OpenAICompatClient,
    config: Config,
    image_paths: list,
    ocr_pages: list[str],
    page_summaries: list[str],
    ui: "UI",
) -> str:
    """Integrate → critique/revise loop → final Markdown."""
    ui.section("🔗", "Integrate")
    draft = integrate(client, config, image_paths, ocr_pages, page_summaries, ui)

    if config.agent.max_iterations > 0:
        ui.section("🔄", f"Review loop · max {config.agent.max_iterations}")

    for iteration in range(config.agent.max_iterations):
        ui.info(f"Iteration {iteration + 1}/{config.agent.max_iterations}")
        critique_text = critique(
            client, config, draft, ocr_pages, page_summaries, ui
        )
        if has_no_issues(critique_text):
            ui.info("→ no issues, review complete ✓")
            break
        draft = revise(
            client, config, draft, critique_text, ocr_pages, page_summaries, ui
        )

    return draft
