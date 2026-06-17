# aiocr

機械OCRとループを持ったAI Agentを組み合わせて、複数の画面撮影画像から1つのMarkdownを生成するCLIです。

- **入力**: スクロールで分割された複数枚の画像（PNG/JPEG/WebP/TIFF/BMP/GIF など）。
- **出力**: 統合・構造化された1つの Markdown（stdout へ出力）。
- **特徴**:
  - 機械OCR（RapidOCR / Tesseract）を第1パスとして利用。
  - マルチモーダルモデル、または「テキスト専用モデル + Visionモデル」のハイブリッド構成に対応。
  - OpenRouter 互換の API エンドポイントでモデルを設定可能。
  - 抽出 → 統合 → 批判・修正 のループにより、1ショット推論より高い精度を目指す。

## セットアップ

```bash
cd ~/aiocr
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

`pip install -e .` でも動作します。

### Tesseractを使いたい場合

```bash
uv pip install -e ".[tesseract]"
```

Tesseract本体も別途インストールが必要です（macOS: `brew install tesseract`）。

## 設定

設定ファイルは `~/.config/aiocr/config.toml` です。APIキーは環境変数名で指定し、実行時に読み出します。

### 最小構成

全項目にデフォルト値があるため、空のファイルでも動きます。

```toml
# ~/.config/aiocr/config.toml（空でもOK）
```

APIキーを環境変数に設定するだけで利用できます。

```bash
export OPENROUTER_API_KEY="sk-..."
```

上書きしたい項目だけ記述すれば十分です。

### 全項目とデフォルト値

```toml
# ~/.config/aiocr/config.toml

[api]
base_url    = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
timeout     = 120

[ocr]
engine    = "rapidocr"
languages = ["eng", "jpn"]

[model]
mode          = "multimodal"
primary_model = "google/gemini-2.0-flash-001"
vision_model  = "openai/gpt-4o-mini"
temperature = 0.2
# top_p      = 0.9
# max_tokens = 8192

[agent]
max_iterations    = 3
# extract_system   = "…"
# integrate_system = "…"
# critique_system  = "…"
```

### 設定値一覧

| セクション | キー | デフォルト | 説明 |
|---|---|---|---|
| `api` | `base_url` | `https://openrouter.ai/api/v1` | OpenRouter互換のAPIエンドポイント |
| | `api_key_env` | `OPENROUTER_API_KEY` | APIキーを格納する環境変数の名前 |
| | `timeout` | `120` | APIリクエストのタイムアウト（秒） |
| `ocr` | `engine` | `rapidocr` | `rapidocr` または `pytesseract` |
| | `languages` | `["eng", "jpn"]` | pytesseract使用時のOCR言語（`+`区切りで渡される） |
| `model` | `mode` | `multimodal` | `multimodal` または `hybrid` |
| | `primary_model` | `google/gemini-2.0-flash-001` | 統合・校正・抽出（multimodal時）に使うモデル |
| | `vision_model` | `openai/gpt-4o-mini` | hybridモードのページ抽出用Visionモデル |
| | `temperature` | `0.2` | 生成のランダム性 |
| | `top_p` | *(未指定)* | nucleus sampling。省略時はAPI側デフォルト |
| | `max_tokens` | *(未指定)* | 最大トークン数。省略時はAPI側デフォルト |
| `agent` | `max_iterations` | `3` | 批判/修正ループの最大反復回数（1〜10） |
| | `extract_system` | *(内蔵)* | ページ抽出のシステムプロンプト |
| | `integrate_system` | *(内蔵)* | 統合のシステムプロンプト |
| | `critique_system` | *(内蔵)* | 批判のシステムプロンプト |

### APIキーの設定

環境変数として設定するか、プロジェクトディレクトリに `.env` を配置します（実行時に自動読み込み・既存の環境変数を上書きしません）。

```bash
export OPENROUTER_API_KEY="sk-..."
# または
echo 'OPENROUTER_API_KEY="sk-..."' > ~/aiocr/.env
```

### マルチモーダルモード（デフォルト）

1つのマルチモーダルモデルが画像全体を見て抽出・統合を行います。

### ハイブリッドモード

Visionモデルで各ページのテキスト・レイアウトを書き起こし、テキスト専用モデルがそれらを統合します。

```toml
[model]
mode = "hybrid"
primary_model = "google/gemini-2.0-flash-001"  # テキスト統合・校正用
vision_model = "openai/gpt-4o"                  # ページ抽出用（より高精度なVisionモデルに変更する例）
```

## 使い方

```bash
# 複数の画像ファイルを指定（stdout へ出力）
aiocr page1.png page2.png page3.png > out.md

# シェルのglobで一括指定
aiocr screenshots/*.png > out.md

# ストリーミング出力を省略せず全表示
aiocr -v screenshots/*.png > out.md
```

結果は標準出力に書き出されるため、リダイレクト（`> out.md`）でファイルに保存します。進行状況・ログはすべて stderr に出力されます。

### CLIオプション

| オプション | 説明 |
|---|---|
| `PATHS...` | 画像ファイル（必須・複数指定可） |
| `-v`, `--verbose` | AIのストリーミング出力を途中で省略せず全行表示 |
| `--completion` | シェル補完スクリプトを表示 |

## 仕組み

```
画像群 → 機械OCR → 抽出Agent → 統合Agent → 批判/修正ループ → Markdown
```

1. **機械OCR**: RapidOCR（デフォルト）またはTesseractで各画像のテキストを抽出。
2. **抽出 Agent**: 画像 + 機械OCRヒントから、各ページの視覚的書き起こしを生成。ハイブリッド時はVisionモデル、マルチモーダル時はマルチモーダルモデルが1ページずつ処理。
3. **統合 Agent**: 全ページのOCR/書き起こしを元に重複除去・文章接続・構造化を行い、1つのMarkdownへ。
4. **批判/修正ループ**: 統合結果を元資料と照合し、残った問題を指摘 → 修正。問題がなければ早期終了、最大 `agent.max_iterations` 回反復。

画像はAPI送信前に自動でリサイズ（最大幅2048px）・圧縮（5MB以下）されます。

## カスタマイズ

プロンプトを `config.toml` で差し替え可能です。

```toml
[agent]
max_iterations = 3
extract_system = "..."     # ページ抽出時のシステムプロンプト
integrate_system = "..."   # 統合時のシステムプロンプト
critique_system = "..."    # 批判時のシステムプロンプト
```

省略時は `src/aiocr/agents.py` のデフォルトプロンプトが使われます。
