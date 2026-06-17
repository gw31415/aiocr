import pathlib

from aiocr.agents import clean_markdown, has_no_issues
from aiocr.config import SAMPLE_CONFIG, Config


def test_clean_markdown_strips_fences():
    raw = "```markdown\n# Heading\n```"
    assert clean_markdown(raw) == "# Heading"


def test_has_no_issues_marker():
    assert has_no_issues("Everything looks good. <no_issues/>")
    assert has_no_issues("<NO_ISSUES />")
    assert not has_no_issues("There is a typo.")


def test_sample_config_can_parse(tmp_path: pathlib.Path):
    path = tmp_path / "config.toml"
    path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    # Load via tomllib manually to verify shape
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    data = tomllib.load(path.open("rb"))
    config = Config.model_validate(data)
    assert config.model.mode in ("multimodal", "hybrid")
