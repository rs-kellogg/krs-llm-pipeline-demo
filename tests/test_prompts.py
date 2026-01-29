from pathlib import Path
import pytest

from ollama_runner.prompts import (
    load_prompts_from_file,
    resolve_prompt,
    PROMPT_SEPARATOR,
)


def test_load_single_prompt(tmp_path: Path):
    p = tmp_path / "prompt.txt"
    p.write_text("Hello world", encoding="utf-8")

    prompts = load_prompts_from_file(p)
    assert prompts == ["Hello world"]


def test_load_multiple_prompts_with_separator(tmp_path: Path):
    p = tmp_path / "prompt.txt"
    p.write_text(
        f"One{PROMPT_SEPARATOR}Two{PROMPT_SEPARATOR}Three",
        encoding="utf-8",
    )

    prompts = load_prompts_from_file(p)
    assert prompts == ["One", "Two", "Three"]


def test_resolve_prompt_success():
    registry = {"p1": {"prompt": "Hi"}}
    assert resolve_prompt("p1", registry)["prompt"] == "Hi"


def test_resolve_prompt_missing():
    with pytest.raises(KeyError):
        resolve_prompt("missing", {})

