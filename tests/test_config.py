from pathlib import Path
from ollama_runner.config import load_config


def test_load_config(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
        [ollama]
        stream = true
        """,
        encoding="utf-8",
    )

    result = load_config(cfg)
    assert result["ollama"]["stream"] is True

