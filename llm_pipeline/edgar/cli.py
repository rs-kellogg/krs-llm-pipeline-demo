import argparse
import logging
from pathlib import Path

from .extract import extract_mda_from_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def write_toml(
    *,
    mda_file: Path,
    toml_path: Path,
    model: str,
    prompt_id: str,
):
    toml = f"""
[prompts.{prompt_id}]
prompt_file = "{mda_file.as_posix()}"
system = "You are a financial analyst. Analyze the following MD&A section."

[[models]]
name = "{model}"
prompts = ["{prompt_id}"]
""".lstrip()

    toml_path.write_text(toml, encoding="utf-8")
    logger.info("Ollama TOML written to: %s", toml_path)


def process_file(
    input_file: Path,
    output_file: Path,
    *,
    emit_toml: Path | None,
    model: str,
):
    logger.info("Parsing file: %s", input_file)

    mda = extract_mda_from_file(input_file)
    if not mda:
        logger.warning("MD&A not found in %s", input_file)
        return

    output_file.write_text(mda, encoding="utf-8")
    logger.info("MD&A written to: %s", output_file)

    if emit_toml:
        prompt_id = output_file.stem
        write_toml(
            mda_file=output_file,
            toml_path=emit_toml,
            model=model,
            prompt_id=prompt_id,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract MD&A (Item 7) from EDGAR 10-K HTML filings"
    )
    parser.add_argument("input_path")
    parser.add_argument("output_path")

    parser.add_argument(
        "--emit-toml",
        type=Path,
        help="Generate an Ollama-runner TOML config",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama model name for generated TOML",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if input_path.is_file():
        process_file(
            input_path,
            output_path,
            emit_toml=args.emit_toml,
            model=args.model,
        )

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

        for html_file in input_path.glob("*.txt*"):
            out_file = output_path / f"{html_file.stem}_MD-and-A.txt"
            process_file(
                html_file,
                out_file,
                emit_toml=None,  # avoid clobbering
                model=args.model,
            )
    else:
        logger.error("Invalid input path provided.")


if __name__ == "__main__":
    main()

