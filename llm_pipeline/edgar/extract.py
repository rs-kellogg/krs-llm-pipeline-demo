import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

PARSERS = ["lxml", "html5lib", "html.parser"]


def html_to_text(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    last_exception = None

    for parser in PARSERS:
        try:
            soup = BeautifulSoup(html, parser)

            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator=" ")
            if text.strip():
                return text

        except Exception as e:
            last_exception = e
            logger.warning(
                "Parser '%s' failed for %s: %s",
                parser,
                file_path,
                e,
            )

    raise RuntimeError(
        f"All HTML parsers failed for {file_path}"
    ) from last_exception


def extract_mda_section(text: str) -> str | None:
    clean_text = re.sub(r"\s+", " ", text)
    upper = clean_text.upper()

    item7_match = re.search(r"\bITEM\s+7\b", upper)
    if not item7_match:
        return None

    item7_end_match = re.search(
        r"\bITEM\s+7A\b|\bITEM\s+8\b",
        upper[item7_match.end():],
    )

    if not item7_end_match:
        return None

    start = item7_match.start()
    end = item7_match.end() + item7_end_match.start()

    candidate = clean_text[start:end].strip()

    if len(candidate) < 500:
        logger.warning("Extracted MD&A is unusually short")

    return candidate


def extract_mda_from_file(input_file: Path) -> str | None:
    text = html_to_text(input_file)
    return extract_mda_section(text)

