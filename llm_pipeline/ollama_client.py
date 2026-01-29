import time
from typing import Dict, Any
import ollama


def run_prompt(
    client: ollama.Client,
    model: str,
    prompt: str,
    *,
    stream: bool = False,
    system: str | None = None,
    temperature: float | None = None,
    options: dict | None = None,
) -> Dict[str, Any]:
    ollama_options = options.copy() if options else {}
    if temperature is not None:
        ollama_options["temperature"] = temperature

    start_time = time.perf_counter()

    response = client.generate(
        model=model,
        prompt=prompt,
        system=system,
        stream=stream,
        options=ollama_options or None,
    )

    elapsed = time.perf_counter() - start_time

    if stream:
        chunks = []
        stats = {}
        for chunk in response:
            chunks.append(chunk.get("response", ""))
            stats = chunk
        text = "".join(chunks)
    else:
        text = response.get("response", "")
        stats = response

    return {
        "text": text,
        "stats": stats,
        "wall_time_s": elapsed,
    }

