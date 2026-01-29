from time import perf_counter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import torch

from .base import LLMBackend


class HuggingFaceBackend(LLMBackend):
    def __init__(
        self,
        device: str | None = None,
        dtype: str | None = None,
        **_ignored,  # <-- accept unused args like `host`
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._pipelines = {}

    def ensure_model(self, model: str) -> None:
        # Trigger download & cache
        AutoTokenizer.from_pretrained(model)
        AutoModelForCausalLM.from_pretrained(model)

    def _get_pipeline(self, model: str):
        if model not in self._pipelines:
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=getattr(torch, self.dtype) if self.dtype else None,
            ).to(self.device)

            self._pipelines[model] = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

        return self._pipelines[model]

    def run_prompt(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        temperature: float | None,
        options: dict | None,
        stream: bool,
    ):
        pipe = self._get_pipeline(model)

        full_prompt = (
            f"{system}\n\n{prompt}" if system else prompt
        )

        gen_kwargs = options.copy() if options else {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        start = perf_counter()

        out = pipe(
            full_prompt,
            max_new_tokens=gen_kwargs.pop("max_new_tokens", 512),
            **gen_kwargs,
        )

        text = out[0]["generated_text"][len(full_prompt):]

        return {
            "text": text.strip(),
            "stats": {
                "backend": "huggingface",
                "model": model,
                "device": self.device,
            },
            "wall_time_s": perf_counter() - start,
        }

