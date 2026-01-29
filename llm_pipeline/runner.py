import sys
from pathlib import Path

from .prompts import resolve_prompt, load_prompts_from_file
from .output import save_result


def run_model_prompts(
    *,
    backend,
    model_cfg: dict,
    prompt_registry: dict,
    stream: bool,
    output_dir: Path,
    filter_prompt: str | None = None,
):
    model_name = model_cfg["name"]

    # Ensure model exists / is downloaded
    try:
        backend.ensure_model(model_name)
    except Exception as e:
        print(
            f"❌ Failed to prepare model '{model_name}': {e}",
            file=sys.stderr,
        )
        return

    prompt_ids = model_cfg.get("prompts", [])

    print(f"\n=== Model: {model_name} ===")

    for prompt_id in prompt_ids:
        if filter_prompt and filter_prompt not in prompt_id:
            continue

        try:
            pdef = resolve_prompt(prompt_id, prompt_registry)
        except KeyError as e:
            print(e, file=sys.stderr)
            continue

        # Merge model + prompt configuration
        system = pdef.get("system", model_cfg.get("system"))
        temperature = pdef.get("temperature", model_cfg.get("temperature"))
        options = {
            **model_cfg.get("options", {}),
            **pdef.get("options", {}),
        }

        # Load prompt text(s)
        if "prompt_file" in pdef:
            prompt_file = Path(pdef["prompt_file"])
            if not prompt_file.exists():
                print(
                    f"Prompt file not found: {prompt_file}",
                    file=sys.stderr,
                )
                continue
            prompts = load_prompts_from_file(prompt_file)
        else:
            prompts = [pdef["prompt"]]

        for idx, prompt_text in enumerate(prompts, start=1):
            run_id = (
                f"{prompt_id}-{idx}"
                if len(prompts) > 1
                else prompt_id
            )

            print(f"\n--- Prompt: {run_id} ---")

            try:
                result = backend.run_prompt(
                    model=model_name,
                    prompt=prompt_text,
                    system=system,
                    temperature=temperature,
                    options=options,
                    stream=stream,
                )

                print(result["text"])

                save_result(
                    output_dir=output_dir,
                    model=model_name,
                    prompt_id=run_id,
                    prompt=prompt_text,
                    system=system,
                    result=result,
                )

            except Exception as e:
                print(
                    f"Error running prompt '{run_id}': {e}",
                    file=sys.stderr,
                )

