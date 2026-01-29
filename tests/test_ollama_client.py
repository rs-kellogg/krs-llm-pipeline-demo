from ollama_runner.ollama_client import run_prompt


class DummyClient:
    def generate(self, **kwargs):
        return {
            "response": "hello",
            "prompt_eval_count": 3,
            "eval_count": 2,
            "total_duration": 1_000_000,
        }


def test_run_prompt_non_stream():
    client = DummyClient()

    result = run_prompt(
        client=client,
        model="dummy",
        prompt="test",
        stream=False,
    )

    assert result["text"] == "hello"
    assert result["stats"]["eval_count"] == 2
    assert result["wall_time_s"] >= 0

