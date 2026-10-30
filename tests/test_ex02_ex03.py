import pytest

from ex02_tokens_and_similarity import cosine_similarity, count_vector, toy_tokenize
import ex03_cli_assistant as cli


def test_toy_tokenize_splits_words_and_punctuation():
    assert toy_tokenize("Hi, Bob!") == ["hi", ",", "bob", "!"]


def test_cosine_identical_and_orthogonal():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero_not_error():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1], [1, 2])


def test_count_vector():
    assert count_vector(["a", "b", "a"], ["a", "b", "c"]) == [2.0, 1.0, 0.0]


class FakeResponse:
    def __init__(self, text):
        self.output_text = text
        self.usage = type("U", (), {"total_tokens": 42})()


class FakeClient:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("rate limited")
        return FakeResponse("Hello from the fake model")


def scripted_input(lines):
    it = iter(lines)
    return lambda prompt="": next(it)


def test_chat_loop_keeps_history_and_prints_tokens():
    out = []
    history = cli.chat_loop(FakeClient(), "any-model",
                            input_fn=scripted_input(["What is a list?", "quit"]),
                            output_fn=out.append)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert any("tokens this turn: 42" in line for line in out)


def test_chat_loop_recovers_from_api_error_and_drops_failed_question():
    out = []
    history = cli.chat_loop(FakeClient(fail_first=True), "m",
                            input_fn=scripted_input(["first", "second", "quit"]),
                            output_fn=out.append)
    assert any("[error] RuntimeError" in line for line in out)
    assert [m["content"] for m in history if m["role"] == "user"] == ["second"]


def test_history_is_trimmed():
    long_history = [{"role": "user", "content": str(i)} for i in range(100)]
    assert len(cli.trim_history(long_history)) == cli.MAX_HISTORY_MESSAGES


def test_model_name_is_passed_through_not_hardcoded():
    client = FakeClient()
    cli.ask_model(client, "my-configured-model", [{"role": "user", "content": "hi"}])
    assert client.calls[0]["model"] == "my-configured-model"
