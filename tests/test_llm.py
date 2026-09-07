import pytest
from pydantic import BaseModel

from agent.llm import StructuredOutputError, generate_structured


class DummySchema(BaseModel):
    value: int


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns each item in `responses` in order, one per .invoke() call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, _messages):
        content = self.responses[self.calls]
        self.calls += 1
        return FakeMessage(content)


def test_succeeds_on_first_valid_response():
    llm = FakeLLM(['{"value": 42}'])
    result = generate_structured(llm, "sys", "user", DummySchema)
    assert result.value == 42
    assert llm.calls == 1


def test_retries_once_on_malformed_json_then_succeeds():
    llm = FakeLLM(["not json at all", '{"value": 7}'])
    result = generate_structured(llm, "sys", "user", DummySchema)
    assert result.value == 7
    assert llm.calls == 2


def test_strips_markdown_code_fences():
    llm = FakeLLM(['```json\n{"value": 1}\n```'])
    result = generate_structured(llm, "sys", "user", DummySchema)
    assert result.value == 1


def test_raises_after_second_failure():
    llm = FakeLLM(["not json", "still not json"])
    with pytest.raises(StructuredOutputError):
        generate_structured(llm, "sys", "user", DummySchema)
    assert llm.calls == 2


def test_content_block_list_format_is_handled():
    llm = FakeLLM([[{"type": "text", "text": '{"value": 9}'}]])
    result = generate_structured(llm, "sys", "user", DummySchema)
    assert result.value == 9
