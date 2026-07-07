import pytest

from conftest import MULTIBYTE_PAIR_IDS, FakeLlmjp4Tokenizer

pytest.importorskip("vllm")

from llm_jp_vllm.llmjp4.reasoning_parser import Llmjp4ReasoningParser  # noqa: E402


@pytest.fixture
def reasoning_parser(fake_tokenizer: FakeLlmjp4Tokenizer) -> Llmjp4ReasoningParser:
    return Llmjp4ReasoningParser(fake_tokenizer)


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content",
            ("Reasoning", "Content"),
        ),
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>Preamble",
            ("Reasoning", "Preamble"),
        ),
    ],
    ids=["final", "preamble-commentary"],
)
def test_extract_reasoning_splits_content_from_analysis(
    reasoning_parser, trace: str, expected: tuple[str, str]
) -> None:
    assert reasoning_parser.extract_reasoning(trace, request=None) == expected


def test_extract_reasoning_keeps_markers_for_tool_calls_and_collects_late_analysis(
    reasoning_parser,
) -> None:
    reasoning, content = reasoning_parser.extract_reasoning(
        "<|channel|>analysis<|message|>Reasoning-1<|end|>"
        + "<|start|>assistant to=functions.get_current_weather"
        + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|call|>'
        + "<|start|>assistant<|channel|>analysis<|message|>Reasoning-2",
        request=None,
    )

    assert reasoning == "Reasoning-1\nReasoning-2"
    # The tool parser re-parses the content, so the markers must survive.
    assert content == (
        "<|start|>assistant to=functions.get_current_weather"
        + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|call|>'
        + "<|start|>assistant<|channel|>analysis<|message|>Reasoning-2"
    )


@pytest.mark.parametrize(
    ("stripped_output", "expected"),
    [
        ("analysis Reasoning", ("Reasoning", None)),
        ("analysis Reasoning assistant final Content", ("Reasoning", "Content")),
        ("Content", (None, "Content")),
    ],
    ids=[
        "truncated-analysis-must-not-leak-cot-into-content",
        "content-excludes-the-marker-words",
        "plain-text-is-the-content",
    ],
)
def test_extract_reasoning_from_stripped_text(
    reasoning_parser, stripped_output: str, expected: tuple[str | None, str | None]
) -> None:
    assert reasoning_parser.extract_reasoning(stripped_output, request=None) == expected


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content",
            "<|start|>assistant<|channel|>final<|message|>Content",
        ),
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>Preamble",
            "<|start|>assistant<|channel|>commentary<|message|>Preamble",
        ),
    ],
    ids=["final", "preamble-commentary"],
)
def test_extract_content_ids_returns_first_non_analysis_message_with_header(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer, trace: str, expected: str
) -> None:
    content_ids = reasoning_parser.extract_content_ids(fake_tokenizer.encode(trace))

    assert fake_tokenizer.decode(content_ids) == expected


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content",
            True,
        ),
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant to=functions.get_current_weather"
            + '<|channel|>commentary json<|message|>{"location": "Berlin"}',
            True,
        ),
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>Preamble",
            True,
        ),
        (
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content<|end|>"
            + "<|start|>assistant<|channel|>",
            False,
        ),
    ],
    ids=[
        "open-final-message-ends-reasoning",
        "open-tool-call-ends-reasoning",
        "open-preamble-commentary-ends-reasoning",
        "only-the-open-message-may-end-reasoning-not-a-completed-one",
    ],
)
def test_is_reasoning_end(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer, trace: str, expected: bool
) -> None:
    assert reasoning_parser.is_reasoning_end(fake_tokenizer.encode(trace)) is expected


def test_streaming_withholds_split_multibyte_character(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    prefix = fake_tokenizer.encode("<|channel|>analysis<|message|>R")
    first = prefix + [MULTIBYTE_PAIR_IDS[0]]
    second = first + [MULTIBYTE_PAIR_IDS[1]]

    delta = reasoning_parser.extract_reasoning_streaming(
        "", "", "", [], first, first[-1:]
    )
    assert delta is not None and delta.reasoning == "R"

    delta = reasoning_parser.extract_reasoning_streaming(
        "", "", "", first, second, second[-1:]
    )
    assert delta is not None and delta.reasoning == "あ"
