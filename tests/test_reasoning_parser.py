import pytest

from conftest import MULTIBYTE_CHAR_IDS, FakeLlmjp4Tokenizer

pytest.importorskip("vllm")

from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage  # noqa: E402

from llm_jp_vllm.llmjp4.reasoning_parser import Llmjp4ReasoningParser  # noqa: E402


@pytest.fixture
def reasoning_parser(fake_tokenizer: FakeLlmjp4Tokenizer) -> Llmjp4ReasoningParser:
    return Llmjp4ReasoningParser(fake_tokenizer)


def stream_deltas(
    reasoning_parser: Llmjp4ReasoningParser, token_ids: list[int]
) -> list[DeltaMessage | None]:
    """Feed tokens one by one and collect the per-step deltas."""
    return [
        reasoning_parser.extract_reasoning_streaming(
            "",
            "",
            "",
            token_ids[: step - 1],
            token_ids[:step],
            token_ids[step - 1 : step],
        )
        for step in range(1, len(token_ids) + 1)
    ]


@pytest.mark.parametrize(
    ("model_output", "expected"),
    [
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content",
            ("Reasoning", "Content"),
            id="final",
        ),
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>Preamble",
            ("Reasoning", "Preamble"),
            id="preamble-commentary",
        ),
        pytest.param(
            "<|channel|>final json<|message|>Content",
            (None, "Content"),
            id="final-with-bare-word-recipient",
        ),
        pytest.param(
            "<|message|>Content",
            (None, "Content"),
            id="headerless-message",
        ),
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning-1<|end|>"
            + "<|start|>assistant to=functions.get_current_weather"
            + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|call|>'
            + "<|start|>assistant<|channel|>analysis<|message|>Reasoning-2",
            (
                "Reasoning-1\nReasoning-2",
                "<|start|>assistant to=functions.get_current_weather"
                + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|call|>'
                + "<|start|>assistant<|channel|>analysis<|message|>Reasoning-2",
            ),
            id="tool-call-markers-survive-for-the-tool-parser",
        ),
        pytest.param(
            "Here is my analysis of the data.",
            (None, "Here is my analysis of the data."),
            id="marker-less-output-is-plain-content",
        ),
    ],
)
def test_extract_reasoning(
    reasoning_parser, model_output: str, expected: tuple[str | None, str | None]
) -> None:
    assert reasoning_parser.extract_reasoning(model_output, request=None) == expected


def test_adjust_request_keeps_special_tokens(reasoning_parser) -> None:
    request = ChatCompletionRequest(model="llm-jp-4", messages=[])

    assert reasoning_parser.adjust_request(request).skip_special_tokens is False


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content",
            True,
            id="open-final-message-ends-reasoning",
        ),
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant to=functions.get_current_weather"
            + '<|channel|>commentary json<|message|>{"location": "Berlin"}',
            True,
            id="open-tool-call-ends-reasoning",
        ),
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>commentary<|message|>Preamble",
            True,
            id="open-preamble-commentary-ends-reasoning",
        ),
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Content<|end|>"
            + "<|start|>assistant<|channel|>",
            False,
            id="only-the-open-message-may-end-reasoning-not-a-completed-one",
        ),
        pytest.param(
            "<|channel|>analysis<|message|>Reasoning<|end|>"
            + "<|channel|>final<|message|>Content",
            True,
            id="final-without-start-marker-ends-reasoning",
        ),
    ],
)
def test_is_reasoning_end(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer, trace: str, expected: bool
) -> None:
    assert reasoning_parser.is_reasoning_end(fake_tokenizer.encode(trace)) is expected


def test_is_reasoning_end_streaming_tracks_cumulative_ids(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    token_ids = fake_tokenizer.encode(
        "<|channel|>analysis<|message|>Reasoning<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>Content"
    )

    # Structured-output engines pass cumulative ids each decode step
    # without ever calling extract_reasoning_streaming.
    results = [
        reasoning_parser.is_reasoning_end_streaming(
            token_ids[:step], token_ids[step - 1 : step]
        )
        for step in range(1, len(token_ids) + 1)
    ]

    assert results[0] is False
    assert results[-1] is True
    assert results == sorted(results)  # flips exactly once


def test_streaming_splits_messages_and_records_the_content_handover(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    token_ids = fake_tokenizer.encode(
        "<|channel|>analysis<|message|>Reasoning-1<|end|>"
        + "<|start|>assistant<|channel|>analysis<|message|>Reasoning-2<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>Content"
    )

    deltas = stream_deltas(reasoning_parser, token_ids)

    reasoning = "".join(d.reasoning or "" for d in deltas if d is not None)
    content = "".join(d.content or "" for d in deltas if d is not None)
    assert (reasoning, content) == ("Reasoning-1\nReasoning-2", "Content")
    # Full ids resolve the handover by scanning; per-step delta ids (all
    # the serving layer passes) resolve it from the streaming state.
    content_handover = "<|start|>assistant<|channel|>final<|message|>Content"
    assert (
        fake_tokenizer.decode(reasoning_parser.extract_content_ids(token_ids))
        == content_handover
    )
    assert (
        fake_tokenizer.decode(reasoning_parser.extract_content_ids(token_ids[-1:]))
        == content_handover
    )


def test_streaming_treats_marker_lookalike_text_as_body(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    # "<|end|>" spelled out as ordinary text tokens must not end the
    # message; only the dedicated special token id does.
    token_ids = (
        fake_tokenizer.encode("<|channel|>analysis<|message|>quote ")
        + [fake_tokenizer.encode(char)[0] for char in "<|end|>"]
        + fake_tokenizer.encode(" done<|end|>")
    )

    deltas = stream_deltas(reasoning_parser, token_ids)

    assert "".join(d.reasoning or "" for d in deltas if d is not None) == (
        "quote <|end|> done"
    )


def test_streaming_withholds_split_multibyte_characters(
    reasoning_parser, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    # Premise: the ids are the byte tokens of "あ", and like real
    # byte-fallback a run decodes as one unit — partial runs are U+FFFD.
    assert fake_tokenizer.decode(list(MULTIBYTE_CHAR_IDS)) == "あ"
    assert fake_tokenizer.decode(list(MULTIBYTE_CHAR_IDS) * 2) == "ああ"
    assert fake_tokenizer.decode(list(MULTIBYTE_CHAR_IDS[:2])) == "��"

    token_ids = (
        fake_tokenizer.encode("<|channel|>analysis<|message|>R")
        + list(MULTIBYTE_CHAR_IDS) * 3
    )

    deltas = stream_deltas(reasoning_parser, token_ids)

    # Bytes of an incomplete character produce no delta; each character
    # arrives whole even when byte runs cross the decode boundary.
    reasonings = [d.reasoning for d in deltas if d is not None]
    assert reasonings[-4:] == ["R", "あ", "あ", "あ"]
