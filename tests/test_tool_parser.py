import json

import pytest

from conftest import MULTIBYTE_CHAR_IDS, FakeLlmjp4Tokenizer

pytest.importorskip("vllm")

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest  # noqa: E402

from llm_jp_vllm.llmjp4.reasoning_parser import Llmjp4ReasoningParser  # noqa: E402
from llm_jp_vllm.llmjp4.tool_parser import (  # noqa: E402
    ChatCompletionRequest,
    DeltaMessage,
    Llmjp4ToolParser,
)

SINGLE_TOOL_CALL_TRACE = (
    "<|channel|>analysis<|message|>Reasoning<|end|>"
    + "<|start|>assistant to=functions.get_current_weather"
    + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|call|>'
)

# Parallel call extension: non-final calls end with <|end|>, only the
# last with <|call|>.
PARALLEL_TOOL_CALL_TRACE = (
    "<|channel|>analysis<|message|>Reasoning<|end|>"
    + "<|start|>assistant to=functions.get_current_weather"
    + '<|channel|>commentary json<|message|>{"location": "Paris"}<|end|>'
    + "<|start|>assistant to=functions.get_current_time"
    + '<|channel|>commentary json<|message|>{"timezone": "Europe/London"}<|end|>'
    + "<|start|>assistant to=functions.get_current_weather"
    + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|call|>'
)


@pytest.fixture
def tool_parser(fake_tokenizer: FakeLlmjp4Tokenizer) -> Llmjp4ToolParser:
    return Llmjp4ToolParser(fake_tokenizer)


@pytest.fixture
def chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="llm-jp-4", messages=[])


def stream_tokens(
    tool_parser: Llmjp4ToolParser,
    chat_request: ChatCompletionRequest,
    content_ids: list[int],
    chunk_size: int = 1,
) -> list[DeltaMessage]:
    """Feed tokens chunk by chunk and collect the emitted deltas."""
    deltas: list[DeltaMessage] = []
    for start in range(0, len(content_ids), chunk_size):
        end = min(start + chunk_size, len(content_ids))
        delta = tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=content_ids[:start],
            current_token_ids=content_ids[:end],
            delta_token_ids=content_ids[start:end],
            request=chat_request,
        )
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_tool_calls(deltas: list[DeltaMessage]) -> list[tuple[str, str]]:
    """Reassemble (name, arguments) per call index from the deltas."""
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for delta in deltas:
        for tool_delta in delta.tool_calls:
            if tool_delta.function.name:
                names[tool_delta.index] = tool_delta.function.name
            if tool_delta.function.arguments:
                arguments[tool_delta.index] = (
                    arguments.get(tool_delta.index, "") + tool_delta.function.arguments
                )
    return [(names[index], arguments.get(index, "")) for index in sorted(names)]


@pytest.mark.parametrize(
    ("trace", "expected_calls"),
    [
        pytest.param(
            PARALLEL_TOOL_CALL_TRACE,
            [
                ("get_current_weather", '{"location": "Paris"}'),
                ("get_current_time", '{"timezone": "Europe/London"}'),
                ("get_current_weather", '{"location": "Berlin"}'),
            ],
            id="parallel",
        ),
        pytest.param(
            "<|start|>assistant to=functions.get_current_time"
            + "<|channel|>commentary json<|message|><|call|>",
            [("get_current_time", "{}")],
            id="no-arguments",
        ),
    ],
)
def test_extract_tool_calls(
    tool_parser, chat_request, trace: str, expected_calls: list[tuple[str, str]]
) -> None:
    result = tool_parser.extract_tool_calls(trace, chat_request)

    assert result.tools_called is True
    assert result.content is None
    assert [
        (tool_call.function.name, tool_call.function.arguments)
        for tool_call in result.tool_calls
    ] == expected_calls
    # Each call is a separate invocation with a distinct call id.
    assert len({tool_call.id for tool_call in result.tool_calls}) == len(expected_calls)


def test_adjust_request_ignores_responses_api_requests(tool_parser) -> None:
    # ResponsesRequest has no stop_token_ids; the Harmony adjustments
    # only apply to Chat Completions requests.
    request = ResponsesRequest(
        input="Hi", tools=[{"type": "function", "name": "echo", "parameters": {}}]
    )

    adjusted = tool_parser.adjust_request(request)

    assert adjusted.skip_special_tokens is True


def test_streaming_delta_protocol(
    tool_parser, chat_request, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    deltas = stream_tokens(
        tool_parser, chat_request, fake_tokenizer.encode(PARALLEL_TOOL_CALL_TRACE)
    )

    expected_names = ["get_current_weather", "get_current_time", "get_current_weather"]
    opened: list[int] = []
    for tool_delta in (d for delta in deltas for d in delta.tool_calls):
        if tool_delta.index == len(opened):
            # The first delta of a call carries id/type/name, exactly once.
            assert tool_delta.id
            assert tool_delta.type == "function"
            assert tool_delta.function.name == expected_names[tool_delta.index]
            opened.append(tool_delta.index)
        else:
            # Argument deltas only ever extend the newest call (the
            # serving layer flushes only the last call at end of stream).
            assert tool_delta.index == len(opened) - 1
            assert not tool_delta.id and not tool_delta.function.name
    assert opened == [0, 1, 2]


@pytest.mark.parametrize(
    ("trace", "expected_calls", "expected_reasoning"),
    [
        # After the serving layer's one-way switch to the tool phase, an
        # analysis message reaches only this parser.
        pytest.param(
            "<|start|>assistant to=functions.get_current_weather"
            + '<|channel|>commentary json<|message|>{"location": "Berlin"}<|end|>'
            + "<|start|>assistant<|channel|>analysis<|message|>Late reasoning",
            [("get_current_weather", '{"location": "Berlin"}')],
            "Late reasoning",
            id="late-analysis-routes-to-reasoning",
        ),
        pytest.param(
            "<|start|>assistant to=functions.echo"
            + '<|channel|>commentary json<|message|>{"text"'
            + '<|message|>: "Hi"}<|call|>',
            [("echo", '{"text"<|message|>: "Hi"}')],
            "",
            id="duplicate-message-marker-is-body-text",
        ),
        # The serving layer cannot flush "{}" for a call whose final
        # step produced no delta; the parser must stream it.
        pytest.param(
            "<|start|>assistant to=functions.get_current_time"
            + "<|channel|>commentary json<|message|><|call|>",
            [("get_current_time", "{}")],
            "",
            id="empty-arguments-stream-as-json",
        ),
    ],
)
def test_streaming_reconstructs_tool_calls(
    tool_parser,
    chat_request,
    fake_tokenizer: FakeLlmjp4Tokenizer,
    trace: str,
    expected_calls: list[tuple[str, str]],
    expected_reasoning: str,
) -> None:
    deltas = stream_tokens(tool_parser, chat_request, fake_tokenizer.encode(trace))

    assert reconstruct_tool_calls(deltas) == expected_calls
    assert "".join(delta.reasoning or "" for delta in deltas) == expected_reasoning


def test_streaming_withholds_split_multibyte_in_arguments(
    tool_parser, chat_request, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    # Premise: the ids are the byte tokens of "あ" (see conftest).
    assert fake_tokenizer.decode(list(MULTIBYTE_CHAR_IDS)) == "あ"

    content_ids = (
        fake_tokenizer.encode(
            "<|start|>assistant to=functions.echo"
            + '<|channel|>commentary json<|message|>{"text": "'
        )
        + list(MULTIBYTE_CHAR_IDS)
        # Long enough for the incremental decode to continue past the
        # multi-byte pair.
        + fake_tokenizer.encode('yes"}<|call|>')
    )

    deltas = stream_tokens(tool_parser, chat_request, content_ids)

    assert reconstruct_tool_calls(deltas) == [("echo", '{"text": "あyes"}')]


def test_streaming_state_resets_between_streams(
    tool_parser, chat_request, fake_tokenizer: FakeLlmjp4Tokenizer
) -> None:
    content_ids = fake_tokenizer.encode(SINGLE_TOOL_CALL_TRACE)
    stream_tokens(tool_parser, chat_request, content_ids)

    deltas = stream_tokens(tool_parser, chat_request, content_ids)

    assert reconstruct_tool_calls(deltas) == [
        ("get_current_weather", '{"location": "Berlin"}')
    ]
    assert len(tool_parser.prev_tool_call_arr) == 1


@pytest.mark.parametrize(
    "chunk_size",
    [pytest.param(1, id="token-by-token"), pytest.param(3, id="chunked")],
)
def test_streaming_matches_non_streaming(
    tool_parser,
    chat_request,
    fake_tokenizer: FakeLlmjp4Tokenizer,
    chunk_size: int,
) -> None:
    trace = PARALLEL_TOOL_CALL_TRACE
    # Equivalence test: the expected values are the non-streaming results.
    reasoning_parser = Llmjp4ReasoningParser(fake_tokenizer)
    full_ids = fake_tokenizer.encode(trace)
    # Replicate the serving layer: reasoning is streamed until
    # is_reasoning_end() flips, then the tool parser receives the
    # extract_content_ids() handover followed by the remaining tokens.
    switch_step = next(
        step
        for step in range(1, len(full_ids) + 1)
        if reasoning_parser.is_reasoning_end(full_ids[:step])
    )
    content_ids = reasoning_parser.extract_content_ids(full_ids[:switch_step])
    assert content_ids
    content_ids = content_ids + full_ids[switch_step:]

    deltas = stream_tokens(tool_parser, chat_request, content_ids, chunk_size)

    expected = tool_parser.extract_tool_calls(trace, chat_request)
    assert [
        (name, json.loads(arguments))
        for name, arguments in reconstruct_tool_calls(deltas)
    ] == [
        (tool_call.function.name, json.loads(tool_call.function.arguments))
        for tool_call in expected.tool_calls
    ]
    assert "".join(delta.content or "" for delta in deltas) == ""
    # The serving layer reads these to flush unstreamed arguments and to
    # set finish_reason="tool_calls".
    assert len(tool_parser.prev_tool_call_arr) == len(expected.tool_calls)
    assert tool_parser.streamed_args_for_tool == [
        tool_call["arguments"] for tool_call in tool_parser.prev_tool_call_arr
    ]
