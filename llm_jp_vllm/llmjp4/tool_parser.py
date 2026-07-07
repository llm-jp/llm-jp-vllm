# vLLM Tool parser implementation for llm-jp-4 models.
# Parses Harmony-format tool calls with the parallel call extension:
# non-final calls end with <|end|>, the last call with <|call|>.

import json
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

from llm_jp_vllm.llmjp4.harmony import (
    FUNCTION_NAMESPACE,
    HarmonyHeader,
    HarmonyMessageKind,
    HarmonyMessageParser,
    strip_incomplete_decode,
)

logger = init_logger(__name__)


@ToolParserManager.register_module(["llmjp4"])  # type: ignore[arg-type]
class Llmjp4ToolParser(ToolParser):
    # Harmony models cannot emit the bare JSON that the standard
    # required/named tool_choice handling expects; route those requests
    # through this parser like "auto".
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        tokenizer = self.model_tokenizer
        vocab = self.vocab

        self._parser = HarmonyMessageParser(tokenizer)
        self._call_id: int = vocab["<|call|>"]
        self._return_id: int = vocab["<|return|>"]
        self._prefill_ids: list[int] = tokenizer.encode(
            "<|start|>assistant", add_special_tokens=False
        )

        self._sent_content_length: int = 0
        self._sent_reasoning_length: int = 0

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if not request.tools or request.tool_choice == "none":
            return super().adjust_request(request)

        # Skip the base implementation for required/named tool_choice:
        # its bare-JSON structured outputs conflict with Harmony headers.
        if request.tool_choice in ("auto", None):
            request = super().adjust_request(request)
        elif not hasattr(ToolParser, "supports_required_and_named"):
            # Without this attribute, serving would return raw Harmony
            # text as the tool arguments; fail fast.
            raise ValueError(
                f"tool_choice={request.tool_choice!r} is not supported by the "
                "llmjp4 tool parser on this vLLM version"
            )

        request.skip_special_tokens = False

        # Without these stop tokens generation does not stop at the
        # final tool call.
        request.stop_token_ids = sorted(
            set(request.stop_token_ids or []) | {self._call_id, self._return_id}
        )
        return request

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        try:
            if "<|channel|>" not in model_output and "<|start|>" not in model_output:
                # adjust_request keeps the special tokens, so marker-less
                # text is a plain answer.
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )
            token_ids = self.model_tokenizer.encode(
                model_output, add_special_tokens=False
            )
            calls, content, _ = self._split_messages(token_ids)
            return ExtractedToolCallInformation(
                tools_called=bool(calls),
                tool_calls=[
                    ToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=self._function_name(header),
                            arguments=self._normalize_arguments(body),
                        ),
                    )
                    for header, body in calls
                ],
                content=content or None,
            )
        except Exception:
            logger.exception("Failed to extract tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Token ids, not text: the <|call|> stop token never appears in
        # the text deltas.
        try:
            if not previous_token_ids:
                self._reset_streaming_state()

            calls, content, reasoning = self._split_messages(list(current_token_ids))
            tool_deltas = [
                delta
                for index, (header, body) in enumerate(calls)
                for delta in self._tool_deltas_for(index, header, body)
            ]
            # Withhold partially decoded multi-byte characters (U+FFFD);
            # they complete in a later step.
            content = strip_incomplete_decode(content)
            reasoning = strip_incomplete_decode(reasoning)
            content_delta = content[self._sent_content_length :]
            self._sent_content_length = len(content)
            reasoning_delta = reasoning[self._sent_reasoning_length :]
            self._sent_reasoning_length = len(reasoning)

            if not tool_deltas and not content_delta and not reasoning_delta:
                return None
            return DeltaMessage(
                content=content_delta or None,
                reasoning=reasoning_delta or None,
                tool_calls=tool_deltas,
            )
        except Exception:
            logger.exception("Error in streaming tool call extraction.")
            return None

    def _split_messages(
        self, token_ids: list[int]
    ) -> tuple[list[tuple[HarmonyHeader, str]], str, str]:
        """Split parsed messages into tool calls, content and reasoning."""
        calls: list[tuple[HarmonyHeader, str]] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for message in self._parser.get_all_messages(self._prefill_ids + token_ids):
            if message.content is None:
                # Incomplete header; the function name may be truncated.
                continue
            header = self._parser.parse_header(message)
            body = self.model_tokenizer.decode(message.content.token_ids)
            if header.kind is HarmonyMessageKind.TOOL_CALL:
                calls.append((header, body))
            elif not body:
                continue
            elif header.kind is HarmonyMessageKind.CONTENT:
                content_parts.append(body)
            elif header.kind is HarmonyMessageKind.REASONING:
                # Analysis reappearing after the switch to the tool phase
                # (e.g. between parallel calls) still belongs to reasoning.
                reasoning_parts.append(body)
        # Messages are joined by newlines like the gpt-oss parser.
        return calls, "\n".join(content_parts), "\n".join(reasoning_parts)

    def _tool_deltas_for(
        self, index: int, header: HarmonyHeader, body: str
    ) -> list[DeltaToolCall]:
        """Emit the unsent portion of one tool call and update the state."""
        deltas: list[DeltaToolCall] = []

        if index == len(self.prev_tool_call_arr):
            # id/type/name go out exactly once; the state entries must be
            # registered in the same step (the serving layer indexes
            # streamed_args_for_tool as soon as a call appears).
            name = self._function_name(header)
            self.prev_tool_call_arr.append({"name": name, "arguments": ""})
            self.streamed_args_for_tool.append("")
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(name=name).model_dump(exclude_none=True),
                )
            )

        args_delta = strip_incomplete_decode(body)[
            len(self.streamed_args_for_tool[index]) :
        ]
        if args_delta:
            deltas.append(
                DeltaToolCall(
                    index=index,
                    function=DeltaFunctionCall(arguments=args_delta).model_dump(
                        exclude_none=True
                    ),
                )
            )
            self.streamed_args_for_tool[index] += args_delta

        # The serving layer flushes prev_tool_call_arr[i]["arguments"]
        # minus streamed_args_for_tool[i] at end of stream.
        self.prev_tool_call_arr[index]["arguments"] = body
        return deltas

    def _reset_streaming_state(self) -> None:
        self.prev_tool_call_arr: list[dict[str, str]] = []
        self.streamed_args_for_tool: list[str] = []
        self._sent_content_length = 0
        self._sent_reasoning_length = 0

    def _normalize_arguments(self, raw: str) -> str:
        # Invalid JSON passes through unchanged; downstream evaluators
        # decide how to treat it.
        text = raw.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Tool call arguments are not valid JSON: %r", text)
            return text
        return json.dumps(parsed, ensure_ascii=False)

    def _function_name(self, header: HarmonyHeader) -> str:
        assert header.recipient is not None  # guaranteed by TOOL_CALL kind
        return header.recipient[len(FUNCTION_NAMESPACE) :]
