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
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

from llm_jp_vllm.llmjp4.harmony import (
    FUNCTION_NAMESPACE,
    HarmonyHeader,
    HarmonyMessageKind,
    HarmonyMessageParser,
    HarmonyStreamParser,
    iter_text_messages,
)

logger = init_logger(__name__)

# Serving layers without this attribute do not route named/required
# tool_choice through the parser.
_HONORS_REQUIRED_AND_NAMED = hasattr(ToolParser, "supports_required_and_named")


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
        # Literal ids of tokenizer.encode("<|start|>assistant"):
        # NOTE(odashi): Prevent accessing to tokenizer methods
        # https://zenn.dev/yay1/articles/ad6958086670b0
        self._prefill_ids: list[int] = [10, 12811]

        self._stream = HarmonyStreamParser(self._parser, self._prefill_ids)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # Only Chat Completions requests carry the fields adjusted below
        # (ResponsesRequest has no stop_token_ids).
        if (
            not isinstance(request, ChatCompletionRequest)
            or not request.tools
            or request.tool_choice == "none"
        ):
            return super().adjust_request(request)

        # Skip the base implementation for required/named tool_choice:
        # its bare-JSON structured outputs conflict with Harmony headers.
        if request.tool_choice in ("auto", None):
            request = super().adjust_request(request)
        elif not _HONORS_REQUIRED_AND_NAMED:
            # Serving would return raw Harmony text as the tool
            # arguments; fail fast.
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
            if not any(
                marker in model_output
                for marker in ("<|channel|>", "<|start|>", "<|message|>")
            ):
                # adjust_request keeps the special tokens, so marker-less
                # text is a plain answer.
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )
            # The output text is lexed directly; re-encoding it would
            # touch the tokenizer's mutable state.
            tool_calls: list[ToolCall] = []
            content_parts: list[str] = []
            for message in iter_text_messages(model_output):
                if message.header.kind is HarmonyMessageKind.TOOL_CALL:
                    tool_calls.append(
                        ToolCall(
                            id=make_tool_call_id(),
                            type="function",
                            function=FunctionCall(
                                name=self._function_name(message.header),
                                arguments=self._normalize_arguments(message.body),
                            ),
                        )
                    )
                elif message.header.kind is HarmonyMessageKind.CONTENT and message.body:
                    content_parts.append(message.body)
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                # Messages are joined by newlines like the gpt-oss parser.
                content="\n".join(content_parts) or None,
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

            reasoning_delta, content_delta = self._stream.advance(
                len(previous_token_ids), current_token_ids
            )
            tool_deltas = [
                delta
                for index, (name, arguments) in enumerate(self._stream.calls)
                for delta in self._tool_deltas_for(index, name, arguments)
            ]

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

    def _tool_deltas_for(
        self, index: int, name: str, arguments: str
    ) -> list[DeltaToolCall]:
        """Emit the unsent portion of one tool call and update the state."""
        deltas: list[DeltaToolCall] = []

        if index == len(self.prev_tool_call_arr):
            # id/type/name go out exactly once; the state entries must be
            # registered in the same step (the serving layer indexes
            # streamed_args_for_tool as soon as a call appears).
            self.prev_tool_call_arr.append({"name": name, "arguments": ""})
            self.streamed_args_for_tool.append("")
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(name=name),
                )
            )

        args_delta = arguments[len(self.streamed_args_for_tool[index]) :]
        if args_delta:
            deltas.append(
                DeltaToolCall(
                    index=index,
                    function=DeltaFunctionCall(arguments=args_delta),
                )
            )
            self.streamed_args_for_tool[index] = arguments

        # The serving layer flushes prev_tool_call_arr[i]["arguments"]
        # minus streamed_args_for_tool[i] at end of stream.
        self.prev_tool_call_arr[index]["arguments"] = arguments or "{}"
        return deltas

    def _reset_streaming_state(self) -> None:
        self.prev_tool_call_arr: list[dict[str, str]] = []
        self.streamed_args_for_tool: list[str] = []
        self._stream = HarmonyStreamParser(self._parser, self._prefill_ids)

    def _normalize_arguments(self, raw: str) -> str:
        # Invalid JSON passes through unchanged; downstream evaluators
        # decide how to treat it.
        text = raw.strip()
        if not text:
            # Clients json.loads() the arguments even for parameterless
            # calls.
            return "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Tool call arguments are not valid JSON: %r", text)
            return text
        return json.dumps(parsed, ensure_ascii=False)

    def _function_name(self, header: HarmonyHeader) -> str:
        assert header.recipient is not None  # guaranteed by TOOL_CALL kind
        return header.recipient[len(FUNCTION_NAMESPACE) :]
