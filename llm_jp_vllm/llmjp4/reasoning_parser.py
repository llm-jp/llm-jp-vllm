# vLLM Reasoning parser implementation for llm-jp-4 models.
# The overall algorithm is based on `GptOssReasoningParser`,
# but applies some modification.
# https://github.com/llm-jp/vllm/blob/4383f1532e87e77b6f961e633230f47467cbd072/vllm/reasoning/gptoss_reasoning_parser.py#L65

from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)
from vllm.tokenizers import TokenizerLike

from llm_jp_vllm.llmjp4.harmony import (
    HarmonyHeader,
    HarmonyMessage,
    HarmonyMessageKind,
    HarmonyMessageParser,
    HarmonyStreamParser,
    iter_text_messages,
)


@ReasoningParserManager.register_module(["llmjp4"])  # type: ignore[arg-type]
class Llmjp4ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        tokenizer = self.model_tokenizer

        self._parser = HarmonyMessageParser(tokenizer)
        # Generation resumes after this prefill, so the first generated
        # message has no <|start|> of its own. Literal ids of
        # tokenizer.encode("<|start|>assistant"):
        # NOTE(odashi): Prevent accessing to tokenizer methods
        # https://zenn.dev/yay1/articles/ad6958086670b0
        self._reasoning_prefill: list[int] = [10, 12811]

        # The serving layer calls is_reasoning_end / extract_content_ids
        # with per-step delta ids only; the reasoning boundary is tracked
        # on this cumulative stream advanced in extract_reasoning_streaming.
        self._stream = HarmonyStreamParser(self._parser, self._reasoning_prefill)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # Marker-based extraction needs the special tokens kept in the
        # output text.
        request.skip_special_tokens = False
        return super().adjust_request(request)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._stream.content_started:
            return True
        # Only the last (still open) message can end the reasoning
        # prefix; completed final messages from previous turns in a
        # multi-turn prompt must not count.
        last_message: HarmonyMessage | None = None
        for last_message in self._parser.iter_messages(
            self._reasoning_prefill + list(input_ids)
        ):
            pass
        return last_message is not None and self._message_ends_reasoning(last_message)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # Structured-output engines call only this method, with cumulative
        # ids, so the stream may still need advancing here; on the chat
        # path extract_reasoning_streaming has already consumed the ids.
        if self._stream.consumed != len(input_ids):
            self._stream.advance(len(input_ids) - len(list(delta_ids)), input_ids)
        return self._stream.content_started

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Returns everything from the <|start|> of the first non-analysis
        # message: the tool parser needs a well-formed message sequence.
        start = self._content_start(input_ids)
        if start is not None:
            return input_ids[start:]
        # input_ids may be a mid-message delta; use the streaming state.
        return self._stream.content_ids

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not any(
            marker in model_output
            for marker in ("<|channel|>", "<|start|>", "<|message|>")
        ):
            # adjust_request keeps the special tokens, so marker-less
            # text is a plain answer.
            return None, model_output

        # The output text is lexed directly; re-encoding it would touch
        # the tokenizer's mutable state (see the NOTE in __init__).
        messages = list(iter_text_messages(model_output))
        # Analysis may reappear after the first content/tool message,
        # so reasoning is collected from the whole sequence.
        reasoning = "\n".join(
            message.body
            for message in messages
            if message.header.kind is HarmonyMessageKind.REASONING and message.body
        )
        offset = next(
            (
                message.start_offset
                for message in messages
                if self._ends_reasoning(message.header)
            ),
            None,
        )
        if offset is not None and any(
            message.header.kind is HarmonyMessageKind.TOOL_CALL for message in messages
        ):
            # The tool parser re-parses this text and needs the raw markers.
            return reasoning or None, model_output[offset:]
        # When no message properly ends the reasoning phase, classified
        # content (e.g. from a headerless message) must still not be lost.
        content = "\n".join(
            message.body
            for message in messages
            if message.header.kind is HarmonyMessageKind.CONTENT
            and message.body
            and (offset is None or message.start_offset >= offset)
        )
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # A fresh stream reusing this instance shows up as a mismatched
        # previous length, which makes advance() rebuild from scratch.
        reasoning_delta, content_delta = self._stream.advance(
            len(previous_token_ids), current_token_ids
        )

        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
        )

    def _message_ends_reasoning(self, message: HarmonyMessage) -> bool:
        # The header is still incomplete until <|message|> appears.
        return message.content is not None and self._ends_reasoning(
            self._parser.parse_header(message)
        )

    def _ends_reasoning(self, header: HarmonyHeader) -> bool:
        if header.channel is None and header.recipient is None:
            # A mid-message fragment (single-step delta) has no header to
            # classify.
            return False
        # IGNORE (built-in tool recipients such as "python") stays inside
        # reasoning, matching the gpt-oss semantics where built-in tool
        # calls happen while reasoning.
        return header.kind in (HarmonyMessageKind.CONTENT, HarmonyMessageKind.TOOL_CALL)

    def _content_start(self, input_ids: list[int]) -> int | None:
        """Position where the first non-analysis message starts."""
        # Prepending is harmless even when input_ids already starts
        # with <|start|>.
        padded = self._reasoning_prefill + list(input_ids)
        for message in self._parser.iter_messages(padded):
            if self._message_ends_reasoning(message):
                return max(message.start_position - len(self._reasoning_prefill), 0)
        return None
