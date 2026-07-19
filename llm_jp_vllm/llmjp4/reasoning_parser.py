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
    HarmonyMessage,
    HarmonyMessageKind,
    HarmonyMessageParser,
    HarmonyStreamParser,
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
        return last_message is not None and self._ends_reasoning(last_message)

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

        token_ids = self.model_tokenizer.encode(model_output, add_special_tokens=False)
        # Analysis may reappear after the first content/tool message,
        # so reasoning is collected from the whole sequence.
        reasoning, content = self._classified_texts(token_ids)
        start = self._content_start(token_ids)
        if start is not None:
            return reasoning or None, self._content_text(token_ids, start)
        # No message properly ends the reasoning phase, but classified
        # content (e.g. from a headerless message) must not be lost.
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

    def _ends_reasoning(self, message: HarmonyMessage) -> bool:
        if message.content is None:
            # The header is still incomplete: <|message|> has not appeared.
            return False
        header = self._parser.parse_header(message)
        if header.channel is None and header.recipient is None:
            # A mid-message fragment (single-step delta) has no header to
            # classify.
            return False
        # IGNORE (built-in tool recipients such as "python") stays inside
        # reasoning, matching the gpt-oss semantics where built-in tool
        # calls happen while reasoning.
        return header.kind in (HarmonyMessageKind.CONTENT, HarmonyMessageKind.TOOL_CALL)

    def _content_text(self, token_ids: list[int], start: int | None) -> str | None:
        """User-visible content of the ids from ``start`` on."""
        if start is None:
            return None
        content_ids = token_ids[start:]
        if self._has_tool_call(content_ids):
            # The tool parser re-parses this text and needs the raw markers.
            return self.model_tokenizer.decode(content_ids)
        # Plain answers reach the client without tool-parser cleanup,
        # so the markers must be dropped here.
        _, content = self._classified_texts(content_ids)
        return content or None

    def _has_tool_call(self, token_ids: list[int]) -> bool:
        return any(
            message.content is not None
            and self._parser.parse_header(message).kind is HarmonyMessageKind.TOOL_CALL
            for message in self._parser.iter_messages(
                self._reasoning_prefill + list(token_ids)
            )
        )

    def _content_start(self, input_ids: list[int]) -> int | None:
        """Position where the first non-analysis message starts."""
        # Prepending is harmless even when input_ids already starts
        # with <|start|>.
        padded = self._reasoning_prefill + list(input_ids)
        for message in self._parser.iter_messages(padded):
            if self._ends_reasoning(message):
                return max(message.start - len(self._reasoning_prefill), 0)
        return None

    def _classified_texts(self, token_ids: list[int]) -> tuple[str, str]:
        """Reasoning/content texts of a token sequence, with messages
        joined by newlines like the gpt-oss parser."""
        # Tool-call bodies are excluded; they belong to the tool parser.
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for message in self._parser.iter_messages(
            self._reasoning_prefill + list(token_ids)
        ):
            if message.content is None:
                continue
            text = self.model_tokenizer.decode(message.content.token_ids)
            if not text:
                continue
            kind = self._parser.parse_header(message).kind
            if kind is HarmonyMessageKind.REASONING:
                reasoning_parts.append(text)
            elif kind is HarmonyMessageKind.CONTENT:
                content_parts.append(text)
        return "\n".join(reasoning_parts), "\n".join(content_parts)
