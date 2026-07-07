# vLLM Reasoning parser implementation for llm-jp-4 models.
# The overall algorithm is based on `GptOssReasoningParser`,
# but applies some modification.
# https://github.com/llm-jp/vllm/blob/4383f1532e87e77b6f961e633230f47467cbd072/vllm/reasoning/gptoss_reasoning_parser.py#L65

import re
from collections.abc import Sequence

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
    strip_incomplete_decode,
)


@ReasoningParserManager.register_module(["llmjp4"])  # type: ignore[arg-type]
class Llmjp4ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        tokenizer = self.model_tokenizer

        self._parser = HarmonyMessageParser(tokenizer)
        # Generation resumes after this prefill, so the first generated
        # message has no <|start|> of its own. Encoded once here:
        # NOTE(odashi): Prevent accessing to tokenizer methods
        # https://zenn.dev/yay1/articles/ad6958086670b0
        self._reasoning_prefill: list[int] = tokenizer.encode(
            "<|start|>assistant", add_special_tokens=False
        )

        # The serving layer calls is_reasoning_end / extract_content_ids
        # with per-step delta ids only; the boundary observed on the
        # cumulative ids in extract_reasoning_streaming is recorded here.
        self._streaming_reasoning_ended: bool = False
        self._pending_content_ids: list[int] = []

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._streaming_reasoning_ended:
            return True
        # Only the last (still open) message can end the reasoning
        # prefix; completed final messages from previous turns in a
        # multi-turn prompt must not count.
        last_message = next(
            self._parser.reverse_iter_messages(
                self._reasoning_prefill + list(input_ids)
            ),
            None,
        )
        return last_message is not None and self._ends_reasoning(last_message)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Returns everything from the <|start|> of the first non-analysis
        # message: the tool parser needs a well-formed message sequence.
        start = self._content_start(input_ids)
        if start is not None:
            return input_ids[start:]
        # input_ids may be a mid-message delta; use the recorded state.
        if self._streaming_reasoning_ended:
            return list(self._pending_content_ids)
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if "<|channel|>" not in model_output and "<|start|>" not in model_output:
            return self._extract_reasoning_from_stripped_text(model_output)

        token_ids = self.model_tokenizer.encode(model_output, add_special_tokens=False)
        # Analysis may reappear after the first content/tool message,
        # so reasoning is collected from the whole sequence.
        reasoning, _ = self._classified_texts(token_ids)
        start = self._content_start(token_ids)
        return reasoning or None, self._content_text(token_ids, start)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # A fresh stream may reuse this instance; drop recorded state.
        if not previous_token_ids:
            self._streaming_reasoning_ended = False
            self._pending_content_ids = []

        start = self._content_start(list(current_token_ids))
        if start is not None:
            self._streaming_reasoning_ended = True
            self._pending_content_ids = list(current_token_ids[start:])

        previous_reasoning, previous_content = self._classified_texts(
            list(previous_token_ids)
        )
        current_reasoning, current_content = self._classified_texts(
            list(current_token_ids)
        )
        # Withhold U+FFFD from partially decoded multi-byte characters;
        # emitting it would corrupt the length-based delta.
        reasoning_delta = strip_incomplete_decode(current_reasoning)[
            len(strip_incomplete_decode(previous_reasoning)) :
        ]
        content_delta = strip_incomplete_decode(current_content)[
            len(strip_incomplete_decode(previous_content)) :
        ]

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

    def _extract_reasoning_from_stripped_text(
        self, model_output: str
    ) -> tuple[str | None, str | None]:
        # "|\s*$" also captures an analysis message truncated by max_tokens.
        reasoning_parts = [
            match.group(1).strip()
            for match in re.finditer(
                r"(?:^|\s)analysis\s+(.*?)(?=\s+assistant\b|\s*$)",
                model_output,
                re.DOTALL,
            )
        ]
        final_match = re.search(
            r"(?:^|\s)assistant\s+final\s+(.*)", model_output, re.DOTALL
        )
        if not reasoning_parts and final_match is None:
            # No Harmony header words at all: plain text is the answer.
            return None, model_output
        # A generation truncated inside analysis must not fall back to
        # the raw output: that would serve the chain-of-thought as content.
        content = final_match.group(1).strip() if final_match else None
        return "\n".join(reasoning_parts) or None, content
