# vLLM Reasoning parser implementation for llm-jp-4 models.
# The overall algorithm is based on `GptOssReasoningParser`,
# but applies some modification.
# https://github.com/llm-jp/vllm/blob/4383f1532e87e77b6f961e633230f47467cbd072/vllm/reasoning/gptoss_reasoning_parser.py#L65

import warnings
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)
from vllm.tokenizers import TokenizerLike

from llm_jp_vllm.llm_jp_4.harmony import HarmonyMessageParser


@ReasoningParserManager.register_module(["llmjp4"])  # type: ignore[arg-type]
class Llmjp4ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        tokenizer = self.model_tokenizer
        vocab = self.vocab

        self._parser = HarmonyMessageParser(tokenizer)
        self._start_id = vocab["<|start|>"]
        self._end_id = vocab["<|end|>"]
        self._message_id = vocab["<|message|>"]

        # NOTE(odashi): Prevent accessing to tokenizer methods
        # https://zenn.dev/yay1/articles/ad6958086670b0
        self._reasoning_end_prefix = [9, 2520]  # tokenizer.encode("<|channel|>final")
        self._reasoning_prefill = [10, 12811]  # tokenizer.encode("<|start|>assistant")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Find the final message pattern: <|channel|>final ... <|message|>
        end_prefix = self._reasoning_end_prefix
        message_id_found = False

        for i in range(len(input_ids) - len(end_prefix), -1, -1):
            if input_ids[i] == self._end_id:
                # Reached the previous message
                return False
            elif input_ids[i] == self._message_id:
                message_id_found = True
            elif input_ids[i] == end_prefix[0]:
                if (
                    input_ids[i : i + len(end_prefix)] == end_prefix
                    and message_id_found
                ):
                    return True

        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        input_ids = self._reasoning_prefill + input_ids

        for message in self._parser.reverse_iter_messages(input_ids):
            if message.role is None:
                continue
            channel_str = self.model_tokenizer.decode(message.role.token_ids)
            if channel_str.startswith("final"):
                return message.content.token_ids if message.content else []

        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return 0

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        warnings.warn(
            "Non-streaming response is not correctly implemented"
            " due to the limitation of the current vLLM interface."
        )
        # NOTE(odashi):
        # This is a workaround implementation,
        # should be replaced with a proper implementation the interface is updated.
        # We still can not handle the preceding reasoning parts appropriately.
        marker = " assistant final "
        marker_index = model_output.rfind(marker)
        if marker_index == -1:
            return None, None
        content = model_output[marker_index + len(marker) :].strip()
        return None, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        previous_token_ids = self._reasoning_prefill + list(previous_token_ids)
        current_token_ids = self._reasoning_prefill + list(current_token_ids)

        last_start_index: int | None = None
        for i in range(len(previous_token_ids) - 1, -1, -1):
            if previous_token_ids[i] == self._start_id:
                last_start_index = i
                break
        assert last_start_index is not None, "<|start|> must exists."

        previous_messages = self._parser.get_all_messages(
            previous_token_ids[last_start_index:]
        )
        current_messages = self._parser.get_all_messages(
            current_token_ids[last_start_index:]
        )
        assert len(previous_messages) == 1
        assert len(current_messages) >= 1

        reasoning_delta: list[str] = []
        content_delta: list[str] = []

        # Continuation of the last message
        previous_content = previous_messages[0].content
        previous_text = self.model_tokenizer.decode(
            previous_content.token_ids if previous_content else []
        )

        # Analyse messages
        for i, message in enumerate(current_messages):
            if (
                message.role is None
                or message.channel is None
                or message.content is None
            ):
                continue

            role_text = self.model_tokenizer.decode(message.role.token_ids)
            channel_text = self.model_tokenizer.decode(message.channel.token_ids)
            current_text = self.model_tokenizer.decode(message.content.token_ids)
            prefix = previous_text if i == 0 else ""
            assert current_text.startswith(prefix)
            delta_text = current_text[len(prefix) :]

            if role_text != "assistant" or not delta_text:
                continue

            if channel_text.startswith("final"):
                content_delta.append(delta_text)
            else:
                reasoning_delta.append(delta_text)

        return DeltaMessage(
            reasoning="".join(reasoning_delta) if reasoning_delta else None,
            content="".join(content_delta) if content_delta else None,
        )
