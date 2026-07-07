import pytest

from conftest import FakeLlmjp4Tokenizer

pytest.importorskip("vllm")

from llm_jp_vllm.llmjp4.harmony import (  # noqa: E402
    HarmonyMessageEndType,
    HarmonyMessageParser,
)


def test_parser_splits_analysis_and_final_messages(
    fake_tokenizer: FakeLlmjp4Tokenizer,
) -> None:
    parser = HarmonyMessageParser(fake_tokenizer)
    token_ids = fake_tokenizer.encode(
        "<|start|>assistant<|channel|>analysis<|message|>Reasoning<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>Content"
    )

    messages = parser.get_all_messages(token_ids)

    assert [
        (
            fake_tokenizer.decode(message.role.token_ids if message.role else []),
            fake_tokenizer.decode(message.channel.token_ids if message.channel else []),
            fake_tokenizer.decode(message.content.token_ids if message.content else []),
            message.end,
        )
        for message in messages
    ] == [
        ("assistant", "analysis", "Reasoning", HarmonyMessageEndType.END),
        ("assistant", "final", "Content", HarmonyMessageEndType.INCOMPLETE),
    ]
