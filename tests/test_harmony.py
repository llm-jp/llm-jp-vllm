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
            fake_tokenizer.decode(message.role or []),
            fake_tokenizer.decode(message.channel or []),
            fake_tokenizer.decode(message.content or []),
            message.end,
        )
        for message in messages
    ] == [
        ("assistant", "analysis", "Reasoning", HarmonyMessageEndType.END),
        ("assistant", "final", "Content", HarmonyMessageEndType.INCOMPLETE),
    ]


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        pytest.param(
            "<|start|>assistant<|channel|>final<|message|>A<|channel|>B<|end|>",
            [("assistant", "final", "A<|channel|>B", HarmonyMessageEndType.END)],
            id="marker-after-message-is-body-text",
        ),
        pytest.param(
            "<|end|><|start|>assistant<|channel|>final<|message|>A",
            [("assistant", "final", "A", HarmonyMessageEndType.INCOMPLETE)],
            id="stray-end-token-yields-no-message",
        ),
        pytest.param(
            "<|start|>assistant<|channel|>final<|message|>A"
            + "<|start|>assistant<|channel|>analysis<|message|>B<|end|>",
            [
                (
                    "assistant",
                    "final",
                    "A<|start|>assistant<|channel|>analysis<|message|>B",
                    HarmonyMessageEndType.END,
                )
            ],
            id="start-without-preceding-end-is-body-text",
        ),
    ],
)
def test_iter_messages_follows_official_state_machine(
    fake_tokenizer: FakeLlmjp4Tokenizer,
    trace: str,
    expected: list[tuple[str, str, str, HarmonyMessageEndType]],
) -> None:
    parser = HarmonyMessageParser(fake_tokenizer)

    messages = parser.get_all_messages(fake_tokenizer.encode(trace))

    assert [
        (
            fake_tokenizer.decode(message.role or []),
            fake_tokenizer.decode(message.channel or []),
            fake_tokenizer.decode(message.content or []),
            message.end,
        )
        for message in messages
    ] == expected
