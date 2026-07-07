"""A fake LLM-jp-4 tokenizer implementing the subset of the tokenizer
API used by the parsers (get_vocab / encode / decode)."""

import pytest

# Harmony special tokens as single tokens, with the ids of the real
# LLM-jp-4 tokenizer where known; the rest are arbitrary but must not
# collide with the dynamically assigned ids (>= 1000).
_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|return|>": 2,
    "<|end|>": 7,
    "<|message|>": 8,
    "<|channel|>": 9,
    "<|start|>": 10,
    "<|constrain|>": 12,
    "<|call|>": 13,
}

# Words that the real LLM-jp-4 tokenizer encodes as single tokens.
_WORD_TOKEN_IDS: dict[str, int] = {
    "assistant": 12811,
    "final": 2520,
}

# Byte-fallback simulation: the first id alone decodes to U+FFFD, the
# pair to the complete character "あ".
MULTIBYTE_PAIR_IDS: tuple[int, int] = (900, 901)


class FakeLlmjp4Tokenizer:
    """Seeded tokens are matched greedily; everything else is tokenized
    per character, so encode/decode round-trip exactly."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            **_SPECIAL_TOKEN_IDS,
            **_WORD_TOKEN_IDS,
        }
        self._id_to_token: dict[int, str] = {
            token_id: token for token, token_id in self._token_to_id.items()
        }
        self._seeded_tokens: list[str] = sorted(
            self._token_to_id, key=len, reverse=True
        )
        self._next_id: int = 1000

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids: list[int] = []
        position = 0
        while position < len(text):
            for token in self._seeded_tokens:
                if text.startswith(token, position):
                    token_ids.append(self._token_to_id[token])
                    position += len(token)
                    break
            else:
                token_ids.append(self._char_id(text[position]))
                position += 1
        return token_ids

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        parts: list[str] = []
        position = 0
        while position < len(token_ids):
            if token_ids[position] == MULTIBYTE_PAIR_IDS[0]:
                if token_ids[position + 1 : position + 2] == [MULTIBYTE_PAIR_IDS[1]]:
                    parts.append("あ")
                    position += 2
                else:
                    parts.append("�")
                    position += 1
                continue
            parts.append(self._id_to_token[token_ids[position]])
            position += 1
        return "".join(parts)

    def _char_id(self, char: str) -> int:
        if char not in self._token_to_id:
            self._token_to_id[char] = self._next_id
            self._id_to_token[self._next_id] = char
            self._next_id += 1
        return self._token_to_id[char]


@pytest.fixture
def fake_tokenizer() -> FakeLlmjp4Tokenizer:
    return FakeLlmjp4Tokenizer()
