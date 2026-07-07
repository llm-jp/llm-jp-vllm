# Generic parser for OpenAI Harmony format.
# This is basically identical with the bundled parser in LLM-jp-4 models,
# but typing annotation is modified to follow vLLM standards.

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Sequence

from vllm.tokenizers import TokenizerLike

# Tool calls are addressed to the "functions" namespace; other recipients
# (browser.*, python, ...) are built-in tools that must not be converted
# into OpenAI tool_calls.
FUNCTION_NAMESPACE = "functions."

_RECIPIENT_PREFIX = "to="


def strip_incomplete_decode(text: str) -> str:
    """Drop the trailing U+FFFD of a partially decoded multi-byte
    character; streaming callers withhold it until it completes."""
    return text.rstrip("�")


class HarmonyMessageEndType(Enum):
    INCOMPLETE = 0
    END = 1
    CALL = 2


class HarmonyMessageKind(Enum):
    """Classification of a Harmony message, mirroring gpt-oss's rules
    (`_SegmentType.from_channel_and_recipient` in vllm/parser/harmony.py).
    """

    REASONING = auto()
    CONTENT = auto()
    TOOL_CALL = auto()
    IGNORE = auto()


@dataclass(frozen=True)
class HarmonySequence:
    """A data class representing a sequence of tokens in the Harmony format."""

    token_ids: list[int]
    start: int  # Start position of the sequence in the original token sequence


@dataclass(frozen=True)
class HarmonyMessage:
    """A data class representing a message in the Harmony format."""

    end: HarmonyMessageEndType
    role: HarmonySequence | None = None
    channel: HarmonySequence | None = None
    constrain: HarmonySequence | None = None
    content: HarmonySequence | None = None

    @property
    def start(self) -> int:
        """Position of the message's <|start|> in the parsed sequence.

        Section starts point just after their marker token; 0 when the
        message opens the sequence without a <|start|>.
        """
        first_section = min(
            (
                section.start
                for section in (self.role, self.channel, self.constrain, self.content)
                if section is not None
            ),
            default=1,
        )
        return max(first_section - 1, 0)


@dataclass(frozen=True)
class HarmonyHeader:
    """Parsed header metadata of a Harmony message."""

    role: str | None = None
    channel: str | None = None
    recipient: str | None = None
    content_type: str | None = None

    @property
    def kind(self) -> HarmonyMessageKind:
        """Classification of the message this header belongs to."""
        if self.recipient is not None:
            if self.recipient.startswith(FUNCTION_NAMESPACE):
                return HarmonyMessageKind.TOOL_CALL
            return HarmonyMessageKind.IGNORE
        if self.channel == "analysis":
            return HarmonyMessageKind.REASONING
        return HarmonyMessageKind.CONTENT


class HarmonyMessageParser:
    """A parser that performs lexical analysis to extract Harmony messages."""

    def __init__(self, tokenizer: TokenizerLike):
        vocab = tokenizer.get_vocab()
        self._tokenizer = tokenizer
        self._start_id = vocab["<|start|>"]
        self._begin_map = {
            vocab["<|start|>"]: "role",
            vocab["<|channel|>"]: "channel",
            vocab["<|constrain|>"]: "constrain",
            vocab["<|message|>"]: "content",
        }
        self._end_map = {
            vocab["<|end|>"]: HarmonyMessageEndType.END,
            vocab["<|return|>"]: HarmonyMessageEndType.END,
            vocab["<|call|>"]: HarmonyMessageEndType.CALL,
        }

    def parse_header(self, message: HarmonyMessage) -> HarmonyHeader:
        """Parse the header sections of a message into structured metadata.

        Follows openai/harmony's `parse_header_from_string`: recipient
        ("to=...") and content-type are read from the tail of the header
        words, so a recipient is recognized on both sides of <|channel|>
        (the LLM-jp-4 template and official examples place it differently).
        """
        role_words = self._section_words(message.role)
        channel_words = self._section_words(message.channel)

        role = role_words[0] if role_words else None
        channel = channel_words[0] if channel_words else None
        parts = role_words[1:] + channel_words[1:]

        recipient: str | None = None
        content_type: str | None = None
        if parts:
            last = parts[-1]
            if last.startswith(_RECIPIENT_PREFIX):
                recipient = last[len(_RECIPIENT_PREFIX) :]
            elif len(parts) == 1:
                # A single word that is not "to=..." is a bare recipient.
                recipient = last
            else:
                # e.g. "to=functions.x json": content-type last, recipient before it
                content_type = last
                recipient = parts[-2].removeprefix(_RECIPIENT_PREFIX)

        constrain_words = self._section_words(message.constrain)
        if constrain_words:
            content_type = constrain_words[0]

        return HarmonyHeader(
            role=role,
            channel=channel,
            recipient=recipient,
            content_type=content_type,
        )

    def _section_words(self, section: HarmonySequence | None) -> list[str]:
        if section is None:
            return []
        return self._tokenizer.decode(section.token_ids).split()

    def iter_messages(self, token_ids: Sequence[int]) -> Iterator[HarmonyMessage]:
        """
        Parse given token ids into messages.

        Args:
            token_ids: A sequence of token ids to be parsed.

        Yields:
            Detected HarmonyMessages.
        """

        message_dict: dict[str, HarmonySequence] = {}
        section: str | None = None  # None indicates out-of-message.
        text_ids: list[int] = []
        text_start: int | None = None

        for token_position, token_id in enumerate(token_ids):
            if token_id in self._begin_map:
                if section is not None:
                    assert text_start is not None
                    message_dict[section] = HarmonySequence(
                        token_ids=text_ids,
                        start=text_start,
                    )
                section = self._begin_map[token_id]
                text_ids = []
                text_start = token_position + 1

            elif token_id in self._end_map:
                if section is not None:
                    assert text_start is not None
                    message_dict[section] = HarmonySequence(
                        token_ids=text_ids,
                        start=text_start,
                    )

                yield HarmonyMessage(**message_dict, end=self._end_map[token_id])

                message_dict = {}
                section = None
                text_ids = []
                text_start = None

            else:
                if section is not None:
                    text_ids.append(token_id)

        if section is not None:
            assert text_start is not None
            message_dict[section] = HarmonySequence(
                token_ids=text_ids,
                start=text_start,
            )
            yield HarmonyMessage(**message_dict, end=HarmonyMessageEndType.INCOMPLETE)

    def get_all_messages(self, token_ids: Sequence[int]) -> list[HarmonyMessage]:
        """
        Parse given token ids into messages.

        Args:
            token_ids: A sequence of token ids to be parsed.

        Returns:
            A list of detected HarmonyMessages.
        """
        return list(self.iter_messages(token_ids))

    def reverse_iter_messages(
        self, token_ids: Sequence[int]
    ) -> Iterator[HarmonyMessage]:
        """
        Parse given token ids into messages in reverse order.

        Args:
            token_ids: A sequence of token ids to be parsed.

        Yields:
            Detected HarmonyMessages in reverse order.
        """
        end_position = len(token_ids)

        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == self._start_id:
                yield next(self.iter_messages(token_ids[i:end_position]))
                end_position = i
