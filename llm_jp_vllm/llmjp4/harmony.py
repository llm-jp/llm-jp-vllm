# Generic parser for OpenAI Harmony format.
# This is basically identical with the bundled parser in LLM-jp-4 models,
# but typing annotation is modified to follow vLLM standards.

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Sequence

from vllm.tokenizers import TokenizerLike

# Tool calls are addressed to the "functions" namespace; other recipients
# (browser.*, python, ...) are built-in tools that must not be converted
# into OpenAI tool_calls.
_FUNCTION_NAMESPACE = "functions."


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
    # Position of the message's <|start|> in the parsed sequence: section
    # starts point just after their marker token; 0 when the message
    # opens the sequence without a <|start|>.
    start_position: int = field(init=False)

    def __post_init__(self) -> None:
        first_section = min(
            (
                section.start
                for section in (self.role, self.channel, self.constrain, self.content)
                if section is not None
            ),
            default=1,
        )
        object.__setattr__(self, "start_position", max(first_section - 1, 0))


@dataclass(frozen=True)
class HarmonyHeader:
    """Parsed header metadata of a Harmony message."""

    role: str | None = None
    channel: str | None = None
    recipient: str | None = None
    # Called "content type" by openai/harmony; named after the marker here.
    constrain: str | None = None
    # Classification of the message this header belongs to.
    kind: HarmonyMessageKind = field(init=False)

    def __post_init__(self) -> None:
        if self.recipient is not None and self.recipient.startswith(
            _FUNCTION_NAMESPACE
        ):
            kind = HarmonyMessageKind.TOOL_CALL
        elif self.channel == "analysis":
            kind = HarmonyMessageKind.REASONING
        elif self.recipient is not None and self.channel != "final":
            # Built-in tool recipients such as "python" or "browser.*".
            kind = HarmonyMessageKind.IGNORE
        else:
            kind = HarmonyMessageKind.CONTENT
        object.__setattr__(self, "kind", kind)


def function_name(header: HarmonyHeader) -> str:
    """Tool function name addressed by a TOOL_CALL message's recipient."""
    assert header.recipient is not None  # guaranteed by the TOOL_CALL kind
    return header.recipient[len(_FUNCTION_NAMESPACE) :]


def _header_from_words(
    role_words: list[str],
    channel_words: list[str],
    constrain_words: list[str],
) -> HarmonyHeader:
    """Build a header from the whitespace-split words of each section.

    Header sections carry space-separated words (the tokenizer preserves
    the spaces of the header text), so splitting on whitespace recovers
    them. Follows openai/harmony's ``parse_header_from_string``:
    recipient ("to=...") and content type are read from the tail of the
    header words, so a recipient is recognized on both sides of
    <|channel|> (the LLM-jp-4 template and official examples place it
    differently).
    """
    recipient_prefix = "to="
    role = role_words[0] if role_words else None
    channel = channel_words[0] if channel_words else None
    parts = role_words[1:] + channel_words[1:]

    recipient: str | None = None
    constrain: str | None = None
    if parts:
        last = parts[-1]
        if last.startswith(recipient_prefix):
            recipient = last[len(recipient_prefix) :]
        elif len(parts) == 1:
            # A single word that is not "to=..." is a bare recipient.
            recipient = last
        else:
            # e.g. "to=functions.x json": content type last, recipient before it
            constrain = last
            recipient = parts[-2].removeprefix(recipient_prefix)

    if constrain_words:
        constrain = constrain_words[0]

    return HarmonyHeader(
        role=role,
        channel=channel,
        recipient=recipient,
        constrain=constrain,
    )


@dataclass(frozen=True)
class HarmonyTextMessage:
    """A Harmony message lexed from decoded output text."""

    header: HarmonyHeader
    body: str
    start_offset: int  # Character offset of the message start in the text.


_MARKER_RE = re.compile(r"<\|(?:start|channel|constrain|message|end|return|call)\|>")

_END_MARKERS = frozenset({"<|end|>", "<|return|>", "<|call|>"})


def iter_text_messages(text: str) -> Iterator[HarmonyTextMessage]:
    """Lex decoded model output into Harmony messages.

    Applies the same rules as ``HarmonyStreamLexer``: <|start|> always
    delimits messages, a marker after <|message|> cannot reclassify a
    message, and a message whose header never reaches <|message|> is
    dropped. The text is assumed to continue the "<|start|>assistant"
    prefill, so the first message's role is pre-seeded.
    """
    sections: dict[str, str] = {"role": "assistant"}
    section: str | None = "role"
    header: HarmonyHeader | None = None
    body_parts: list[str] = []
    message_start = 0
    position = 0

    for match in _MARKER_RE.finditer(text):
        segment = text[position : match.start()]
        position = match.end()
        if header is not None:
            body_parts.append(segment)
        elif section is not None:
            sections[section] = sections.get(section, "") + segment

        marker = match.group()
        if marker == "<|start|>" or marker in _END_MARKERS:
            if header is not None:
                yield HarmonyTextMessage(header, "".join(body_parts), message_start)
            sections = {}
            header = None
            body_parts = []
            if marker == "<|start|>":
                # <|start|> always delimits messages; a missing end token
                # must not merge two messages.
                section = "role"
                message_start = match.start()
            else:
                section = None
                message_start = position
        elif header is not None:
            # A marker after <|message|> cannot reclassify a message
            # whose text was already collected.
            pass
        elif marker == "<|message|>":
            header = _header_from_words(
                sections.get("role", "").split(),
                sections.get("channel", "").split(),
                sections.get("constrain", "").split(),
            )
        else:
            section = "channel" if marker == "<|channel|>" else "constrain"
            sections[section] = ""

    if header is not None:
        body_parts.append(text[position:])
        yield HarmonyTextMessage(header, "".join(body_parts), message_start)


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
        """Parse the header sections of a message into structured metadata."""
        return _header_from_words(
            self._section_words(message.role),
            self._section_words(message.channel),
            self._section_words(message.constrain),
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


class HarmonyStreamLexer:
    """Streaming lexer for Harmony messages, modeled on openai/harmony's
    ``StreamableParser``: header tokens accumulate until <|message|>,
    where the parsed header is reported to the sink once; body text is
    then decoded incrementally and emitted as soon as it is complete.
    Message kinds are the sink's concern.

    Body decoding uses the two-offset scheme of vLLM's incremental
    detokenization: the decode window always starts where a previous
    clean decode ended, so byte-fallback runs are never split mid-run,
    and a delta ending in U+FFFD (a partially decoded multi-byte
    character) is withheld until it completes.
    """

    def __init__(
        self,
        parser: HarmonyMessageParser,
        sink: "HarmonyStreamParser",
        role_ids: Sequence[int],
    ):
        self._parser = parser
        self._tokenizer = parser._tokenizer
        self._sink = sink
        self._position: int = 0
        self._message_start: int = 0
        # Message-scoped state, reset at every message boundary. The
        # stream resumes after the "<|start|>assistant" prefill, so it
        # opens inside that message's role section, seeded with role_ids.
        self._section: str | None = "role"
        self._sections: dict[str, list[int]] = {"role": list(role_ids)}
        self._header: HarmonyHeader | None = None
        self._body_ids: list[int] = []
        self._prefix_offset: int = 0
        self._read_offset: int = 0
        self._prefix_text: str = ""

    def push(self, token_id: int) -> None:
        begin = self._parser._begin_map.get(token_id)
        if begin == "role":
            # <|start|> always delimits messages (as reverse_iter_messages
            # assumes); a missing end token must not merge two messages.
            self._finish_message()
        self._position += 1
        if begin is not None:
            if self._header is not None:
                # A marker after <|message|> cannot reclassify a message
                # whose text was already emitted.
                return
            if begin == "content":
                self._header = self._parser.parse_header(
                    HarmonyMessage(
                        end=HarmonyMessageEndType.INCOMPLETE,
                        **{
                            # Start positions are unused on this path.
                            name: HarmonySequence(token_ids=ids, start=0)
                            for name, ids in self._sections.items()
                        },
                    )
                )
                self._sink.on_message_begin(self._header, self._message_start)
            else:
                self._section = begin
                self._sections[begin] = []
        elif token_id in self._parser._end_map:
            self._finish_message()
        elif self._header is not None:
            self._push_body(token_id)
        elif self._section is not None:
            self._sections[self._section].append(token_id)

    def _push_body(self, token_id: int) -> None:
        self._body_ids.append(token_id)
        delta = self._pending_text()
        if delta and not delta.endswith("�"):
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._body_ids)
            self._prefix_text = self._tokenizer.decode(
                self._body_ids[self._prefix_offset :]
            )
            self._sink.on_body_delta(delta)

    def _pending_text(self) -> str:
        new_text = self._tokenizer.decode(self._body_ids[self._prefix_offset :])
        return new_text[len(self._prefix_text) :]

    def _finish_message(self) -> None:
        if self._header is not None:
            # A message that ends mid-character emits the partial
            # character now; nothing will complete it anymore.
            self._sink.on_message_end(self._pending_text())
        self._section = None
        self._sections = {}
        self._header = None
        self._body_ids = []
        self._prefix_offset = 0
        self._read_offset = 0
        self._prefix_text = ""
        self._message_start = self._position


class HarmonyStreamParser:
    """Routes the lexed message stream into reasoning/content text,
    tool calls and the handover ids, mirroring how vLLM's gpt-oss
    integration classifies ``StreamableParser`` output into segments.

    Re-parsing the cumulative sequence every step would be O(N^2) over
    a stream; advancing the lexer over the unseen tokens keeps each
    step O(delta).
    """

    def __init__(self, parser: HarmonyMessageParser, prefill_ids: Sequence[int]):
        self._parser = parser
        self._prefill_ids = list(prefill_ids)
        self._reset()

    def _reset(self) -> None:
        # The prefill's <|start|> already opened a message; the lexer is
        # seeded with the remaining ids as that message's role section.
        self._lexer = HarmonyStreamLexer(self._parser, self, self._prefill_ids[1:])
        self._consumed: int = 0
        self._ids: list[int] = []
        self._content_start: int | None = None
        # The open message's kind, and the pending newline that joins
        # messages like the gpt-oss parser (skipping empty ones).
        self._kind: HarmonyMessageKind | None = None
        self._separator: str = ""
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._reasoning_seen: bool = False
        self._content_seen: bool = False
        self.calls: list[tuple[str, str]] = []

    @property
    def consumed(self) -> int:
        """Number of stream tokens consumed so far."""
        return self._consumed

    @property
    def content_started(self) -> bool:
        """Whether a message ending the reasoning phase has begun."""
        return self._content_start is not None

    @property
    def content_ids(self) -> list[int]:
        """Tokens from the first non-reasoning message on, for the tool
        parser handover."""
        if self._content_start is None:
            return []
        return self._ids[self._content_start :]

    def advance(self, previous_len: int, current_ids: Sequence[int]) -> tuple[str, str]:
        """Consume the tokens beyond ``previous_len`` and return the new
        (reasoning, content) text."""
        if previous_len != self._consumed:
            # The caller broke the cumulative-stream contract (a fresh
            # request reusing this instance, or interleaved n>1 choices);
            # rebuilding once is cheaper than misclassifying.
            self._reset()
        new_ids = current_ids[self._consumed :]
        self._ids.extend(new_ids)
        for token_id in new_ids:
            self._lexer.push(token_id)
        self._consumed = len(current_ids)
        reasoning = "".join(self._reasoning_parts)
        content = "".join(self._content_parts)
        self._reasoning_parts.clear()
        self._content_parts.clear()
        return reasoning, content

    def on_message_begin(self, header: HarmonyHeader, start: int) -> None:
        self._kind = header.kind
        if header.kind is HarmonyMessageKind.TOOL_CALL:
            self.calls.append((function_name(header), ""))
        elif header.kind is HarmonyMessageKind.REASONING:
            self._separator = "\n" if self._reasoning_seen else ""
        elif header.kind is HarmonyMessageKind.CONTENT:
            self._separator = "\n" if self._content_seen else ""
        if (
            self._content_start is None
            and header.kind
            in (HarmonyMessageKind.CONTENT, HarmonyMessageKind.TOOL_CALL)
            # A headerless fragment must not end the reasoning phase,
            # mirroring the reasoning parser's _ends_reasoning.
            and (header.channel is not None or header.recipient is not None)
        ):
            self._content_start = start

    def on_body_delta(self, delta: str) -> None:
        if not delta:
            return
        if self._kind is HarmonyMessageKind.TOOL_CALL:
            name, arguments = self.calls[-1]
            self.calls[-1] = (name, arguments + delta)
        elif self._kind is HarmonyMessageKind.REASONING:
            self._reasoning_parts.append(self._separator + delta)
            self._separator = ""
            self._reasoning_seen = True
        elif self._kind is HarmonyMessageKind.CONTENT:
            self._content_parts.append(self._separator + delta)
            self._separator = ""
            self._content_seen = True

    def on_message_end(self, tail: str) -> None:
        self.on_body_delta(tail)
        if self._kind is HarmonyMessageKind.TOOL_CALL and not self.calls[-1][1]:
            # The serving layer flushes unstreamed arguments only onto an
            # existing tool-call delta, so a parameterless call must
            # stream its "{}" itself.
            self.calls[-1] = (self.calls[-1][0], "{}")
        self._kind = None
