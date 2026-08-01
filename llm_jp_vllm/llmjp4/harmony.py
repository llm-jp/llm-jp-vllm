# Generic parser for OpenAI Harmony format.
# Based on the parser bundled with LLM-jp-4 models, restructured after
# openai/harmony's StreamableParser state machine.

import re
from dataclasses import dataclass, field
from enum import Enum
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

    REASONING = 1
    CONTENT = 2
    TOOL_CALL = 3
    IGNORE = 4


@dataclass(frozen=True)
class HarmonyMessage:
    """A data class representing a message in the Harmony format.

    Each section holds the raw token ids of its span, e.g.
    "<|start|>assistant<|channel|>final<|message|>Hi<|end|>" fills
    role, channel and content with the ids of "assistant", "final"
    and "Hi".
    """

    end: HarmonyMessageEndType
    # Position of the marker that opened this message (normally its
    # <|start|>) in the parsed sequence.
    start_position: int
    role: list[int] | None = None
    channel: list[int] | None = None
    constrain: list[int] | None = None
    content: list[int] | None = None


@dataclass(frozen=True)
class HarmonyHeader:
    """Parsed header metadata of a Harmony message."""

    role: str | None = None
    channel: str | None = None
    recipient: str | None = None
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

    The recipient ("to=...") appears in either the role or the channel
    section::

        assistant to=functions.x<|channel|>commentary json   (chat template)
        assistant<|channel|>commentary to=functions.x json   (docs example)

    The role-side placement is not explicit in the harmony spec, but
    openai/harmony's parser tests cover it and the LLM-jp-4 chat template
    renders tool calls this way; see
    https://github.com/llm-jp/llm-jp-vllm/pull/6#discussion_r3655430969.

    Following upstream ``parse_header_from_string``, the role and channel
    values are stripped and recipient / constrain value are read from the
    tail of the remaining words, so one rule recognizes both shapes.
    """
    recipient_prefix = "to="
    role = role_words[0] if role_words else None
    channel = channel_words[0] if channel_words else None
    remaining_words = role_words[1:] + channel_words[1:]

    recipient: str | None = None
    constrain: str | None = None
    if remaining_words:
        last = remaining_words[-1]
        if last.startswith(recipient_prefix):
            # e.g. "to=functions.x": a recipient and no constrain value.
            recipient = last[len(recipient_prefix) :]
        elif len(remaining_words) == 1:
            # A single word that is not "to=..." is a bare recipient.
            recipient = last
        else:
            # e.g. "to=functions.x json": constrain value last, recipient
            # before it.
            constrain = last
            recipient = remaining_words[-2].removeprefix(recipient_prefix)

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

    Applies the same rules as ``HarmonyMessageParser.iter_messages``,
    except that a message whose header never reaches <|message|> is
    dropped. The text is assumed to continue the "<|start|>assistant"
    prefill, so the first message's role is pre-seeded.
    """
    sections: dict[str, str] = {"role": "assistant"}
    section: str | None = "role"
    header: HarmonyHeader | None = None
    body_parts: list[str] = []
    message_start = 0
    position = 0

    def parse_sections() -> HarmonyHeader:
        return _header_from_words(
            role_words=sections.get("role", "").split(),
            channel_words=sections.get("channel", "").split(),
            constrain_words=sections.get("constrain", "").split(),
        )

    for match in _MARKER_RE.finditer(text):
        segment = text[position : match.start()]
        marker = match.group()
        position = match.end()

        if header is not None:
            body_parts.append(segment)
            if marker in _END_MARKERS:
                yield HarmonyTextMessage(header, "".join(body_parts), message_start)
                sections = {}
                section = None
                header = None
                body_parts = []
            else:
                # Any marker after <|message|> is body text.
                body_parts.append(marker)
        elif section is not None:
            sections[section] = sections.get(section, "") + segment
            if marker in _END_MARKERS:
                # A message whose header never reached <|message|> has
                # no body to report.
                sections = {}
                section = None
            elif marker == "<|message|>":
                header = parse_sections()
            elif marker == "<|start|>":
                # <|start|> inside a header is header text.
                sections[section] += marker
            else:
                section = "channel" if marker == "<|channel|>" else "constrain"
                sections[section] = ""
        elif marker not in _END_MARKERS:
            # This marker opens the message; a stray end marker or text
            # between messages is dropped.
            message_start = match.start()
            if marker == "<|start|>":
                section = "role"
                sections = {"role": ""}
            elif marker == "<|message|>":
                header = parse_sections()
            else:
                section = "channel" if marker == "<|channel|>" else "constrain"
                sections = {section: ""}

    if header is not None:
        body_parts.append(text[position:])
        yield HarmonyTextMessage(header, "".join(body_parts), message_start)


@dataclass
class _ExpectStart:
    """Between messages, waiting for a marker that opens one."""


@dataclass
class _Header:
    """A message under construction: its start position, raw sections
    and the section now being collected ("content" after <|message|> in
    the batch parser; the stream lexer switches to _Content instead).
    """

    start: int
    sections: dict[str, list[int]]
    section: str


@dataclass
class _Content:
    """Streaming a body after <|message|>: the parsed header plus the
    incremental-decode window of ``HarmonyStreamLexer``.
    """

    header: HarmonyHeader
    body_ids: list[int] = field(default_factory=list)
    prefix_offset: int = 0
    read_offset: int = 0
    prefix_text: str = ""


class HarmonyMessageParser:
    """A parser that performs lexical analysis to extract Harmony messages.

    A message is a marker-delimited span such as::

        <|start|>assistant<|channel|>analysis<|message|>Reasoning<|end|>

    where the header sections (role, channel, constrain) precede the
    <|message|> body and <|end|> / <|return|> / <|call|> close the
    message.
    """

    def __init__(self, tokenizer: TokenizerLike):
        vocab = tokenizer.get_vocab()
        self._tokenizer = tokenizer
        self._start_id = vocab["<|start|>"]
        self._begin_map = {
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

        The tokenizer preserves the spaces of the header text, so
        decoding a section and splitting on whitespace recovers its
        space-separated words.
        """
        role_words, channel_words, constrain_words = (
            self._tokenizer.decode(section).split() if section else []
            for section in (message.role, message.channel, message.constrain)
        )
        return _header_from_words(
            role_words=role_words,
            channel_words=channel_words,
            constrain_words=constrain_words,
        )

    def iter_messages(self, token_ids: Sequence[int]) -> Iterator[HarmonyMessage]:
        """
        Parse given token ids into messages.

        Follows the state machine of openai/harmony's ``StreamableParser``
        (ExpectStart / Header / Content): a marker after <|message|> is
        body text, only an end marker closes a message, and tokens between
        messages are dropped. A header marker may open a message without
        <|start|>.

        Args:
            token_ids: A sequence of token ids to be parsed.

        Yields:
            Detected HarmonyMessages.
        """

        state: _ExpectStart | _Header = _ExpectStart()

        for token_position, token_id in enumerate(token_ids):
            begin = self._begin_map.get(token_id)

            if token_id in self._end_map:
                # A stray end token between messages yields nothing.
                if isinstance(state, _Header):
                    yield HarmonyMessage(
                        **state.sections,
                        end=self._end_map[token_id],
                        start_position=state.start,
                    )
                    state = _ExpectStart()

            elif isinstance(state, _Header):
                if state.section == "content":
                    # Any marker after <|message|> is body text.
                    state.sections["content"].append(token_id)
                elif begin is None:
                    # Text tokens and a nested <|start|> are header text.
                    state.sections[state.section].append(token_id)
                else:
                    state.section = begin
                    state.sections[begin] = []

            elif token_id == self._start_id:
                state = _Header(
                    start=token_position, sections={"role": []}, section="role"
                )

            elif begin is not None:
                # A section marker may open a message without <|start|>
                # so that a degraded sequence keeps its final answer.
                state = _Header(
                    start=token_position, sections={begin: []}, section=begin
                )

        if isinstance(state, _Header):
            yield HarmonyMessage(
                **state.sections,
                end=HarmonyMessageEndType.INCOMPLETE,
                start_position=state.start,
            )

    def get_all_messages(self, token_ids: Sequence[int]) -> list[HarmonyMessage]:
        """
        Parse given token ids into messages.

        Args:
            token_ids: A sequence of token ids to be parsed.

        Returns:
            A list of detected HarmonyMessages.
        """
        return list(self.iter_messages(token_ids))


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
        # The stream resumes after the "<|start|>assistant" prefill, so
        # it opens inside that message's role section, seeded with
        # role_ids.
        self._state: _ExpectStart | _Header | _Content = _Header(
            start=0, sections={"role": list(role_ids)}, section="role"
        )

    def push(self, token_id: int) -> None:
        state = self._state
        begin = self._parser._begin_map.get(token_id)
        self._position += 1

        if token_id in self._parser._end_map:
            # Between messages this is a stray end token and a no-op.
            self._finish_message()
        elif isinstance(state, _Content):
            # Any marker after <|message|> is body text.
            self._push_body(state, token_id)
        elif isinstance(state, _Header):
            if begin is None:
                # Text tokens and a nested <|start|> are header text.
                state.sections[state.section].append(token_id)
            elif begin == "content":
                self._begin_content(state)
            else:
                state.section = begin
                state.sections[begin] = []
        elif token_id == self._parser._start_id:
            self._state = _Header(
                start=self._position - 1, sections={"role": []}, section="role"
            )
        elif begin is not None:
            # A section marker may open a message without <|start|> so
            # that a degraded sequence keeps its final answer.
            opened = _Header(start=self._position - 1, sections={}, section=begin)
            if begin == "content":
                self._begin_content(opened)
            else:
                opened.sections[begin] = []
                self._state = opened

    def _begin_content(self, state: _Header) -> None:
        header = self._parser.parse_header(
            HarmonyMessage(
                end=HarmonyMessageEndType.INCOMPLETE,
                start_position=state.start,
                **state.sections,
            )
        )
        self._state = _Content(header=header)
        self._sink.on_message_begin(header, state.start)

    def _push_body(self, state: _Content, token_id: int) -> None:
        state.body_ids.append(token_id)
        delta = self._pending_text(state)
        if delta and not delta.endswith("�"):
            state.prefix_offset = state.read_offset
            state.read_offset = len(state.body_ids)
            state.prefix_text = self._tokenizer.decode(
                state.body_ids[state.prefix_offset :]
            )
            self._sink.on_body_delta(delta)

    def _pending_text(self, state: _Content) -> str:
        new_text = self._tokenizer.decode(state.body_ids[state.prefix_offset :])
        return new_text[len(state.prefix_text) :]

    def _finish_message(self) -> None:
        if isinstance(self._state, _Content):
            # A message that ends mid-character emits the partial
            # character now; nothing will complete it anymore.
            self._sink.on_message_end(self._pending_text(self._state))
        self._state = _ExpectStart()


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
