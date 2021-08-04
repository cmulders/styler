import codecs
import copy
import enum
import io
import itertools
import logging
import re
import string
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import dropwhile, tee
from typing import (
    IO,
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

# CSS Syntax Module Level 3
# Ref: https://www.w3.org/TR/css-syntax-3/


logger = logging.getLogger()

# logging.basicConfig(level=logging.NOTSET)

TValue = TypeVar("TValue")


class NoDefault(enum.Enum):
    # We make this an Enum
    # 1) because it round-trips through pickle correctly (see GH#40397)
    # 2) because mypy does not understand singletons
    no_default = "NO_DEFAULT"


class PeekableIterable(Iterable[TValue]):
    """Wraps an iterator to make it peekable

    Uses itertools.tee() internally.

    """

    def __init__(self, it: Iterable[TValue]) -> None:
        # Make a tee-wrapped Iterator
        self.parent_it = itertools.tee(it, 1)[0]

        # Variables for reconsumption
        self._last: Union[TValue, NoDefault] = NoDefault.no_default
        self._reconsume: Union[TValue, NoDefault] = NoDefault.no_default

    def peek(self, items: int = 1) -> Optional[Union[TValue, Tuple[TValue, ...]]]:
        if items <= 0:
            raise ValueError("Number of items should be a positive integer.")

        peeked = self.peek_slice(slice(None, items))

        if items == 1:
            return next(peeked, None)
        else:
            return tuple(peeked)

    def peek_slice(self, s: slice) -> Iterator[TValue]:
        assert isinstance(s, slice), "Slice required."

        # We make a copy of the tee-wrapped iterator
        child_it = copy.copy(self.parent_it)
        return itertools.islice(child_it, s.start, s.stop, s.step)

    def reconsume(self):
        self._reconsume = self._last

    def __iter__(self):
        return self

    def __next__(self) -> TValue:
        if self._reconsume is not NoDefault.no_default:
            value = self._reconsume
        else:
            value = next(self.parent_it)

        self._last = value
        self._reconsume = NoDefault.no_default
        return value


class ParseError(Exception):
    pass


def peek_buffered(buffer: Union[io.BufferedReader, io.BufferedIOBase], size: int):
    if not isinstance(buffer, io.BufferedReader):
        assert buffer.seekable, "Need a seekable buffer"

        peek = buffer.read(size)
        # Rewind
        buffer.seek(-size, io.SEEK_CUR)
    else:
        peek = buffer.peek(size)

    return peek


def detect_bom(
    buffer: Union[io.BufferedReader, io.BufferedIOBase]
) -> Optional[codecs.CodecInfo]:
    sample = peek_buffered(buffer, 3)

    if sample.startswith(codecs.BOM_UTF8):
        # Use UTF-8-sig to skip the BOM during reading
        buffer.read(3)  # Advance 3 bytes
        return codecs.lookup("utf-8")

    elif sample.startswith(codecs.BOM_UTF16_BE):
        buffer.read(2)  # Advance 2 bytes
        return codecs.lookup("utf-16-be")

    elif sample.startswith(codecs.BOM_UTF16_LE):
        buffer.read(2)  # Advance 2 bytes
        return codecs.lookup("utf-16-le")

    return None


def detect_charset(sample: bytes) -> Optional[codecs.CodecInfo]:

    prefix = '@charset "'.encode("ascii")
    suffix = '";'.encode("ascii")

    prefix_pos = sample.find(prefix)
    if prefix_pos == -1:
        return None
    suffix_pos = sample.find(suffix, prefix_pos)
    if suffix_pos == -1:
        return None

    charset_value = sample[
        prefix_pos + len(prefix) : suffix_pos
    ]  # The value of @charset "<value>";
    encoding = charset_value.decode("ascii")  # We need to assume ASCII compatible bytes

    try:
        codec_info = codecs.lookup(encoding)
    except LookupError:
        warnings.warn(f"Ignored @charset encoding, invalid encoding: {encoding}.")
        # Could not detect specified encoding in the @charset
        return None

    if codec_info.name in ("utf-16-be", "utf-16-le"):
        # According to the spec, UTF-16 could not be detected relaibly, and UTF-8 is a decent fallback in these cases
        codec_info = codecs.lookup("utf-8")

    return codec_info


def decode(
    stream: Union[io.RawIOBase, io.BufferedIOBase, io.TextIOBase, IO],
    protocol_encoding=None,
    environment_encoding=None,
) -> io.TextIOBase:
    """Decodes a stream
    If the stream is a TextIOBase, it is assumed to be decoded, and returned unchanged.

    For byte streams the encoding is detected in the following order:
    1. BOM marker
    2. Protocol encoding
    3. @charset rule
    4. Environment encoding
    5. Assume UTF-8

    Ref: https://www.w3.org/TR/css-syntax-3/#input-byte-stream
    """
    if isinstance(stream, io.TextIOBase):
        return stream

    buffered: Union[io.BufferedReader, io.BufferedIOBase]
    if isinstance(stream, io.RawIOBase):
        buffered = io.BufferedReader(stream)
    elif isinstance(stream, (io.BufferedReader, io.BufferedIOBase)):
        buffered = stream
    else:
        raise ValueError(f"Unexpected steam: {stream!r}")

    # 1. BOM marker
    codec_info = detect_bom(buffered)

    if not codec_info and protocol_encoding:
        # 2. Protocol encoding
        try:
            codec_info = codecs.lookup(protocol_encoding)
        except LookupError:
            warnings.warn(
                f"Ignored protocol encoding, invalid encoding: {protocol_encoding}."
            )
            pass

    if not codec_info:
        # 3. @charset rule
        sample = peek_buffered(buffered, 1024)
        codec_info = detect_charset(sample)

    if not codec_info and environment_encoding:
        # 4. Environment encoding
        try:
            codec_info = codecs.lookup(environment_encoding)
        except LookupError:
            warnings.warn(
                f"Ignored environment encoding, invalid encoding: {environment_encoding}."
            )
            pass

    if not codec_info:
        # 5. Default to UTF-8
        codec_info = codecs.lookup("utf-8")

    # Decoding and Preprocessing
    # Errors during decoding will be replaced by the U+FFFD REPLACEMENT CHARACTER (�)
    #
    # Wrapping the stream in TextIOWrapper will ensure the line endings will be normalized
    # to single U+000A LINE FEED (LF) code point
    # Ref: https://www.w3.org/TR/css-syntax-3/#input-preprocessing
    return io.TextIOWrapper(
        cast(BinaryIO, buffered), encoding=codec_info.name, errors="replace"
    )


# region tokens

# Ref: https://www.w3.org/TR/css-syntax-3/#tokenization
@dataclass(frozen=True)
class Token:
    position: int


class EOF(Token):
    pass


class Comma(Token):
    pass


class Colon(Token):
    pass


class Semicolon(Token):
    pass


class CDO(Token):
    # <!--
    pass


class CDC(Token):
    # -->
    pass


class BlockToken(Token):
    pass


class OpenBlockToken(BlockToken):
    @property
    @abstractmethod
    def matching(self) -> Type["CloseBlockToken"]:
        ...


class CloseBlockToken(BlockToken):
    @property
    @abstractmethod
    def matching(self) -> Type["OpenBlockToken"]:
        ...


class OpenParenthesis(OpenBlockToken):
    @property
    def matching(self):
        return CloseParenthesis


class CloseParenthesis(CloseBlockToken):
    @property
    def matching(self):
        return OpenParenthesis


class OpenCurlyBracket(OpenBlockToken):
    @property
    def matching(self):
        return CloseCurlyBracket


class CloseCurlyBracket(CloseBlockToken):
    @property
    def matching(self):
        return OpenCurlyBracket


class OpenBracket(OpenBlockToken):
    @property
    def matching(self):
        return CloseBracket


class CloseBracket(CloseBlockToken):
    @property
    def matching(self):
        return OpenBracket


@dataclass(frozen=True)
class ValueToken(Token):
    value: str


class Whitespace(ValueToken):
    pass


class Comment(Whitespace):
    pass


class Delim(ValueToken):
    pass


class Ident(ValueToken):
    pass


class String(ValueToken):
    pass


class BadString(Token):
    pass


class HashType(enum.Enum):
    UNRESTRICTED = enum.auto()
    ID = enum.auto()


@dataclass(frozen=True)
class Hash(ValueToken):
    hash_type: HashType = HashType.UNRESTRICTED


class Function(ValueToken):
    pass


class Url(ValueToken):
    pass


class AtKeyword(ValueToken):
    pass


class BadUrl(Token):
    pass


@dataclass(frozen=True)
class Percentage(Token):
    value: float


class NumericType(enum.Enum):
    INTEGER = enum.auto()
    NUMBER = enum.auto()


@dataclass(frozen=True)
class Number(Token):
    value: Union[int, float]
    numeric_type: NumericType


@dataclass(frozen=True)
class Dimension(Number):
    dimension: str


# endregion

# Ref: https://www.w3.org/TR/css-syntax-3/#whitespace
WHITESPACE = string.whitespace

# Ref: https://www.w3.org/TR/css-syntax-3/#non-printable-code-point
NON_PRINTABLE_CODE_POINT = (
    # U+0000 NULL and U+0008 BACKSPACE inclusive
    "\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    # U+000B LINE TABULATION
    "\x0b"
    # U+000E SHIFT OUT and U+001F INFORMATION SEPARATOR ONE inclusive
    "\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    # U+007F DELETE
    "\x7f"
)

T = TypeVar("T")


class Tokenizer:
    def __init__(self, content: str) -> None:
        self.content = content
        self.pos = 0
        self.exhausted = False
        self.errors: List[str] = []

    @classmethod
    def from_reader(cls, reader: io.TextIOBase):
        return cls(reader.read())

    @classmethod
    def from_str(cls, content: str):
        return cls(content)

    @contextmanager
    def memo(self):
        pos = self.pos
        try:
            yield
        finally:
            self.pos = pos

    def peek(self, n: int = 1) -> str:
        # Returns the next N characters without modifying the position
        return self.content[self.pos : self.pos + n]

    def advance(self, n: int) -> int:
        # Increase the position
        self.pos += n
        return self.pos

    def consume(self, n: int) -> str:
        # Returns the next N characters and increases the position
        start = self.pos
        end = self.advance(n)
        return self.content[start:end]

    @property
    def eof(self) -> bool:
        return self.pos >= len(self.content)

    def __iter__(self):
        return self

    def __next__(self) -> Token:
        if self.exhausted:
            raise StopIteration

        token = self.next_token()
        if isinstance(token, EOF):
            self.exhausted = True

        return token

    def next_token(self) -> Token:
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-token
        c = self.peek(1)
        pos = self.pos
        if c == "":
            # Empty, so EOF
            return EOF(pos)
        elif c == "/":
            if self.peek(2) == "/*":
                return self.consume_comment()
        elif c in WHITESPACE:
            return self.consume_whitespace()
        elif c in '"':
            return self.consume_string()
        elif c in "#":
            with self.memo():
                self.advance(1)
                is_ident = self.at_ident_start()
                is_name = is_ident or self.is_name_char(self.peek(1))
                is_eof = self.eof
            if is_ident:
                return self.consume_hash(HashType.ID)
            elif not is_eof and is_name:
                return self.consume_hash(HashType.UNRESTRICTED)
        elif c in "'":
            return self.consume_string()
        elif c == "(":
            self.advance(1)
            return OpenParenthesis(pos)
        elif c == ")":
            self.advance(1)
            return CloseParenthesis(pos)
        elif c == "+":
            if self.at_number_start():
                return self.consume_numeric()
        elif c == ",":
            self.advance(1)
            return Comma(pos)
        elif c == "-":
            if self.at_number_start():
                return self.consume_numeric()
            if self.peek(3) == "-->":
                self.advance(3)
                return CDC(pos)
            if self.at_ident_start():
                return self.consume_ident_like()
        elif c == ".":
            if self.at_number_start():
                return self.consume_numeric()
        elif c == ":":
            self.advance(1)
            return Colon(pos)
        elif c == ";":
            self.advance(1)
            return Semicolon(pos)
        elif c == "<":
            if self.peek(4) == "<!--":
                self.advance(4)
                return CDO(pos)
        elif c == "@":
            with self.memo():
                self.advance(1)
                is_ident = self.at_ident_start()
            if is_ident:
                self.advance(1)
                name = self.consume_name()
                return AtKeyword(pos, name)
        elif c == "[":
            self.advance(1)
            return OpenBracket(pos)
        elif c == "\\":
            if self.at_escape():
                return self.consume_ident_like()
            self.errors.append("Invalid escape.")
        elif c == "]":
            self.advance(1)
            return CloseBracket(pos)
        elif c == "{":
            self.advance(1)
            return OpenCurlyBracket(pos)
        elif c == "}":
            self.advance(1)
            return CloseCurlyBracket(pos)
        elif c in string.digits:
            return self.consume_numeric()
        elif self.is_name_start(c):
            return self.consume_ident_like()

        # If not anything above, default to delim token
        return Delim(pos, self.consume(1))

    def at_escape(self) -> bool:
        peeked = self.peek(2)
        if len(peeked) != 2:
            return False
        return self.is_escape(peeked)

    @classmethod
    def is_escape(cls, chars: str) -> bool:
        assert len(chars) >= 2

        if chars[0] != "\\":
            return False
        elif chars[1] == "\n":
            return False

        return True

    @classmethod
    def is_non_printable(cls, c: str) -> bool:
        assert len(c) == 1
        return c in NON_PRINTABLE_CODE_POINT

    @classmethod
    def is_whitespace(cls, c: str) -> bool:
        assert len(c) == 1
        return c in WHITESPACE

    @classmethod
    def is_name_char(cls, c: str) -> bool:
        if c == "":
            return False
        assert len(c) == 1
        return cls.is_name_start(c) or c in string.digits + "-"

    @classmethod
    def is_name_start(cls, c: str) -> bool:
        assert len(c) == 1
        return c in string.ascii_letters + "_" or c > "\u0080"

    def at_number_start(self) -> bool:
        # Ref: https://www.w3.org/TR/css-syntax-3/#check-if-three-code-points-would-start-a-number
        c = self.peek(3)

        if c[0] in "+-":
            if len(c) < 2:
                return False
            elif c[1] in string.digits:
                return True
            elif c[1] == "." and len(c) == 3:
                return c[2] in string.digits

            return False
        elif c[0] == ".":
            if len(c) < 2:
                return False
            else:
                return c[1] in string.digits

        elif c[0] in string.digits:
            return True

        return False

    def at_ident_start(self) -> bool:
        # Ref: https://www.w3.org/TR/css-syntax-3/#would-start-an-identifier
        c = self.peek(3)

        if len(c) < 2:
            return False

        if c[0] == "-":
            c = c[1:]  # Shift the code points left

        if self.is_name_start(c[0]) or c[0] == "-":
            return True
        elif len(c) >= 2 and self.is_escape(c[:2]):
            return True
        else:
            return False

    def consume_comment(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-comments
        start = self.pos
        self.advance(2)  # Consume the /*

        value = ""
        while not self.eof and self.peek(2) != "*/":
            value += self.consume(1)

        if self.peek(2) == "*/":
            self.advance(2)
        else:
            self.errors.append("Comment did not close.")

        return Comment(start, value)

    def consume_whitespace(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#whitespace
        start = self.pos

        value = ""
        while not self.eof and self.is_whitespace(self.peek(1)):
            value += self.consume(1)

        return Whitespace(start, value)

    def consume_string(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-string-token
        ending = self.consume(1)
        start = self.pos

        value = ""
        while not self.eof and self.peek(1) != ending:
            if self.at_escape():
                value += self.consume_escape()
            elif self.peek(1) == "\n":
                self.errors.append(f"String contains newline.")
                return BadString(start)
            else:
                value += self.consume(1)

        if self.eof or self.peek(1) != ending:
            self.errors.append(f"String not ended with matching `{ending}`.")
            return String(start, value)
        else:
            self.advance(1)
            return String(start, value)

    def consume_escape(self) -> str:
        assert self.consume(1) == "\\"

        if self.eof:
            self.errors.append(f"Unexpected EOF while parsing an escape.")
            return "\N{REPLACEMENT CHARACTER}"

        if self.peek(1) not in string.hexdigits:
            # Return code point, as it is not a hex digit
            return self.consume(1)

        hex_chars = ""
        while not self.eof and self.peek(1) in string.hexdigits and len(hex_chars) < 6:
            hex_chars += self.consume(1)

        if not self.eof and self.is_whitespace(self.peek(1)):
            # If the next token is whitespace, consume it
            self.advance(1)

        codepoint = chr(int(hex_chars, 16))
        # Check valid unicode range and it is not a surrogate (https://infra.spec.whatwg.org/#surrogate)
        if "\u0000" < codepoint <= "\U0010ffff" and not (
            "\uD800" <= codepoint <= "\uDFFF"
        ):
            return codepoint
        else:
            # Invalid range, use fallback
            return "\N{REPLACEMENT CHARACTER}"

    def consume_name(self) -> str:
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-name
        name = ""
        while not self.eof and self.is_name_char(self.peek(1)) or self.at_escape():
            if self.at_escape():
                name += self.consume_escape()
            else:
                name += self.consume(1)

        assert name, "Did not parse a name."

        return name

    def consume_hash(self, hash_type: HashType):
        start = self.pos
        assert self.consume(1) == "#"

        return Hash(start, self.consume_name(), hash_type)

    def consume_number(self) -> Tuple[Union[int, float], NumericType]:
        n_type = NumericType.INTEGER
        _repr = ""
        value = 0

        def _consume_digits() -> Iterable[str]:
            while not self.eof and self.peek(1) in string.digits:
                yield self.consume(1)

            return ""

        if self.peek(1) in "+-":
            _repr += self.consume(1)

        _repr += "".join(_consume_digits())

        p2 = self.peek(2)
        if len(p2) == 2 and p2[0] == "." and p2[1] in string.digits:
            n_type = NumericType.NUMBER
            _repr += self.consume(2)
            _repr += "".join(_consume_digits())

        p3 = self.peek(3)
        if len(p3) >= 2 and p3[0] in "eE":
            signed = len(p3) == 3 and p3[1] in "+-" and p3[2] in string.digits
            if signed or p3[1] in string.digits:
                n_type = NumericType.NUMBER
                _repr += self.consume(2)
                _repr += "".join(_consume_digits())

        # Ref: https://www.w3.org/TR/css-syntax-3/#convert-a-string-to-a-number
        components = re.fullmatch(
            r"""
            (?P<sign>[+-]?)
            (?P<integer>\d*)
            (?P<decimal>\.?)
            (?P<fractional>\d*)
            (?P<expind>[eE]?)
            (?P<expsign>[+-]?)
            (?P<exponent>\d*)
            """,
            _repr,
            re.VERBOSE,
        )
        if components is None:
            raise ValueError("Not a number")
        s = -1 if components.group("sign") == "-" else 1
        i = int(components.group("integer") or "0")
        f = int(components.group("fractional") or "0")
        d = len(components.group("fractional"))
        t = -1 if components.group("expsign") == "-" else 1
        e = int(components.group("exponent") or "0")
        value = s * (i + f * pow(10, -d)) * pow(10, t * e)

        return value, n_type

    def consume_numeric(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-a-numeric-token
        start = self.pos
        value, n_type = self.consume_number()

        if self.at_ident_start():
            unit = self.consume_name()
            return Dimension(start, value, n_type, unit)

        if self.peek(1) == "%":
            self.advance(1)
            return Percentage(start, value)

        return Number(start, value, n_type)

    def consume_ident_like(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-an-ident-like-token
        pos = self.pos
        name = self.consume_name()

        if name.lower() == "url" and self.peek(1) == "(":
            self.advance(1)
            while not self.eof and all(self.is_whitespace(c) for c in self.peek(2)):
                self.advance(1)
            chrs = self.peek(2)
            if any(c in "\"'" for c in chrs):
                return Function(pos, name)
            else:
                return self.consume_url(pos)

        elif self.peek(1) == "(":
            self.advance(1)
            return Function(pos, name)
        else:
            return Ident(pos, name)

    def consume_bad_url_remnants(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-remnants-of-bad-url
        while not self.eof and self.peek(1) != ")":
            if self.at_escape():
                self.consume_escape()
            else:
                self.consume(1)

        if self.peek(1) == ")":
            self.consume(1)

        return

    def consume_url(self, pos: int = None):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-url-token
        assert pos is not None, "Must pass the start of the url(."
        value = ""

        while not self.eof and self.is_whitespace(self.peek(1)):
            self.advance(1)

        while not self.eof:
            if self.peek(1) == "\\":
                if self.at_escape():
                    value += self.consume_escape()
                else:
                    self.errors.append("Invalid escape while parsing Url")
                    self.consume_bad_url_remnants()
                    return BadUrl(pos)

            c = self.consume(1)

            if c == ")":
                return Url(pos, value)
            elif self.is_whitespace(c):
                # Whitespace is not allowed whithin a url, so parse the rest untill ')'
                while not self.eof and self.is_whitespace(self.peek(1)):
                    self.advance(1)

                if self.eof:
                    self.errors.append("Unexpected EOF while parsing Url.")
                    return Url(pos, value)
                elif self.peek(1) == ")":
                    self.advance(1)
                    return Url(pos, value)

                self.consume_bad_url_remnants()
                return BadUrl(pos)

            elif c in "\"'(" or self.is_non_printable(c):
                self.errors.append(f"Url contains invalid character: {c}.")
                self.consume_bad_url_remnants()
                return BadUrl(pos)
            else:
                # Append valid character to url
                value += c

        self.errors.append("Unexpected EOF while parsing Url.")
        return Url(pos, value)


# region ast
@dataclass(frozen=True)
class CssAst:
    pass


@dataclass(frozen=True)
class BlockAst(CssAst):
    block: BlockToken
    content: List[Token]


@dataclass(frozen=True)
class FunctionAst(CssAst):
    name: str
    value: List[Token]


@dataclass(frozen=True)
class AtRuleAst(CssAst):
    name: str
    prelude: List[Union[BlockAst, FunctionAst, Token]] = field(default_factory=list)
    block: Optional[BlockAst] = None


@dataclass(frozen=True)
class QualifiedRuleAst(CssAst):
    prelude: List[Union[BlockAst, FunctionAst, Token]] = field(default_factory=list)
    block: Optional[BlockAst] = None


RuleAst = Union[AtRuleAst, QualifiedRuleAst]


@dataclass(frozen=True)
class DeclarationAst(CssAst):
    name: str
    value: List[Token]
    important: bool = False


@dataclass(frozen=True)
class StyleSheetAst(CssAst):
    rules: List[RuleAst]


class ASTSyntaxError(Exception):
    pass


# endregion


class Parser:
    def __init__(self, token_stream: Iterable[Token], include_whitespace=False) -> None:
        if not include_whitespace:
            non_whitespace = lambda tok: not isinstance(tok, Whitespace)
            token_stream = filter(non_whitespace, token_stream)

        self.tokens = PeekableIterable(token_stream)

        self.errors: List[str] = []

    def skip_whitespace(self):
        while isinstance(self.tokens.peek(), Whitespace):
            next(self.tokens)

    def parse_css_grammar(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-grammar
        raise NotImplementedError

    def parse_stylesheet(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-stylesheet
        return StyleSheetAst(self.consume_rule_list(top_level=True))

    def parse_rule_list(self) -> List[RuleAst]:
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-list-of-rules
        return self.consume_rule_list(top_level=False)

    def parse_rule(self) -> RuleAst:
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-rule
        self.skip_whitespace()
        token = self.tokens.peek()
        if isinstance(token, EOF):
            raise ASTSyntaxError("Unexpected EOF.")

        if isinstance(token, AtKeyword):
            rule = self.consume_at_rule()
        else:
            rule = self.consume_qualified_rule()

        self.skip_whitespace()
        last = self.tokens.peek()
        if not isinstance(last, EOF) and last is not None:
            raise ASTSyntaxError(f"Expected EOF, but got {self.tokens.peek()!r}.")

        return rule

    def parse_declaration(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-declaration
        self.skip_whitespace()

        if not isinstance(self.tokens.peek(), Ident):
            raise ASTSyntaxError(f"Expected Ident, but got {self.tokens.peek()!r}.")

        return self.consume_declaration()

    def parse_declaration_list(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-list-of-declarations
        return self.consume_declaration_list()

    def parse_component_value(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-component-value
        self.skip_whitespace()
        if isinstance(self.tokens.peek(), EOF):
            raise ASTSyntaxError(f"Unexpected EOF.")

        cpv = self.consume_component_value()

        self.skip_whitespace()
        last = self.tokens.peek()
        if not isinstance(last, EOF) and last is not None:
            raise ASTSyntaxError(f"Expected EOF, but got {self.tokens.peek()!r}.")

        return cpv

    def parse_component_value_list(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#parse-list-of-component-values
        cpv_list = []
        for token in self.tokens:
            if isinstance(token, EOF):
                continue
            self.tokens.reconsume()
            cpv = self.consume_component_value()
            cpv_list.append(cpv)

        return cpv_list

    def parse_component_value_list_comma_separated(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/parse-comma-separated-list-of-component-values
        cvls = []
        cvl_current = []
        for token in self.tokens:
            if isinstance(token, EOF):
                continue

            elif isinstance(token, Comma):
                cvls.append(cvl_current)
                cvl_current = []
            else:
                self.tokens.reconsume()
                cpv = self.consume_component_value()
                cvl_current.append(cpv)

        if cvl_current:
            cvls.append(cvl_current)
        return cvls

    def consume_rule_list(self, top_level: bool):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-list-of-rules
        rule_list: List[RuleAst] = []

        for token in self.tokens:
            if isinstance(token, Whitespace):
                continue

            elif isinstance(token, EOF):
                break

            elif isinstance(token, (CDO, CDC)):
                if top_level:
                    continue
                self.tokens.reconsume()
                rule = self.consume_qualified_rule()
                if rule:
                    rule_list.append(rule)

            elif isinstance(token, AtKeyword):
                self.tokens.reconsume()
                rule = self.consume_at_rule()
                if rule:
                    rule_list.append(rule)

            else:
                self.tokens.reconsume()
                rule = self.consume_qualified_rule()
                if rule:
                    rule_list.append(rule)

        return rule_list

    def consume_at_rule(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-at-rule
        name = next(self.tokens)
        assert isinstance(name, AtKeyword)
        prelude = []
        block = None

        for token in self.tokens:
            if isinstance(token, EOF):
                self.errors.append("Unexpected EOF while consuming a at-rule.")
                break
            elif isinstance(token, Semicolon):
                break
            elif isinstance(token, OpenCurlyBracket):
                block = self.consume_simple_block(token)
                break
            else:
                self.tokens.reconsume()
                prelude.append(self.consume_component_value())

        return AtRuleAst(
            name.value,  # Should be a name
            prelude,
            block,
        )

    def consume_qualified_rule(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-qualified-rule
        prelude = []
        block = None
        for token in self.tokens:
            if isinstance(token, EOF):
                self.errors.append("Unexpected EOF while consuming a qualified rule.")
                break
            elif isinstance(token, OpenCurlyBracket):
                block = self.consume_simple_block(token)
                break
            else:
                self.tokens.reconsume()
                prelude.append(self.consume_component_value())

        return QualifiedRuleAst(prelude, block)

    def consume_declaration_list(self) -> List[Union[DeclarationAst, AtRuleAst]]:
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-list-of-declarations
        decl_list = []

        for token in self.tokens:
            if isinstance(token, (Whitespace, Semicolon)):
                continue
            elif isinstance(token, EOF):
                break
            elif isinstance(token, AtKeyword):
                self.tokens.reconsume()
                decl_list.append(self.consume_at_rule())
            elif isinstance(token, Ident):
                temp_tokens: List[Token] = [token]
                while not isinstance(self.tokens.peek(), (EOF, Semicolon)):
                    temp_tokens.append(next(self.tokens))
                temp_parser = type(self)(temp_tokens)
                decl = temp_parser.consume_declaration()
                if decl:
                    decl_list.append(decl)
                self.errors.extend(temp_parser.errors)
            else:
                # Invalid declaration, consume until next.
                self.errors.append(f"Invalid declaration.")
                while not isinstance(self.tokens.peek(), (EOF, Semicolon)):
                    # Throw away value
                    _ = self.consume_component_value()

        return decl_list

    def consume_declaration(self) -> Optional[DeclarationAst]:
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-declaration
        name = next(self.tokens)
        assert isinstance(name, Ident)
        value = []
        important = False

        self.skip_whitespace()

        if not isinstance(self.tokens.peek(), Colon):
            self.errors.append(f"Expected a colon, got {self.tokens.peek()!r}.")
            return None
        # Consume the Colon
        next(self.tokens)

        self.skip_whitespace()

        for token in self.tokens:
            if isinstance(token, EOF):
                break
            self.tokens.reconsume()
            value.append(self.consume_component_value())

        if len(value) >= 2:
            prev2, prev1 = value[-2:]
            is_exclamation = isinstance(prev2, Delim) and prev2.value == "!"
            is_important = (
                isinstance(prev1, Ident) and prev1.value.lower() == "important"
            )
            if is_exclamation and is_important:
                important = True
                value = value[:-2]

        self.skip_whitespace()

        return DeclarationAst(name.value, value, important)

    def consume_component_value(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-component-value
        token = next(self.tokens)

        if isinstance(token, OpenBlockToken):
            return self.consume_simple_block(token)
        elif isinstance(token, Function):
            self.tokens.reconsume()
            return self.consume_function()
        else:
            return token

    def consume_simple_block(self, block_open: OpenBlockToken):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-simple-block
        MatchingType = block_open.matching
        content = []
        for token in self.tokens:
            if isinstance(token, Whitespace):
                continue
            elif isinstance(token, EOF):
                self.errors.append("Unexpected EOF while consuming a block.")
                break
            elif isinstance(token, MatchingType):
                break
            else:
                self.tokens.reconsume()
                content.append(self.consume_component_value())

        return BlockAst(block_open, content)

    def consume_function(self):
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-function
        fn_tok = next(self.tokens)
        assert isinstance(fn_tok, Function)
        value = []

        for token in self.tokens:
            if isinstance(token, CloseParenthesis):
                break
            elif isinstance(token, EOF):
                self.errors.append("Unexpected EOF while consuming a function.")
            else:
                self.tokens.reconsume()
                value.append(self.consume_component_value())

        return FunctionAst(
            fn_tok.value,
            value,
        )
