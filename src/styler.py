from contextlib import contextmanager
import codecs
import enum
import re
from typing import (
    BinaryIO,
    Callable,
    Iterable,
    List,
    TextIO,
    TypeVar,
    Union,
    cast,
    Tuple,
    Optional,
)
import codecs

import io
import string

# CSS Syntax Module Level 3
# Ref: https://www.w3.org/TR/css-syntax-3/

import logging
from unittest.loader import TestLoader
from itertools import tee, takewhile
from dataclasses import dataclass

logger = logging.getLogger()

# logging.basicConfig(level=logging.NOTSET)


class ParseError(Exception):
    pass


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
        logger.warning(f"Ignored @charset encoding, invalid encoding: {encoding}.")
        # Could not detect specified encoding in the @charset
        return None

    if codec_info.name in ("utf-16-be", "utf-16-le"):
        # According to the spec, UTF-16 could not be detected relaibly, and UTF-8 is a decent fallback in these cases
        codec_info = codecs.lookup("utf-8")

    return codec_info


def decode(
    stream: Union[io.RawIOBase, io.BufferedIOBase, io.TextIOBase],
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
    else:
        buffered = stream

    # 1. BOM marker
    codec_info = detect_bom(buffered)

    if not codec_info and protocol_encoding:
        # 2. Protocol encoding
        try:
            codec_info = codecs.lookup(protocol_encoding)
        except LookupError:
            logger.warning(
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
            logger.warning(
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


class OpenParenthesis(Token):
    pass


class CloseParenthesis(Token):
    pass


class OpenCurlyBracket(Token):
    pass


class CloseCurlyBracket(Token):
    pass


class OpenBracket(Token):
    pass


class CloseBracket(Token):
    pass


@dataclass(frozen=True)
class ValueToken(Token):
    value: str


class Comment(ValueToken):
    pass


class Whitespace(ValueToken):
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

    @property
    def remaining(self):
        return len(self.content) - self.pos

    @contextmanager
    def memo(self):
        pos = self.pos
        try:
            yield
        finally:
            self.pos = pos

    def with_memo(self, fn: Callable[["Tokenizer"], T]) -> T:
        with self.memo():
            return fn(self)

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

    def consume_all(self) -> str:
        start = self.pos
        self.pos = len(self.content)  # Set to end
        return self.content[start:]

    def content_iter(self) -> Iterable[str]:
        return iter(self.content[self.pos :])

    @property
    def eof(self) -> bool:
        return self.pos >= len(self.content)

    def __iter__(self):
        return self

    def __next__(self) -> Token:
        if self.exhausted:
            raise StopIteration

        if self.eof:
            self.exhausted = True
            return EOF(self.pos)

        token = self.next_token()
        if token is not None:
            return token

        raise StopIteration

    def next_token(self) -> Optional[Token]:
        # Ref: https://www.w3.org/TR/css-syntax-3/#consume-token
        c = self.peek(1)
        pos = self.pos
        if c == "/":
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
        if self.remaining < 2:
            # We need at least 2 code points
            return False

        c = self.peek(3)

        if c[0] in "+-":
            if c[1] in string.digits:
                return True
            elif c[1] == "." and len(c) == 3:
                return c[2] in string.digits

            return False
        elif c[0] == ".":
            return c[1] in string.digits

        return c[0] in string.digits

    def at_ident_start(self) -> bool:
        # Ref: https://www.w3.org/TR/css-syntax-3/#would-start-an-identifier
        if self.remaining < 2:
            return False

        c = self.peek(3)

        if c[0] == "-":
            c = c[1:]  # Shift the code points left

        if self.is_name_start(c[0]) or c[0] == "-":
            return True
        elif len(c) >= 2 is not None and self.is_escape(c[:2]):
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
                return BadString(start)
            else:
                value += self.consume(1)

        if self.eof or self.peek(1) != ending:
            return BadString(start)
        else:
            self.advance(1)
            return String(start, value)

    def consume_escape(self) -> str:
        assert self.consume(1) == "\\"

        if self.eof:
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
            while all(self.is_whitespace(c) for c in self.peek(2)):
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
                    self.errors.append("Unexpected EOF while parsing Url")
                    return Url(pos, value)
                elif self.peek(1) == ")":
                    self.advance(1)
                    return Url(pos, value)

                self.consume_bad_url_remnants()
                return BadUrl(pos)

            elif c in "\"'(" or self.is_non_printable(c):
                self.consume_bad_url_remnants()
                return BadUrl(pos)
            else:
                # Append valid character to url
                value += c

        self.errors.append("Unexpected EOF while parsing Url")
        return BadUrl(pos)
