import codecs
from typing import (
    BinaryIO,
    Iterable,
    NamedTuple,
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


def detect_bom(buffer: io.BufferedReader) -> Optional[codecs.CodecInfo]:
    sample = buffer.peek(3)
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
    stream: Union[TextIO, BinaryIO, io.RawIOBase],
    protocol_encoding=None,
    environment_encoding=None,
) -> TextIO:
    """
    Try to detect encoding in the following order:
    1. BOM marker
    2. Protocol encoding
    3. @charset rule
    4. Environment encoding
    5. Assume UTF-8

    Ref: https://www.w3.org/TR/css-syntax-3/#input-byte-stream
    """
    buffered = io.BufferedReader(cast(io.RawIOBase, stream))

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
        codec_info = detect_charset(buffered.peek(1024))

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
    return io.TextIOWrapper(buffered, encoding=codec_info.name, errors="replace")


