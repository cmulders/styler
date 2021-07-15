import codecs
import re
from collections import namedtuple
import unittest

from typing import Collection, Iterable, Sequence, Tuple, Type

import io
from pathlib import Path
from styler import decode
import json
import logging
from itertools import islice

logger = logging.getLogger(__name__)

CSS_PARSING_TESTS_DIR = Path(__file__).parent / "css-parsing-tests"

JSONCase = namedtuple("JSONCase", "case, expectation")


def pairs(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    return zip(
        islice(iterable, 0, None, 2),
        islice(iterable, 1, None, 2),
    )


class CSSParseTestCaseMeta(type):
    """Metaclass for dynanic test loading"""

    @classmethod
    def __prepare__(cls, clsname, bases, **kwargs):
        namespace = dict()

        if not "cases" in kwargs or unittest.TestCase not in bases:
            logger.warning(
                f"Class `{cls}` should specify a name as intialize argument and must base unittest.TestCase, nothing loaded"
            )
            return namespace

        namespace["cases"] = list(cls.load_cases(kwargs["cases"]))

        for idx, case in enumerate(namespace["cases"]):
            name, fn = cls.create_test(idx, case)
            namespace[name] = fn

        return namespace

    def __new__(cls, name, bases, namespace, **kwargs):
        kwargs.pop("cases")  # Already processd this in the __prepare__
        return super().__new__(cls, name, bases, namespace, **kwargs)

    @classmethod
    def load_cases(cls, name) -> Iterable[JSONCase]:
        json_path = (CSS_PARSING_TESTS_DIR / name).with_suffix(".json")
        assert json_path.exists(), f"JSON cases file does not exists: {json_path}."
        with json_path.open("rb") as fd:
            raw_cases = json.load(fd)

        return map(JSONCase._make, pairs(raw_cases))

    @staticmethod
    def create_test(idx, case: JSONCase):
        def inner(self):
            self.run_case(case.case, case.expectation)

        if isinstance(case.case, dict) and "comment" in case.case:
            case_str = case.case["comment"]
        elif isinstance(case.case, dict) and "css_bytes" in case.case:
            case_str = case.case["css_bytes"]
        else:
            case_str = ""

        case_str = re.sub(r"[^\w]+", "_", case_str).strip("_").strip()
        if case_str:
            return f"test_{idx:03}_{case_str}", inner
        else:
            return f"test_{idx:03}", inner


class StylesheetBytesTestCase(
    unittest.TestCase,
    metaclass=CSSParseTestCaseMeta,
    cases="stylesheet_bytes",
):
    def run_case(self, case, expectation: Tuple[Iterable, str]):
        css_bytes = str(case["css_bytes"]).encode("latin1")
        protocol_encoding = case.get("protocol_encoding")
        environment_encoding = case.get("environment_encoding")

        expected_ast, expected_encoding = expectation

        stream, encoding = decode(
            io.BytesIO(css_bytes),
            protocol_encoding=protocol_encoding,
            environment_encoding=environment_encoding,
        )

        # Encoding matches with expectation
        self.assertEqual(
            codecs.lookup(expected_encoding),
            encoding,
            f"Detected encoding {encoding.name} instead of {expected_encoding}",
        )
