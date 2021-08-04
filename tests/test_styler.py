import io
import textwrap
import unittest
from os import SEEK_SET
from tempfile import TemporaryFile
from typing import Iterable

import styler


class DecoderTestCase(unittest.TestCase):
    def test_utf16_be_bom(self):
        stream = io.BytesIO(
            '\u00EF\u00BB\u00BF@charset "ISO-8859-5"; @\u00C3\u00A9'.encode("latin1")
        )
        decoded = styler.decode(stream)
        self.assertEqual('@charset "ISO-8859-5"; @é', decoded.read())

    def test_text_io(self):
        stream = io.StringIO('@charset "ISO-8859-5"; @é')
        decoded = styler.decode(stream)
        self.assertEqual('@charset "ISO-8859-5"; @é', decoded.read())

    def test_binary_io(self):
        with TemporaryFile("wb+") as tmp:
            tmp.write(
                '\u00EF\u00BB\u00BF@charset "ISO-8859-5"; @\u00C3\u00A9'.encode(
                    "latin1"
                )
            )
            tmp.seek(0, SEEK_SET)

            decoded = styler.decode(tmp)
            self.assertEqual('@charset "ISO-8859-5"; @é', decoded.read())

    def test_binary_io_no_buffer(self):
        with TemporaryFile("wb+", buffering=0) as tmp:
            tmp.write(
                '\u00EF\u00BB\u00BF@charset "ISO-8859-5"; @\u00C3\u00A9'.encode(
                    "latin1"
                )
            )
            tmp.seek(0, SEEK_SET)

            decoded = styler.decode(tmp)
            self.assertEqual('@charset "ISO-8859-5"; @é', decoded.read())

    def test_invalid_charset_encoding(self):
        stream = io.BytesIO(b'@charset "kamoulox"; a')
        with self.assertWarns(Warning):
            decoded = styler.decode(stream)
        self.assertEqual('@charset "kamoulox"; a', decoded.read())

    def test_invalid_protocol_encoding(self):
        stream = io.BytesIO(b"a")
        with self.assertWarns(Warning):
            decoded = styler.decode(stream, protocol_encoding="kamoulox")
        self.assertEqual("a", decoded.read())

    def test_invalid_environment_encoding(self):
        stream = io.BytesIO(b"a")
        with self.assertWarns(Warning):
            decoded = styler.decode(stream, environment_encoding="kamoulox")
        self.assertEqual("a", decoded.read())


class TokenizerTestCase(unittest.TestCase):
    def assertTokenEqual(self, css, token, parse_errors=None):
        expected_tokens = [token] if not isinstance(token, Iterable) else list(token)

        tokenizer = styler.Tokenizer(css)
        found_tokens = list(tokenizer)

        if not expected_tokens or not isinstance(expected_tokens[-1], styler.EOF):
            # Strip EOF if we did not specify
            found_tokens = found_tokens[:-1]

        self.assertListEqual(
            found_tokens,
            expected_tokens,
            msg=f"Tokenization of `{css}` did not result in expected tokens.",
        )
        self.assertListEqual(tokenizer.errors, parse_errors or [])

    def test_empty(self):
        self.assertTokenEqual("", [])
        self.assertTokenEqual("", [styler.EOF(0)])

    def test_comment(self):
        self.assertTokenEqual("/* test */", styler.Comment(0, " test "))
        self.assertTokenEqual("/**/", styler.Comment(0, ""))
        self.assertTokenEqual("/* test /* nest */", styler.Comment(0, " test /* nest "))
        self.assertTokenEqual(
            "/*test \n /*test",
            styler.Comment(0, "test \n /*test"),
            parse_errors=["Comment did not close."],
        )

    def test_whitespace(self):
        self.assertTokenEqual(" ", styler.Whitespace(0, " "))
        self.assertTokenEqual(
            "/*   \t*/ /**/",
            [
                styler.Comment(0, "   \t"),
                styler.Whitespace(8, " "),
                styler.Comment(9, ""),
                styler.EOF(13),
            ],
        )

    def test_string(self):
        self.assertTokenEqual('"Test"', styler.String(1, "Test"))
        self.assertTokenEqual("'Test'", styler.String(1, "Test"))
        self.assertTokenEqual("'T\"est'", styler.String(1, 'T"est'))
        self.assertTokenEqual('"T\'est"', styler.String(1, "T'est"))
        # Unterminated, gives warning
        self.assertTokenEqual(
            '"T',
            styler.String(1, "T"),
            ['String not ended with matching `"`.'],
        )
        self.assertTokenEqual(
            "'T",
            styler.String(1, "T"),
            ["String not ended with matching `'`."],
        )

    def test_badstring(self):
        self.assertTokenEqual(
            '"T\n',
            [styler.BadString(1), styler.Whitespace(2, "\n")],
            parse_errors=["String contains newline."],
        )

    def test_curly_brackets(self):
        self.assertTokenEqual(
            "/**/{ }",
            [
                styler.Comment(0, ""),
                styler.OpenCurlyBracket(4),
                styler.Whitespace(5, " "),
                styler.CloseCurlyBracket(6),
            ],
        )

    def test_brackets(self):
        self.assertTokenEqual(
            "[ ]",
            [
                styler.OpenBracket(0),
                styler.Whitespace(1, " "),
                styler.CloseBracket(2),
            ],
        )

    def test_parenthesis(self):
        self.assertTokenEqual(
            "( ) )",
            [
                styler.OpenParenthesis(0),
                styler.Whitespace(1, " "),
                styler.CloseParenthesis(2),
                styler.Whitespace(3, " "),
                styler.CloseParenthesis(4),
            ],
        )

    def test_plus_not_a_number(self):
        self.assertTokenEqual("+", [styler.Delim(0, "+")])
        self.assertTokenEqual("+A", [styler.Delim(0, "+"), styler.Ident(1, "A")])

    def test_dot_not_a_number(self):
        self.assertTokenEqual(".", [styler.Delim(0, ".")])
        self.assertTokenEqual(".A", [styler.Delim(0, "."), styler.Ident(1, "A")])

    def test_number_percentage(self):
        self.assertTokenEqual("100%", styler.Percentage(0, 100))
        self.assertTokenEqual("-100%", styler.Percentage(0, -100))
        self.assertTokenEqual("15.5%", styler.Percentage(0, 15.5))
        self.assertTokenEqual("-15.5%", styler.Percentage(0, -15.5))
        self.assertTokenEqual("2e3%", styler.Percentage(0, 2e3))
        self.assertTokenEqual("-2e3%", styler.Percentage(0, -2e3))

    def test_number_dimension(self):
        NUMBER = styler.NumericType.NUMBER
        INTEGER = styler.NumericType.INTEGER
        self.assertTokenEqual("100em", styler.Dimension(0, 100, INTEGER, "em"))
        self.assertTokenEqual("-100px", styler.Dimension(0, -100, INTEGER, "px"))
        self.assertTokenEqual("15.5text", styler.Dimension(0, 15.5, NUMBER, "text"))
        self.assertTokenEqual("-15.5pt", styler.Dimension(0, -15.5, NUMBER, "pt"))
        self.assertTokenEqual("2e3em", styler.Dimension(0, 2e3, NUMBER, "em"))
        self.assertTokenEqual("-2e3rem", styler.Dimension(0, -2e3, NUMBER, "rem"))

    def test_number_integer(self):
        INTEGER = styler.NumericType.INTEGER
        self.assertTokenEqual("0", styler.Number(0, 0, INTEGER))
        self.assertTokenEqual("1", styler.Number(0, 1, INTEGER))
        self.assertTokenEqual("+1", styler.Number(0, 1, INTEGER))
        self.assertTokenEqual("-1", styler.Number(0, -1, INTEGER))

    def test_number_number(self):
        NUMBER = styler.NumericType.NUMBER
        self.assertTokenEqual("+1.5", styler.Number(0, 1.5, NUMBER))
        self.assertTokenEqual("-1.5", styler.Number(0, -1.5, NUMBER))
        self.assertTokenEqual(".5", styler.Number(0, 0.5, NUMBER))
        self.assertTokenEqual("-.5", styler.Number(0, -0.5, NUMBER))

    def test_number_number_exp(self):
        NUMBER = styler.NumericType.NUMBER
        self.assertTokenEqual("2e2", styler.Number(0, 2e2, NUMBER))
        self.assertTokenEqual("-2e2", styler.Number(0, -2e2, NUMBER))
        self.assertTokenEqual("2e-2", styler.Number(0, 2e-2, NUMBER))
        self.assertTokenEqual("-2e-2", styler.Number(0, -2e-2, NUMBER))
        self.assertTokenEqual("2.5e2", styler.Number(0, 2.5e2, NUMBER))
        self.assertTokenEqual("-2.5e2", styler.Number(0, -2.5e2, NUMBER))
        self.assertTokenEqual("2.5e-2", styler.Number(0, 2.5e-2, NUMBER))
        self.assertTokenEqual("-2.5e-2", styler.Number(0, -2.5e-2, NUMBER))

    def test_hash_without_name(self):
        self.assertTokenEqual("#", styler.Delim(0, "#"))

    def test_hash_id(self):
        self.assertTokenEqual("#test", styler.Hash(0, "test", styler.HashType.ID))
        self.assertTokenEqual("#--test", styler.Hash(0, "--test", styler.HashType.ID))

    def test_hash_unrestricted(self):
        self.assertTokenEqual("#0red", styler.Hash(0, "0red"))
        self.assertTokenEqual("#-0red", styler.Hash(0, "-0red"))

    def test_comma(self):
        self.assertTokenEqual(",", styler.Comma(0))
        self.assertTokenEqual(
            "#a,#",
            [
                styler.Hash(0, "a", styler.HashType.ID),
                styler.Comma(2),
                styler.Delim(3, "#"),
            ],
        )

    def test_cdo(self):
        self.assertTokenEqual("<!--", styler.CDO(0))

    def test_cdc(self):
        self.assertTokenEqual("-->", styler.CDC(0))

    def test_ident_url(self):
        self.assertTokenEqual(
            "url('')",
            [
                styler.Function(0, "url"),
                styler.String(5, ""),
                styler.CloseParenthesis(6),
            ],
        )

    def test_minus_ident(self):
        self.assertTokenEqual("-a", styler.Ident(0, "-a"))
        self.assertTokenEqual("--a", styler.Ident(0, "--a"))
        self.assertTokenEqual("--a-b", styler.Ident(0, "--a-b"))

    def test_escapes(self):
        self.assertTokenEqual("-\\000394a", styler.Ident(0, "-\u0394a"))
        self.assertTokenEqual('"\\000394a"', styler.String(1, "\u0394a"))
        self.assertTokenEqual("\\", styler.Delim(0, "\\"), ["Invalid escape."])
        self.assertTokenEqual("\\ ", styler.Ident(0, " "))
        self.assertTokenEqual("\\Z", [styler.Ident(0, "Z")])
        self.assertTokenEqual("\\0", styler.Ident(0, "\N{REPLACEMENT CHARACTER}"))

    def test_function(self):
        self.assertTokenEqual(
            "func1('arg1', 'arg2')",
            [
                styler.Function(0, "func1"),
                styler.String(7, "arg1"),
                styler.Comma(12),
                styler.Whitespace(13, " "),
                styler.String(15, "arg2"),
                styler.CloseParenthesis(20),
            ],
        )

    def test_not_a_url(self):
        self.assertTokenEqual(
            "url('arg1')",
            [
                styler.Function(0, "url"),
                styler.String(5, "arg1"),
                styler.CloseParenthesis(10),
            ],
        )

    def test_url(self):
        self.assertTokenEqual("url()", styler.Url(0, ""))
        self.assertTokenEqual(
            "url(http", styler.Url(0, "http"), ["Unexpected EOF while parsing Url."]
        )
        self.assertTokenEqual(
            "url(http  ", styler.Url(0, "http"), ["Unexpected EOF while parsing Url."]
        )
        self.assertTokenEqual(
            "url(https://www.w3.org/TR/css-syntax-3/#consume-url-token)",
            styler.Url(0, "https://www.w3.org/TR/css-syntax-3/#consume-url-token"),
        )
        # Url should ignore whitespace
        self.assertTokenEqual("url(  \t\n)", styler.Url(0, ""))
        # Should handle escapes
        self.assertTokenEqual(f"url(\\{ord('d'):x})", styler.Url(0, "d"))
        self.assertTokenEqual(f"url(\\{ord('d'):x})", styler.Url(0, "d"))
        self.assertTokenEqual("url( http:// )", styler.Url(0, "http://"))
        self.assertTokenEqual("url( http:// \n)", styler.Url(0, "http://"))

    def test_bad_url(self):
        # Space is not allowed
        self.assertTokenEqual("url( http // )", styler.BadUrl(0))

        self.assertTokenEqual(
            "url( http(// )", styler.BadUrl(0), ["Url contains invalid character: (."]
        )
        self.assertTokenEqual(
            'url( http"// )', styler.BadUrl(0), ['Url contains invalid character: ".']
        )
        self.assertTokenEqual(
            "url( http'// )", styler.BadUrl(0), ["Url contains invalid character: '."]
        )
        self.assertTokenEqual(
            "url( http(\a )", styler.BadUrl(0), ["Url contains invalid character: (."]
        )
        self.assertTokenEqual(
            "url(\\", styler.BadUrl(0), ["Invalid escape while parsing Url"]
        )
        self.assertTokenEqual(
            "url(\\", styler.BadUrl(0), ["Invalid escape while parsing Url"]
        )

    def test_colon(self):
        self.assertTokenEqual(":", styler.Colon(0))

    def test_semicolon(self):
        self.assertTokenEqual(";", styler.Semicolon(0))

    def test_declaration(self):
        self.assertTokenEqual(
            "border-width: 10px;",
            [
                styler.Ident(0, "border-width"),
                styler.Colon(12),
                styler.Whitespace(13, " "),
                styler.Dimension(14, 10, styler.NumericType.INTEGER, "px"),
                styler.Semicolon(18),
            ],
        )

    def test_at_keyword(self):
        self.assertTokenEqual("@media", styler.AtKeyword(0, "media"))
        self.assertTokenEqual("@--some", styler.AtKeyword(0, "--some"))


class DecoderToTokensTestCase(unittest.TestCase):
    def test_bootstrap_snippet(self):
        stream = io.StringIO(
            textwrap.dedent(
                """
                /*!
                 * Bootstrap Reboot v5.0.2
                 */
                *,
                *::before,
                *::after {
                  box-sizing: border-box;
                }
                
                @media (prefers-reduced-motion: no-preference) {
                  :root {
                    scroll-behavior: smooth;
                  }
                }
                
                body {
                  margin: 0;
                  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
                  font-size: 1rem;
                  font-weight: 400;
                  line-height: 1.5;
                  color: #212529;
                  background-color: #fff;
                  -webkit-text-size-adjust: 100%;
                  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
                }"""
            )
        )
        decoded = styler.decode(stream)
        tokens = list(styler.Tokenizer.from_reader(decoded))
        without_whitespace = [
            tok for tok in tokens if not isinstance(tok, styler.Whitespace)
        ]
        expected = [
            styler.Delim(36, "*"),
            styler.Comma(37),
            styler.Delim(39, "*"),
            styler.Colon(40),
            styler.Colon(41),
            styler.Ident(42, "before"),
            styler.Comma(48),
            styler.Delim(50, "*"),
            styler.Colon(51),
            styler.Colon(52),
            styler.Ident(53, "after"),
            styler.OpenCurlyBracket(59),
            styler.Ident(63, "box-sizing"),
            styler.Colon(73),
            styler.Ident(75, "border-box"),
            styler.Semicolon(85),
            styler.CloseCurlyBracket(87),
            styler.AtKeyword(90, "media"),
            styler.OpenParenthesis(97),
            styler.Ident(98, "prefers-reduced-motion"),
            styler.Colon(120),
            styler.Ident(122, "no-preference"),
            styler.CloseParenthesis(135),
            styler.OpenCurlyBracket(137),
            styler.Colon(141),
            styler.Ident(142, "root"),
            styler.OpenCurlyBracket(147),
            styler.Ident(153, "scroll-behavior"),
            styler.Colon(168),
            styler.Ident(170, "smooth"),
            styler.Semicolon(176),
            styler.CloseCurlyBracket(180),
            styler.CloseCurlyBracket(182),
            styler.Ident(185, "body"),
            styler.OpenCurlyBracket(190),
            styler.Ident(194, "margin"),
            styler.Colon(200),
            styler.Number(202, 0, styler.NumericType.INTEGER),
            styler.Semicolon(203),
            styler.Ident(207, "font-family"),
            styler.Colon(218),
            styler.Ident(220, "system-ui"),
            styler.Comma(229),
            styler.Ident(231, "-apple-system"),
            styler.Comma(244),
            styler.String(247, "Segoe UI"),
            styler.Comma(256),
            styler.Ident(258, "Roboto"),
            styler.Comma(264),
            styler.String(267, "Helvetica Neue"),
            styler.Comma(282),
            styler.Ident(284, "Arial"),
            styler.Comma(289),
            styler.String(292, "Noto Sans"),
            styler.Comma(302),
            styler.String(305, "Liberation Sans"),
            styler.Comma(321),
            styler.Ident(323, "sans-serif"),
            styler.Comma(333),
            styler.String(336, "Apple Color Emoji"),
            styler.Comma(354),
            styler.String(357, "Segoe UI Emoji"),
            styler.Comma(372),
            styler.String(375, "Segoe UI Symbol"),
            styler.Comma(391),
            styler.String(394, "Noto Color Emoji"),
            styler.Semicolon(411),
            styler.Ident(415, "font-size"),
            styler.Colon(424),
            styler.Dimension(426, 1, styler.NumericType.INTEGER, "rem"),
            styler.Semicolon(430),
            styler.Ident(434, "font-weight"),
            styler.Colon(445),
            styler.Number(447, 400, styler.NumericType.INTEGER),
            styler.Semicolon(450),
            styler.Ident(454, "line-height"),
            styler.Colon(465),
            styler.Number(467, 1.5, styler.NumericType.NUMBER),
            styler.Semicolon(470),
            styler.Ident(474, "color"),
            styler.Colon(479),
            styler.Hash(481, "212529", hash_type=styler.HashType.UNRESTRICTED),
            styler.Semicolon(488),
            styler.Ident(492, "background-color"),
            styler.Colon(508),
            styler.Hash(510, "fff", hash_type=styler.HashType.ID),
            styler.Semicolon(514),
            styler.Ident(518, "-webkit-text-size-adjust"),
            styler.Colon(542),
            styler.Percentage(544, 100),
            styler.Semicolon(548),
            styler.Ident(552, "-webkit-tap-highlight-color"),
            styler.Colon(579),
            styler.Function(581, "rgba"),
            styler.Number(586, 0, styler.NumericType.INTEGER),
            styler.Comma(587),
            styler.Number(589, 0, styler.NumericType.INTEGER),
            styler.Comma(590),
            styler.Number(592, 0, styler.NumericType.INTEGER),
            styler.Comma(593),
            styler.Number(595, 0, styler.NumericType.INTEGER),
            styler.CloseParenthesis(596),
            styler.Semicolon(597),
            styler.CloseCurlyBracket(599),
            styler.EOF(600),
        ]

        self.assertEqual(without_whitespace, expected)


class PeekableIterableTestCase(unittest.TestCase):
    def test_peek_reclamation(self):
        import gc

        # Disable GC to test reclamation of consumed values
        gc.set_threshold(0, 0, 0)
        deleted = set()

        class LogDel:
            def __init__(self, value: int) -> None:
                self.value = value

            def __eq__(self, o: object) -> bool:
                return self.value == o

            def __del__(self):
                deleted.add(self.value)

        it = iter(map(LogDel, range(100)))
        p = styler.PeekableIterable(it)

        self.assertEqual(p.peek(2), (0, 1))
        self.assertSetEqual(
            deleted, set(), "Nothing should have been collected yet, buffer of 57."
        )

        self.assertEqual(next(p), 0)
        self.assertEqual(next(p), 1)
        self.assertEqual(p.peek(2), (2, 3))
        self.assertEqual(next(p), 2)
        self.assertEqual(next(p), 3)

        # Here we hold a reference inside the to be dropped tee data.
        v = p.peek()
        assert v is not None

        for i in range(60):
            next(p)

        # itertools.tee caches 57 items
        # https://github.com/python/cpython/blob/f64de53ff01e734d48d1d42195443d7d1646f220/Modules/itertoolsmodule.c#L575
        self.assertSetEqual(
            deleted,
            set(range(57)) - {v.value},
            "Not all objects except a single one was collected.",
        )

        # Release the reference, so it can be dropped
        del v

        self.assertSetEqual(deleted, set(range(57)), "Not all objects were collected.")

    def test_reconsumption(self):
        it = styler.PeekableIterable(range(50))

        self.assertEqual(it.peek(), 0)
        self.assertEqual(next(it), 0)

        self.assertEqual(it.peek(), 1)
        self.assertEqual(next(it), 1)

        it.reconsume()
        self.assertEqual(next(it), 1)

        self.assertEqual(it.peek(), 2)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)

        it.reconsume()
        self.assertEqual(next(it), 3)

    def test_iteration_with_reconsumption(self):
        it = styler.PeekableIterable(range(50))
        real_it = iter(it)

        self.assertEqual(it.peek(), 0)
        self.assertEqual(next(real_it), 0)
        it.reconsume()
        self.assertEqual(next(real_it), 0, "Reconsumption did not reset the iterator.")
        self.assertEqual(next(real_it), 1)

    def test_peek(self):
        it = styler.PeekableIterable(range(50))
        self.assertEqual(it.peek(), 0)
        self.assertEqual(it.peek(1), 0)
        self.assertEqual(it.peek(2), (0, 1))
        with self.assertRaises(ValueError):
            it.peek(0)
        with self.assertRaises(ValueError):
            it.peek(-1)


class ParserTestCase(unittest.TestCase):
    def test_parse_rule_eof(self):
        tokens = styler.Tokenizer.from_str(" ")
        parser = styler.Parser(tokens)

        with self.assertRaisesRegex(styler.ASTSyntaxError, "Unexpected EOF.*"):
            parser.parse_rule()

    def test_parse_rule_should_end_with_eof(self):
        tokens = styler.Tokenizer.from_str("div{} span{}")
        parser = styler.Parser(tokens)

        with self.assertRaisesRegex(styler.ASTSyntaxError, "Expected EOF.*"):
            parser.parse_rule()

    def test_parse_rule_at_rule_only_name(self):
        tokens = styler.Tokenizer.from_str("@media;")
        parser = styler.Parser(tokens)

        rule = parser.parse_rule()
        self.assertIsInstance(rule, styler.AtRuleAst)
        assert isinstance(rule, styler.AtRuleAst)
        self.assertEqual(rule, styler.AtRuleAst("media", [], None))

    def test_parse_rule_at_rule_no_block(self):
        tokens = styler.Tokenizer.from_str("@media screen;")
        parser = styler.Parser(tokens)

        rule = parser.parse_rule()
        self.assertIsInstance(rule, styler.AtRuleAst)
        assert isinstance(rule, styler.AtRuleAst)
        self.assertEqual(rule.name, "media")
        self.assertSequenceEqual(rule.prelude, [styler.Ident(7, "screen")])
        self.assertEqual(rule.block, None)

    def test_parse_rule_at_rule(self):
        tokens = styler.Tokenizer.from_str(
            """@media (prefers-reduced-motion: no-preference) {
                  :root {
                    scroll-behavior: smooth;
                  }
                }"""
        )
        parser = styler.Parser(tokens)

        rule = parser.parse_rule()
        self.assertIsInstance(rule, styler.AtRuleAst)
        assert isinstance(rule, styler.AtRuleAst)
        self.assertEqual(rule.name, "media")
        self.assertSequenceEqual(
            rule.prelude,
            [
                styler.BlockAst(
                    styler.OpenParenthesis(7),
                    [
                        styler.Ident(8, "prefers-reduced-motion"),
                        styler.Colon(30),
                        styler.Ident(32, "no-preference"),
                    ],
                )
            ],
        )
        self.assertSequenceEqual(
            getattr(rule.block, "content", []),
            [
                styler.Colon(67),
                styler.Ident(68, "root"),
                styler.BlockAst(
                    styler.OpenCurlyBracket(73),
                    [
                        styler.Ident(95, "scroll-behavior"),
                        styler.Colon(110),
                        styler.Ident(112, "smooth"),
                        styler.Semicolon(118),
                    ],
                ),
            ],
        )

    def test_parse_rule_qualified_rule(self):
        tokens = styler.Tokenizer.from_str("div{ border-width: 10px; }")
        parser = styler.Parser(tokens)

        rule = parser.parse_rule()
        self.assertIsInstance(rule, styler.QualifiedRuleAst)
        assert isinstance(rule, styler.QualifiedRuleAst)
        self.assertEqual(
            rule,
            styler.QualifiedRuleAst(
                prelude=[styler.Ident(0, "div")],
                block=styler.BlockAst(
                    block=styler.OpenCurlyBracket(3),
                    content=[
                        styler.Ident(5, "border-width"),
                        styler.Colon(17),
                        styler.Dimension(19, 10, styler.NumericType.INTEGER, "px"),
                        styler.Semicolon(23),
                    ],
                ),
            ),
        )

    def test_parse_rule_qualified_rule_no_block(self):
        tokens = styler.Tokenizer.from_str("div")
        parser = styler.Parser(tokens)

        rule = parser.parse_rule()
        self.assertIsInstance(rule, styler.QualifiedRuleAst)
        assert isinstance(rule, styler.QualifiedRuleAst)
        self.assertSequenceEqual(rule.prelude, [styler.Ident(0, "div")])

    def test_parse_rule_list(self):
        tokens = styler.Tokenizer.from_str("div{} @media;")
        parser = styler.Parser(tokens)

        rule_list = parser.parse_rule_list()
        self.assertEqual(rule_list[0].prelude, [styler.Ident(0, "div")])

        self.assertEqual(rule_list[1].prelude, [])
        self.assertEqual(rule_list[1].block, None)

    def test_parse_rule_list_consume_cdo_cdc(self):
        tokens = styler.Tokenizer.from_str("<!--div{}")
        parser = styler.Parser(tokens)

        rule_list = parser.parse_rule_list()
        self.assertEqual(len(rule_list), 1)
        self.assertEqual(
            rule_list[0],
            styler.QualifiedRuleAst(
                [styler.CDO(0), styler.Ident(4, "div")],
                styler.BlockAst(styler.OpenCurlyBracket(7), []),
            ),
        )

        tokens = styler.Tokenizer.from_str("<!-- div -->{}")
        parser = styler.Parser(tokens)

        rule_list = parser.parse_rule_list()
        self.assertEqual(
            rule_list[0],
            styler.QualifiedRuleAst(
                [styler.CDO(0), styler.Ident(5, "div"), styler.CDC(9)],
                styler.BlockAst(styler.OpenCurlyBracket(12), []),
            ),
        )

    def test_parse_component_value_empty(self):
        tokens = styler.Tokenizer.from_str(" ")
        parser = styler.Parser(tokens)

        with self.assertRaisesRegex(styler.ASTSyntaxError, "Unexpected EOF.*"):
            parser.parse_component_value()

    def test_parse_component_value_function(self):
        tokens = styler.Tokenizer.from_str("calc(10px + 10em)")
        parser = styler.Parser(tokens)

        cpv = parser.parse_component_value()
        self.assertIsInstance(cpv, styler.FunctionAst)
        assert isinstance(cpv, styler.FunctionAst)
        self.assertEqual(
            cpv,
            styler.FunctionAst(
                "calc",
                [
                    styler.Dimension(5, 10, styler.NumericType.INTEGER, "px"),
                    styler.Delim(10, "+"),
                    styler.Dimension(12, 10, styler.NumericType.INTEGER, "em"),
                ],
            ),
        )

    def test_parse_component_value_list(self):
        tokens = styler.Tokenizer.from_str("calc(10px + 10em) test")
        parser = styler.Parser(tokens)

        cpv_list = parser.parse_component_value_list()
        self.assertListEqual(
            cpv_list,
            [
                styler.FunctionAst(
                    "calc",
                    [
                        styler.Dimension(5, 10, styler.NumericType.INTEGER, "px"),
                        styler.Delim(10, "+"),
                        styler.Dimension(12, 10, styler.NumericType.INTEGER, "em"),
                    ],
                ),
                styler.Ident(18, "test"),
            ],
        )

    def test_parse_comma_separated_component_value_list(self):
        tokens = styler.Tokenizer.from_str("calc(10px + 10em) test, div")
        parser = styler.Parser(tokens)

        cpv_list = parser.parse_component_value_list_comma_separated()
        self.assertListEqual(
            cpv_list,
            [
                [
                    styler.FunctionAst(
                        "calc",
                        [
                            styler.Dimension(5, 10, styler.NumericType.INTEGER, "px"),
                            styler.Delim(10, "+"),
                            styler.Dimension(12, 10, styler.NumericType.INTEGER, "em"),
                        ],
                    ),
                    styler.Ident(18, "test"),
                ],
                [styler.Ident(24, "div")],
            ],
        )

    def test_parse_declaration_no_colon(self):
        tokens = styler.Tokenizer.from_str("border")
        parser = styler.Parser(tokens)

        decl = parser.parse_declaration()
        self.assertIsNone(decl)
        self.assertEqual(len(parser.errors), 1, "Should have emitted a parse error.")

    def test_parse_declaration(self):
        tokens = styler.Tokenizer.from_str("border: 10px")
        parser = styler.Parser(tokens)

        decl = parser.parse_declaration()
        self.assertIsInstance(decl, styler.DeclarationAst)
        assert isinstance(decl, styler.DeclarationAst)
        self.assertEqual(
            decl,
            styler.DeclarationAst(
                "border", [styler.Dimension(8, 10, styler.NumericType.INTEGER, "px")]
            ),
        )

    def test_parse_declaration_important(self):
        tokens = styler.Tokenizer.from_str("background-color: red !important")
        parser = styler.Parser(tokens)

        decl = parser.parse_declaration()
        self.assertIsInstance(decl, styler.DeclarationAst)
        assert isinstance(decl, styler.DeclarationAst)
        self.assertEqual(
            decl,
            styler.DeclarationAst(
                "background-color",
                [styler.Ident(18, "red")],
                True,
            ),
        )

    def test_parse_declaration_list(self):
        tokens = styler.Tokenizer.from_str(
            "border: 10px; background-color: red !important;"
        )
        parser = styler.Parser(tokens)

        decl_list = parser.parse_declaration_list()
        self.assertEqual(
            decl_list,
            [
                styler.DeclarationAst(
                    "border",
                    [styler.Dimension(8, 10, styler.NumericType.INTEGER, "px")],
                ),
                styler.DeclarationAst(
                    "background-color", [styler.Ident(32, "red")], True
                ),
            ],
        )

    def test_parse_declaration_list_invalid_not_a_ident(self):
        tokens = styler.Tokenizer.from_str(
            ">border: 10px; background-color: red !important;"
        )
        parser = styler.Parser(tokens)

        decl_list = parser.parse_declaration_list()
        self.assertEqual(
            decl_list,
            [
                styler.DeclarationAst(
                    "background-color", [styler.Ident(33, "red")], True
                ),
            ],
        )

    def test_parse_declaration_list_invalid_missing_semicolon(self):
        tokens = styler.Tokenizer.from_str(
            "border 10px; background-color: red !important;"
        )
        parser = styler.Parser(tokens)

        decl_list = parser.parse_declaration_list()
        self.assertEqual(
            decl_list,
            [
                styler.DeclarationAst(
                    "background-color", [styler.Ident(31, "red")], True
                ),
            ],
        )

    def test_parse_declaration_list_at_rule(self):
        tokens = styler.Tokenizer.from_str(
            "border: 10px; @media screen {background-color: red !important;}"
        )
        parser = styler.Parser(tokens)

        decl_list = parser.parse_declaration_list()
        self.assertEqual(
            decl_list,
            [
                styler.DeclarationAst(
                    "border",
                    [styler.Dimension(8, 10, styler.NumericType.INTEGER, "px")],
                ),
                styler.AtRuleAst(
                    "media",
                    [styler.Ident(21, "screen")],
                    styler.BlockAst(
                        styler.OpenCurlyBracket(28),
                        [
                            styler.Ident(29, "background-color"),
                            styler.Colon(45),
                            styler.Ident(47, "red"),
                            styler.Delim(51, "!"),
                            styler.Ident(52, "important"),
                            styler.Semicolon(61),
                        ],
                    ),
                ),
            ],
        )

    def test_parse_stylesheet(self):
        tokens = styler.Tokenizer.from_str(
            "div {background-color: red !important;} span {color: #123456;}"
        )
        parser = styler.Parser(tokens)

        stylesheet = parser.parse_stylesheet()
        self.assertEqual(
            stylesheet,
            styler.StyleSheetAst(
                [
                    styler.QualifiedRuleAst(
                        [styler.Ident(0, "div")],
                        styler.BlockAst(
                            styler.OpenCurlyBracket(4),
                            [
                                styler.Ident(5, "background-color"),
                                styler.Colon(21),
                                styler.Ident(23, "red"),
                                styler.Delim(27, "!"),
                                styler.Ident(28, "important"),
                                styler.Semicolon(37),
                            ],
                        ),
                    ),
                    styler.QualifiedRuleAst(
                        [styler.Ident(40, "span")],
                        styler.BlockAst(
                            styler.OpenCurlyBracket(45),
                            [
                                styler.Ident(46, "color"),
                                styler.Colon(51),
                                styler.Hash(53, "123456"),
                                styler.Semicolon(60),
                            ],
                        ),
                    ),
                ]
            ),
        )
