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
        self.assertTokenEqual("", [styler.EOF()])

    def test_comment(self):
        self.assertTokenEqual("/* test */", styler.Comment(" test "))
        self.assertTokenEqual("/**/", styler.Comment(""))
        self.assertTokenEqual("/* test /* nest */", styler.Comment(" test /* nest "))
        self.assertTokenEqual(
            "/*test \n /*test",
            styler.Comment("test \n /*test"),
            parse_errors=["Comment did not close."],
        )

    def test_whitespace(self):
        self.assertTokenEqual(" ", styler.Whitespace(" "))
        self.assertTokenEqual(
            "/*   \t*/ /**/",
            [
                styler.Comment("   \t"),
                styler.Whitespace(" "),
                styler.Comment(""),
                styler.EOF(),
            ],
        )

    def test_string(self):
        self.assertTokenEqual('"Test"', styler.String("Test"))
        self.assertTokenEqual("'Test'", styler.String("Test"))
        self.assertTokenEqual("'T\"est'", styler.String('T"est'))
        self.assertTokenEqual('"T\'est"', styler.String("T'est"))
        # Unterminated, gives warning
        self.assertTokenEqual(
            '"T',
            styler.String("T"),
            ['String not ended with matching `"`.'],
        )
        self.assertTokenEqual(
            "'T",
            styler.String("T"),
            ["String not ended with matching `'`."],
        )

    def test_badstring(self):
        self.assertTokenEqual(
            '"T\n',
            [styler.BadString(), styler.Whitespace("\n")],
            parse_errors=["String contains newline."],
        )

    def test_curly_brackets(self):
        self.assertTokenEqual(
            "/**/{ }",
            [
                styler.Comment(""),
                styler.OpenCurlyBracket(),
                styler.Whitespace(" "),
                styler.CloseCurlyBracket(),
            ],
        )

    def test_brackets(self):
        self.assertTokenEqual(
            "[ ]",
            [
                styler.OpenBracket(),
                styler.Whitespace(" "),
                styler.CloseBracket(),
            ],
        )

    def test_parenthesis(self):
        self.assertTokenEqual(
            "( ) )",
            [
                styler.OpenParenthesis(),
                styler.Whitespace(" "),
                styler.CloseParenthesis(),
                styler.Whitespace(" "),
                styler.CloseParenthesis(),
            ],
        )

    def test_plus_not_a_number(self):
        self.assertTokenEqual("+", [styler.Delim("+")])
        self.assertTokenEqual("+A", [styler.Delim("+"), styler.Ident("A")])

    def test_dot_not_a_number(self):
        self.assertTokenEqual(".", [styler.Delim(".")])
        self.assertTokenEqual(".A", [styler.Delim("."), styler.Ident("A")])

    def test_number_percentage(self):
        self.assertTokenEqual("100%", styler.Percentage(100))
        self.assertTokenEqual("-100%", styler.Percentage(-100))
        self.assertTokenEqual("15.5%", styler.Percentage(15.5))
        self.assertTokenEqual("-15.5%", styler.Percentage(-15.5))
        self.assertTokenEqual("2e3%", styler.Percentage(2e3))
        self.assertTokenEqual("-2e3%", styler.Percentage(-2e3))

    def test_number_dimension(self):
        NUMBER = styler.NumericType.NUMBER
        INTEGER = styler.NumericType.INTEGER
        self.assertTokenEqual("100em", styler.Dimension(100, INTEGER, "em"))
        self.assertTokenEqual("-100px", styler.Dimension(-100, INTEGER, "px"))
        self.assertTokenEqual("15.5text", styler.Dimension(15.5, NUMBER, "text"))
        self.assertTokenEqual("-15.5pt", styler.Dimension(-15.5, NUMBER, "pt"))
        self.assertTokenEqual("2e3em", styler.Dimension(2e3, NUMBER, "em"))
        self.assertTokenEqual("-2e3rem", styler.Dimension(-2e3, NUMBER, "rem"))

    def test_number_integer(self):
        INTEGER = styler.NumericType.INTEGER
        self.assertTokenEqual("0", styler.Number(0, INTEGER))
        self.assertTokenEqual("1", styler.Number(1, INTEGER))
        self.assertTokenEqual("+1", styler.Number(1, INTEGER))
        self.assertTokenEqual("-1", styler.Number(-1, INTEGER))

    def test_number_number(self):
        NUMBER = styler.NumericType.NUMBER
        self.assertTokenEqual("+1.5", styler.Number(1.5, NUMBER))
        self.assertTokenEqual("-1.5", styler.Number(-1.5, NUMBER))
        self.assertTokenEqual(".5", styler.Number(0.5, NUMBER))
        self.assertTokenEqual("-.5", styler.Number(-0.5, NUMBER))

    def test_number_number_exp(self):
        NUMBER = styler.NumericType.NUMBER
        self.assertTokenEqual("2e2", styler.Number(2e2, NUMBER))
        self.assertTokenEqual("-2e2", styler.Number(-2e2, NUMBER))
        self.assertTokenEqual("2e-2", styler.Number(2e-2, NUMBER))
        self.assertTokenEqual("-2e-2", styler.Number(-2e-2, NUMBER))
        self.assertTokenEqual("2.5e2", styler.Number(2.5e2, NUMBER))
        self.assertTokenEqual("-2.5e2", styler.Number(-2.5e2, NUMBER))
        self.assertTokenEqual("2.5e-2", styler.Number(2.5e-2, NUMBER))
        self.assertTokenEqual("-2.5e-2", styler.Number(-2.5e-2, NUMBER))

    def test_hash_without_name(self):
        self.assertTokenEqual("#", styler.Delim("#"))

    def test_hash_id(self):
        self.assertTokenEqual("#test", styler.Hash("test", styler.HashType.ID))
        self.assertTokenEqual("#--test", styler.Hash("--test", styler.HashType.ID))

    def test_hash_unrestricted(self):
        self.assertTokenEqual("#0red", styler.Hash("0red"))
        self.assertTokenEqual("#-0red", styler.Hash("-0red"))

    def test_comma(self):
        self.assertTokenEqual(",", styler.Comma())
        self.assertTokenEqual(
            "#a,#",
            [
                styler.Hash("a", styler.HashType.ID),
                styler.Comma(),
                styler.Delim("#"),
            ],
        )

    def test_cdo(self):
        self.assertTokenEqual("<!--", styler.CDO())

    def test_cdc(self):
        self.assertTokenEqual("-->", styler.CDC())

    def test_ident_url(self):
        self.assertTokenEqual(
            "url('')",
            [
                styler.Function("url"),
                styler.String(""),
                styler.CloseParenthesis(),
            ],
        )

    def test_minus_ident(self):
        self.assertTokenEqual("-a", styler.Ident("-a"))
        self.assertTokenEqual("--a", styler.Ident("--a"))
        self.assertTokenEqual("--a-b", styler.Ident("--a-b"))

    def test_escapes(self):
        self.assertTokenEqual("-\\000394a", styler.Ident("-\u0394a"))
        self.assertTokenEqual('"\\000394a"', styler.String("\u0394a"))
        self.assertTokenEqual("\\", styler.Delim("\\"), ["Invalid escape."])
        self.assertTokenEqual("\\ ", styler.Ident(" "))
        self.assertTokenEqual("\\Z", [styler.Ident("Z")])
        self.assertTokenEqual("\\0", styler.Ident("\N{REPLACEMENT CHARACTER}"))

    def test_function(self):
        self.assertTokenEqual(
            "func1('arg1', 'arg2')",
            [
                styler.Function("func1"),
                styler.String("arg1"),
                styler.Comma(),
                styler.Whitespace(" "),
                styler.String("arg2"),
                styler.CloseParenthesis(),
            ],
        )

    def test_not_a_url(self):
        self.assertTokenEqual(
            "url('arg1')",
            [
                styler.Function("url"),
                styler.String("arg1"),
                styler.CloseParenthesis(),
            ],
        )

    def test_url(self):
        self.assertTokenEqual("url()", styler.Url(""))
        self.assertTokenEqual(
            "url(http", styler.Url("http"), ["Unexpected EOF while parsing Url."]
        )
        self.assertTokenEqual(
            "url(http  ", styler.Url("http"), ["Unexpected EOF while parsing Url."]
        )
        self.assertTokenEqual(
            "url(https://www.w3.org/TR/css-syntax-3/#consume-url-token)",
            styler.Url("https://www.w3.org/TR/css-syntax-3/#consume-url-token"),
        )
        # Url should ignore whitespace
        self.assertTokenEqual("url(  \t\n)", styler.Url(""))
        # Should handle escapes
        self.assertTokenEqual(f"url(\\{ord('d'):x})", styler.Url("d"))
        self.assertTokenEqual(f"url(\\{ord('d'):x})", styler.Url("d"))
        self.assertTokenEqual("url( http:// )", styler.Url("http://"))
        self.assertTokenEqual("url( http:// \n)", styler.Url("http://"))

    def test_bad_url(self):
        # Space is not allowed
        self.assertTokenEqual("url( http // )", styler.BadUrl())

        self.assertTokenEqual(
            "url( http(// )", styler.BadUrl(), ["Url contains invalid character: (."]
        )
        self.assertTokenEqual(
            'url( http"// )', styler.BadUrl(), ['Url contains invalid character: ".']
        )
        self.assertTokenEqual(
            "url( http'// )", styler.BadUrl(), ["Url contains invalid character: '."]
        )
        self.assertTokenEqual(
            "url( http(\a )", styler.BadUrl(), ["Url contains invalid character: (."]
        )
        self.assertTokenEqual(
            "url(\\", styler.BadUrl(), ["Invalid escape while parsing Url"]
        )
        self.assertTokenEqual(
            "url(\\", styler.BadUrl(), ["Invalid escape while parsing Url"]
        )

    def test_colon(self):
        self.assertTokenEqual(":", styler.Colon())

    def test_semicolon(self):
        self.assertTokenEqual(";", styler.Semicolon())

    def test_declaration(self):
        self.assertTokenEqual(
            "border-width: 10px;",
            [
                styler.Ident("border-width"),
                styler.Colon(),
                styler.Whitespace(" "),
                styler.Dimension(10, styler.NumericType.INTEGER, "px"),
                styler.Semicolon(),
            ],
        )

    def test_at_keyword(self):
        self.assertTokenEqual("@media", styler.AtKeyword("media"))
        self.assertTokenEqual("@--some", styler.AtKeyword("--some"))


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
            styler.Delim("*"),
            styler.Comma(),
            styler.Delim("*"),
            styler.Colon(),
            styler.Colon(),
            styler.Ident("before"),
            styler.Comma(),
            styler.Delim("*"),
            styler.Colon(),
            styler.Colon(),
            styler.Ident("after"),
            styler.OpenCurlyBracket(),
            styler.Ident("box-sizing"),
            styler.Colon(),
            styler.Ident("border-box"),
            styler.Semicolon(),
            styler.CloseCurlyBracket(),
            styler.AtKeyword("media"),
            styler.OpenParenthesis(),
            styler.Ident("prefers-reduced-motion"),
            styler.Colon(),
            styler.Ident("no-preference"),
            styler.CloseParenthesis(),
            styler.OpenCurlyBracket(),
            styler.Colon(),
            styler.Ident("root"),
            styler.OpenCurlyBracket(),
            styler.Ident("scroll-behavior"),
            styler.Colon(),
            styler.Ident("smooth"),
            styler.Semicolon(),
            styler.CloseCurlyBracket(),
            styler.CloseCurlyBracket(),
            styler.Ident("body"),
            styler.OpenCurlyBracket(),
            styler.Ident("margin"),
            styler.Colon(),
            styler.Number(0, styler.NumericType.INTEGER),
            styler.Semicolon(),
            styler.Ident("font-family"),
            styler.Colon(),
            styler.Ident("system-ui"),
            styler.Comma(),
            styler.Ident("-apple-system"),
            styler.Comma(),
            styler.String("Segoe UI"),
            styler.Comma(),
            styler.Ident("Roboto"),
            styler.Comma(),
            styler.String("Helvetica Neue"),
            styler.Comma(),
            styler.Ident("Arial"),
            styler.Comma(),
            styler.String("Noto Sans"),
            styler.Comma(),
            styler.String("Liberation Sans"),
            styler.Comma(),
            styler.Ident("sans-serif"),
            styler.Comma(),
            styler.String("Apple Color Emoji"),
            styler.Comma(),
            styler.String("Segoe UI Emoji"),
            styler.Comma(),
            styler.String("Segoe UI Symbol"),
            styler.Comma(),
            styler.String("Noto Color Emoji"),
            styler.Semicolon(),
            styler.Ident("font-size"),
            styler.Colon(),
            styler.Dimension(1, styler.NumericType.INTEGER, "rem"),
            styler.Semicolon(),
            styler.Ident("font-weight"),
            styler.Colon(),
            styler.Number(400, styler.NumericType.INTEGER),
            styler.Semicolon(),
            styler.Ident("line-height"),
            styler.Colon(),
            styler.Number(1.5, styler.NumericType.NUMBER),
            styler.Semicolon(),
            styler.Ident("color"),
            styler.Colon(),
            styler.Hash("212529", hash_type=styler.HashType.UNRESTRICTED),
            styler.Semicolon(),
            styler.Ident("background-color"),
            styler.Colon(),
            styler.Hash("fff", hash_type=styler.HashType.ID),
            styler.Semicolon(),
            styler.Ident("-webkit-text-size-adjust"),
            styler.Colon(),
            styler.Percentage(100),
            styler.Semicolon(),
            styler.Ident("-webkit-tap-highlight-color"),
            styler.Colon(),
            styler.Function("rgba"),
            styler.Number(0, styler.NumericType.INTEGER),
            styler.Comma(),
            styler.Number(0, styler.NumericType.INTEGER),
            styler.Comma(),
            styler.Number(0, styler.NumericType.INTEGER),
            styler.Comma(),
            styler.Number(0, styler.NumericType.INTEGER),
            styler.CloseParenthesis(),
            styler.Semicolon(),
            styler.CloseCurlyBracket(),
            styler.EOF(),
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
        self.assertSequenceEqual(rule.prelude, [styler.Ident("screen")])
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
                    styler.OpenParenthesis(),
                    [
                        styler.Ident("prefers-reduced-motion"),
                        styler.Colon(),
                        styler.Ident("no-preference"),
                    ],
                )
            ],
        )
        self.assertSequenceEqual(
            getattr(rule.block, "content", []),
            [
                styler.Colon(),
                styler.Ident("root"),
                styler.BlockAst(
                    styler.OpenCurlyBracket(),
                    [
                        styler.Ident("scroll-behavior"),
                        styler.Colon(),
                        styler.Ident("smooth"),
                        styler.Semicolon(),
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
                prelude=[styler.Ident("div")],
                block=styler.BlockAst(
                    block=styler.OpenCurlyBracket(),
                    content=[
                        styler.Ident("border-width"),
                        styler.Colon(),
                        styler.Dimension(10, styler.NumericType.INTEGER, "px"),
                        styler.Semicolon(),
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
        self.assertSequenceEqual(rule.prelude, [styler.Ident("div")])

    def test_parse_rule_list(self):
        tokens = styler.Tokenizer.from_str("div{} @media;")
        parser = styler.Parser(tokens)

        rule_list = parser.parse_rule_list()
        self.assertEqual(rule_list[0].prelude, [styler.Ident("div")])

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
                [styler.CDO(), styler.Ident("div")],
                styler.BlockAst(styler.OpenCurlyBracket(), []),
            ),
        )

        tokens = styler.Tokenizer.from_str("<!-- div -->{}")
        parser = styler.Parser(tokens)

        rule_list = parser.parse_rule_list()
        self.assertEqual(
            rule_list[0],
            styler.QualifiedRuleAst(
                [styler.CDO(), styler.Ident("div"), styler.CDC()],
                styler.BlockAst(styler.OpenCurlyBracket(), []),
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
                    styler.Dimension(10, styler.NumericType.INTEGER, "px"),
                    styler.Delim("+"),
                    styler.Dimension(10, styler.NumericType.INTEGER, "em"),
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
                        styler.Dimension(10, styler.NumericType.INTEGER, "px"),
                        styler.Delim("+"),
                        styler.Dimension(10, styler.NumericType.INTEGER, "em"),
                    ],
                ),
                styler.Ident("test"),
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
                            styler.Dimension(10, styler.NumericType.INTEGER, "px"),
                            styler.Delim("+"),
                            styler.Dimension(10, styler.NumericType.INTEGER, "em"),
                        ],
                    ),
                    styler.Ident("test"),
                ],
                [styler.Ident("div")],
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
                "border", [styler.Dimension(10, styler.NumericType.INTEGER, "px")]
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
                [styler.Ident("red")],
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
                    [styler.Dimension(10, styler.NumericType.INTEGER, "px")],
                ),
                styler.DeclarationAst("background-color", [styler.Ident("red")], True),
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
                styler.DeclarationAst("background-color", [styler.Ident("red")], True),
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
                styler.DeclarationAst("background-color", [styler.Ident("red")], True),
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
                    [styler.Dimension(10, styler.NumericType.INTEGER, "px")],
                ),
                styler.AtRuleAst(
                    "media",
                    [styler.Ident("screen")],
                    styler.BlockAst(
                        styler.OpenCurlyBracket(),
                        [
                            styler.Ident("background-color"),
                            styler.Colon(),
                            styler.Ident("red"),
                            styler.Delim("!"),
                            styler.Ident("important"),
                            styler.Semicolon(),
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
                        [styler.Ident("div")],
                        styler.BlockAst(
                            styler.OpenCurlyBracket(),
                            [
                                styler.Ident("background-color"),
                                styler.Colon(),
                                styler.Ident("red"),
                                styler.Delim("!"),
                                styler.Ident("important"),
                                styler.Semicolon(),
                            ],
                        ),
                    ),
                    styler.QualifiedRuleAst(
                        [styler.Ident("span")],
                        styler.BlockAst(
                            styler.OpenCurlyBracket(),
                            [
                                styler.Ident("color"),
                                styler.Colon(),
                                styler.Hash("123456"),
                                styler.Semicolon(),
                            ],
                        ),
                    ),
                ]
            ),
        )
