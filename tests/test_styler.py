from typing import Iterable
import unittest

import io
import styler


class DecoderTestCase(unittest.TestCase):
    def test_utf16_be_bom(self):
        stream = io.BytesIO(
            '\u00EF\u00BB\u00BF@charset "ISO-8859-5"; @\u00C3\u00A9'.encode("latin1")
        )
        decoded = styler.decode(stream)
        self.assertEqual('@charset "ISO-8859-5"; @é', decoded.read())


class TokenizerTestCase(unittest.TestCase):
    def assertTokenEqual(self, css, token):
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

    def test_empty(self):
        self.assertTokenEqual("", [])
        self.assertTokenEqual("", [styler.EOF(0)])

    def test_comment(self):
        self.assertTokenEqual("/* test */", styler.Comment(0, " test "))
        self.assertTokenEqual("/**/", styler.Comment(0, ""))
        self.assertTokenEqual("/* test /* nest */", styler.Comment(0, " test /* nest "))
        self.assertTokenEqual("/*test \n /*test", styler.Comment(0, "test \n /*test"))

    def test_whitespace(self):
        self.assertTokenEqual(" ", styler.Whitespace(0, " "))
        self.assertTokenEqual(
            "/*   \t*/ /*",
            [
                styler.Comment(0, "   \t"),
                styler.Whitespace(8, " "),
                styler.Comment(9, ""),
                styler.EOF(11),
            ],
        )

    def test_string(self):
        self.assertTokenEqual('"Test"', styler.String(1, "Test"))
        self.assertTokenEqual("'Test'", styler.String(1, "Test"))
        self.assertTokenEqual("'T\"est'", styler.String(1, 'T"est'))
        self.assertTokenEqual('"T\'est"', styler.String(1, "T'est"))

    def test_badstring(self):
        self.assertTokenEqual('"T\n', [styler.BadString(1), styler.Whitespace(2, "\n")])
        # Unterminated
        self.assertTokenEqual('"T', styler.BadString(1))

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
        self.assertTokenEqual(
            "+A",
            [styler.Delim(0, "+"), styler.Ident(1, "A")],
        )

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
        # self.assertTokenEqual("100em", styler.Dimension(0, 100, INTEGER, "em"))
        # self.assertTokenEqual("-100px", styler.Dimension(0, -100, INTEGER, "px"))
        # self.assertTokenEqual("15.5text", styler.Dimension(0, 15.5, NUMBER, "text"))
        # self.assertTokenEqual("-15.5pt", styler.Dimension(0, -15.5, NUMBER, "pt"))
        self.assertTokenEqual("2e3em", styler.Dimension(0, 2e3, NUMBER, "em"))
        self.assertTokenEqual("-2e3rem", styler.Dimension(0, -2e3, NUMBER, "rem"))

    def test_number_integer(self):
        INTEGER = styler.NumericType.INTEGER
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
        # self.assertTokenEqual("2e2", styler.Number(0, 2e2, NUMBER))
        # self.assertTokenEqual("-2e2", styler.Number(0, -2e2, NUMBER))
        # self.assertTokenEqual("2e-2", styler.Number(0, 2e-2, NUMBER))
        # self.assertTokenEqual("-2e-2", styler.Number(0, -2e-2, NUMBER))
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
        # Url should ignore whitespace
        self.assertTokenEqual("url( \t\n)", styler.Url(0, ""))

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
