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
