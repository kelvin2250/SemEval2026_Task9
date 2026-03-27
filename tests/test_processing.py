import unittest

from src.utils.processing import clean_text


class TestProcessing(unittest.TestCase):
    def test_clean_text_normalizes_url_and_mentions(self):
        text = "Hey @alice check https://example.com now"
        cleaned = clean_text(text)

        self.assertIn("@USER", cleaned)
        self.assertIn("HTTPURL", cleaned)
        self.assertNotIn("@alice", cleaned)
        self.assertNotIn("https://example.com", cleaned)

    def test_clean_text_handles_non_string(self):
        self.assertEqual(clean_text(None), "")


if __name__ == "__main__":
    unittest.main()
