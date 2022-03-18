from unittest import TestCase

import spacy


class TestTokens(TestCase):
    def test_get_doc_phonemes(self):
        """can request phonemes for entire doc"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        doc._.phonemes = ["wʌn", "tuː", "θriː"]
        self.assertEqual(doc._.phonemes, ["wʌn", "tuː", "θriː"])

    def test_get_span_phonemes(self):
        """can request phonemes for span"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        doc._.phonemes = ["wʌn", "tuː", "θriː"]
        self.assertEqual(doc[1:]._.phonemes, ["tuː", "θriː"])

    def test_get_token_phonemes(self):
        """can request phonemes for single token"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        doc._.phonemes = ["wʌn", "tuː", "θriː"]
        self.assertEqual(doc[1]._.phonemes, "tuː")
