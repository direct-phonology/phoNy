from unittest import TestCase

import spacy

from phony.tokens.doc import to_phonemes_array


class TestDoc(TestCase):
    def test_to_phonemes_array(self):
        """outputs phoneme data as an array of hashes"""
        nlp = spacy.blank("en")
        doc = nlp("one two three")

        # store phonemes in the string store and pretend we set them on doc
        one_hash = nlp.vocab.strings.add("wʌn")
        two_hash = nlp.vocab.strings.add("tuː")
        three_hash = nlp.vocab.strings.add("θriː")
        doc[0]._.phonemes = one_hash
        doc[1]._.phonemes = two_hash
        doc[2]._.phonemes = three_hash

        # retrieved values can be converted back into strings
        phoneme_strings = [nlp.vocab.strings[p] for p in to_phonemes_array(doc)]
        self.assertEqual(phoneme_strings, ["wʌn", "tuː", "θriː"])

    def test_to_phonemes_array_empty(self):
        """outputs empty phoneme data as zeroes"""
        nlp = spacy.blank("en")
        doc = nlp("one two three")
        self.assertEqual(to_phonemes_array(doc).tolist(), [0, 0, 0])
