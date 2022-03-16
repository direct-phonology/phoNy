from unittest import TestCase

import spacy
from spacy.training import Example

from phony.training import get_aligned_phonemes


class TestGetAlignedPhonemes(TestCase):
    def test_aligned(self):
        """returns aligned phoneme data from training examples"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")

        # set some phonemes on the reference doc
        one_hash = nlp.vocab.strings.add("wʌn")
        two_hash = nlp.vocab.strings.add("tuː")
        three_hash = nlp.vocab.strings.add("θriː")
        example = Example.from_dict(doc, {})
        example.reference[0]._.phonemes = one_hash
        example.reference[1]._.phonemes = two_hash
        example.reference[2]._.phonemes = three_hash

        # if aligment is perfect, results should match the reference exactly
        self.assertEqual(
            get_aligned_phonemes(example),
            [one_hash, two_hash, three_hash],
        )

    def test_misaligned(self):
        """returns aligned phoneme data from misaligned training examples"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")

        # create example with differing tokenization and set some phonemes on it
        wu_hash = nlp.vocab.strings.add("wʌ")
        n_hash = nlp.vocab.strings.add("n")
        two_hash = nlp.vocab.strings.add("tuː")
        three_hash = nlp.vocab.strings.add("θriː")
        example = Example.from_dict(doc, {"words": ["on", "e", "two", "three"]})
        example.reference[0]._.phonemes = wu_hash
        example.reference[1]._.phonemes = n_hash
        example.reference[2]._.phonemes = two_hash
        example.reference[3]._.phonemes = three_hash

        # aligned tokens should have values; misaligned should be None
        self.assertEqual(
            get_aligned_phonemes(example),
            [None, two_hash, three_hash],
        )

    def test_as_string(self):
        """returns aligned phoneme data from training examples as strings"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")

        # set some phonemes on the reference doc
        one_hash = nlp.vocab.strings.add("wʌn")
        two_hash = nlp.vocab.strings.add("tuː")
        three_hash = nlp.vocab.strings.add("θriː")
        example = Example.from_dict(doc, {})
        example.reference[0]._.phonemes = one_hash
        example.reference[1]._.phonemes = two_hash
        example.reference[2]._.phonemes = three_hash

        # aligned output should be strings, not hashes
        self.assertEqual(
            get_aligned_phonemes(example, as_string=True),
            ["wʌn", "tuː", "θriː"],
        )
