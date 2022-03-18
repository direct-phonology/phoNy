from unittest import TestCase

import spacy
from spacy.training import Example

from phony.training import example_from_phonemes_dict, get_aligned_phonemes


class TestGetAlignedPhonemes(TestCase):
    def test_aligned(self):
        """returns aligned phoneme data from training examples"""
        nlp = spacy.blank("en")
        example = example_from_phonemes_dict(
            nlp.make_doc("one two three"),
            {"phonemes": ["wʌn", "tuː", "θriː"]},
        )

        # if aligment is perfect, results should match the reference exactly
        self.assertEqual(
            get_aligned_phonemes(example),
            ["wʌn", "tuː", "θriː"],
        )

    def test_misaligned(self):
        """returns aligned phoneme data from misaligned training examples"""
        nlp = spacy.blank("en")
        example = example_from_phonemes_dict(
            nlp.make_doc("one two three"),
            {
                "words": ["on", "e", "two", "three"],
                "phonemes": ["wʌ", "n", "tuː", "θriː"],
            },
        )

        # aligned tokens should have values; misaligned should be None
        self.assertEqual(
            get_aligned_phonemes(example),
            [None, "tuː", "θriː"],
        )


class TestExampleFromPhonemesDict(TestCase):
    def test_no_phonemes_data(self):
        """with no supplied phonemes data, works just like Example.from_dict"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        example = example_from_phonemes_dict(
            doc,
            {
                "words": ["on", "e", "two", "three"],
                "SPACY": [True, True, True, False],
            },
        )
        self.assertEqual(example.predicted, doc)
        self.assertEqual(example.reference.text, "on e two three")

    def test_phonemes_data(self):
        """when phonemes are supplied, sets them on Example.reference"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        example = example_from_phonemes_dict(doc, {"phonemes": ["wʌn", "tuː", "θriː"]})
        self.assertEqual(example.reference._.phonemes, ["wʌn", "tuː", "θriː"])

    def test_wrong_length_phonemes_data(self):
        """raises an error if the wrong number of phonemes are supplied"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        with self.assertRaises(ValueError):
            example_from_phonemes_dict(doc, {"phonemes": ["wʌn", "tuː", "θriː", "f"]})
