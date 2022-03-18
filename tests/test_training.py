from unittest import TestCase

import spacy

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
