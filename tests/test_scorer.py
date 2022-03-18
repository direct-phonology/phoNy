from unittest import TestCase

import spacy

from phony.scorer import phoneme_score
from phony.training import example_from_phonemes_dict


class TestPhonemeScorer(TestCase):
    def test_perfect_score(self):
        """returns a perfect score for a perfect example"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        doc._.phonemes = ["wʌn", "tuː", "θriː"]
        example = example_from_phonemes_dict(
            doc,
            {"phonemes": ["wʌn", "tuː", "θriː"]},
        )
        scores = phoneme_score([example])
        self.assertEqual(scores["phonemes_acc"], 1.0)

    def test_imperfect_score(self):
        """score is proportional to length of document"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three four")
        doc._.phonemes = ["wʌn", "tuː", "θriː", "fɔr"]
        example = example_from_phonemes_dict(
            doc,
            {"phonemes": ["wʌn", "wʌn", "θriː", "fɔr"]},  # 1 error of 4
        )
        scores = phoneme_score([example])
        self.assertEqual(scores["phonemes_acc"], 0.75)
