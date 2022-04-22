from unittest import TestCase

import numpy as np

from phony.phonology import (
    Backness,
    Consonant,
    Height,
    Length,
    Manner,
    Place,
    Roundedness,
    Syllable,
    Voicing,
    Vowel,
)

# some examples for testing
p = Consonant(Voicing.voiceless, Place.bilabial, Manner.stop)
b = Consonant(Voicing.voiced, Place.bilabial, Manner.stop)
t = Consonant(Voicing.voiceless, Place.alveolar, Manner.stop)
ɪ = Vowel(Height.near_close, Backness.front, Roundedness.unrounded)

pit = Syllable(p, ɪ, t)  # /pɪt/ english "pit"
bit = Syllable(b, ɪ, t)  # /bɪt/ english "bit"


class TestPhoneme(TestCase):
    def test_phoneme_features(self):
        """phonemes are sets of phonological features"""
        # check presence of features
        self.assertIn(Voicing.voiceless, p.features)
        self.assertIn(Place.bilabial, b.features)
        self.assertIn(Manner.stop, t.features)

        # features that weren't explicitly specified should still be listed
        self.assertEqual(len(ɪ.features), 4)
        self.assertIn(Length.short, ɪ.features)  # short is the default

        # can make an identical phoneme from same features
        new_p = Consonant(*p.features)
        self.assertEqual(p, new_p)

        # features can be added and subtracted from the phoneme

        # adding a copy of an existing feature does nothing

    def test_phoneme_array(self):
        """phonemes can be converted to/from numpy arrays"""
        # conversion into arrays
        self.assertEqual(p.to_array().tolist(), [1, 1, 1, 1])
        self.assertEqual(b.to_array().tolist(), [3, 1, 1, 1])
        self.assertEqual(ɪ.to_array().tolist(), [2, 1, 1, 1])

        # generating from an array
        self.assertEqual(Consonant.from_array(np.array([1, 1, 1, 1])), p)
        self.assertEqual(Consonant.from_array(np.array([3, 1, 1, 1])), b)
        self.assertEqual(Vowel.from_array(np.array([2, 1, 1, 1])), ɪ)

        # equality
        self.assertEqual(p, Consonant.from_array(p.to_array()))
        self.assertEqual(b, Consonant.from_array(b.to_array()))
        self.assertEqual(ɪ, Vowel.from_array(ɪ.to_array()))

    def test_phoneme_str(self):
        """phonemes have a useful string representation"""
        # default representation
        self.assertEqual(str(p), "voiceless bilabial stop")

        # drop the "modal" in modal voiced
        self.assertEqual(str(b), "voiced bilabial stop")

        # hyphenate multi-word features
        self.assertEqual(str(ɪ), "near-close front unrounded vowel")

    def test_phoneme_similarity(self):
        pass


class TestSyllable(TestCase):
    def test_syllable_features(self):
        """syllables are sets of phonological features"""
        # check presence of features
        self.assertEqual(len(pit.features), 12)
        self.assertIn(Voicing.voiceless, pit.features)
        self.assertIn(Voicing.voiced, pit.features)

        # only unique features are stored

    def test_syllable_phonemes(self):
        """syllables are ordered groups of phonemes"""
        self.assertEqual(pit.phonemes, (p, ɪ, t))
        self.assertEqual(bit.phonemes, (b, ɪ, t))

    def test_syllable_str(self):
        pass

    def test_syllable_array(self):
        pass

    def test_syllable_similarity(self):
        pass
