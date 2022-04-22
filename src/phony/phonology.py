from enum import IntEnum, auto
from typing import Iterator, List, Optional, Sequence, Set, Union

import numpy as np

from .ml import Arrayable


class PhonologicalFeature(IntEnum):
    def __str__(self) -> str:
        return self.name.replace("_", "-")


class Voicing(PhonologicalFeature):
    """
    Voicing is the tension of the vocal cords when producing a sound.

    Values are ordered by increasing levels of tension in the glottis, with
    "voiceless" indicating total relaxation of the vocal cords, as with
    obstruents. Glottal closure commonly occurs in stops, while "voiced"
    refers to the normal state for vowels and sonorants, also called "modal
    voice".

    Some languages use contrasting but weaker departures from the modal voice,
    called "slack voice" and "stiff voice", or instead distinguish voicing
    among consonants via "tension". We alias these values so that the
    appropriate terminology can be used with the same numeric result.

    For more, see: https://en.wikipedia.org/wiki/Phonation
    """

    voiceless = auto()
    breathy_voiced = slack_voiced = lax = auto()
    voiced = modal_voiced = plain = auto()
    creaky_voiced = stiff_voiced = tense = auto()
    glottal_closure = auto()


class Place(PhonologicalFeature):
    """
    Place describes the location in the mouth where a consonant is articulated.

    Values are ordered by location in the mouth from front to back, focusing
    on the place of passive articulation, which is the more stationary part
    of the vocal tract.

    Many "in-between" values, such as "pre-velar", "post-velar", etc. which
    may be used to more precisely specify articulation are not represented here.
    The intent is to represent articulation broadly on a continuum.

    For more, see: https://en.wikipedia.org/wiki/Place_of_articulation
    """

    bilabial = auto()
    labiodental = auto()
    dental = auto()
    alveolar = auto()
    postalveolar = auto()
    retroflex = auto()
    palatal = auto()
    velar = auto()
    uvular = auto()
    pharyngeal = auto()
    glottal = auto()


class Manner(PhonologicalFeature):
    stop = auto()
    nasal = auto()
    trill = auto()
    tap = flap = auto()
    affricate = auto()
    fricative = auto()
    approximant = auto()


class Aspiration(PhonologicalFeature):
    """
    Aspiration is a burst of air in the pronunciation of a consonant.

    Typically only voiceless consonants can be aspirated, with stops and
    affricates being the most common. Voiced aspirated consonants can be
    represented instead using breathy voice.

    For more, see: https://en.wikipedia.org/wiki/Aspirated_consonant
    """

    unaspirated = auto()
    aspirated = auto()


class Roundedness(PhonologicalFeature):
    """
    Roundedness is the amount of rounding in the lips when articulating a vowel.

    In general, front vowels tend to be unrounded, while back vowels tend to be
    rounded. Roundedness is analogous to labialization in consonants, and the
    two are often related at the level of the syllable.

    For more, see: https://en.wikipedia.org/wiki/Roundedness
    """

    unrounded = auto()
    rounded = auto()


class Length(PhonologicalFeature):
    short = auto()
    long = auto()
    overlong = auto()


class Gemination(PhonologicalFeature):
    ungeminated = auto()
    geminated = auto()


# order for Height, Backness, and Manner is important
# the feature values must be ordered by *increasing sonority*
class Height(PhonologicalFeature):
    close = auto()
    near_close = auto()
    close_mid = auto()
    mid = auto()
    open_mid = auto()
    near_open = auto()
    open = auto()


class Backness(PhonologicalFeature):
    front = auto()
    near_front = auto()
    central = auto()
    back = auto()
    near_back = auto()


# TODO look at
# https://github.com/explosion/spaCy/blob/master/spacy/tokens/morphanalysis.pyx
# also
# https://github.com/explosion/spaCy/blob/master/spacy/lexeme.pyx


class Phoneme(Arrayable):
    def __init__(self, *features: PhonologicalFeature) -> None:
        self.features = set(features)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}{self.features}>"

    def __iter__(self) -> Iterator[PhonologicalFeature]:
        return iter(self.features)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.features == other.features

    def to_array(self) -> np.ndarray:
        return np.array([feat.value for feat in self.features], dtype=np.uint8)


class Tone(PhonologicalFeature):
    pass


# voicing, aspiration, place, manner, length


class Consonant(Phoneme):
    def __init__(
        self,
        voicing: Voicing,
        place: Place,
        manner: Manner,
        aspiration: Aspiration = Aspiration.unaspirated,
        gemination: Gemination = Gemination.ungeminated,
    ) -> None:
        super().__init__(voicing, place, manner, aspiration, gemination)
        self.voicing = voicing
        self.place = place
        self.manner = manner
        self.aspiration = aspiration
        self.gemination = gemination

    def __str__(self) -> str:
        return f"{self.voicing} {self.place} {self.manner}"

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Consonant":
        return cls(
            Voicing(array[0]),
            Place(array[1]),
            Manner(array[2]),
            Aspiration(array[3]),
            Gemination(array[4]),
        )


# height, backness, roundedness, length


class Vowel(Phoneme):
    def __init__(
        self,
        height: Height,
        backness: Backness,
        roundedness: Roundedness,
        length: Length = Length.short,
    ) -> None:
        super().__init__(height, backness, roundedness, length)
        self.height = height
        self.backness = backness
        self.roundedness = roundedness
        self.length = length

    def __str__(self) -> str:
        return f"{self.height} {self.backness} {self.roundedness} vowel"

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Vowel":
        return cls(
            Height(array[0]),
            Backness(array[1]),
            Roundedness(array[2]),
            Length(array[3]),
        )


# see https://github.com/explosion/spaCy/blob/master/spacy/vocab.pyx
class Inventory:
    pass


Segment = Union[Phoneme, Sequence[Phoneme]]
ConsonantSegment = Union[Consonant, Sequence[Consonant]]
VowelSegment = Union[Vowel, Sequence[Vowel]]


class Syllable(Arrayable):
    def __init__(
        self,
        initial: Optional[ConsonantSegment],
        nucleus: VowelSegment,
        coda: Optional[ConsonantSegment],
        **suprasegmentals: PhonologicalFeature,
    ) -> None:
        self.initial = initial
        self.nucleus = nucleus
        self.coda = coda
        self.suprasegmentals = suprasegmentals

    @property
    def features(self) -> Set[PhonologicalFeature]:
        seg_feats = set([feat for phon in self.phonemes for feat in phon])
        supra_feats = set(self.suprasegmentals.values())
        return seg_feats | supra_feats

    @property
    def segments(self) -> List[Segment]:
        return [seg for seg in [self.initial, self.nucleus, self.coda] if seg]

    @property
    def phonemes(self) -> List[Phoneme]:
        phonemes: List[Phoneme] = []
        for seg in [self.initial, self.nucleus, self.coda]:
            phonemes.extend(list(*seg))
        return phonemes

    @property
    def final(self) -> List[Segment]:
        return [seg for seg in [self.nucleus, self.coda] if seg]

    @property
    def rime(self) -> List[Segment]:
        return self.final

    def to_array(self) -> np.ndarray:
        seg_vals = [feat.value for phon in self.phonemes for feat in phon]
        supra_vals = [feat.value for _key, feat in self.suprasegmentals.items()]
        return np.array(seg_vals + supra_vals, dtype=np.uint8)
