from typing import List

from spacy.tokens import Doc

from ..util import register_attrs

register_attrs()


def to_phonemes_array(doc: Doc) -> List[int]:
    """Get the phoneme data for a Doc as an array of hashes."""
    # replacement for spacy's Doc.to_array(), which doesn't work on custom
    # extension attributes.
    gold_values = [0] * len(doc)

    for i, token in enumerate(doc):
        if token._.phonemes:
            gold_values[i] = token._.phonemes

    return gold_values
