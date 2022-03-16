from typing import List

import numpy as np
from spacy.tokens import Doc, Token


def to_phonemes_array(doc: Doc) -> np.ndarray:
    """Get the phoneme data for a Doc as a numpy array of IDs."""
    # replacement for spacy's Doc.to_array(), which doesn't work on custom
    # extension attributes.
    gold_values = [0] * len(doc)

    for i, token in enumerate(doc):
        if token._.phonemes:
            gold_values[i] = token._.phonemes

    return np.asarray(gold_values, dtype=np.uint64)


def get_token_phonemes_(token: Token) -> str:
    """Get the phoneme data for a Token as a string."""
    return token.vocab.strings[token._.phonemes]  # type: ignore


def set_token_phonemes_(token: Token, phonemes: str) -> None:
    """Set the phoneme data for a Token by providing a string."""
    token._.phonemes = token.vocab.strings.add(phonemes)  # type: ignore


def get_doc_phonemes(doc: Doc) -> List[int]:
    """Get the phoneme data for a Doc as a list of IDs."""
    return [token._.phonemes for token in doc]


def get_doc_phonemes_(doc: Doc) -> List[str]:
    """Get the phoneme data for a Doc as a list of strings."""
    return [token._.phonemes_ for token in doc]


def register_attrs():
    """Helper function to register custom extension attributes."""
    # token phonemes (assigned by phonemizer)
    if not Token.has_extension("phonemes"):
        Token.set_extension("phonemes", default="")
    if not Token.has_extension("phonemes_"):
        Token.set_extension(
            "phonemes_",
            getter=get_token_phonemes_,
            setter=set_token_phonemes_,
        )

    # doc phonemes (delegates to tokens)
    if not Doc.has_extension("phonemes"):
        Doc.set_extension("phonemes", getter=get_doc_phonemes)
    if not Doc.has_extension("phonemes_"):
        Doc.set_extension("phonemes_", getter=get_doc_phonemes_)


register_attrs()
