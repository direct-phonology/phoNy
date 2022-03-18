from typing import Iterable, List

from spacy.tokens import Doc, Span, Token


def get_doc_phonemes(doc: Doc) -> List[str]:
    """Get the phoneme data for a Doc as a list of strings."""
    return [token._.phonemes for token in doc]


def set_doc_phonemes(doc: Doc, phonemes: Iterable[str]):
    """Set the phoneme data for a Doc by providing a list of strings."""
    for token, phoneme in zip(doc, phonemes):
        token._.phonemes = phoneme


def get_span_phonemes(span: Span) -> List[str]:
    """Get the phoneme data for a Span as a list of strings."""
    return [token._.phonemes for token in span]


def register_attrs():
    """Helper function to register custom extension attributes."""
    # token phonemes (assigned by phonemizer)
    if not Token.has_extension("phonemes"):
        Token.set_extension("phonemes", default=None)

    # span phonemes (delegates to tokens)
    if not Span.has_extension("phonemes"):
        Span.set_extension("phonemes", getter=get_span_phonemes)

    # doc phonemes (delegates to tokens)
    if not Doc.has_extension("phonemes"):
        Doc.set_extension(
            "phonemes",
            getter=get_doc_phonemes,
            setter=set_doc_phonemes,
        )


register_attrs()
