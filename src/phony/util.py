from spacy.tokens import Token


def register_attrs():
    """Helper function to register custom extension attributes."""
    # token phonemes (assigned by phonemizer)
    if not Token.has_extension("phonemes"):
        Token.set_extension("phonemes", default=None)
