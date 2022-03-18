from typing import List, Optional

import numpy as np
from spacy.tokens import Doc
from spacy.training import Example


def get_aligned_phonemes(example: Example) -> List[Optional[str]]:
    """Get the aligned phoneme data for a training Example."""
    # replacement for spacy's Example.get_aligned(), which doesn't work on
    # custom extension attributes.
    align = example.alignment.x2y
    gold_values = np.asarray(example.reference._.phonemes)
    output: List[Optional[str]] = [None] * len(example.predicted)
    for token in example.predicted:
        values = gold_values[align[token.i].dataXd]
        values = values.ravel()
        if len(values) == 0:
            output[token.i] = None
        elif len(values) == 1:
            output[token.i] = values[0]
        elif len(set(list(values))) == 1:
            output[token.i] = values[0]
        else:
            output[token.i] = None

    return output


def example_from_phonemes_dict(predicted: Doc, data: dict) -> Example:
    """Create an Example from an existing Doc plus phoneme data."""
    # replacement for spacy's Example.from_dict(), which doesn't work on
    # custom extension attributes.
    phonemes_data = data.pop("phonemes", None)
    example = Example.from_dict(predicted, data)

    # if no phoneme data, just return the Example as normal
    if not phonemes_data:
        return example

    # otherwise add provided phoneme data to the reference doc
    if len(phonemes_data) != len(example.reference):
        raise ValueError("Wrong number of phonemes in example data dict")
    for i, p in enumerate(phonemes_data):
        example.reference[i]._.phonemes = p

    return example
