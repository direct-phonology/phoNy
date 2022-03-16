from typing import List, Optional, Union

from spacy.tokens import Doc
from spacy.training import Example

from .tokens import to_phonemes_array


def get_aligned_phonemes(
    example: Example, as_string: bool = False
) -> Union[List[Optional[str]], List[Optional[int]]]:
    """Get the aligned phoneme data for a training Example."""
    # replacement for spacy's Example.get_aligned(), which doesn't work on
    # custom extension attributes.
    align = example.alignment.x2y

    # construct an aligned list of phonemes from the example
    vocab = example.reference.vocab
    gold_values = to_phonemes_array(example.reference)
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

    # convert to string if requested; otherwise output hashes
    if as_string:
        return [vocab.strings[o] if o is not None else o for o in output]

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

    # otherwise hash and add provided phoneme data to the reference doc
    vocab = predicted.vocab
    if "phonemes" in data:
        if len(data["phonemes"] != len(example.reference)):
            raise ValueError("Wrong number of phonemes in example data dict")
        for i, p in enumerate(data["phonemes"]):
            example.reference[i]._.phonemes = vocab.strings[p]

    return example
