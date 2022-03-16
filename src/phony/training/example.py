from typing import List, Optional

from spacy.training import Example

from ..tokens.doc import to_phonemes_array


def get_aligned_phonemes(
    example: Example, as_string: bool = False
) -> List[Optional[str]]:
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
