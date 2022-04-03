from typing import List, Tuple

from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Model, MultiSoftmax, chain, with_array
from thinc.types import Floats2d


@registry.architectures("phony.MultiTagger.v1")
def build_multi_tagger_model(
    tok2vec: Model[List[Doc], List[Floats2d]], nOs: Tuple[int]
) -> Model[List[Doc], List[Floats2d]]:
    """Build a tagger model, using a provided token-to-vector component. The tagger
    model adds a linear layer with multi-softmax activation to predict several
    multi-class attributes at once based on the token vectors.

    Args:
        tok2vec: The token-to-vector component.
        nOs: The number of variables per class.
    """
    # based on https://github.com/explosion/spaCy/blob/master/spacy/ml/models/tagger.py
    t2v_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else None
    output_layer = MultiSoftmax(nOs, t2v_width)
    softmax = with_array(output_layer)  # type: ignore
    model = chain(tok2vec, softmax)
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("softmax", output_layer)
    model.set_ref("output_layer", output_layer)
    return model
