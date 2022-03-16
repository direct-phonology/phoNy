from typing import Any, Dict, Iterable

from spacy.scorer import Scorer
from spacy.training import Example
from spacy.util import registry


def phoneme_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    return Scorer.score_token_attr(
        examples,
        attr="phonemes",
        getter=lambda t, attr: t._.get(attr),
        missing_values=set("_"),
        **kwargs,
    )


@registry.scorers("phoneme_scorer.v1")
def make_phoneme_scorer():
    return phoneme_score
