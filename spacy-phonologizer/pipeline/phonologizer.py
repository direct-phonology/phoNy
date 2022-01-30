from typing import Callable, Dict, Optional, Union, Any

from spacy.language import Language
from spacy.pipeline.tagger import Tagger
from spacy.scorer import Scorer
from spacy.util import registry
from spacy.vocab import Vocab
from thinc.api import Config, Model

default_model_config = """
[model]
@architectures = "spacy.Tagger.v1"

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[model.tok2vec.embed]
@architectures = "spacy.CharacterEmbed.v2"
width = 128
rows = 7000
nM = 64
nC = 8
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 128
depth = 4
window_size = 1
maxout_pieces = 3
"""

DEFAULT_MORPH_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "phonologizer",
    assigns=["token.phon"],
    default_config={
        "model": DEFAULT_MORPH_MODEL,
        "scorer": {"@scorers": "spacy.phonologizer_scorer.v1"},
    },
    default_score_weights={"phon_acc": 0.5, "phon_per_feat": None},
)
def make_phonologizer(
    nlp: Language,
    model: Model,
    name: str,
    scorer: Optional[Callable],
):
    return Phonologizer(nlp.vocab, model, name, scorer=scorer)


def phonologizer_score(examples, **kwargs):
    def phon_key_getter(token, attr):
        return getattr(token, attr).key

    results = {}
    results.update(Scorer.score_token_attr(examples, "pos", **kwargs))
    results.update(
        Scorer.score_token_attr(examples, "phon", getter=phon_key_getter, **kwargs)
    )
    results.update(
        Scorer.score_token_attr_per_feat(
            examples, "phon", getter=phon_key_getter, **kwargs
        )
    )
    return results


@registry.scorers("spacy.phonologizer_scorer.v1")
def make_phonologizer_scorer():
    return phonologizer_score


class Phonologizer(Tagger):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "phonologizer",
        *,
        scorer: Optional[Callable] = phonologizer_score,
    ):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.scorer = scorer
        self._rehearsal_model = None
        self.labels_phon: Any = {}

    @property
    def labels(self):
        """The labels currently added to the component."""
        return tuple(self.labels_phon.keys())

    @property
    def label_data(self) -> Dict[str, Dict[str, Union[str, float, int, None]]]:
        """A dictionary with all labels data."""
        return self.labels_phon

    def add_label(self, label):
        """Add a new label to the pipe."""
        pass

    def initialize(self, get_examples, *, nlp=None, labels=None):
        """Initialize the pipe."""
        pass

    def set_annotations(self, docs, batch_tag_ids):
        pass

    def get_loss(self, examples, scores):
        pass
