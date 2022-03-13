from itertools import islice, zip_longest
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy
from spacy.errors import Errors
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.scorer import Scorer
from spacy.tokens import Doc, Token
from spacy.training import Example, validate_examples, validate_get_examples
from spacy.util import check_lexeme_norms, registry
from spacy.vocab import Vocab
from thinc.api import Config, Model, SequenceCategoricalCrossentropy
from thinc.types import Floats2d, Ints1d

# register custom phonemes property
if not Token.has_extension("phonemes"):
    Token.set_extension("phonemes", default=None)


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


default_model_config = """
[model]
@architectures = "spacy.Tagger.v1"

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 96
depth = 4
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = false
"""
DEFAULT_PHONEMIZER_MODEL = Config().from_str(default_model_config)["model"]


class Phonemizer(TrainablePipe):
    """Pipeline component for grapheme-to-phoneme conversion."""

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "phon",
        *,
        scorer=phoneme_score,
    ) -> None:
        """Initialize a grapheme-to-phoneme converter."""
        self.vocab = vocab
        self.model = model
        self.name = name
        cfg: Dict[str, Any] = {"labels": []}
        self.cfg = dict(sorted(cfg.items()))
        self.scorer = scorer

    @property
    def labels(self) -> Tuple[str, ...]:
        """Return the labels currently added to the pipe."""
        return tuple(self.cfg["labels"])

    def add_label(self, label: str) -> int:
        """Add a label to the pipe. Return 0 if label already exists, else 1."""
        if not isinstance(label, str):
            raise ValueError("Phonemizer labels must be strings")
        if label in self.labels:
            return 0
        self.cfg["labels"].append(label)
        return 1

    def predict(self, docs: List[Doc]) -> List[Ints1d]:
        """Predict annotations for a batch of Docs, without modifying them."""
        # Handle cases where there are no tokens in any docs: return empty set
        # of scores for each doc
        if not any(len(doc) for doc in docs):
            n_labels = len(self.labels)
            guesses = [self.model.ops.alloc1i(n_labels) for _ in docs]
            return guesses

        # Get the scores predicted by the model; we should have have the same
        # (non-zero) number of both score sets and docs
        scores = self.model.predict(docs)
        assert len(scores) == len(docs), (len(scores), len(docs))

        # Pick the highest-scoring guess for each token in each doc; we should
        # (still) have the same number of guess sets as docs
        guesses = []
        for doc_scores in scores:
            doc_guesses = doc_scores.argmax(axis=1)
            if not isinstance(doc_guesses, numpy.ndarray):
                doc_guesses = doc_guesses.get()
            guesses.append(doc_guesses)
        assert len(guesses) == len(docs)
        return guesses

    def set_annotations(self, docs: Iterable[Doc], tag_ids: List[Ints1d]) -> None:
        """Annotate a batch of Docs, using pre-computed IDs."""
        labels = self.labels
        for doc, doc_tag_ids in zip_longest(docs, tag_ids):
            if not doc or not doc_tag_ids.any():
                continue
            if hasattr(doc_tag_ids, "get"):
                doc_tag_ids = doc_tag_ids.get()
            for token, tag_id in zip(list(doc), doc_tag_ids):
                token._.phonemes = labels[tag_id]

    def get_loss(
        self,
        examples: Iterable[Example],
        guesses: List[Floats2d],
    ) -> Tuple[float, List[Floats2d]]:
        """Compute the loss and gradient for a batch of examples and guesses."""
        # Create loss function
        validate_examples(examples, "Phonemizer.get_loss")
        loss_func = SequenceCategoricalCrossentropy(normalize=False)

        # Compute loss & gradient
        truths = self._examples_to_truth(examples)
        gradient, loss = loss_func(guesses, truths)  # type: ignore
        if self.model.ops.xp.isnan(loss):
            raise ValueError(Errors.E910.format(name=self.name))
        return float(loss), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
    ):
        """Initialize the pipe for training using a set of examples."""
        validate_get_examples(get_examples, "Phonemizer.initialize")
        check_lexeme_norms(self.vocab, "phonemizer")

        # Read all unique tags from the examples and add them
        tags = set()
        for example in get_examples():
            for token in example.predicted:
                if token._.phonemes:
                    tags.add(token._.phonemes)
        for tag in sorted(tags):
            self.add_label(tag)

        # Use the first 10 examples to sample Docs and labels
        doc_sample = []
        label_sample = []
        n_labels = len(self.labels)
        for example in islice(get_examples(), 10):
            doc_sample.append(example.reference)
            labeled = self._examples_to_truth([example])
            if labeled:
                label_sample += labeled
            else:
                label_sample.append(
                    self.model.ops.alloc2f(len(example.reference), n_labels)
                )

        # Initialize the model
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(
        self,
        examples: Iterable[Example],
    ) -> Optional[List[Floats2d]]:
        """Get the gold-standard labels for a batch of examples."""
        # Handle cases where there are no annotations in any examples
        tag_count = 0
        for example in examples:
            tag_count += len(
                list(filter(None, [token._.phonemes for token in example.reference]))
            )
        if tag_count == 0:
            return None

        # Get all the true labels
        truths = []
        for example in examples:
            gold_tags = self._get_aligned_phonemes(example)

            # Make a one-hot array for correct tag for each token
            gold_array = [
                [1.0 if tag == gold_tag else 0.0 for tag in self.labels]
                for gold_tag in gold_tags
            ]
            truths.append(self.model.ops.asarray2f(gold_array))  # type: ignore

        return truths

    def _get_aligned_phonemes(self, example: Example) -> List[Optional[str]]:
        """Get the aligned phonology data for a training Example."""
        alignment = example.alignment.x2y
        gold_phon = [token._.phonemes for token in example.reference]
        output: List[Optional[str]] = [None] * len(example.predicted)

        # Handle case where example has no tokens
        if not output:
            return []

        for token in example.predicted:
            if not token.is_alpha:
                output[token.i] = None
            else:
                output[token.i] = gold_phon[alignment[token.i].dataXd[0][0]]

        return output


@Language.factory(
    "phonemizer",
    assigns=["token._.phonemes"],
    default_config={
        "model": DEFAULT_PHONEMIZER_MODEL,
        "scorer": {"@scorers": "phoneme_scorer.v1"},
    },
    default_score_weights={"phon_acc": 1.0},
)
def make_phonemizer(
    nlp: Language,
    model: Model[List[Doc], List[Floats2d]],
    name: str,
    scorer: Optional[Callable],
) -> Phonemizer:
    """Construct a Phonemizer component."""
    return Phonemizer(nlp.vocab, model, name=name, scorer=scorer)
