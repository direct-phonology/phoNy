from unittest import TestCase, skip
from unittest.mock import Mock

import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch
from thinc.api import compounding

from phony.training import example_from_phonemes_dict

from . import MockCupyNdarray


class TestPhonemizer(TestCase):
    def test_set_annotations(self):
        """should set provided annotations for provided list of docs"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        doc = nlp.make_doc("one two three")
        tag_ids = np.asarray([0, 1, 2])  # wʌn, tuː, θriː
        phonemizer.set_annotations([doc], [tag_ids])
        self.assertEqual(doc._.phonemes, ["wʌn", "tuː", "θriː"])

    @skip("fixme — see https://github.com/direct-phonology/och-g2p/issues/12")
    def test_set_punct_annotations(self):
        """should not set an annotation for non-alphabetic tokens"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        doc = nlp.make_doc("one. two")
        tag_ids = np.asarray([1, 1, 2])  # pretend we predicted "wʌn" for "."
        phonemizer.set_annotations([doc], [tag_ids])
        self.assertEqual(doc._.phonemes, ["wʌn", None, "θriː"])

    def test_set_annotations_gpu(self):
        """should handle setting annotations based on predictions using gpu"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        doc = nlp.make_doc("one two three")
        tag_ids = MockCupyNdarray(np.asarray([0, 1, 2]))  # wʌn, tuː, θriː
        phonemizer.set_annotations([doc], [tag_ids])
        self.assertEqual(doc._.phonemes, ["wʌn", "tuː", "θriː"])

    def test_add_new_label(self):
        """should add provided label and return 1 if it didn't exist"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        self.assertEqual(phonemizer.add_label("foo"), 1)
        self.assertIn("foo", phonemizer.labels)

    def test_add_existing_label(self):
        """should return 0 if provided label to add already exists"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        self.assertEqual(phonemizer.add_label("wʌn"), 0)
        self.assertIn("wʌn", phonemizer.labels)

    def test_add_invalid_label(self):
        """should raise exception if provided label to add isn't a string"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        with self.assertRaises(ValueError):
            phonemizer.add_label(1)

    def test_get_labels(self):
        """should return labels as tuple if requested"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        self.assertEqual(phonemizer.labels, ("wʌn", "tuː", "θriː"))

    def test_predict(self):
        """should choose highest-scoring guess as prediction for each token"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        predictions = [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        phonemizer.model = Mock()
        phonemizer.model.predict.return_value = np.array(predictions, dtype=np.float32)
        doc = nlp.make_doc("one two three")
        guesses = phonemizer.predict([doc])
        self.assertEqual(guesses[0].tolist(), [0, 1, 2])  # wʌn, tuː, θriː

    def test_predict_empty_docs(self):
        """predicting on docs without any tokens shouldn't cause errors"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        doc = nlp.make_doc("")
        phonemizer.model = Mock()
        phonemizer.model.ops.alloc1i = lambda size: np.zeros(size, dtype=np.int32)
        guesses = phonemizer.predict([doc])
        self.assertEqual(guesses[0].tolist(), [])

    def test_predict_gpu(self):
        """should handle predictions made on the gpu"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.model = Mock()
        predictions = [
            MockCupyNdarray(
                np.array(
                    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                    dtype=np.float32,
                )
            )
        ]
        phonemizer.model.predict.return_value = predictions
        doc = nlp.make_doc("one two three")
        guesses = phonemizer.predict([doc])
        self.assertEqual(guesses[0].tolist(), [0, 1, 2])  # wʌn, tuː, θriː

    def test_initialize(self):
        """should initialize component with sorted labels from training data"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        doc = nlp.make_doc("one two three")
        example = example_from_phonemes_dict(
            doc,
            {"phonemes": ["wʌn", "tuː", "θriː"]},
        )
        phonemizer.initialize(lambda: [example])
        self.assertEqual(phonemizer.labels, ("tuː", "wʌn", "θriː"))

    def test_initialize_no_data(self):
        """should error if initialized with docs missing training data"""
        nlp = spacy.blank("en")
        doc = nlp.make_doc("one two three")
        example = Example.from_dict(doc, {})
        nlp.add_pipe("phonemizer")
        with self.assertRaises(ValueError):
            nlp.initialize(get_examples=lambda: [example])

    def test_initialize_incomplete_data(self):
        """should handle initialization with misaligned/partial training data"""
        nlp = spacy.blank("en")
        nlp.add_pipe("phonemizer")

        # partial data
        example1 = example_from_phonemes_dict(
            nlp.make_doc("four one two three five"),
            {"phonemes": [None, "wʌn", "tuː", "θriː", None]},
        )

        # misaligned partial data
        example2 = example_from_phonemes_dict(
            nlp.make_doc("five one two three four"),
            {
                "words": ["five", "on", "e", "two", "three", "four"],
                "phonemes": [None, "wʌ", "n", "tuː", "θriː", None],
            },
        )

        # train for awhile
        optimizer = nlp.initialize(get_examples=lambda: [example1, example2])
        for _ in range(50):
            losses = {}
            nlp.update([example1, example2], sgd=optimizer, losses=losses)

        # should still make correct predictions
        doc = nlp("two three")
        self.assertEqual(doc._.phonemes, ["tuː", "θriː"])

    def test_train(self):
        """training should produce predictable results"""
        # set up pipeline and training docs
        nlp = spacy.blank("en")
        nlp.add_pipe("phonemizer")
        examples = []
        examples.append(
            example_from_phonemes_dict(
                nlp.make_doc("one two three"),
                {"phonemes": ["wʌn", "tuː", "θriː"]},
            )
        )
        examples.append(
            example_from_phonemes_dict(
                nlp.make_doc("three two one one three"),
                {"phonemes": ["θriː", "tuː", "wʌn", "wʌn", "θriː"]},
            )
        )

        # train for awhile, loss should resolve to 0
        optimizer = nlp.initialize(get_examples=lambda: examples)
        for _ in range(50):
            losses = {}
            nlp.update(examples, sgd=optimizer, losses=losses)
        self.assertLess(losses["phonemizer"], 0.00001)

        # test the trained model
        doc = nlp("one two one three")
        self.assertEqual(doc._.phonemes, ["wʌn", "tuː", "wʌn", "θriː"])

    def test_train_empty_data(self):
        """data with empty annotations shouldn't cause errors during training"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        example = Example.from_dict(nlp.make_doc(""), {})
        train_data = [example, example]
        optimizer = nlp.initialize()
        for _ in range(5):
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)
