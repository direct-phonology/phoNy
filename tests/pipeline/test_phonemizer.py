from unittest import TestCase, skip
from unittest.mock import ANY, Mock

import numpy as np
import spacy
from spacy.training import Example

from phony.pipeline import Phonemizer


class TestPhonemizer(TestCase):
    def setUp(self):
        # create a mock spacy language, ml model, and pipeline component
        self.nlp = spacy.blank("en")
        self.model = Mock()
        self.phonemizer = Phonemizer(self.nlp.vocab, self.model)

        # add some labels to the pipe
        self.phonemizer.add_label("wʌn")
        self.phonemizer.add_label("tuː")
        self.phonemizer.add_label("θriː")

    def test_set_annotations(self):
        """should set provided annotations for provided list of docs"""
        doc = self.nlp.make_doc("one two three")
        tag_ids = np.asarray([0, 1, 2])  # wʌn, tuː, θriː
        self.phonemizer.set_annotations([doc], [tag_ids])
        self.assertEqual(doc[0]._.phonemes, "wʌn")
        self.assertEqual(doc[1]._.phonemes, "tuː")
        self.assertEqual(doc[2]._.phonemes, "θriː")

    @skip("fixme — see https://github.com/direct-phonology/och-g2p/issues/12")
    def test_set_punct_annotations(self):
        """should not set an annotation for non-alphabetic tokens"""
        doc = self.nlp.make_doc("one. two")
        tag_ids = np.asarray([1, 1, 2])  # pretend we predicted "wʌn" for "."
        self.phonemizer.set_annotations([doc], [tag_ids])
        self.assertEqual(doc[0]._.phonemes, "wʌn")
        self.assertEqual(doc[1]._.phonemes, None)
        self.assertEqual(doc[2]._.phonemes, "θriː")

    def test_add_new_label(self):
        """should add provided label and return 1 if it didn't exist"""
        self.assertEqual(self.phonemizer.add_label("foo"), 1)
        self.assertIn("foo", self.phonemizer.labels)

    def test_add_existing_label(self):
        """should return 0 if provided label to add already exists"""
        self.assertEqual(self.phonemizer.add_label("wʌn"), 0)
        self.assertIn("wʌn", self.phonemizer.labels)

    def test_add_invalid_label(self):
        """should raise exception if provided label to add isn't a string"""
        with self.assertRaises(ValueError):
            self.phonemizer.add_label(1)

    def test_get_labels(self):
        """should return labels as tuple if requested"""
        self.assertEqual(self.phonemizer.labels, ("wʌn", "tuː", "θriː"))

    def test_predict(self):
        """should produce predictions for provided list of docs using model"""
        doc = self.nlp.make_doc("one two three")
        predictions = [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
        self.model.predict.return_value = np.array(predictions, dtype=np.float32)
        guesses = self.phonemizer.predict([doc])
        self.assertEqual(guesses[0].tolist(), [0, 1, 2])  # wʌn, tuː, θriː

    def test_predict_empty_docs(self):
        """should handle cases where predicted docs have no tokens"""
        doc = self.nlp.make_doc("")
        self.model.ops.alloc1i = lambda size: np.zeros(size, dtype=np.int32)
        guesses = self.phonemizer.predict([doc])
        self.assertEqual(guesses[0].tolist(), [0, 0, 0])

    def test_predict_gpu(self):
        """should handle predictions made on the gpu"""

        # create a mock version of cupy's ndarray that implements get()
        class MockCupyNdarray:
            def __init__(self, data: np.ndarray):
                self.data = data

            def get(self):
                return self.data

            def argmax(self, axis: int):
                return MockCupyNdarray(np.argmax(self.data, axis=axis))

        doc = self.nlp.make_doc("one two three")
        predictions = [
            MockCupyNdarray(
                np.array(
                    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
                    dtype=np.float32,
                )
            )
        ]
        self.model.predict.return_value = predictions
        guesses = self.phonemizer.predict([doc])
        self.assertEqual(guesses[0].tolist(), [0, 1, 2])  # wʌn, tuː, θriː

    @skip("fixme — loss is NaN")
    def test_get_loss(self):
        """should calculate loss and gradient for examples and guesses"""
        doc = self.nlp.make_doc("one two three")
        doc[0]._.phonemes = "wʌn"
        doc[1]._.phonemes = "tuː"
        doc[2]._.phonemes = "θriː"
        example = Example(doc, doc)
        guesses = np.asarray(
            [[0.4, 0.5, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]],
            dtype=np.float32,
        )
        self.model.ops.asarray2f = lambda d: np.asarray(d, dtype=np.float32)
        loss, grad = self.phonemizer.get_loss([example], [guesses])
        self.assertEqual(loss, 0.0)

    @skip("todo")
    def test_get_loss_aligned(self):
        """should calculate correct loss for differing example alignments"""
        pass

    @skip("todo")
    def test_get_loss_no_truths(self):
        """should calculate correct loss when no examples have annotations"""
        pass

    def test_initialize_labels(self):
        """should initialize component with labels sample from training data"""
        doc = self.nlp.make_doc("one two three")
        doc[0]._.phonemes = "wʌn"
        doc[1]._.phonemes = "tuː"
        doc[2]._.phonemes = "θriː"
        example = Example(doc, doc)
        get_examples = Mock(return_value=[example])
        self.phonemizer.initialize(get_examples)
        self.assertEqual(self.phonemizer.labels, ("wʌn", "tuː", "θriː"))

    def test_initialize_docs(self):
        """should initialize component with doc sample from training data"""
        doc = self.nlp.make_doc("one two three")
        example = Example(doc, doc)
        get_examples = Mock(return_value=[example])
        self.phonemizer.initialize(get_examples)
        self.model.initialize.assert_called_with(X=[doc], Y=ANY)
