from unittest import TestCase, skip
from unittest.mock import Mock

import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch
from thinc.api import compounding

from .. import MockCupyNdarray


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
        self.assertEqual(doc[0]._.phonemes, "wʌn")
        self.assertEqual(doc[1]._.phonemes, "tuː")
        self.assertEqual(doc[2]._.phonemes, "θriː")

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
        self.phonemizer.set_annotations([doc], [tag_ids])
        self.assertEqual(doc[0]._.phonemes, "wʌn")
        self.assertEqual(doc[1]._.phonemes, None)
        self.assertEqual(doc[2]._.phonemes, "θriː")

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
        self.assertEqual(doc[0]._.phonemes, "wʌn")
        self.assertEqual(doc[1]._.phonemes, "tuː")
        self.assertEqual(doc[2]._.phonemes, "θriː")

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

    def test_get_loss(self):
        """should calculate loss and gradient for examples and guesses"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.add_label("wʌn")
        phonemizer.add_label("tuː")
        phonemizer.add_label("θriː")
        phonemizer.model = Mock()
        doc = nlp.make_doc("one two three")
        doc[0]._.phonemes = "wʌn"
        doc[1]._.phonemes = "tuː"
        doc[2]._.phonemes = "θriː"
        example = Example.from_dict(doc, {})
        guesses = np.asarray(
            [[0.4, 0.5, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]],
            dtype=np.float32,
        )
        phonemizer.model.ops.asarray2f = lambda d: np.asarray(d, dtype=np.float32)
        loss, grad = phonemizer.get_loss([example], [guesses])
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
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        doc = nlp.make_doc("one two three")
        doc[0]._.phonemes = "wʌn"
        doc[1]._.phonemes = "tuː"
        doc[2]._.phonemes = "θriː"
        example = Example.from_dict(doc, {})
        phonemizer.initialize(lambda: [example])
        self.assertEqual(phonemizer.labels, ("wʌn", "tuː", "θriː"))

    def test_initialize_docs(self):
        """should initialize component with doc sample from training data"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        phonemizer.model = Mock()
        doc = nlp.make_doc("one two three")
        doc[0]._.phonemes = "wʌn"
        doc[1]._.phonemes = "tuː"
        doc[2]._.phonemes = "θriː"
        example = Example(doc, doc)
        example = Example.from_dict(doc, {})
        phonemizer.initialize(lambda: [example])
        phonemizer.model.initialize.assert_called_with(X=[doc], Y=ANY)

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
        doc1 = nlp.make_doc("one two three four")
        doc2 = nlp.make_doc("on e two three four")
        doc1[0]._.phonemes = "wʌn"
        doc1[1]._.phonemes = "tuː"
        doc1[2]._.phonemes = "θriː"
        doc2[0]._.phonemes = "ɒn"
        doc2[2]._.phonemes = "tuː"
        doc2[3]._.phonemes = "θriː"
        examples = [Example.from_dict(doc1, {}), Example.from_dict(doc2, {})]
        optimizer = nlp.initialize(get_examples=lambda: examples)
        for _ in range(50):
            losses = {}
            self.nlp.update(examples, sgd=optimizer, losses=losses)
        self.assertLess(losses["phonemizer"], 0.00001)

    @skip("todo")
    def test_train(self):
        """results after training should be predictable on sample data"""
        nlp = spacy.blank("en")
        phonemizer = nlp.add_pipe("phonemizer")
        doc1 = nlp.make_doc("one two three")
        doc2 = nlp.make_doc("three two one two")
        doc1[0]._.phonemes = "wʌn"
        doc1[1]._.phonemes = "tuː"
        doc1[2]._.phonemes = "θriː"
        doc2[0]._.phonemes = "θriː"
        doc2[1]._.phonemes = "tuː"
        doc2[2]._.phonemes = "wʌn"
        doc2[3]._.phonemes = "tuː"
        examples = [Example.from_dict(doc1, {}), Example.from_dict(doc2, {})]
        optimizer = phonemizer.train(lambda: examples)
        for _ in range(50):
            losses = {}
            phonemizer.update(examples, sgd=optimizer, losses=losses)

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
