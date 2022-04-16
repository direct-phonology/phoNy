from unittest import TestCase

import numpy as np

from phony.ml import Arrayable


class TestArrayable(TestCase):
    class MyArrayable(Arrayable):
        """Arrayable implementor for testing"""

        def __init__(self, data: list) -> None:
            self.data = data

        def to_array(self) -> np.ndarray:
            return np.array(self.data)

        @classmethod
        def from_array(cls, array: np.ndarray) -> Arrayable:
            return cls(list(array))

    def test_similarity(self):
        """should score similarity based on euclidean distance"""
        a = self.MyArrayable([1, 2, 3])
        b = self.MyArrayable([1, 2, 3])
        self.assertEqual(a.similarity(b), 1.0)

        with self.assertRaises(ValueError):
            a = self.MyArrayable([1, 2, 3])
            b = self.MyArrayable([1, 2, 3, 4])
            a.similarity(b)
