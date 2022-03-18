import numpy as np


class MockCupyNdarray:
    """Mock cupy.ndarray for testing GPU-based logic."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def get(self):
        return self.data

    def any(self):
        return self.data.any()

    def argmax(self, axis: int):
        return MockCupyNdarray(np.argmax(self.data, axis=axis))
