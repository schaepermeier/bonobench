import unittest
import numpy as np
import bonobench.bonoproblem as bp
from bonobench.utils import random_rotation_matrix

class TestPeakFunction(unittest.TestCase):
    """Test the PeakFunction class"""

    def test_evaluation(self):
        """Validate that evaluate and bulk_eval do the same."""
        R = random_rotation_matrix(2)
        D = np.array([[10,0],[0,1]])
        H = R.T @ D @ R

        pf = bp.PeakFunction([-3,-1], H, 0)

        self.assertEqual(pf.evaluate([-3, -1]), 0)

        X = np.random.uniform(-5, 5, size = (100, 2))

        y1 = pf.bulk_eval(X)
        y2 = np.array([float(pf.evaluate(x)) for x in X])

        np.testing.assert_almost_equal(y1, y2)
