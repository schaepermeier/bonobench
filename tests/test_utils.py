"""Test utility functions"""
import unittest
import numpy as np
import bonobench.bonoproblem as bp
from bonobench.utils import step_round, generate_optima

class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def test_peak_addition(self):
        """Test whether the addition of two peak problems works as expected"""
        np.random.seed(0x7EE)

        xopt_1 = np.array([1,1])
        xopt_2 = np.array([2,2])
        yopt_1 = 99
        yopt_2 = 42
        H1 = np.array([[10,3],[3,10]])
        # H2 = H1
        H2 = np.array([[1,1],[1,1]])

        pf1 = bp.PeakFunction(xopt_1, H1, yopt_1)
        pf2 = bp.PeakFunction(xopt_2, H2, yopt_2)
        pf_add = pf1 + pf2

        np.testing.assert_almost_equal(H1 + H2, pf_add.get_hessian())

        for x in np.random.uniform(-10, 10, (10001, 2)):
            self.assertTrue(pf1.evaluate(x) <= pf_add.evaluate(x))
            self.assertTrue(pf2.evaluate(x) <= pf_add.evaluate(x))
            self.assertAlmostEqual(pf1.evaluate(x) + pf2.evaluate(x), pf_add.evaluate(x))

    def test_step_round(self):
        """Test that step_round method works as expected"""
        self.assertEqual(step_round(y = 2.5, ymin = 2, step = 1), 2)
        self.assertEqual(step_round(y = 3, ymin = 2, step = 1), 3)
        self.assertEqual(step_round(y = 2, ymin = 2, step = 1), 2)

    def test_optima_creation(self):
        """Test that optima creation works as expected"""
        np.random.seed(0xC0FFEE)

        for _ in range(100):
            d = np.random.randint(2, 40)
            xl = np.random.uniform(-10, 10, size = d)
            xu = xl + np.random.uniform(0,20, size = d)
            min_dist = np.random.uniform(0, min(xu - xl))
            fixed_dimensions = np.random.randint(0, d - 1)

            x1, x2 = generate_optima(d = d, xl = xl, xu = xu, min_dist = min_dist,
                                     fixed_dimensions = fixed_dimensions)

            self.assertEqual(len(x1), d)
            self.assertEqual(len(x2), d)
            self.assertEqual(np.sum(x1 == x2), fixed_dimensions)
            self.assertTrue(np.all(x1 <= xu))
            self.assertTrue(np.all(x2 <= xu))
            self.assertTrue(np.all(x1 >= xl))
            self.assertTrue(np.all(x2 >= xl))
            self.assertTrue(np.linalg.norm(x1 - x2) >= min_dist)
