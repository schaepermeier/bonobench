"Tests the ParetoFront object"
import unittest
import numpy as np
from bonobench.pareto import ParetoFront

class TestParetoFront(unittest.TestCase):
    "Tests the ParetoFront object"

    def test_nondominated(self):
        """Test that the front object accepts nondominated points."""
        front = ParetoFront(ideal = (0,0), ref = (1,1))

        for y1 in np.linspace(0, 1, 101):
            front.add((y1, 1 - y1))

        self.assertEqual(len(front), 101)
        self.assertTrue(front.dominates((1e-8, 1)))
        self.assertTrue(front.dominates((1, 1e-8)))
        self.assertTrue(front.dominates((1, 1)))

    def test_pf_ignore_dominance(self):
        """Test that the front object ignores dominated points."""
        front = ParetoFront(ideal = (0,0), ref = (1,1))

        for y1 in np.linspace(0, 1, 101):
            front.add((y1, 1))

        for _ in range(0, 100):
            front.add((0, 1))

        self.assertEqual(len(front), 1)
        self.assertEqual(front.get_r2(), 0.5)
        self.assertEqual(front.get_hv(), 0)

    def test_pf_incremental_updates(self):
        """Test that the indicator values are approximately hit when adding lots of points."""
        np.random.seed(0xC0FFEE)
        front = ParetoFront(ideal = (0,0), ref = (1,1))

        for y1 in np.random.uniform(0, 1, 100_001):
            front.add((y1, 1 - y1 + np.random.uniform(0, 1)))

        self.assertAlmostEqual(front.get_hv(), 0.5, delta = 0.01)
        self.assertAlmostEqual(front.get_r2(), 1 / 6, delta = 0.001)

    def test_pf_basic_indicators(self):
        """Test that the front object outputs some known PF values, including obj. space shifts."""
        front = ParetoFront(ideal = (0,0), ref = (1,1))
        front.add((0, 0))
        self.assertEqual(front.get_r2(), 0)
        self.assertEqual(front.get_hv(), 1)

        front = ParetoFront(ideal = (0,0), ref = (1,1))
        front.add((1, 1))
        self.assertEqual(front.get_r2(), 0.75)
        self.assertEqual(front.get_hv(), 0)

        front = ParetoFront(ideal = (0, 0), ref = (1, 1))
        front.add((1, 0))
        self.assertEqual(front.get_r2(), 0.5)
        self.assertEqual(front.get_hv(), 0)
        front.add((0, 1))
        self.assertEqual(front.get_r2(), 0.25)
        self.assertEqual(front.get_hv(), 0)

        for shift in np.random.uniform(-1000, 1000, size = (100, 2)):
            front = ParetoFront(ideal = tuple(shift), ref = tuple(shift + 1))
            front.add(tuple(shift))
            self.assertAlmostEqual(front.get_r2(), 0)
            self.assertAlmostEqual(front.get_hv(), 1)

            front = ParetoFront(ideal = tuple(shift), ref = tuple(shift + 1))
            front.add(tuple(shift + 1))
            self.assertAlmostEqual(front.get_r2(), 0.75)
            self.assertAlmostEqual(front.get_hv(), 0)
