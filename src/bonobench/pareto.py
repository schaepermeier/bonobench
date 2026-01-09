from dataclasses import dataclass
import numpy as np
from sortedcontainers import SortedList
import matplotlib.pyplot as plt

@dataclass
class ObjectivePoint:
    """Stores a point in objective space (y) with optional decision space coordinates (x)"""
    y: np.array
    x: np.array = None
    peaks: np.array = None

    def dominates(self, p):
        """Returns true if self dominates p"""
        return self.y[0] <= p.y[0] and self.y[1] <= p.y[1] and np.any(self.y != p.y)

    def __lt__(self, p):
        return (self.y[0] < p.y[0]) or (self.y[0] == p.y[0] and self.y[1] < p.y[1])

class ParetoFront:
    """Managing a Pareto front and associated indicators"""

    pf: SortedList
    ideal: ObjectivePoint
    ref: ObjectivePoint
    r2: np.float64
    hv: np.float64
    norm_area = np.float64

    def __init__(self, ideal, ref):
        self.pf = SortedList()

        self.ideal = ObjectivePoint(np.array(ideal))
        self.ref = ObjectivePoint(np.array(ref))
        assert self.ideal.dominates(self.ref)
        self.norm_area = np.prod(self.ref.y - self.ideal.y)

        self.r2 = np.inf
        self.hv = 0

    def __len__(self):
        return len(self.pf)

    def add(self, y, x = None, peaks = None):
        """Try to add y to pf. Returns if p was previously nondominated and pf was changed."""

        added = False
        p = ObjectivePoint(y, x, peaks)

        if len(self.pf) == 0:
            self.pf.add(p)
            added = True

            idx = 0
            self.r2 = self._exclusive_r2(idx)
            self.hv = self._exclusive_hv(idx)
        else:
            idx = self.pf.bisect(p)
            left_boundary = (idx == 0)
            idx = max(0, idx - 1)

            if not self.pf[idx].dominates(p) and np.all(self.pf[idx].y != p.y):
                if not left_boundary:
                    idx += 1

                while idx < len(self.pf) and p.dominates(self.pf[idx]):
                    self.r2 -= self._exclusive_r2(idx)
                    self.hv -= self._exclusive_hv(idx)
                    self.pf.remove(self.pf[idx])

                self.pf.add(p)
                added = True

                self.r2 += self._exclusive_r2(idx)
                self.hv += self._exclusive_hv(idx)

        return added

    def dominates(self, p):
        """Returns whether pf dominates p."""
        if len(self.pf) == 0:
            return False

        if not isinstance(p, ObjectivePoint):
            p = ObjectivePoint(p)
        idx = self.pf.bisect(p)
        idx = max(0, idx - 1)
        return self.pf[idx].dominates(p)

    def get_r2(self):
        """Returns the current R2 indicator value."""
        return self.r2

    def get_hv(self):
        """Returns the current hypervolume indicator value."""
        return self.hv

    def get_normalized_hv(self):
        """Returns the current normalized hypervolume indicator value."""
        return self.hv / self.norm_area

    def get_points(self):
        """Returns the objective values from the archive."""
        return np.array([p.y for p in self.pf])

    def get_dec_points(self):
        """Returns the decision vectors from the archive."""
        return np.array([p.x for p in self.pf])

    def _exclusive_hv(self, idx):
        """
        Computes the exclusive hypervolume of sl[idx] w.r.t. sl and reference point ref.
        """
        if not self.pf[idx].dominates(self.ref):
            return 0

        if idx == 0:
            left_neighbor_f2 = self.ref.y[1]
        else:
            left_neighbor_f2 = self.pf[idx - 1].y[1]

        if idx == len(self.pf) - 1:
            right_neighbor_f1 = self.ref.y[0]
        else:
            right_neighbor_f1 = self.pf[idx + 1].y[0]

        right_neighbor_f1 = min(right_neighbor_f1, self.ref.y[0])
        left_neighbor_f2 = min(left_neighbor_f2, self.ref.y[1])

        return (left_neighbor_f2 - self.pf[idx].y[1]) * (right_neighbor_f1 - self.pf[idx].y[0])

    def _exclusive_r2(self, idx):
        """
        Compute the exclusive R2 contribution of sl[idx] w.r.t. sl and ideal point ideal.
        ref is used for normalizing objective values.
        """
        scale = self.ref.y - self.ideal.y

        center = self.pf[idx].y - self.ideal.y
        center = center / scale

        if idx == 0:
            left_neighbor = np.array([np.inf, np.inf])
        else:
            left_neighbor = (self.pf[idx - 1].y - self.ideal.y) / scale

        if idx == len(self.pf) - 1:
            right_neighbor = np.array([np.inf, np.inf])
        else:
            right_neighbor = (self.pf[idx + 1].y - self.ideal.y) / scale

        r2_f1 = utility(center[0], center[1], left_neighbor[1]) - \
                utility(right_neighbor[0], center[1], left_neighbor[1])
        r2_f2 = utility(center[1], center[0], right_neighbor[0]) - \
                utility(left_neighbor[1], center[0], right_neighbor[0])

        return r2_f1 + r2_f2

    def plot(self, show = True):
        """Plot the points stored in front."""
        Y = self.get_points()
        Y1 = Y[:,0]
        Y2 = Y[:,1]

        ideal = self.ideal.y
        nadir = self.ref.y
        delta = nadir - ideal

        plt.scatter(Y1, Y2)
        plt.axis("auto")
        plt.plot(ideal[0], ideal[1], 'r+')
        plt.plot(nadir[0], nadir[1], 'r+')
        plt.xlim(ideal[0] - 0.02 * delta[0], nadir[0] + 0.1 * delta[0])
        plt.ylim(ideal[1] - 0.02 * delta[1], nadir[1] + 0.1 * delta[1])
        
        if show:
            plt.show()

def utility(y1, y2, y2p):
    """
    Compute the Tchebycheff utility of the axis-parallel segment u(y1, [y2,y2p]).
    """
    if y1 == 0 and y2 == 0:
        return 0

    if y1 == np.inf:
        return 0

    if y2p == np.inf:
        wp = 1
    else:
        wp = y2p / (y1 + y2p)

    if y2 == np.inf:
        w = 1
    else:
        w = y2 / (y1 + y2)

    return 0.5 * y1 * (wp**2 - w**2)
