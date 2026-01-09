"""
Module includes all functionality to define a BONOProblem.
"""

import numpy as np
from sortedcontainers import SortedList
import moocore
import matplotlib.pyplot as plt
from .pareto import ParetoFront, ObjectivePoint, utility
from .utils import step_round

class Problem:
    """Template class for single-objective problems"""
    xopt: np.array
    dim: int
    yopt: np.float64

    def get_xopt(self):
        """Return xopt"""
        return self.xopt

    def get_yopt(self):
        """Return yopt"""
        return self.yopt

    def get_dim(self):
        """Return dimensionality"""
        return self.dim

class PeakFunction(Problem):
    """Creates a unimodal quadratic (Peak) function"""
    H: np.array

    def __init__(self, xopt: np.array, H: np.array, yopt: np.float64 = 0):
        self.xopt = np.array(xopt)
        self.dim = len(self.xopt)
        self.yopt = np.float64(yopt)
        self.H = np.array(H)

        assert self.dim == self.H.shape[0] == self.H.shape[1]

    def evaluate(self, x: np.array):
        """Evaluate the peak function at x"""
        dx = x - self.xopt
        quadratic_value = 0.5 * dx.T @ self.H @ dx

        return quadratic_value + self.yopt

    def bulk_eval(self, X: np.array):
        """Bulk Evaluation"""
        dx = X - self.xopt
        quadratic_value = 0.5 * np.sum(dx @ self.H * dx, axis=1)

        return quadratic_value + self.yopt

    def get_hessian(self):
        "Get Hessian of peak function"
        return self.H

    def get_conditioning(self):
        """Return the conditioning of the Hessian function."""
        return np.linalg.cond(self.H)

    def __add__(self, other):
        H = self.H + other.H
        xopt = np.linalg.inv(H) @ (self.H @ self.get_xopt() + other.H @ other.get_xopt())
        yopt = self.evaluate(xopt) + other.evaluate(xopt)

        return PeakFunction(xopt, H, yopt)

class MultiplePeakProblem(Problem):
    """Create a multiple peak problem."""
    peaks: list[PeakFunction]
    s: np.float64
    p: np.float64
    yopt: np.float64
    step: np.float64

    def __init__(self, peaks: list[PeakFunction], s: np.float64, p: np.float64,
                 yopt: np.float64, step: np.float64 = None):
        self.peaks = peaks
        self.s = s
        self.p = p
        np.testing.assert_allclose(np.min([peak.get_yopt() for peak in self.peaks]), 0, atol = 1e-10)
        self.xopt = self.peaks[np.argmin([peak.get_yopt() for peak in self.peaks])].get_xopt()
        self.yopt = yopt
        self.step = step
        self.dim = len(self.xopt)

    def monotone_trafo(self, raw_value):
        """Apply monotone transformations to raw objective value."""
        y_trafo = self.s * np.power(raw_value, self.p / 2) + self.yopt
        if self.step is not None:
            return step_round(y_trafo, self.yopt, self.step)
        else:
            return y_trafo

    def evaluate(self, x):
        """Evaluate the problem at x"""
        raw_value = np.min([peak.evaluate(x) for peak in self.peaks])
        return self.monotone_trafo(raw_value)

    def bulk_eval(self, X):
        """Bulk evaluation of X"""
        raw_value = np.amin(np.array([peak.bulk_eval(X) for peak in self.peaks]), axis = 0)
        return self.monotone_trafo(raw_value)

    def evaluate_peak(self, x, peak_id):
        """Evaluate self.peaks[peak_id] at x"""
        raw_value = self.peaks[peak_id].evaluate(x)
        return self.monotone_trafo(raw_value)

    def bulk_eval_peak(self, X, peak_id):
        """Bulk evaluate self.peaks[peak_id] at X"""
        raw_value = self.peaks[peak_id].bulk_eval(X)
        return self.monotone_trafo(raw_value)

    def get_peaks(self):
        """Get constitutent peaks."""
        return self.peaks

    def get_scale(self):
        """Get scale."""
        return self.s

    def get_distance_parameter(self):
        """Get the p from the applied Lp norm."""
        return self.p

    def get_step(self):
        """Get the step applied in stepped function."""
        return self.step

    def set_step(self, step):
        """Modify the step applied in stepped function."""
        self.step = step

class BONOProblem:
    """Class hosting a bi-objective numerical optimization problem."""

    f1: MultiplePeakProblem
    f2: MultiplePeakProblem

    ideal: np.array
    nadir: np.array

    ps: np.float64
    ss: np.float64

    def get_f1(self):
        """Return the first objective function"""
        return self.f1

    def get_f2(self):
        """Return the second objective function"""
        return self.f2

    def evaluate(self, x: np.array):
        """Evaluate the problem at x"""
        y1 = self.f1.evaluate(x)
        y2 = self.f2.evaluate(x)

        return (y1, y2)

    def bulk_eval(self, X: np.array):
        """Bulk evaluation of X"""
        Y1 = self.f1.bulk_eval(X)
        Y2 = self.f2.bulk_eval(X)

        return (Y1, Y2)

    def __init__(self, f1: MultiplePeakProblem, f2: MultiplePeakProblem):
        assert f1.get_dim() == f2.get_dim()

        self.f1 = f1
        self.f2 = f2

        self.ideal = np.array([self.f1.get_yopt(),
                               self.f2.get_yopt()])
        self.nadir = np.array([self.f1.evaluate(self.f2.get_xopt()),
                               self.f2.evaluate(self.f1.get_xopt())])

    def get_ideal(self):
        """Get the problems' ideal point."""
        return self.ideal

    def get_nadir(self):
        """Get the problems' nadir point."""
        return self.nadir

    def get_dim(self):
        """Get the problems' dimensionality (decision space)."""
        return self.f1.get_dim()

    def dominance_plot(self, resolution = 301):
        """Visualize with dominance plot"""
        assert self.get_dim() == 2

        f1_opt = self.f1.get_xopt()
        f2_opt = self.f2.get_xopt()

        x1 = np.linspace(-5, 5, resolution)
        x2 = np.linspace(-5, 5, resolution)

        dec_space = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
        obj_1, obj_2 = self.bulk_eval(dec_space)
        obj_space = np.array([obj_1, obj_2]).T
        obj_rank = moocore.pareto_rank(obj_space)

        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (9,4), constrained_layout = True)
        ax1, ax2 = axs

        ax1.contourf(x1, x2, np.log10(np.array(obj_rank).reshape(resolution, resolution).T), levels = 100)
        ax1.axis('scaled')
        ax1.plot(f1_opt[0], f1_opt[1], 'w+')
        ax1.plot(f2_opt[0], f2_opt[1], 'w+')

        ideal = self.get_ideal()
        nadir = self.get_nadir()
        delta = nadir - ideal
        order = np.argsort(-obj_rank)

        fig2 = ax2.scatter(obj_1[order], obj_2[order], c = np.array(obj_rank)[order], norm = "log")
        ax2.axis("auto")
        ax2.plot(ideal[0], ideal[1], 'r+')
        ax2.plot(nadir[0], nadir[1], 'r+')
        ax2.set_xlim(ideal[0] - 0.02 * delta[0], nadir[0] + 1 * delta[0])
        ax2.set_ylim(ideal[1] - 0.02 * delta[1], nadir[1] + 1 * delta[1])

        fig.colorbar(fig2, ax=axs)

        # plt.tight_layout()
        plt.show()

    def approximate_indicators(self, delta_hv = 1e-5, delta_r2 = 1e-6, n_split = 2, it_max = np.inf, recompute_interval = 1e5, log = False, keep_dec = True):
        """Approximate optimal indicator values."""

        ideal = self.get_ideal()
        nadir = self.get_nadir()
        norm_scales = nadir - ideal
        norm_area = np.prod(norm_scales)

        front = ParetoFront(ideal, nadir)

        evaluation_queue = SortedList()
        total_uncertainty_hv = 0
        total_uncertainty_r2 = 0

        f1 = self.get_f1()
        f1_peaks = f1.get_peaks()
        f2 = self.get_f2()
        f2_peaks = f2.get_peaks()

        f1_ids, f2_ids = self._pareto_relevant_peaks()

        # f1_ids = np.argsort([p.get_yopt() for p in f1.get_peaks()])
        # f2_ids = np.argsort([p.get_yopt() for p in f2.get_peaks()])

        it = 0

        for f1_peak_id in f1_ids:
            for f2_peak_id in f2_ids:
                t_low = 0
                x_low = f1_peaks[f1_peak_id].get_xopt()
                y_low = (f1.evaluate_peak(x_low, f1_peak_id),
                         f2.evaluate_peak(x_low, f2_peak_id))

                t_high = 1
                x_high = f2_peaks[f2_peak_id].get_xopt()
                y_high = (f1.evaluate_peak(x_high, f1_peak_id),
                          f2.evaluate_peak(x_high, f2_peak_id))

                if not front.dominates(np.minimum(y_low, y_high)):
                    front.add(y_low, x_low if keep_dec else None, np.array([f1_peak_id, f2_peak_id]))
                    front.add(y_high, x_high if keep_dec else None, np.array([f1_peak_id, f2_peak_id]))

                    uncertainty_hv = (y_high[0] - y_low[0]) * (y_low[1] - y_high[1])
                    uncertainty_r2 = r2_uncertainty(y_low, y_high, ideal, norm_scales)

                    evaluation_queue.add((uncertainty_hv, uncertainty_r2, f1_peak_id, f2_peak_id, y_low, y_high, t_low, t_high))
                    total_uncertainty_hv += uncertainty_hv
                    total_uncertainty_r2 += uncertainty_r2

                if log and it % 10000 == 0:
                    print(f"{front.get_normalized_hv():.5f}; delta {total_uncertainty_hv / norm_area:.2e}, {total_uncertainty_r2:.2e}; it: {it}, queue {len(evaluation_queue)}     ", end = "\r")

                it += 1

        it = 0

        while it < it_max and (total_uncertainty_hv >= delta_hv * norm_area or total_uncertainty_r2 >= delta_r2):
            it += 1

            uncertainty_hv, uncertainty_r2, f1_peak_id, f2_peak_id, y_low, y_high, t_low, t_high = evaluation_queue.pop()
            total_uncertainty_hv -= uncertainty_hv
            total_uncertainty_r2 -= uncertainty_r2

            if front.dominates(np.minimum(y_low, y_high)):
                # skip if ideal of area already dominated
                continue

            p1 = f1_peaks[f1_peak_id]
            p2 = f2_peaks[f2_peak_id]

            if t_high - t_low < 1e-10:
                # skip if ts are too close
                x_low = solve_hessians(p1, p2, t_low)
                x_high = solve_hessians(p1, p2, t_high)
                if np.linalg.norm(x_high - x_low) < 1e-10:
                    continue
            
            t_it_low = t_low
            y_it_low = y_low

            ts = np.linspace(t_low, t_high, num = n_split + 1)
            xs = [solve_hessians(p1, p2, t) for t in ts]
            f1s = f1.bulk_eval_peak(np.stack(xs), f1_peak_id)
            f2s = f2.bulk_eval_peak(np.stack(xs), f2_peak_id)

            for i in range(n_split):
                t_it_low = ts[i]
                t_it_high = ts[i + 1]

                y_it_low = (f1s[i], f2s[i])

                x_it_high = xs[i + 1]
                y_it_high = (f1s[i + 1], f2s[i + 1])

                front.add(y_it_high, x_it_high if keep_dec else None, np.array([f1_peak_id, f2_peak_id]))

                # Potentially add area from lower subinterval

                new_uncertainty_hv_it = (y_it_high[0] - y_it_low[0]) * (y_it_low[1] - y_it_high[1])
                new_uncertainty_r2_it = r2_uncertainty(y_it_low, y_it_high, ideal, norm_scales)

                if not front.dominates(np.minimum(y_it_low, y_it_high)) and new_uncertainty_hv_it > 0:
                    evaluation_queue.add((new_uncertainty_hv_it, new_uncertainty_r2_it, f1_peak_id, f2_peak_id, y_it_low, y_it_high, t_it_low, t_it_high))
                    total_uncertainty_hv += new_uncertainty_hv_it
                    total_uncertainty_r2 += new_uncertainty_r2_it

            # t_mid = (t_low + t_high) / 2
            # x_mid = solve_hessians(p1, p2, t_mid)

            # y_mid = (f1.evaluate_peak(x_mid, f1_peak_id),
            #          f2.evaluate_peak(x_mid, f2_peak_id))
            # front.add(y_mid, x_mid if keep_dec else None, np.array([f1_peak_id, f2_peak_id]))

            # # Potentially add area from lower subinterval

            # new_uncertainty_hv_low = (y_mid[0] - y_low[0]) * (y_low[1] - y_mid[1])
            # new_uncertainty_r2_low = r2_uncertainty(y_low, y_mid, ideal, norm_scales)

            # if not front.dominates(np.minimum(y_low, y_mid)) and new_uncertainty_hv_low > 0:
            #     evaluation_queue.add((new_uncertainty_hv_low, new_uncertainty_r2_low, f1_peak_id, f2_peak_id, y_low, y_mid, t_low, t_mid))
            #     total_uncertainty_hv += new_uncertainty_hv_low
            #     total_uncertainty_r2 += new_uncertainty_r2_low

            # # Potentially add area from upper subinterval

            # new_uncertainty_hv_high = (y_high[0] - y_mid[0]) * (y_mid[1] - y_high[1])
            # new_uncertainty_r2_high = r2_uncertainty(y_mid, y_high, ideal, norm_scales)

            # if not front.dominates(np.minimum(y_mid, y_high)) and new_uncertainty_hv_high > 0:
            #     evaluation_queue.add((new_uncertainty_hv_high, new_uncertainty_r2_high, f1_peak_id, f2_peak_id, y_mid, y_high, t_mid, t_high))
            #     total_uncertainty_hv += new_uncertainty_hv_high
            #     total_uncertainty_r2 += new_uncertainty_r2_high

            if it % recompute_interval == 0:
                # to combat numerical issues, recompute indicators from scratch every so often
                total_uncertainty_hv = np.sum([uncertainty_hv for (uncertainty_hv, _, _, _, _, _, _, _) in evaluation_queue])
                total_uncertainty_r2 = np.sum([uncertainty_r2 for (_, uncertainty_r2, _, _, _, _, _, _) in evaluation_queue])

            if log and it % 10000 == 0:
                print(f"{front.get_normalized_hv():.5f}; delta {total_uncertainty_hv / norm_area:.2e}, {total_uncertainty_r2:.2e}; it: {it}, queue {len(evaluation_queue)}     ", end = "\r")


        if log:
            print(f"Final approximation: {front.get_normalized_hv():.5f}; delta {total_uncertainty_hv / norm_area:.2e}, {total_uncertainty_r2:.2e} (total it.s: {it})")

        # front.peak_combinations = [(f1_peak_id, f2_peak_id) for (_,_,f1_peak_id,f2_peak_id,_,_,_,_) in evaluation_queue]
        front.peak_combinations = [point.peaks for point in front.pf]

        return front

    def _pareto_relevant_peaks(self):
        f1 = self.get_f1()
        f2 = self.get_f2()
        nadir = self.get_nadir()

        relevant_peak_ids_f1 = [id for id in range(len(f1.get_peaks()))
                                if f1.evaluate_peak(f1.get_peaks()[id].get_xopt(), id) <= nadir[0] + 1e-6]

        relevant_peak_ids_f2 = [id for id in range(len(f2.get_peaks()))
                                if f2.evaluate_peak(f2.get_peaks()[id].get_xopt(), id) <= nadir[1] + 1e-6]

        return (relevant_peak_ids_f1, relevant_peak_ids_f2)

class TrackedBONOProblem:
    """A BONOProblem with internal HV and R2 tracking."""

    bonoproblem: BONOProblem

    reference_front: ParetoFront
    reference_hv: np.float64
    reference_r2: np.float64

    run_front: ParetoFront

    hv_targets: np.array
    hv_solved: int
    hv_hit_times: np.array

    r2_targets: np.array
    r2_solved: int
    r2_hit_times: np.array

    fevals: int

    def __init__(self, bonoproblem: BONOProblem, indicator_history = True):
        self.bonoproblem = bonoproblem
        self.indicator_history = indicator_history
        self.hv_hit_times = []
        self.r2_hit_times = []
        self.reset()

    def determine_targets(self, delta_hv = 1e-5, delta_r2 = 1e-6, reference_hv = None, reference_r2 = None, hv_targets = None, r2_targets = None, log = True):
        """Approximate reference indicators"""
        self.reference_hv = reference_hv
        self.reference_r2 = reference_r2

        if reference_hv is None or reference_r2 is None:
            self.reference_front = self.bonoproblem.approximate_indicators(delta_hv, delta_r2, log = log, keep_dec=False)
            if reference_hv is None:
                self.reference_hv = self.reference_front.get_normalized_hv()

            if reference_r2 is None:
                self.reference_r2 = self.reference_front.get_r2()

        if hv_targets is None:
            hv_targets = np.pow(10, np.linspace(0, -5, num = 51))

        if r2_targets is None:
            r2_targets = np.pow(10, np.linspace(0, -5, num = 51))

        self.hv_targets = hv_targets
        self.r2_targets = r2_targets

        self.hv_solved = 0
        self.r2_solved = 0

        self.hv_hit_times = np.array(len(hv_targets) * [np.inf])
        self.r2_hit_times = np.array(len(hv_targets) * [np.inf])

    def evaluate(self, x):
        """Evaluate single solution x and update front."""
        y = self.bonoproblem.evaluate(x)
        self._update_front(y)

        return y

    def bulk_eval(self, X):
        """Evaluate solution matrix X and update front."""
        Y = self.bonoproblem.bulk_eval(X)

        for i in range(len(Y[0])):
            self._update_front((Y[0][i], Y[1][i]))

        return Y

    def _update_front(self, y):
        self.fevals += 1

        updated = self.run_front.add(y)

        if self.indicator_history:
            self.hv_history.append(self.run_front.get_normalized_hv())
            self.r2_history.append(self.run_front.get_r2())

        if updated:
            dhv = self.reference_hv - self.run_front.get_normalized_hv()
            dr2 = self.run_front.get_r2() - self.reference_r2

            while self.hv_solved < len(self.hv_targets) and dhv <= self.hv_targets[self.hv_solved]:
                self.hv_hit_times[self.hv_solved] = self.fevals
                self.hv_solved += 1

            while self.r2_solved < len(self.r2_targets) and dr2 <= self.r2_targets[self.r2_solved]:
                self.r2_hit_times[self.r2_solved] = self.fevals
                self.r2_solved += 1

    def get_hv_hit_times(self):
        """Get all hv hit times"""
        return self.hv_hit_times

    def get_hv_targets(self):
        """Get all hv targets"""
        return self.hv_targets

    def get_r2_hit_times(self):
        """Get all r2 hit times"""
        return self.r2_hit_times

    def get_r2_targets(self):
        """Get all r2 targets"""
        return self.r2_targets

    def get_targets_solved(self):
        """Get dict of solved targets for hv and r2."""
        return {"hv": self.hv_solved, "r2": self.r2_solved}

    def reset(self):
        """Resets all history"""
        self.hv_solved = 0
        self.r2_solved = 0

        self.hv_hit_times = np.array(len(self.hv_hit_times) * [np.inf])
        self.r2_hit_times = np.array(len(self.r2_hit_times) * [np.inf])

        self.run_front = ParetoFront(self.bonoproblem.get_ideal(), self.bonoproblem.get_nadir())

        if self.indicator_history:
            self.hv_history = []
            self.r2_history = []

        self.fevals = 0

    def all_targets_solved(self):
        return np.all(self.hv_hit_times != np.inf) and np.all(self.r2_hit_times != np.inf)

def solve_hessians(p1: PeakFunction, p2: PeakFunction, t: np.float64):
    """Find optimal x for (1 - t) * p1 + t * p2."""
    H1 = p1.get_hessian()
    H2 = p2.get_hessian()
    xopt_1 = p1.get_xopt()
    xopt_2 = p2.get_xopt()

    return np.linalg.inv((1 - t) * H1 + t * H2) @ ((1 - t) * H1 @ xopt_1 + t * H2 @ xopt_2)

def r2_uncertainty(y_low, y_high, ideal, norm_scales):
    """Determine potential R2 improvement between y_low and y_high"""
    y_low = np.array(y_low - ideal) / norm_scales
    y_high = np.array(y_high - ideal) / norm_scales

    r2u = utility(y_high[0], y_high[1], y_low[1]) \
        + utility(y_low[1], y_low[0], y_high[0]) \
        - utility(y_low[0], y_high[1], y_low[1]) \
        - utility(y_high[1], y_low[0], y_high[0])

    return r2u
