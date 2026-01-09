import numpy as np
from .utils import generate_hessian, generate_optima, sample_loguniform
from .bonoproblem import PeakFunction, MultiplePeakProblem, BONOProblem

class InstanceGenerator:
    d: int
    n_peaks: int
    cond: np.float64
    same_hessians: bool
    rotate_hessians: bool
    round_steps: list
    p: np.float64

    def generate_instance(self, seed: int):
        """Generate a new instance with random seed."""
        pass

    def sample_conditioning(self):
        """Sample a conditioning value from given range"""
        if not isinstance(self.cond, list):
            return self.cond
        else:
            return sample_loguniform(min(self.cond), max(self.cond))

    def sample_p(self):
        """Sample a p value from given range"""
        if not isinstance(self.p, list):
            return self.p
        else:
            return sample_loguniform(min(self.p), max(self.p))

    def sample_round_steps(self):
        """Sample a number of rounding steps from given range"""
        if self.round_steps is not None:
            if not isinstance(self.round_steps, list):
                return self.round_steps
            else:
                return np.floor(sample_loguniform(min(self.round_steps), max(self.round_steps) + 1))
        else:
            return None

    def pareto_front_inbounds(self, bp: BONOProblem, lower = -5, upper = 5,
                              delta_hv = 1e-3, delta_r2 = 1e-4):
        """Check whether all points on the PF are inbounds"""
        front = bp.approximate_indicators(delta_hv = delta_hv, delta_r2 = delta_r2, log = False)
        dec_points = front.get_dec_points()
        return np.all(dec_points > lower) and np.all(dec_points < upper)

class UnimodalGenerator(InstanceGenerator):
    """Generator for all unimodal problem instances"""

    def __init__(self, d: int, cond: list, p: list, round_steps: list, same_hessians = True,
                 rotate_hessians = True, fixed_dimensions = 0):
        self.d = d
        self.cond = cond
        self.p = p
        self.round_steps = round_steps
        self.same_hessians = same_hessians
        self.rotate_hessians = rotate_hessians
        self.fixed_dimensions = fixed_dimensions

    def generate_instance(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        d = self.d
        same_hessians = self.same_hessians
        rotate_hessians = self.rotate_hessians
        fixed_dimensions = self.fixed_dimensions

        while True:
            # Single-objective optima
            xopt_f1, xopt_f2 = generate_optima(d, min_dist = 2, fixed_dimensions = fixed_dimensions)

            # Hessians
            cond = self.sample_conditioning()
            hessian_f1 = generate_hessian(d, cond, rotate_hessians)
            hessian_f2 = hessian_f1 if same_hessians else generate_hessian(d, cond, rotate_hessians)

            base_f1 = MultiplePeakProblem([PeakFunction(xopt_f1, hessian_f1)], s=1, p=2, yopt=0)
            base_f2 = MultiplePeakProblem([PeakFunction(xopt_f2, hessian_f2)], s=1, p=2, yopt=0)
            base_problem = BONOProblem(base_f1, base_f2)

            if self.pareto_front_inbounds(base_problem):
                break

        # Objective scaling
        scale_f1 = np.round(np.pow(10, np.random.uniform(0, 6)), 4)
        scale_f2 = np.round(np.pow(10, np.random.uniform(0, 6)), 4)

        # Optimal y-values
        yopt_f1 = np.round(np.random.uniform(-scale_f1, scale_f1), 4)
        yopt_f2 = np.round(np.random.uniform(-scale_f2, scale_f2), 4)

        # Define component functions
        p = self.sample_p()
        f1 = MultiplePeakProblem([PeakFunction(xopt_f1, hessian_f1)],
                                 s = scale_f1, p = p, yopt = yopt_f1)
        f2 = MultiplePeakProblem([PeakFunction(xopt_f2, hessian_f2)],
                                 s = scale_f2, p = p, yopt = yopt_f2)

        bp = BONOProblem(f1, f2)

        round_steps = self.sample_round_steps()

        # Add plateaus
        if round_steps is not None:
            steps = (bp.get_nadir() - bp.get_ideal()) / round_steps
            bp.get_f1().set_step(steps[0])
            bp.get_f2().set_step(steps[1])

        return bp

class StructuredMultimodalGenerator(InstanceGenerator):
    """Generator for all structured multimodal problems."""

    def __init__(self, global_generator: UnimodalGenerator, n_peaks = 1000, cond = None):
        self.global_generator = global_generator
        self.n_peaks = n_peaks
        self.cond = cond

    def generate_instance(self, seed):
        if seed is not None:
            np.random.seed(seed)

        global_fn = self.global_generator.generate_instance(seed)

        d = self.global_generator.d
        # same_hessians = self.global_generator.same_hessians
        rotate_hessians = self.global_generator.rotate_hessians
        # fixed_dimensions = self.global_generator.fixed_dimensions
        n_peaks = self.n_peaks

        # generate f1

        f1_global = global_fn.get_f1().get_peaks()[0]

        f1_cond = f1_global.get_conditioning()

        if self.cond is not None and self.cond > 0:
            f1_cond = self.cond

        peaks_f1 = [f1_global + f1_global]

        while len(peaks_f1) < n_peaks:
            xlopt, _ = generate_optima(d)
            hessian_local = generate_hessian(d, cond = f1_cond, rotate = rotate_hessians)
            peaks_f1.append(f1_global + PeakFunction(xlopt, hessian_local))

        # generate f2

        f2_global = global_fn.get_f2().get_peaks()[0]

        f2_cond = f2_global.get_conditioning()

        if self.cond is not None and self.cond > 0:
            f2_cond = self.cond

        peaks_f2 = [f2_global + f2_global]

        while len(peaks_f2) < n_peaks:
            xlopt, _ = generate_optima(d)
            hessian_local = generate_hessian(d, cond = f2_cond, rotate = rotate_hessians)
            peaks_f2.append(f2_global + PeakFunction(xlopt, hessian_local))

        # get other copied parameters

        scale_f1 = global_fn.get_f1().get_scale()
        scale_f2 = global_fn.get_f2().get_scale()

        p_f1 = global_fn.get_f1().get_distance_parameter()
        p_f2 = global_fn.get_f2().get_distance_parameter()

        yopt_f1 = global_fn.get_f1().get_yopt()
        yopt_f2 = global_fn.get_f2().get_yopt()

        # combine fs

        f1 = MultiplePeakProblem(peaks_f1, s = scale_f1, p = p_f1, yopt = yopt_f1)
        f2 = MultiplePeakProblem(peaks_f2, s = scale_f2, p = p_f2, yopt = yopt_f2)

        bp = BONOProblem(f1, f2)

        if global_fn.get_f1().get_step() is not None:
            bp.get_f1().set_step(global_fn.get_f1().get_step())
        if global_fn.get_f2().get_step() is not None:
            bp.get_f2().set_step(global_fn.get_f2().get_step())

        return bp

class RandomMultimodalGenerator(InstanceGenerator):
    """Generates bi-objective problems with random topology"""

    def __init__(self, d: int, n_peaks: int, cond: list, p: list, round_steps: list, rotate_hessians = True):
        self.d = d
        self.n_peaks = n_peaks
        self.cond = cond
        self.p = p
        self.round_steps = round_steps
        self.rotate_hessians = rotate_hessians

    def generate_instance(self, seed: int):
        if seed is not None:
            np.random.seed(seed)

        d = self.d
        n_peaks = self.n_peaks
        rotate_hessians = self.rotate_hessians

        xopt_f1, xopt_f2 = generate_optima(d, min_dist = 2)

        hessian = generate_hessian(d, self.sample_conditioning(), rotate = rotate_hessians)
        peaks_f1 = [PeakFunction(xopt_f1, hessian)]

        while len(peaks_f1) < n_peaks:
            xlopt, _ = generate_optima(d)
            ylopt = np.random.uniform(1, 10)

            hessian = generate_hessian(d, cond = self.sample_conditioning(), rotate = rotate_hessians)

            peaks_f1.append(PeakFunction(xlopt, hessian, ylopt))

        hessian = generate_hessian(d, cond = self.sample_conditioning(), rotate = rotate_hessians)

        peaks_f2 = [PeakFunction(xopt_f2, hessian)]

        while len(peaks_f2) < n_peaks:
            xlopt, _ = generate_optima(d)
            ylopt = np.random.uniform(1, 10)

            hessian = generate_hessian(d, cond = self.sample_conditioning(), rotate = rotate_hessians)

            peaks_f2.append(PeakFunction(xlopt, hessian, ylopt))

        scale_f1 = np.round(np.pow(10, np.random.uniform(0, 6)), 4)
        scale_f2 = np.round(np.pow(10, np.random.uniform(0, 6)), 4)

        yopt_f1 = np.round(np.random.uniform(-scale_f1, scale_f1), 4)
        yopt_f2 = np.round(np.random.uniform(-scale_f2, scale_f2), 4)

        # Define component functions
        p = self.sample_p()
        f1 = MultiplePeakProblem(peaks_f1, s = scale_f1, p = p, yopt = yopt_f1)
        f2 = MultiplePeakProblem(peaks_f2, s = scale_f2, p = p, yopt = yopt_f2)

        bp = BONOProblem(f1, f2)

        round_steps = self.sample_round_steps()

        # Add plateaus
        if round_steps is not None:
            steps = (bp.get_nadir() - bp.get_ideal()) / round_steps
            bp.get_f1().set_step(steps[0])
            bp.get_f2().set_step(steps[1])

        return bp
