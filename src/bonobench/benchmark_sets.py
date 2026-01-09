"""Simplified generators for particular problem classes"""

from .problem_generators import UnimodalGenerator, StructuredMultimodalGenerator, RandomMultimodalGenerator

# Unimodal problems

class AxisAlignedSphere(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = 1, p = 2, round_steps = None,
                         same_hessians = True, rotate_hessians = False, fixed_dimensions = d - 1)

class AxisAlignedEllipsoid(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = [1e5, 1e6], p = 2, round_steps = None,
                         same_hessians = True, rotate_hessians = False, fixed_dimensions = d - 1)

class ConvexFrontEllipsoid(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = [50, 200], p = [1.5,3], round_steps = None,
                         same_hessians = True, rotate_hessians = True)

class LinearFrontEllipsoid(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = [50, 200], p = 1, round_steps = None,
                         same_hessians = True, rotate_hessians = True)

class ConcaveFrontEllipsoid(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = [50, 200], p = [1/3, 2/3], round_steps = None,
                         same_hessians = True, rotate_hessians = True)

class FreeEllipsoid(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = [50, 200], p = [1/3, 3], round_steps = None,
                         same_hessians = False, rotate_hessians = True)

class SteppedEllipsoid(UnimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, cond = [50, 200], p = [1/3, 3], round_steps = [50, 200],
                         same_hessians = False, rotate_hessians = True)

# Structured multimodal problems

N_PEAKS = 500

class MultimodalAxisAlignedSphere(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = AxisAlignedSphere(d), n_peaks = N_PEAKS)

class MultimodalAxisAlignedEllipsoid(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = AxisAlignedEllipsoid(d), n_peaks = N_PEAKS)

class MultimodalConvexFrontEllipsoid(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = ConvexFrontEllipsoid(d), n_peaks = N_PEAKS)

class MultimodalLinearFrontEllipsoid(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = LinearFrontEllipsoid(d), n_peaks = N_PEAKS)

class MultimodalConcaveFrontEllipsoid(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = ConcaveFrontEllipsoid(d), n_peaks = N_PEAKS)

class MultimodalFreeEllipsoid(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = FreeEllipsoid(d), n_peaks = N_PEAKS)

class MultimodalSteppedEllipsoid(StructuredMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(global_generator = SteppedEllipsoid(d), n_peaks = N_PEAKS)

# Random multimodal problems

N_FEW = 10
N_MANY = 100

class FewSpheres(RandomMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, n_peaks = N_FEW, cond = 1, p = 2, round_steps = None, rotate_hessians = False)

class ManySpheres(RandomMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, n_peaks = N_MANY, cond = 1, p = [1/3, 3], round_steps = None, rotate_hessians = False)

class SteppedManySpheres(RandomMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, n_peaks = N_MANY, cond = 1, p = [1/3, 3], round_steps = [50, 200], rotate_hessians = False)

class FewEllipsoids(RandomMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, n_peaks = N_FEW, cond = [50, 200], p = [1/3, 3], round_steps = None, rotate_hessians = True)

class ManyEllipsoids(RandomMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, n_peaks = N_MANY, cond = [50, 200], p = [1/3, 3], round_steps = None, rotate_hessians = True)

class SteppedManyEllipsoids(RandomMultimodalGenerator):

    def __init__(self, d: int):
        super().__init__(d, n_peaks = N_MANY, cond = [50, 200], p = [1/3, 3], round_steps = [50, 200], rotate_hessians = True)
