import numpy as np

def generate_optima(d: int, xl = -4, xu = 4, min_dist = 1, fixed_dimensions: int = 0):
    """Sample two points within [-4,4]^d with minimal distance min_dist"""
    assert fixed_dimensions < d

    x1 = np.zeros(d)
    x2 = np.zeros(d)

    while np.linalg.norm(x1 - x2) < min_dist:
        x1 = np.round(np.random.uniform(xl, xu, size = d), 4)
        x2 = np.round(np.random.uniform(xl, xu, size = d), 4)
        if fixed_dimensions > 0:
            fixed_idx = np.random.choice(range(d), fixed_dimensions, replace = False)
            x2[fixed_idx] = x1[fixed_idx]

    return (x1, x2)

def generate_hessian(d: int, cond: np.float64, rotate = True):
    """Generate a d-dimensional Hessian matrix with conditioning cond."""
    D = np.diag(np.append([1, cond], sample_loguniform(1, cond, size = d - 2)))

    if rotate:
        R = random_rotation_matrix(d)
    else:
        # permute values to shuffle Hessian
        R = np.eye(d)[np.random.permutation(d)]

    H = np.round(R.T @ D @ R, 4)

    return H

def sample_loguniform(minval, maxval, size = None):
    """Sample log-uniformly beetween min and max"""
    return np.exp(np.random.uniform(np.log(minval), np.log(maxval), size = size))

def random_rotation_matrix(d: int):
    """Create a random rotation matrix in d dimensions."""
    N = np.random.randn(d, d)
    Q, _ = np.linalg.qr(N)

    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q

def step_round(y, ymin, step):
    """Floor y to nearest value in [ymin, ymin + step, ymin + 2 * step, ...]"""
    return step * np.floor((y - ymin) / step) + ymin
