import numpy as np


# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):
    """
    Joint CDF F_XY(x, y) for (X, Y) ~ Uniform(0,1) x (0,1)
    """
    if x <= 0 or y <= 0:
        return 0.0
    elif 0 < x < 1 and 0 < y < 1:
        return x * y
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    elif x >= 1 and y >= 1:
        return 1.0
    else:
        return 0.0


def rectangle_probability(x1, x2, y1, y2):
    """
    Compute P(x1 < X <= x2, y1 < Y <= y2)
    using joint CDF
    """
    return (
        joint_cdf_unit_square(x2, y2)
        - joint_cdf_unit_square(x1, y2)
        - joint_cdf_unit_square(x2, y1)
        + joint_cdf_unit_square(x1, y1)
    )


def marginal_fx_unit_square(x):
    """
    Marginal PDF of X
    """
    return 1.0 if 0 < x < 1 else 0.0


def marginal_fy_unit_square(y):
    """
    Marginal PDF of Y
    """
    return 1.0 if 0 < y < 1 else 0.0


# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

def joint_pmf_heads(x, y):
    """
    Joint PMF for:
    X = heads in first toss
    Y = total heads in two tosses
    """
    pmf = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (0, 2): 0.0,
        (1, 0): 0.0,
        (1, 1): 0.25,
        (1, 2): 0.25,
    }
    return pmf.get((x, y), 0.0)


def marginal_px_heads(x):
    """
    Marginal PMF of X
    """
    return sum(joint_pmf_heads(x, y) for y in [0, 1, 2])


def marginal_py_heads(y):
    """
    Marginal PMF of Y
    """
    return sum(joint_pmf_heads(x, y) for x in [0, 1])


def check_independence_heads():
    """
    Check if X and Y are independent
    """
    for x in [0, 1]:
        for y in [0, 1, 2]:
            if not np.isclose(
                joint_pmf_heads(x, y),
                marginal_px_heads(x) * marginal_py_heads(y)
            ):
                return False
    return True
