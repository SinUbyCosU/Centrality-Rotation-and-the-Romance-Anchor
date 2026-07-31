import numpy as np
from utils.stats_utils import compute_cohens_d, compute_loo_r2, compute_bootstrap_r2

def assert_raises_value_error(func, *args, match=""):
    try:
        func(*args)
    except ValueError as e:
        if match in str(e):
            return True
        else:
            raise AssertionError(f"Expected ValueError matching '{match}', got '{str(e)}'")
    except Exception as e:
        raise AssertionError(f"Expected ValueError, got {type(e).__name__}")
    
    raise AssertionError("Expected ValueError, but no exception was raised")

def test_cohens_d_empty_group():
    assert_raises_value_error(compute_cohens_d, [], [1, 2, 3], match="undefined_empty_group")
    assert_raises_value_error(compute_cohens_d, [1, 2, 3], [], match="undefined_empty_group")

def test_cohens_d_zero_variance():
    assert_raises_value_error(compute_cohens_d, [1, 1, 1], [1, 1, 1], match="undefined_zero_variance")

def test_cohens_d_valid():
    d = compute_cohens_d(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert isinstance(d, float)
    assert d < 0  # Group 1 mean is lower than group 2 mean

def test_loo_r2_zero_variance():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 2, 2, 2, 2])
    assert_raises_value_error(compute_loo_r2, X, y, match="undefined_zero_variance")

def test_loo_r2_valid():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    r2 = compute_loo_r2(X, y)
    assert isinstance(r2, float)
    assert np.isclose(r2, 1.0)  # perfect correlation

def test_bootstrap_r2_zero_variance():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 2, 2, 2, 2])
    assert_raises_value_error(compute_bootstrap_r2, X, y, match="undefined_zero_variance")

def test_bootstrap_r2_valid():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    boot_mean, boot_ci = compute_bootstrap_r2(X, y, n_boot=10, random_state=42)
    assert isinstance(boot_mean, float)
    assert len(boot_ci) == 2
    assert boot_mean > 0

if __name__ == '__main__':
    test_cohens_d_empty_group()
    test_cohens_d_zero_variance()
    test_cohens_d_valid()
    test_loo_r2_zero_variance()
    test_loo_r2_valid()
    test_bootstrap_r2_zero_variance()
    test_bootstrap_r2_valid()
    print("All tests passed!")
