import numpy as np
from ferrodispcalc.compute import calculate_dielectric_constant


def _expected_eps(pol, volume, temperature):
    """Compute expected dielectric tensor directly for cross-checking."""
    eps0 = 8.854187817e-12
    kB = 1.380649e-23
    factor = volume * 1.0e-30 / (eps0 * kB * temperature)
    pairs = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
             "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    result = {}
    for label, (i, j) in pairs.items():
        mean_pi = np.mean(pol[..., i], axis=0)
        mean_pj = np.mean(pol[..., j], axis=0)
        mean_pipj = np.mean(pol[..., i] * pol[..., j], axis=0)
        result[f"eps_{label}"] = factor * (mean_pipj - mean_pi * mean_pj)
    return result


def test_dielectric_constant_global():
    rng = np.random.RandomState(42)
    P = rng.randn(20, 8, 3) * 0.1  # 20 frames, 8 cells
    volume = 500.0
    temperature = 300.0

    result = calculate_dielectric_constant(P, volume, temperature, atomic=False)

    P_avg = np.mean(P, axis=1)  # (20, 3)
    expected = _expected_eps(P_avg, volume, temperature)

    for key in expected:
        assert key in result
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-10)


def test_dielectric_constant_atomic():
    rng = np.random.RandomState(123)
    P = rng.randn(20, 4, 3) * 0.1  # 20 frames, 4 cells
    volume = 60.0
    temperature = 300.0

    result = calculate_dielectric_constant(P, volume, temperature, atomic=True)

    expected = _expected_eps(P, volume, temperature)

    for key in expected:
        assert key in result
        assert result[key].shape == (4,)
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-10)


def test_dielectric_constant_2d_input():
    rng = np.random.RandomState(99)
    P = rng.randn(30, 3) * 0.05  # already averaged, 30 frames
    volume = 500.0
    temperature = 300.0

    result = calculate_dielectric_constant(P, volume, temperature, atomic=False)

    expected = _expected_eps(P, volume, temperature)

    for key in expected:
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-10)


def test_dielectric_constant_invalid_inputs():
    P = np.random.randn(10, 3)

    try:
        calculate_dielectric_constant(P, volume=-1.0, temperature=300.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        calculate_dielectric_constant(P, volume=100.0, temperature=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        calculate_dielectric_constant(P, volume=100.0, temperature=300.0, atomic=True)
        assert False, "Should have raised ValueError for atomic=True with 2D input"
    except ValueError:
        pass
