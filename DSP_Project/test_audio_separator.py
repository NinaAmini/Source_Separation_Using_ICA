import numpy as np
import pytest
from cocktail_party import normalize_signal, apply_ica

@pytest.mark.parametrize("signal, expected", [
    ([-10.0, 0.0, 5.0, 20.0], [-0.5, 0.0, 0.25, 1.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.0, 1.0, -1.0, 0.5, -0.5], [0.0, 1.0, -1.0, 0.5, -0.5])
])
def test_normalize_signal(signal, expected):
    """Tests basic normalization, scaling, and zero-array edge cases."""
    np.testing.assert_array_almost_equal(normalize_signal(np.array(signal)), np.array(expected))

def test_apply_ica_pipeline():
    """Validates truncation, int16 conversion, output range, and source separation."""
    s1, s2 = apply_ica(np.random.rand(1050), np.random.rand(1000))
    
    assert len(s1) == len(s2) == 1000
    assert s1.dtype == s2.dtype == np.int16
    assert np.all(np.abs(s1) <= 32767) and np.all(np.abs(s2) <= 32767)
    assert not np.array_equal(s1, s2)