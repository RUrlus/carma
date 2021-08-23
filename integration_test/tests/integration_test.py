"""Test nparray.h."""
import numpy as np

import integration_test_carma as carma


def test_is_f_contiguous():
    """Test is_f_contiguous."""
    m = 'F order array should be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_f_contiguous(sample) is True, m

    m = 'C order array should not be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert carma.is_f_contiguous(sample) is False, m


def test_mat_roundtrip():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)
