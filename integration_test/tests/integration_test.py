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
