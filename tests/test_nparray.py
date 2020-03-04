"""Test nparray.h."""
import numpy as np

import libs.test_carma as carma


def test_is_f_contiguous():
    """Test is_f_contiguous."""
    m = 'F order array should be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_f_contiguous(sample) is True, m

    m = 'C order array should not be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert carma.is_f_contiguous(sample) is False, m


def test_is_c_contiguous():
    """Test is_c_contiguous."""
    m = 'C order array should be C contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert carma.is_c_contiguous(sample) is True, m

    m = 'F order array should not be C contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_c_contiguous(sample) is False, m


def test_is_writeable():
    """Test is_writable."""
    m = 'Array should be writable'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_writable(sample) is True, m

    m = 'Array should not be writable'
    sample.setflags(write=0)
    assert carma.is_writable(sample) is False, m


def test_is_owndata():
    """Test is_writable."""
    m = 'Array should own the data'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_owndata(sample) == sample.flags['OWNDATA'], m

    m = 'Array should not own the data'
    view = sample.reshape(20, 1)
    assert carma.is_owndata(view) == view.flags['OWNDATA'], m


def test_is_aligned():
    """Test is_aligned."""
    m = 'Array should be aligned'
    sample = np.arange(200, dtype=np.float64)
    assert carma.is_aligned(sample) == sample.flags['ALIGNED'], m

    m = 'Array should not be aligned'
    alt = np.frombuffer(sample.data, offset=2, count=100, dtype=np.float64)
    alt.shape = 10, 10
    assert carma.is_aligned(alt) == alt.flags['ALIGNED'], m
