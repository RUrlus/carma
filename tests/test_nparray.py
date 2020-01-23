"""Test nparray.h."""
import pytest
import numpy as np

import libs._test_fsts_core as _core


def test_is_f_contiguous():
    """Test is_f_contiguous."""
    m = 'F order array should be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert _core.is_f_contiguous(sample) is True, m

    m = 'C order array should not be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert _core.is_f_contiguous(sample) is False, m


def test_is_c_contiguous():
    """Test is_c_contiguous."""
    m = 'C order array should be C contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert _core.is_c_contiguous(sample) is True, m

    m = 'F order array should not be C contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert _core.is_c_contiguous(sample) is False, m


def test_is_writeable():
    """Test is_writable."""
    m = 'Array should be writable'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert _core.is_writable(sample) is True, m

    m = 'Array should not be writable'
    sample.setflags(write=0)
    assert _core.is_writable(sample) is False, m


def test_is_owndata():
    """Test is_writable."""
    m = 'Array should own the data'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert _core.is_owndata(sample) == sample.flags['OWNDATA'], m

    m = 'Array should not own the data'
    view = sample.reshape(20, 1)
    assert _core.is_owndata(view) == view.flags['OWNDATA'], m


def test_is_aligned():
    """Test is_aligned."""
    m = 'Array should be aligned'
    sample = np.arange(200, dtype=np.float64)
    assert _core.is_aligned(sample) == sample.flags['ALIGNED'], m

    m = 'Array should not be aligned'
    alt = np.frombuffer(sample.data, offset=2, count=100, dtype=np.float64)
    alt.shape = 10, 10
    assert _core.is_aligned(alt) == alt.flags['ALIGNED'], m


def test_flat_reference():
    """Test flat_reference construct."""
    m = 'Incorrect sample value returned'
    sample = np.random.normal(size=(10, 2))
    assert _core.flat_reference(sample, 3) == sample[1, 1], sample

    sample = np.random.randint(low=-10, high=10, size=(10, 2))
    assert _core.flat_reference(sample, 3) == sample[1, 1], m


def test_flat_reference_fortran():
    """Test flat_reference construct with fortran layout."""
    m = 'Incorrect sample value returned'
    sample = np.asarray(
        np.random.normal(size=(10, 2)), dtype=np.float64, order='F'
    )
    assert _core.flat_reference(sample, 9) == sample[9, 0], m

    sample = np.asarray(
        np.random.randint(low=-10, high=10, size=(10, 2)),
        dtype=np.int64,
        order='F'
    )
    assert _core.flat_reference(sample, 9) == sample[9, 0], m


def test_flat_reference_non_writable():
    """Test flat_reference construct with non-writable array."""
    m = 'Array should not be writable'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    sample.setflags(write=0)
    assert _core.flat_reference(sample, 9) == sample[9, 0], m


def test_mutable_flat_reference():
    """Test flat_reference construct."""
    m = 'Incorrect sample value returned'
    sample = np.random.normal(size=(10, 2))
    assert abs(_core.mutable_flat_reference(sample, 3, 9.56) - 9.56) < 1e-12, m
    assert abs(sample[1, 1] - 9.56) < 1e-12, sample

    sample = np.random.randint(low=-10, high=10, size=(10, 2))
    assert _core.mutable_flat_reference(sample, 3, 9) == 9, m
    assert sample[1, 1] == 9


def test_mutable_flat_reference_fortran():
    """Test flat_reference construct with fortran layout."""
    m = 'Incorrect sample value returned'
    sample = np.asarray(
        np.random.normal(size=(10, 2)), dtype=np.float64, order='F'
    )
    assert abs(_core.mutable_flat_reference(sample, 9, 9.56) - 9.56) < 1e-12, m
    assert abs(sample[9, 0] - 9.56) < 1e-12, sample

    sample = np.asarray(
        np.random.randint(low=-10, high=10, size=(10, 2)),
        dtype=np.int64,
        order='F'
    )
    assert _core.mutable_flat_reference(sample, 9, 9) == 9, m
    assert sample[9, 0] == 9


def test_mutable_flat_reference_non_writable():
    """Test flat_reference construct with non-writable array."""
    m = 'Array should not be writable'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    sample.setflags(write=0)
    with pytest.raises(ValueError):
        assert _core.mutable_flat_reference(sample, 9, 9) == sample[9, 0], m
