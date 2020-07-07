"""Tests for ArrayStore class."""
import pytest
import numpy as np
import test_carma as carma


def test_ArrayStore_get_view():
    """Tests for ArrayStore class.get_view()."""
    sample = np.random.uniform(-1, 1, size=100)
    arraystore = carma.dArrayStore(sample, False)
    arr = arraystore.get_view(False)
    assert arr.flags['OWNDATA'] is False
    np.testing.assert_allclose(arr.flatten(), sample)


def test_ArrayStore_copy():
    """Test ArrayStore class when not stealing."""
    og_sample = np.random.uniform(-1, 1, size=100)
    sample = og_sample.copy()

    arraystore = carma.dArrayStore(sample, False)
    arr = arraystore.get_view(False)
    np.testing.assert_allclose(arr.flatten(), og_sample)

    # trigger descructor
    arraystore = None
    del arraystore
    arr = None
    del arr
    # Validate the memory of sample is untouched
    assert np.allclose(sample, og_sample)


def test_ArrayStore_non_writeable():
    """Test ArrayStore class when marked as non-readable."""
    sample = np.random.uniform(-1, 1, size=100)
    arraystore = carma.dArrayStore(sample, False)
    arr = arraystore.get_view(False)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is False
    with pytest.raises(ValueError):
        arr[0, 0] = 1.0


def test_ArrayStore_writeable():
    """Test ArrayStore class when marked as writeable."""
    sample = np.random.uniform(-1, 1, size=100)
    arraystore = carma.dArrayStore(sample, False)
    arr = arraystore.get_view(True)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is True
    arr[0, 0] = 1.0


def test_ArrayStore_steal():
    """Test ArrayStore class when we steal the memory."""
    og_sample = np.random.uniform(-1, 1, size=100)
    sample = og_sample.copy()

    arraystore = carma.dArrayStore(sample, True)
    arr = arraystore.get_view(True)
    np.testing.assert_allclose(arr.flatten(), sample)

    # trigger destructor
    arraystore = None
    del arraystore
    arr = None
    del arr
    # Validate the memory is now garbage
    assert not np.allclose(sample, og_sample)


def test_ArrayStore_set_data():
    """Test ArrayStore class function set_data."""
    sample1 = np.random.uniform(-1, 1, size=100)
    sample2 = np.random.uniform(-1, 1, size=100)

    arraystore = carma.dArrayStore(sample1, True)
    arr = arraystore.get_view(True)
    np.testing.assert_allclose(arr.flatten(), sample1)

    arraystore.set_data(sample2, True)
    arr = arraystore.get_view(True)
    np.testing.assert_allclose(arr.flatten(), sample2)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is True


def test_ArrayStore_set_data_flags():
    """Test ArrayStore class function set_data."""
    sample1 = np.random.uniform(-1, 1, size=100)
    sample2 = np.random.uniform(-1, 1, size=100)

    arraystore = carma.dArrayStore(sample1, True)
    arr = arraystore.get_view(True)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is True

    arraystore.set_data(sample2, True)
    arr = arraystore.get_view(False)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is False


def test_ArrayStore_get_view_float():
    """Tests for ArrayStore class.get_view()."""
    sample = np.random.uniform(-1, 1, size=100).astype(np.float32)
    arraystore = carma.fArrayStore(sample, False)
    arr = arraystore.get_view(False)
    assert arr.flags['OWNDATA'] is False
    np.testing.assert_allclose(arr.flatten(), sample)


def test_ArrayStore_get_view_long():
    """Tests for ArrayStore class.get_view()."""
    sample = np.random.randint(-10, 10, size=100).astype(np.int)
    arraystore = carma.lArrayStore(sample, False)
    arr = arraystore.get_view(False)
    assert arr.flags['OWNDATA'] is False
    np.testing.assert_allclose(arr.flatten(), sample)


def test_ArrayStore_get_view_int():
    """Tests for ArrayStore class.get_view()."""
    sample = np.random.randint(-10, 10, size=100).astype(np.int32)
    arraystore = carma.iArrayStore(sample, False)
    arr = arraystore.get_view(False)
    assert arr.flags['OWNDATA'] is False
    np.testing.assert_allclose(arr.flatten(), sample)


def test_ArrayStore_get_mat():
    """Tests for ArrayStore C++ api."""
    delta = carma.test_ArrayStore_get_mat()
    assert delta < 1e-6


def test_ArrayStore_get_mat_rvalue():
    """Tests for ArrayStore C++ api."""
    delta = carma.test_ArrayStore_get_mat()
    assert delta < 1e-6


def test_ArrayStore_get_view_cpp():
    """Tests for ArrayStore get_view."""
    arr = carma.test_ArrayStore_get_view(True)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is True

    arr = carma.test_ArrayStore_get_view(False)
    assert arr.flags['OWNDATA'] is False
    assert arr.flags['WRITEABLE'] is False
