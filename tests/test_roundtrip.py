"""Test numpy array to arma and back conversion functions."""
import numpy as np

import test_carma as carma


def test_mat_roundtrip():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)

def test_mat_roundtrip_large():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(1000, 1000)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_mat_roundtrip_small():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(3, 3)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_mat_roundtrip_c_order():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='C'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)

def test_mat_roundtrip_large_c_order():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(1000, 1000)), dtype=np.float64, order='C'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_mat_roundtrip_small_c_order():
    """Test mat_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(3, 3)), dtype=np.float64, order='C'
    )
    sample = og_sample.copy()
    out = carma.mat_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_row_roundtrip():
    """Test row_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(50)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.row_roundtrip(sample)
    assert np.allclose(og_sample, out)

def test_row_roundtrip_large():
    """Test row_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(1000)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.row_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_row_roundtrip_small():
    """Test row_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(3)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.row_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_col_roundtrip():
    """Test col_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(50)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.col_roundtrip(sample)
    assert np.allclose(og_sample, out.ravel())

def test_col_roundtrip_large():
    """Test col_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(1000)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.col_roundtrip(sample)
    assert np.allclose(og_sample, out.ravel())


def test_col_roundtrip_small():
    """Test col_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(3)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.col_roundtrip(sample)
    assert np.allclose(og_sample, out.ravel())


def test_cube_roundtrip():
    """Test cube_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(50, 3, 2)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.cube_roundtrip(sample)
    assert np.allclose(og_sample, out)

def test_cube_roundtrip_large():
    """Test cube_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=(1000, 10, 2)), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.cube_roundtrip(sample)
    assert np.allclose(og_sample, out)


def test_cube_roundtrip_small():
    """Test cube_roundtrip."""
    og_sample = np.asarray(
        np.random.normal(size=((2, 2, 2))), dtype=np.float64, order='F'
    )
    sample = og_sample.copy()
    out = carma.cube_roundtrip(sample)
    assert np.allclose(og_sample, out)
