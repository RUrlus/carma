"""Test armadillo matrix to numpy array type caster."""
import numpy as np

import test_carma as carma


def test_type_caster_in_mat():
    """Test type caster input handling of matrix."""
    sample = np.random.normal(size=(25, 2))
    npsum = sample.sum()
    accu = carma.tc_in_mat(sample)
    assert np.isclose(accu, npsum)

    sample = np.random.normal(size=(2, 25))
    npsum = sample.sum()
    accu = carma.tc_in_mat(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_row():
    """Test type caster input handling of row."""
    sample = np.random.normal(size=(25))
    npsum = sample.sum()
    accu = carma.tc_in_row(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_row_2d():
    """Test type caster input handling of 2d row."""
    sample = np.random.normal(size=(1, 25))
    npsum = sample.sum()
    accu = carma.tc_in_row(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_col():
    """Test type caster input handling of column."""
    sample = np.random.normal(size=(25))
    npsum = sample.sum()
    accu = carma.tc_in_col(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_col_2d():
    """Test type caster input handling of 2d column."""
    sample = np.random.normal(size=(25, 1))
    npsum = sample.sum()
    accu = carma.tc_in_col(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_cube():
    """Test type caster input handling of cube."""
    sample = np.random.normal(size=(25, 2, 3))
    npsum = sample.sum()
    accu = carma.tc_in_cube(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_cube_small_no_uaf():
    """Regression: passing a small cube (n_elem <= Cube_prealloc::mem_n_elem
    = 64) used to trigger a heap-use-after-free in type_caster::load().

    The previous code patched ``tmp.mem_state = 0`` and ``tmp.n_alloc =
    n_elem`` before ``std::move`` to force Armadillo's steal_mem to
    take the steal-pointer path. For ``n_alloc <= mem_n_elem``, that
    routed instead through the copy-then-reset-source branch, whose
    ``reset()`` called ``release(x.mem)`` on the numpy buffer. Numpy
    still held a refcount on the ndarray, so any subsequent read was
    a use-after-free that ASan caught immediately; release builds
    surfaced it as ``malloc(): unaligned tcache chunk`` later.

    This test exercises the small-cube load() path and then forces
    the allocator to reuse the (formerly freed) slot. With the fix
    (steal_mem reached via the ``is_move && mem_state == 2`` clause
    instead of the patching trick), the original ndarray remains
    untouched.
    """
    # 10 elements -- well below Cube_prealloc::mem_n_elem (64).
    sample = np.ones((1, 1, 10), order="F", dtype=np.float64)
    expected = sample.copy()
    accu = carma.tc_in_cube(sample)
    assert np.isclose(accu, 10.0)

    # Force the allocator to reuse any prematurely-freed slot — these
    # spray-allocate ndarrays of the same size as ``sample.data``.
    decoys = [np.zeros((1, 1, 10), dtype=np.float64) for _ in range(64)]
    del decoys

    # ``sample`` must still hold the original values. Under the old
    # buggy code, a subsequent allocator might land on the freed slot
    # and overwrite the data, breaking this assertion deterministically
    # (and any ASan-instrumented build would have caught the UAF
    # before we got here).
    assert np.allclose(sample, expected), (
        "small cube was modified after load() — heap-use-after-free in "
        "type_caster::load() reintroduced?"
    )


def test_type_caster_in_mat_small_no_uaf():
    """Same regression as above but for the matrix path
    (Mat_prealloc::mem_n_elem = 16)."""
    sample = np.ones((2, 5), order="F", dtype=np.float64)  # 10 elements
    expected = sample.copy()
    accu = carma.tc_in_mat(sample)
    assert np.isclose(accu, 10.0)
    decoys = [np.zeros((2, 5), dtype=np.float64) for _ in range(64)]
    del decoys
    assert np.allclose(sample, expected), (
        "small matrix was modified after load() — heap-use-after-free in "
        "type_caster::load() reintroduced?"
    )


def test_type_caster_out_mat():
    """Test type caster output handling of matrix."""
    sample = np.random.normal(size=(25, 2))
    mat = carma.tc_out_mat(sample)
    assert np.allclose(mat, sample + 1)


def test_type_caster_out_mat_const():
    """Test type caster output handling of matrix."""
    sample = np.random.normal(size=(25, 2))
    mat = carma.tc_out_mat_const(sample)
    assert np.allclose(mat, sample + 1)


def test_type_caster_out_mat_rvalue():
    """Test type caster output handling of matrix rvalue."""
    sample = np.random.normal(size=(25, 2))
    mat = carma.tc_out_mat_rvalue(sample)
    assert np.allclose(mat, sample + 1)


def test_type_caster_out_row():
    """Test type caster input handling of matrix."""
    sample = np.random.normal(size=(25))
    mat = carma.tc_out_row(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_row_2d():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(1, 25)), order='F')
    mat = carma.tc_out_row(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_row_rvalue():
    """Test type caster input handling of matrix."""
    sample = np.random.normal(size=(25))
    mat = carma.tc_out_row_rvalue(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_col():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25)), order='F')
    mat = carma.tc_out_col(sample)
    assert np.allclose(mat.flatten(), 1 + sample)


def test_type_caster_out_col_2d():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25, 1)), order='F')
    mat = carma.tc_out_col(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_col_rvalue():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25)), order='F')
    mat = carma.tc_out_col_rvalue(sample)
    assert np.allclose(mat.flatten(), 1 + sample)


def test_type_caster_out_cube():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25, 3, 2)), order='F')
    mat = carma.tc_out_cube(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_cube_rvalue():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25, 3, 2)), order='F')
    mat = carma.tc_out_cube_rvalue(sample)
    assert np.allclose(mat, 1 + sample)
