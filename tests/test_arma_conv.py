"""Test arma.h."""
import numpy as np

import libs._test_fsts_core as _core

test_flags = {
    1: 'Number of elements between array and matrix are not the same',
    2: 'Number of rows between array and matrix are not the same',
    3: 'Number of columns between array and matrix are not the same',
    4: 'Sum of elements between array and matrix is not aproximately equal',
    5: 'Pointer to memory is not as expected',
}


def test_arr_to_mat_double():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(10, 2)), dtype=np.float64, order='F'
    )
    flag = _core.arr_to_mat_double(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_long():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(10, 2)), dtype=np.int64, order='F'
    )
    flag = _core.arr_to_mat_long(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_double_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(np.random.normal(size=(10, 2)), dtype=np.float64)
    flag = _core.arr_to_mat_double(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_long_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(np.random.normal(size=(10, 2)), dtype=np.int64)
    flag = _core.arr_to_mat_long(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_double_copy():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(10, 2)), dtype=np.float64, order='F'
    )
    flag = _core.arr_to_mat_double_copy(sample)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_double_copy_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(10, 2)), dtype=np.float64, order='C'
    )
    flag = _core.arr_to_mat_double_copy(sample)
    assert flag == 0, test_flags[flag]


# #############################################################################
#                                   N-DIM 1                                   #
# #############################################################################
def test_arr_to_mat_1d():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(10)), dtype=np.float64, order='F'
    )
    flag = _core.arr_to_mat_1d(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_1d_copy():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(10)), dtype=np.float64, order='F'
    )
    flag = _core.arr_to_mat_1d(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='F')
    flag = _core.arr_to_col(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_col_C():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='C')
    flag = _core.arr_to_col(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_col_writeable():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='C')
    sample.setflags(write=0)
    flag = _core.arr_to_col(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col_copy():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='F')
    flag = _core.arr_to_col(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col_copy_C():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='C')
    flag = _core.arr_to_col(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='F')
    flag = _core.arr_to_row(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_row_C():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='C')
    flag = _core.arr_to_row(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_row_writeable():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='F')
    sample.setflags(write=0)
    flag = _core.arr_to_row(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row_copy():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='F')
    flag = _core.arr_to_row(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row_copy_C():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=10), dtype=np.float64, order='C')
    flag = _core.arr_to_row(sample, True, False)
    assert flag == 5, test_flags[flag]
