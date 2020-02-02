"""Test armadillo matrix to  numpy array functions."""
import numpy as np

import libs.test_carma as carma

test_flags = {
    1: 'Number of elements between array and matrix are not the same',
    2: 'Number of rows between array and matrix are not the same',
    3: 'Number of columns between array and matrix are not the same',
    4: 'Sum of elements between array and matrix is not aproximately equal',
    5: 'Pointer to memory is not as expected',
}


def test_mat_to_arr_return():
    """Test arr_to_mat."""
    arr = carma.mat_to_arr_return()

    nan_cnt = np.isnan(arr).sum()
    assert nan_cnt == 0, 'Conversion introduced NaNs'

    zero_cnt = (arr == 0.0).sum()
    assert zero_cnt == 0, 'Conversion introduced zeros'

    inf_cnt = (arr == np.inf).sum()
    assert inf_cnt == 0, 'Conversion introduced inf values'


def test_mat_to_arr():
    """Test arr_to_mat."""
    flag = carma.mat_to_arr(False)
    assert flag == 0, test_flags[flag]


def test_mat_to_arr_copy():
    """Test arr_to_mat."""
    flag = carma.mat_to_arr(True)
    assert flag == 5, test_flags[flag]


def test_row_to_arr():
    """Test arr_to_mat."""
    flag = carma.row_to_arr(False)
    assert flag == 0, test_flags[flag]


def test_row_to_arr_copy():
    """Test arr_to_col."""
    flag = carma.row_to_arr(True)
    assert flag == 5, test_flags[flag]


def test_col_to_arr():
    """Test arr_to_col."""
    flag = carma.col_to_arr(False)
    assert flag == 0, test_flags[flag]


def test_col_to_arr_copy():
    """Test arr_to_col."""
    flag = carma.col_to_arr(True)
    assert flag == 5, test_flags[flag]


def test_cube_to_arr():
    """Test arr_to_cube."""
    flag = carma.cube_to_arr(False)
    assert flag == 0, test_flags[flag]


def test_cube_to_arr_copy():
    """Test arr_to_cube."""
    flag = carma.cube_to_arr(True)
    assert flag == 5, test_flags[flag]


def test_to_numpy_mat():
    """Test to_numpy_mat."""
    flag = carma.to_numpy_mat(False)
    assert flag == 0, test_flags[flag]


def test_to_numpy_mat_copy():
    """Test to_numpy_mat."""
    flag = carma.to_numpy_mat(True)
    assert flag == 5, test_flags[flag]


def test_to_numpy_cube():
    """Test to_numpy_cube."""
    flag = carma.to_numpy_cube(False)
    assert flag == 0, test_flags[flag]


def test_to_numpy_cube_copy():
    """Test to_numpy_cube."""
    flag = carma.to_numpy_cube(True)
    assert flag == 5, test_flags[flag]


def test_to_numpy_row():
    """Test to_numpy_row."""
    flag = carma.to_numpy_row(False)
    assert flag == 0, test_flags[flag]


def test_to_numpy_row_copy():
    """Test to_numpy_row."""
    flag = carma.to_numpy_row(True)
    assert flag == 5, test_flags[flag]


def test_to_numpy_col():
    """Test to_numpy_col."""
    flag = carma.to_numpy_col(False)
    assert flag == 0, test_flags[flag]


def test_to_numpy_col_copy():
    """Test to_numpy_col."""
    flag = carma.to_numpy_col(True)
    assert flag == 5, test_flags[flag]


def test_update_array_mat():
    """Test update_array."""
    arr = np.asarray(np.random.normal(size=(10, 2)), dtype=np.float, order='F')
    flag = carma.update_array_mat(arr, 2)
    assert flag == 0, test_flags[flag]
    assert arr.shape == (10, 4)
    assert np.abs(arr[:, 2:].sum()) < 1e-12, arr


def test_update_array_cube():
    """Test update_array."""
    arr = np.asarray(
        np.random.normal(size=(10, 2, 2)), dtype=np.float, order='F'
    )
    flag = carma.update_array_cube(arr, 2)
    assert flag == 0, test_flags[flag]
    assert arr.shape == (10, 4, 2)


def test_update_array_row():
    """Test update_array."""
    arr = np.asarray(
        np.random.normal(size=(1, 10)), dtype=np.float, order='F'
    )
    flag = carma.update_array_row(arr, 2)
    assert flag == 0, test_flags[flag]
    assert arr.shape == (1, 12)
    assert np.abs(arr[:, 10:].sum()) < 1e-12, arr


def test_update_array_col():
    """Test update_array."""
    arr = np.asarray(
        np.random.normal(size=(10, 1)), dtype=np.float, order='F'
    )
    flag = carma.update_array_col(arr, 2)
    assert flag == 0, test_flags[flag]
    assert arr.shape == (12, 1)
    assert np.abs(arr[10:, :].sum()) < 1e-12, arr
