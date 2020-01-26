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
