#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <carma_bits/base/numpy_converters.hpp>
#include <utility>

namespace py = pybind11;

namespace carma {
/*******************************************************************************
 *                               ARR_TO_ROW                                    *
 *******************************************************************************/

/**
 * \brief Converter to arma::Row for l-value references.
 * \details By default the BorrowConverter is used which requires
 *          that the numpy array is mutable, and well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Row<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_row(py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Row<eT>>()(arr);
}

/**
 * \brief Default converter to arma::Row for const l-value references.
 * \details By default the CopyConverter is used.
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Row<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_row(const py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Row<eT>>()(arr);
}

/**
 * \brief Converter to arma::Row for r-value references.
 * \details By default the MoveConverter is used which requires
 *          that the numpy array is well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Row<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_row(py::array_t<eT>&& arr) {
    return internal::DefaultNumpyConverter<arma::Row<eT>>()(std::forward<decltype(arr)>(arr));
}

/*******************************************************************************
 *                               ARR_TO_COL                                    *
 *******************************************************************************/

/**
 * \brief Converter to arma::Col for l-value references.
 * \details By default the BorrowConverter is used which requires
 *          that the numpy array is mutable, and well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Col<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_col(py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Col<eT>>()(arr);
}

/**
 * \brief Default converter to arma::Col for const l-value references.
 * \details By default the CopyConverter is used.
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Col<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_col(const py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Col<eT>>()(arr);
}

/**
 * \brief Converter to arma::Col for r-value references.
 * \details By default the MoveConverter is used which requires
 *          that the numpy array is well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Col<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_col(py::array_t<eT>&& arr) {
    return internal::DefaultNumpyConverter<arma::Col<eT>>()(std::forward<decltype(arr)>(arr));
}

/*******************************************************************************
 *                               ARR_TO_MAT                                    *
 *******************************************************************************/

/**
 * \brief Converter to arma::Mat for l-value references.
 * \details By default the BorrowConverter is used which requires
 *          that the numpy array is mutable, and well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Mat<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_mat(py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Mat<eT>>()(arr);
}

/**
 * \brief Default converter to arma::Mat for const l-value references.
 * \details By default the CopyConverter is used.
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Mat<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_mat(const py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Mat<eT>>()(arr);
}

/**
 * \brief Converter to arma::Mat for r-value references.
 * \details By default the MoveConverter is used which requires
 *          that the numpy array is well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Mat<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_mat(py::array_t<eT>&& arr) {
    return internal::DefaultNumpyConverter<arma::Mat<eT>>()(std::forward<decltype(arr)>(arr));
}
/*******************************************************************************
 *                               ARR_TO_CUBE                                   *
 *******************************************************************************/

/**
 * \brief Converter to arma::Cube for l-value references.
 * \details By default the BorrowConverter is used which requires
 *          that the numpy array is mutable, and well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Cube<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_cube(py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Cube<eT>>()(arr);
}

/**
 * \brief Default converter to arma::Cube for const l-value references.
 * \details By default the CopyConverter is used.
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Cube<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_cube(const py::array_t<eT>& arr) {
    return internal::DefaultNumpyConverter<arma::Cube<eT>>()(arr);
}

/**
 * \brief Converter to arma::Cube for r-value references.
 * \details By default the MoveConverter is used which requires
 *          that the numpy array is well-behaved.
 *
 *
 * \tparam eT element type
 * \param[in] arr pybind11 array to be converted
 * \return arma::Cube<eT> the created armadillo object
 */
template <typename eT>
auto arr_to_cube(py::array_t<eT>&& arr) {
    return internal::DefaultNumpyConverter<arma::Cube<eT>>()(std::forward<decltype(arr)>(arr));
}
}  // namespace carma
