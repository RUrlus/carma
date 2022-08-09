#ifndef INCLUDE_CARMA_BITS_CONVERTERS_HPP_
#define INCLUDE_CARMA_BITS_CONVERTERS_HPP_

#include <armadillo>
#include <carma_bits/array_view.hpp>
#include <carma_bits/common.hpp>
#include <carma_bits/to_arma.hpp>
#include <carma_bits/to_numpy.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace carma {

namespace py = pybind11;

/*******************************************************************************
 *                               npConverter                                   *
 *******************************************************************************/

/**
 * \brief Configurable Numpy to Armadillo converter.
 * \details npConverter should be used when you want to configure a specific conversion strategy for specific arma or
 * numpy types.
 *
 * \tparam armaT     armadillo type
 * \tparam numpyT    pybind11::array_t specialisation
 * \tparam config    carma::ConversionConfig object
 * \return armaT     the created armadillo object
 */
template <typename armaT, typename numpyT, typename config>
struct npConverter {
    using config_ = config;
    armaT operator()(numpyT&& src) {
        static_assert(is_ConversionConfig<config>::value,
                      "|carma| config must be a specialisation of `ConversionConfig`");
        return internal::npConverterImpl<armaT, typename config::converter_, typename config::resolution_,
                                         typename config::mem_order_>()
            .template operator()<decltype(src)>(std::forward<numpyT>(src));
    };
};

/**
 * \brief Compile time conversion assert.
 * \details Should be used in combination with the npConverter.
 *          Certain correct usage cannot be enforced in npConverter.
 *          For example, we cannot enforce a const armaT return type
 *          for the ViewConverter which it assumes.
 *
 * \tparam armaT     armadillo type
 * \tparam numpyT    pybind11::array_t
 * \tparam converter the converter used, i.e. the npConverter functor instance
 */
template <typename armaT, typename numpyT, typename converter>
inline void static_conversion_assert(armaT, numpyT, converter) {
    static_assert(not(is_ViewConverter<typename converter::config_::converter_>::value &&
                      (!std::is_const_v<std::remove_reference_t<numpyT>>)),
                  "numpyT should be const when using the ViewConverter.");
}

/**
 * \brief Generic Numpy to Armadillo converter.
 * \details Default generic converter with support for Row, Col, Mat and Cube.
 *          The converter used is based on the the armaT and numpyT.
 *          If numpyT is an r-value reference the MoveConverter is used.
 *          If armaT is const qualified the ViewConverter is used.
 *          If numpyT is an l-value reference the CARMA_DEFAULT_LVALUE_CONVERTER is used, BorrowConverter by default.
 *          If numpyT is an const l-value reference the CARMA_DEFAULT_CONST_LVALUE_CONVERTER is used, CopyConverter by
 *          default.
 *
 * \tparam armaT     armadillo type, cannot be deduced and must be specified
 * \tparam numpyT    pybind11::array_t specialisation, can often be deduced
 * \param[in] src    the numpy array to be converted
 * \return armaT     the created armadillo object
 */
template <typename armaT, typename numpyT>
armaT to_arma(numpyT&& src) {
    return internal::toArma<armaT>().template operator()<decltype(src)>(std::forward<numpyT>(src));
}

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
    return internal::toArma<arma::Row<eT>>()(arr);
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
    return internal::toArma<arma::Row<eT>>()(arr);
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
    return internal::toArma<arma::Row<eT>>()(arr);
}

/**
 * \brief Configurable Numpy to arma::Row converter.
 * \details this converter should be used when you want to use a specific configuration
 *          for pybind11::array_t specialisations.
 *
 * \tparam eT            element type
 * \tparam config        carma::ConversionConfig object
 * \tparam numpyT        pybind11::array_t specialisation
 * \return arma::Row<eT> the created armadillo object
 */
template <typename eT, typename config, typename numpyT>
auto arr_to_row(numpyT arr) {
    return npConverter<arma::Row<eT>, numpyT, config>()(arr);
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
    return internal::toArma<arma::Col<eT>>()(arr);
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
    return internal::toArma<arma::Col<eT>>()(arr);
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
    return internal::toArma<arma::Col<eT>>()(arr);
}

/**
 * \brief Configurable Numpy to arma::Col converter.
 * \details this converter should be used when you want to use a specific configuration
 *          for pybind11::array_t specialisations.
 *
 * \tparam eT            element type
 * \tparam config        carma::ConversionConfig object
 * \tparam numpyT        pybind11::array_t specialisation
 * \return arma::Col<eT> the created armadillo object
 */
template <typename eT, typename config, typename numpyT>
auto arr_to_col(numpyT arr) {
    return npConverter<arma::Col<eT>, numpyT, config>()(arr);
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
    return internal::toArma<arma::Mat<eT>>()(arr);
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
    return internal::toArma<arma::Mat<eT>>()(arr);
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
    return internal::toArma<arma::Mat<eT>>()(arr);
}

/**
 * \brief Configurable Numpy to arma::Mat converter.
 * \details this converter should be used when you want to use a specific configuration
 *          for pybind11::array_t specialisations.
 *
 * \tparam eT            element type
 * \tparam config        carma::ConversionConfig object
 * \tparam numpyT        pybind11::array_t specialisation
 * \return arma::Mat<eT> the created armadillo object
 */
template <typename eT, typename config, typename numpyT>
auto arr_to_mat(numpyT arr) {
    return npConverter<arma::Mat<eT>, numpyT, config>()(arr);
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
    return internal::toArma<arma::Cube<eT>>()(arr);
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
    return internal::toArma<arma::Cube<eT>>()(arr);
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
    return internal::toArma<arma::Cube<eT>>()(arr);
}

/**
 * \brief Configurable Numpy to arma::Cube converter.
 * \details this converter should be used when you want to use a specific configuration
 *          for pybind11::array_t specialisations.
 *
 * \tparam eT            element type
 * \tparam config        carma::ConversionConfig object
 * \tparam numpyT        pybind11::array_t specialisation
 * \return arma::Cube<eT> the created armadillo object
 */
template <typename eT, typename config, typename numpyT>
auto arr_to_cube(numpyT arr) {
    return npConverter<arma::Cube<eT>, numpyT, config>()(arr);
}

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_CONVERTERS_HPP_
