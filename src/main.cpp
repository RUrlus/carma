// #define ARMA_EXTRA_DEBUG true
// #define CARMA_EXTENSION_MODE

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define CARMA_DEFAULT_MEMORY_ORDER carma::TransposedRowOrder
#include <carma>
#include <random>
// block reordering
#include <armadillo>
#include <iostream>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};

    py::module_ sys = py::module_::import("sys");

#ifdef CARMA_ENV_PATH
    sys.attr("path").attr("append")(CARMA_ENV_PATH);
#endif

    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");
    py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(10, 10));
    py::array_t<double> carr = py::array_t<double>(arr);
    py::array_t<double> farr = py::array_t<double, py::array::f_style | py::array::forcecast>(arr);

    std::cout << "carma version: " << carma::carma_version().as_string() << "\n";

    // using config = carma::NumpyConversionConfig<carma::MoveConverter, carma::CopyResolution,
    // carma::TransposedRowOrder>; using my_custom_converter = carma::NumpyConverter<arma::Mat<double>,
    // py::array_t<double>&&, config>; auto mat = my_custom_converter()(std::move(arr)); auto mat2 =
    // my_custom_converter()(std::move(farr));
    //
    auto mat = carma::arr_to_mat(arr);
    auto mat2 = carma::arr_to_mat(std::move(farr));

    auto matr = arma::Mat<double>(10, 10, arma::fill::randu);

    auto rconv = carma::internal::DefaultArmaConverter();
    py::array_t<double> rarr = rconv(std::move(matr));

    // auto mat = my_custom_converter()(std::move(arr));
    // auto mat2 = my_custom_converter()(std::move(farr));

    // py::array_t<double> farr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(2, 100)));
    // mat = carma::arr_to_mat(farr);

    // auto dtype = py::detail::npy_format_descriptor<double>::dtype();

    // using config = carma::ConversionConfig<carma::ViewConverter, carma::RaiseResolution, carma::TransposedRowOrder>;
    // using my_custom_converter = carma::npConverter<const arma::Mat<double>, const py::array_t<double>&, config>;
    // auto mat = my_custom_converter()(arr);

    // auto mat = carma::arr_to_mat(carr);

    // std::cout << "arr second element: " << *((const double*)(arr.data(0, 1)))
    // << "\n"; std::cout << "mat second element: " << mat.at(1, 0) << "\n";
    // mat.at(1, 0) = -1.111111;
    // std::cout << "arr second element: " << *((const double*)(arr.data(0, 1)))
    // << "\n"; std::cout << "mat second element: " << mat.at(1, 0) << "\n";

    // auto my_custom_converter2 = carma::npConverter<py::array_t<double>,
    // arma::Mat<double>, config>(); auto mat2 = my_custom_converter2(arr);

    std::cout << "n_rows: " << mat.n_rows << "\n";
    std::cout << "n_cols: " << mat.n_cols << "\n";
    std::cout << "n_elem: " << mat.n_elem << "\n";
    std::cout << "arr second element: " << arr.at(0, 1) << "\n";
    std::cout << "mat second element: " << mat.at(0, 1) << "\n";
    std::cout << mat << "\n";

    // using config = carma::ConversionConfig<carma::ViewConverter,
    // carma::CopyResolution, carma::ColumnOrder>; auto converter =
    // carma::npConverter<arma::Mat<double>, const py::array_t<double>&,
    // config>();
    // // auto view = carma::internal::ArrayView(arr);
    // // carma::ColumnOrder().check<arma::Mat<double>>(view);
    // // auto mat = carma::ViewConverter().get<arma::Mat<double>>(view);
    // auto mat = carma::to_arma<const arma::Mat<double>>(arr);
    // auto mat = carma::arr_to_mat(arr);
    // auto mat0 = carma::to_arma<arma::Mat<double>>(arr);
    // auto mat1 = carma::to_arma<arma::Mat<double>>(std::move(arr));
    // auto mat2 = carma::to_arma<arma::Mat<double>>(carr);

    // using config = carma::ConversionConfig<carma::BorrowConverter>;
    // auto converter = carma::npConverter<arma::Mat<double>, const
    // py::array_t<double>&, config>(); auto converter =
    // carma::internal::npConverterImpl<arma::Mat<double>, carma::MoveConverter,
    // carma::CopyResolution,
    //                                                   carma::ColumnOrder>();
    // auto converter = carma::internal::toArma<arma::Mat<double>>();
    // auto mat2 = converter(carr);
    // auto mat2 = carma::to_arma<arma::Mat<double>>(std::move(arr));
    // using config = carma::ConversionConfig<carma::MoveConverter>;
    // auto converter = carma::npConverter<arma::Mat<double>,
    // py::array_t<double>&&, config>(); auto mat = converter(std::move(arr));

    // auto mat = carma::internal::arr_to_mat<double, py::array_t<double>>(arr);
    // auto mat = carma::internal::toArma<arma::Mat<double>>()(arr);

    // using config = carma::ConversionConfig<carma::ViewConverter>;
    // auto converter = carma::npConverter<arma::Mat<double>, const
    // py::array_t<double>&, config>();
    // // auto converter = carma::internal::toArma<arma::Mat<double>, const
    // py::array_t<double>&>(); auto mat2 = converter(arr);
    // // FIXME add more checks
    // carma::static_conversion_assert(arr, mat2, converter);

    // auto mat = converter(arr);
    // auto mat = converter(arr);
    // carma::static_conversion_assert(arr, mat, converter);

    // auto mat = carma::arr_to_mat<double>(arr);
    // std::cout << mat << "\n";
    // std::cout << "arr second element: " << arr.at(0, 1) << "\n";
    // std::cout << "mat second element: " << mat.at(0, 1) << "\n";
    // mat.at(0, 1) = 2.0;
    // std::cout << "arr second element: " << arr.at(0, 1) << "\n";
    // std::cout << "mat second element: " << mat.at(0, 1) << "\n";
    return 0;
};
