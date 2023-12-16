#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <carma_bits/base/config.hpp>
#include <carma_bits/internal/arma_container.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_api.hpp>
#include <carma_bits/internal/type_traits.hpp>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace carma {
namespace internal {

template <typename converter, typename memory_order_policy>
struct ArmaConverter {
    template <typename armaT, typename eT = armaT_eT<armaT>>
    py::array_t<eT> operator()(armaT&& src) {
        static_assert(
            is_Arma<armaT>::value,
            "|carma| `armaT` must be a (subclass of) `arma::Row`, `arma::Col`, "
            "`arma::Mat` or `arma::Cube`."
        );
        static_assert(
            is_Converter<converter>::value,
            "|carma| `converter` must be one of: BorrowConverter, "
            "CopyConverter, ViewConverter or "
            "MoveConverter."
        );
        static_assert(
            is_MemoryOrderPolicy<memory_order_policy>::value,
            "|carma| `memory_order_policy` must be one of: ColumnOrder, "
            "TransposedRowOrder."
        );
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        static_assert(
            not(is_MoveConverter<converter>::value && (!std::is_rvalue_reference_v<armaT>)),
            "|carma| [optional] `MoveConverter` is only enabled for r-value "
            "references"
        );
#endif
        ArmaContainer container(src);
        // FIXME add converter info
        // #ifdef CARMA_EXTRA_DEBUG
        //         NumpyConverterInfo<armaT, armaT, converter, resolution_policy, memory_order_policy>()(container);
        // #endif  // CARMA_EXTRA_DEBUG
        memory_order_policy().template check<armaT>(container);
        return converter().template get<decltype(src)>(std::forward<armaT>(src), container);
    }
};

template <typename memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct DefaultArmaConverter {
    template <typename armaT, typename eT = armaT_eT<armaT>>
    py::array_t<eT> operator()(armaT&& src) {
        if constexpr (std::is_rvalue_reference_v<decltype(src)>) {
            return ArmaConverter<MoveConverter, memory_order_policy>().template operator(
            )<decltype(src)>(std::forward<armaT>(src));
        } else {
            return ArmaConverter<CopyConverter, memory_order_policy>().template operator(
            )<decltype(src)>(std::forward<armaT>(src));
        }
    }
};

}  // namespace internal
}  // namespace carma

// std::vector<py::ssize_t> shape = {size, 1};
// std::vector<py::ssize_t> strides = py::detail::c_strides(shape, sizeof(eT));
// int flags
//     = (py::detail::npy_api::NPY_ARRAY_OWNDATA_ | py::detail::npy_api::NPY_ARRAY_ALIGNED_
//        | py::detail::npy_api::NPY_ARRAY_WRITEABLE_ | py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_
//        | py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_);
//

// template <typename armaT>
// struct ArmaView {
//     using eT = typename armaT::elem_type;
//     static constexpr auto tsize = static_cast<ssize_t>(sizeof(eT));
//     ssize_t n_rows;
//     ssize_t n_cols;
//     ssize_t n_slices;
//     ssize_t* shape;
//     ssize_t* strides;
//     static constexpr int n_dim = is_Vec<armaT>::value ? 1 :
//     arma::is_Mat<armaT>::value ? 2 : 3; eT* data = nullptr; armaT* obj =
//     nullptr;
// };

// target_order | NPY_ARRAY_OWNDATA | NPY_ARRAY_BEHAVED | NPY_ARRAY_WRITEABLE

// template <typename armaT, typename converter, typename resolution_policy,
// typename memory_order_policy> struct armaConverter {
//     using eT = typename armaT::elem_type;
//     py::array_t<eT> operator()(armaT&& src) {
//         auto view = ArmaView<armaT>();
//         // check if arma owns the mem
//         if constexpr (is_BorrowConverter<converter>::value) {
//         }
//     };
// };

// template <typename armaT, typename eT = typename armaT::elem_type>
// inline py::array_t<eT> to_numpy(const ArmaView<armaT>& src) {
//     return py::array_t<eT>(src.shape,                  // shape
//                            src.strides,                // F-style contiguous
//                            strides src.data,                   // the data
//                            pointer create_capsule<armaT>(src)  // numpy array
//                            references this parent
//     );
// };

// template <typename eT>
// inline py::array_t<eT> to_numpy(arma::Col<eT>* src) {
//     constexpr auto tsize = static_cast<ssize_t>(sizeof(eT));
//     auto n_rows = static_cast<ssize_t>(src->n_rows);

//     py::capsule base = create_capsule<arma::Row<eT>>(src);

//     return py::array_t<eT>({n_rows, static_cast<ssize_t>(1)},  // shape
//                            {tsize, tsize},                     // F-style
//                            contiguous strides src->memptr(), // the data
//                            pointer base                                //
//                            numpy array references this parent
//     );
// };
//
// template <typename armaT, iff_Mat<armaT> = 2>
// inline armaT to_numpy(const ArrayView& src) {
//     using eT = typename armaT::elem_type;
//     return arma::Mat<eT>(src.data<eT>(), src.n_rows, src.n_cols, src.copy_in,
//     src.strict);
// };

// template <typename armaT, iff_Cube<armaT> = 3>
// inline armaT to_numpy(const ArrayView& src) {
//     using eT = typename armaT::elem_type;
//     return arma::Cube<eT>(src.data<eT>(), src.n_rows, src.n_cols,
//     src.n_slices, src.copy_in, src.strict);
// };
