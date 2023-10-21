#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>

#include <armadillo>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_api.hpp>

namespace carma {

namespace internal {

// https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
// template <typename armaT>
// inline py::array_t<typename armaT::value_type> as_pyarray(armaT &&seq) {
//     auto size = seq.size();
//     auto data = seq.data();
//     std::unique_ptr<armaT> seq_ptr = std::make_unique<armaT>(std::move(seq));
//     auto capsule = py::capsule(seq_ptr.get(), [](void *p) { std::unique_ptr<armaT>(reinterpret_cast<armaT*>(p)); });
//     seq_ptr.release();
//     return py::array(size, data, capsule);
// }

template <typename armaT>
inline py::capsule create_capsule(armaT data) {
    return py::capsule(data, [](void* f) {
        auto mat = reinterpret_cast<armaT*>(f);
        carma_extra_debug_print("|carma| freeing memory @", mat->memptr());
        delete mat;
    });
} /* create_capsule */

template <typename armaT>
inline py::capsule create_view_capsule(const armaT* data) {
#ifdef CARMA_EXTRA_DEBUG
    return py::capsule(data, [](void* f) {
        auto mat = reinterpret_cast<armaT*>(f);
        std::cout << "|carma| destructing view on memory @" << mat->memptr() << "\n";
    });
#else
    return py::capsule(data, [](void*) {});
#endif
} /* create_view_capsule */

struct ArmaView {
    int n_dim;
    int flags;
    npy_intp* shape;
    npy_intp* strides;
    void* data = nullptr;
    NPY_ORDER target_order;
};

/**
 * \brief Create an array that references the data in a capsule.
 * \details Create an array that references the data.
 *          Lifetime management is handed over to Numpy but destruction is handled by
 *          the referenced-object in the capsule.
 *
 * \tparam eT              the element type
 * \param src              uniform object that provides view on Armadillo's object meta-data
 * \return py::array_t<eT> array with capsule base
 */
template <typename eT>
py::array_t<eT> create_capsule_array(const ArmaView& src);

/**
 * \brief Create an array that owns the data.
 * \details Create an array that owns the data without copying.
 *          Lifetime management is handed over to Numpy which will free the memory with its de-alloctor.
 *          The caller must ensure that memory was allocated using Numpy's allocator.
 *
 * \tparam eT              the element type
 * \param src              uniform object that provides view on Armadillo's object meta-data
 * \return py::array_t<eT> array with own data
 */
template <typename eT>
py::array_t<eT> create_owning_array(const ArmaView& src) {
    auto api = py::detail::npy_api::get();
    // get description from element type
    auto descr = py::dtype::of<eT>();
    auto obj = api.PyArray_NewFromDescr_(
        api.PyArray_Type_, descr.release().ptr(), src.n_dim, src.shape, src.strides, src.data, src.flags, nullptr
    );
    return py::reinterpret_steal<py::array_t<eT>>(obj);
}

/**
 * \brief Create an array that is a read-only view on the data.
 * \details Create an array that references read-only heap allocated data.
 *          Lifetime management remains with the caller and care must be taken to ensure
 *          that the underlying data remains valid for at least the array's lifetime.
 *
 * \tparam eT              the element type
 * \param src              uniform object that provides view on Armadillo's object meta-data
 * \return py::array_t<eT> array with read-only capsule base
 */
template <typename eT>
py::array_t<eT> create_reference_array(const ArmaView& src);

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
