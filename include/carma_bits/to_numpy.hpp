#pragma once

#include <armadillo>
#include <carma_bits/common.hpp>
#include <carma_bits/numpy_api.hpp>
#include <carma_bits/to_arma.hpp>

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
