#ifndef INCLUDE_CARMA_BITS_NPTOARMA_HPP_
#define INCLUDE_CARMA_BITS_NPTOARMA_HPP_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carma {

namespace py = pybind11;

template <typename numpyT, typename eT>
using iffNumpy = std::enable_if_t<std::is_same_v<py::array_t<eT>, numpyT>, int>;

template <typename armaT>
using iffRow = std::enable_if_t<arma::is_Row<armaT>::value, int>;

template <typename armaT>
using iffCol = std::enable_if_t<arma::is_Col<armaT>::value, int>;

template <typename armaT>
using iffVec = std::enable_if_t<arma::is_Col<armaT>::value || arma::is_Row<armaT>::value, int>;

template <typename armaT>
using iffMat = std::enable_if_t<arma::is_Mat_only<armaT>::value, int>;

template <typename armaT>
using iffCube = std::enable_if_t<arma::is_Cube<armaT>::value, int>;

template <typename armaT>
using iffArma = std::enable_if_t<arma::is_Mat<armaT>::value || arma::is_Cube<armaT>::value, int>;

namespace internal {

inline void* get_data(PyObject* src) { return PyArray_DATA(reinterpret_cast<PyArrayObject*>(src)); }

inline void* get_data(PyArrayObject* src) { return PyArray_DATA(src); }

inline bool is_aligned(const PyArrayObject* src) { return PyArray_CHKFLAGS(src, NPY_ARRAY_ALIGNED); }

inline bool is_aligned(const PyObject* src) {
    return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject*>(src), NPY_ARRAY_ALIGNED);
}

inline bool is_f_contiguous(const PyObject* src) {
    return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject*>(src), NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_f_contiguous(const PyArrayObject* src) { return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS); }

inline bool is_c_contiguous(const PyObject* src) {
    return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject*>(src), NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_c_contiguous(const PyArrayObject* src) { return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS); }

struct npy_api {
    typedef struct {
        Py_intptr_t* ptr;
        int len;
    } PyArray_Dims;

    static npy_api& get() {
        static npy_api api = lookup();
        return api;
    }

    void (*PyArray_Free_)(PyArrayObject*, void* ptr);
    int (*PyArray_Size_)(PyObject* src);
    PyObject* (*PyArray_NewCopy_)(PyArrayObject*, int);
    int (*PyArray_CopyInto_)(PyArrayObject* dest, PyArrayObject* src);
    PyObject* (*PyArray_NewLikeArray_)(PyArrayObject* prototype, NPY_ORDER order, PyArray_Descr* descr, int subok);
    PyObject* (*PyArray_NewFromDescr_)(PyTypeObject* subtype, PyArray_Descr* descr, int nd, npy_intp const* dims,
                                       npy_intp const* strides, void* data, int flags, PyObject* obj);
    void* (*PyDataMem_NEW_)(size_t nbytes);
    void (*PyDataMem_FREE_)(void* ptr);

   private:
    enum functions {
        API_PyArray_Free = 165,
        API_PyArray_Size = 59,
        API_PyArray_NewCopy = 85,
        API_PyArray_CopyInto = 82,
        API_PyArray_NewLikeArray = 277,
        API_PyArray_NewFromDescr = 94,
        API_PyDataMem_NEW = 288,
        API_PyDataMem_FREE = 289,
    };

    static npy_api lookup() {
        py::module m = py::module::import("numpy.core.multiarray");
        auto c = m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
        void** api_ptr = reinterpret_cast<void**>(PyCapsule_GetPointer(c.ptr(), nullptr));
#else
        void** api_ptr = reinterpret_cast<void**>(PyCObject_AsVoidPtr(c.ptr()));
#endif
        npy_api api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_))api_ptr[API_##Func];
        DECL_NPY_API(PyArray_Free);
        DECL_NPY_API(PyArray_Size);
        DECL_NPY_API(PyArray_NewCopy);
        DECL_NPY_API(PyArray_CopyInto);
        DECL_NPY_API(PyArray_NewLikeArray);
        DECL_NPY_API(PyArray_NewFromDescr);
        DECL_NPY_API(PyDataMem_NEW);
        DECL_NPY_API(PyDataMem_FREE);
#undef DECL_NPY_API
        return api;
    }
};

class ArrayView {
   public:
    std::array<ssize_t, 3> shape;
    PyObject* obj;
    PyArrayObject* arr;
    void* mem;
    arma::uword n_elem;
    arma::uword n_rows = 0;
    arma::uword n_cols = 0;
    arma::uword n_slices = 0;
    int n_dim;
    // 0 is non-contigous; 1 is C order; 2 is F order
    int contiguous;
    //-1 for any order; 0 for C-order; 1 for F order
    NPY_ORDER target_order = NPY_ANYORDER;
    bool owndata;
    bool writeable;
    bool aligned;
    bool ill_conditioned;
    bool order_copy = false;
    bool copy_in = false;

    template <typename numpyT>
    explicit ArrayView(const numpyT src)
        : obj{src.ptr()},
          arr{reinterpret_cast<PyArrayObject*>(obj)},
          mem{PyArray_DATA(arr)},
          n_elem{static_cast<arma::uword>(src.size())},
          n_dim{static_cast<int>(src.ndim())},
          contiguous{is_f_contiguous(arr)   ? 2
                     : is_c_contiguous(arr) ? 1
                                            : 0},
          owndata{src.owndata()},
          writeable{src.writeable()},
          aligned{is_aligned(arr)} {
        int clipped_n_dim = n_dim < 3 ? n_dim : 3;
        std::memcpy(shape.data(), src.shape(), clipped_n_dim * sizeof(ssize_t));
        ill_conditioned = (!aligned) || (!static_cast<bool>(contiguous));
        copy_in = n_elem <= arma::arma_config::mat_prealloc;
    };

    template <typename eT>
    eT* data() const {
        return static_cast<eT*>(mem);
    }

    void take_ownership() { PyArray_CLEARFLAGS(arr, NPY_ARRAY_OWNDATA); }
};

/* Use Numpy's api to account for stride, order and steal the memory */
inline void steal_copy(ArrayView& src) {
    auto& api = npy_api::get();
    // build an PyArray to do F-order copy
    auto dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewLikeArray_(src.arr, src.target_order, nullptr, 0));

    // copy the array to a well behaved F-order
    int ret_code = api.PyArray_CopyInto_(dest, src.arr);
    if (ret_code != 0) {
        throw std::runtime_error("|carma| Copy of numpy array failed with ret_code: " + std::to_string(ret_code));
    }

    src.mem = PyArray_DATA(dest);
    // set OWNDATA to false such that the newly create
    // memory is not freed when the array is cleared
    PyArray_CLEARFLAGS(dest, NPY_ARRAY_OWNDATA);
    // free the array but not the memory
    api.PyArray_Free_(dest, nullptr);
}  // steal_copy_array

inline void swap_copy(ArrayView& src) {
    auto& api = npy_api::get();
    auto tmp = reinterpret_cast<PyArrayObject*>(api.PyArray_NewLikeArray_(src.arr, src.target_order, nullptr, 0));

    // copy the array to a well behaved target-order
    int ret_code = api.PyArray_CopyInto_(tmp, src.arr);
    if (ret_code != 0) {
        throw std::runtime_error("|carma| Copy of numpy array failed with ret_code: " + std::to_string(ret_code));
    }
    // swap copy into the original array
    auto tmp_of = reinterpret_cast<PyArrayObject_fields*>(tmp);
    auto src_of = reinterpret_cast<PyArrayObject_fields*>(src.arr);
    std::swap(src_of->data, tmp_of->data);

    // fix strides
    std::swap(src_of->strides, tmp_of->strides);

    PyArray_CLEARFLAGS(src.arr, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    PyArray_ENABLEFLAGS(src.arr, src.target_order | NPY_ARRAY_BEHAVED | NPY_ARRAY_OWNDATA);

    // clean up temporary which now contains the old memory
    PyArray_ENABLEFLAGS(tmp, NPY_ARRAY_OWNDATA);
    api.PyArray_Free_(tmp, PyArray_DATA(tmp));
    src.mem = PyArray_DATA(src.arr);
}  // swap_copy

template <typename armaT, iffRow<armaT> = 0>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Row<eT> dest(src.data<eT>(), src.n_elem, src.copy_in, true);
    return dest;
};

template <typename armaT, iffCol<armaT> = 1>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Col<eT> dest(src.data<eT>(), src.n_elem, src.copy_in, true);
    return dest;
};

template <typename armaT, iffMat<armaT> = 2>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Mat<eT> dest(src.data<eT>(), src.n_rows, src.n_cols, src.copy_in, true);
    return dest;
};

template <typename armaT, iffCube<armaT> = 3>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Cube<eT> dest(src.data<eT>(), src.n_rows, src.n_cols, src.n_slices, src.copy_in, true);
    return dest;
};

template <typename armaT, iffArma<armaT> = 0>
inline void give_ownership(armaT& dest, const ArrayView& src) {
    arma::access::rw(dest.n_alloc) = src.n_elem;
    arma::access::rw(dest.mem_state) = 0;
}

#define LIKELY(expr) __builtin_expect((expr), 1)
#define UNLIKELY(expr) __builtin_expect((expr), 0)

class FitsArmaType {
    template <typename armaT, iffVec<armaT> = 0>
    bool fits_vec(const ArrayView& src) {
        return (src.n_dim == 1) || ((src.n_dim == 2) && (src.shape[1] == 1 || src.shape[0] == 1));
    }

    template <typename armaT, iffMat<armaT> = 0>
    bool fits_mat(const ArrayView& src) {
        return (src.n_dim == 2) || ((src.n_dim == 3) && (src.shape[2] == 1 || src.shape[1] == 1 || src.shape[0] == 1));
    }

   public:
    template <typename armaT, iffVec<armaT> = 0>
    void check(const ArrayView& src) {
        if (UNLIKELY((src.n_dim < 1) || (src.n_dim > 2) || (!fits_vec<armaT>(src)))) {
            throw std::runtime_error("|carma| cannot convert array to arma::Vec with dimensions: " +
                                     std::to_string(src.n_dim));
        }
    }

    template <typename armaT, iffMat<armaT> = 0>
    void check(const ArrayView& src) {
        if (UNLIKELY((src.n_dim < 1) || (src.n_dim > 3) || (!fits_mat<armaT>(src)))) {
            throw std::runtime_error("|carma| cannot convert array to arma::Mat with dimensions: " +
                                     std::to_string(src.n_dim));
        }
    }
};

}  // namespace internal

/* --------------------------------------------------------------
                    Ownership policies
-------------------------------------------------------------- */
struct BorrowConverter {
    template <typename armaT, iffArma<armaT> = 0>
    armaT get(const internal::ArrayView& src) const {
        return internal::to_arma<armaT>(src);
    };
};

struct ViewConverter {
    template <typename armaT, iffArma<armaT> = 0>
    const armaT get(const internal::ArrayView& src) const {
        return internal::to_arma<armaT>(src);
    };
};

struct CopyConverter {
    template <typename armaT, iffArma<armaT> = 0>
    armaT get(internal::ArrayView& src) const {
        internal::steal_copy(src);
        auto dest = internal::to_arma<armaT>(src);
        internal::give_ownership(dest, src);
        return dest;
    };
};

struct MoveConverter {
    template <typename armaT, iffArma<armaT> = 0>
    armaT get(internal::ArrayView& src) const {
        src.take_ownership();
        auto dest = internal::to_arma<armaT>(src);
        internal::give_ownership(dest, src);
        return dest;
    };
};

template <typename convertion_policy>
using isViewConverter = std::is_same<convertion_policy, ViewConverter>;

template <typename convertion_policy>
using isMoveConverter = std::is_same<convertion_policy, MoveConverter>;

template <typename convertion_policy>
using isCopyConverter = std::is_same<convertion_policy, CopyConverter>;

template <typename convertion_policy>
using isBorrowConverter = std::is_same<convertion_policy, BorrowConverter>;

template <typename convertion_policy>
struct isConvertionPolicy {
    static constexpr bool value =
        (isBorrowConverter<convertion_policy>::value || isCopyConverter<convertion_policy>::value ||
         isViewConverter<convertion_policy>::value || isMoveConverter<convertion_policy>::value);
};

template <typename armaT>
using iffConst = std::enable_if_t<std::is_const_v<armaT>, int>;

template <typename convertion_policy>
using iffViewConverter = std::enable_if_t<std::is_same_v<convertion_policy, ViewConverter>, int>;

template <typename convertion_policy>
using iffMoveConverter = std::enable_if_t<std::is_same_v<convertion_policy, MoveConverter>, int>;

template <typename convertion_policy>
using iffCopyConverter = std::enable_if_t<std::is_same_v<convertion_policy, CopyConverter>, int>;

template <typename convertion_policy>
using iffBorrowConverter = std::enable_if_t<std::is_same_v<convertion_policy, BorrowConverter>, int>;

template <typename convertion_policy>
using iffBorrowOrCopyConverter = std::enable_if_t<
    std::is_same_v<convertion_policy, BorrowConverter> || std::is_same_v<convertion_policy, CopyConverter>, int>;

template <typename convertion_policy>
using iffConvertionPolicy = std::enable_if_t<
    std::is_same_v<convertion_policy, BorrowConverter> || std::is_same_v<convertion_policy, CopyConverter> ||
        std::is_same_v<convertion_policy, ViewConverter> || std::is_same_v<convertion_policy, MoveConverter>,
    int>;

template <typename convertion_policy>
using iffMutableConvertionPolicy = std::enable_if_t<std::is_same_v<convertion_policy, BorrowConverter> ||
                                                        std::is_same_v<convertion_policy, CopyConverter> ||
                                                        std::is_same_v<convertion_policy, MoveConverter>,
                                                    int>;

/* --------------------------------------------------------------
                    Memory order policies
-------------------------------------------------------------- */
struct TransposedRowOrder {
    template <typename aramT, iffRow<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    };

    template <typename aramT, iffCol<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    };

    template <typename aramT, iffMat<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[1];
        src.n_cols = src.shape[0];
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    };

    template <typename aramT, iffCube<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[2];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[0];
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    };
};

struct ColumnOrder {
    template <typename aramT, iffRow<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    };

    template <typename aramT, iffCol<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    };
    template <typename aramT, iffMat<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    };

    template <typename aramT, iffCube<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[2];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    };
};

template <typename T>
struct isMemoryOrderPolicy {
    static constexpr bool value = (std::is_same_v<T, ColumnOrder> || std::is_same_v<T, TransposedRowOrder>);
};

/* --------------------------------------------------------------
                    Resolution policies
-------------------------------------------------------------- */

struct CopyResolution {
    template <typename armaT, typename convertion_policy, iffBorrowConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        std::cout << "src.order_copy: " << src.order_copy << "\n";
        std::cout << "src.copy_in: " << src.copy_in << "\n";
        std::cout << "src.writeable: " << src.writeable << "\n";
        std::cout << "src.ill_conditioned: " << src.ill_conditioned << "\n";
        if (src.ill_conditioned || src.order_copy || (!src.writeable)) {
            throw std::runtime_error("|carma| Cannot borrow an array that is ill-conditioned");
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffCopyConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffMoveConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            internal::steal_copy(src);
            return MoveConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffViewConverter<convertion_policy> = 0>
    const armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::steal_copy(src);
            return MoveConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    };
};

struct RaiseResolution {
    template <typename armaT, typename convertion_policy, iffBorrowConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy || (!src.writeable))) {
            throw std::runtime_error("|carma| Cannot borrow an array that is ill-conditioned");
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffCopyConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffMoveConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            throw std::runtime_error("|carma| Cannot take ownership of an array that is ill-conditioned");
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffViewConverter<convertion_policy> = 0>
    const armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy)) {
            throw std::runtime_error("|carma| Cannot create view of an array that is ill-conditioned");
        }
        return ViewConverter().get<armaT>(src);
    };
};
struct CopySwapResolution {
    template <typename armaT, typename convertion_policy, iffBorrowConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY((!src.writeable) || (!src.owndata))) {
            throw std::runtime_error("|carma| Cannot CopySwap an array that does not own the data or is not writeable");
        } else if (UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::swap_copy(src);
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffCopyConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffMoveConverter<convertion_policy> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            internal::steal_copy(src);
            return MoveConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename convertion_policy, iffViewConverter<convertion_policy> = 0>
    const armaT resolve(internal::ArrayView& src) {
        if (UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::steal_copy(src);
            return MoveConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    };
};

template <typename T>
struct isResolutionPolicy {
    static constexpr bool value = (std::is_same_v<T, CopyResolution> || std::is_same_v<T, RaiseResolution> ||
                                   std::is_same_v<T, CopySwapResolution>);
};

#ifndef CARMA_DEFAULT_RESOLUTION
#define CARMA_DEFAULT_RESOLUTION carma::CopyResolution
#endif

#ifndef CARMA_DEFAULT_MEMORY_ORDER
#define CARMA_DEFAULT_MEMORY_ORDER carma::ColumnOrder
#endif

template <class convertion_policy, class resolution_policy = CARMA_DEFAULT_RESOLUTION,
          class memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct ConverterConfig {
    static_assert(
        isConvertionPolicy<convertion_policy>::value,
        "|carma| `convertion_policy` must be one of: BorrowConverter, CopyConverter, ViewConverter or MoveConverter.");
    using converter = convertion_policy;
    static_assert(isResolutionPolicy<resolution_policy>::value,
                  "|carma| `resolution_policy` must be one of: CopyResolution, RaiseResolution, CopySwapResolution.");
    using resolution = resolution_policy;
    static_assert(isMemoryOrderPolicy<memory_order_policy>::value,
                  "|carma| `memory_order_policy` must be one of: ColumnOrder, TransposedRowOrder.");
    using mem_order = memory_order_policy;
};

template <typename numpyT, typename armaT, typename convertion_policy, typename resolution_policy,
          typename memory_order_policy, iffArma<armaT> = 0, iffNumpy<numpyT, typename armaT::elem_type> = 0>
struct npConverterImpl {
    armaT operator()(numpyT src) {
        static_assert(not((isMoveConverter<convertion_policy>::value || isBorrowConverter<convertion_policy>::value) &&
                          (std::is_const_v<numpyT>)),
                      "|carma| BorrowConverter and MoveConverter cannot be used with const py::array_t.");
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        static_assert(not(isMoveConverter<convertion_policy>::value && (!std::is_rvalue_reference_v<numpyT>)),
                      "|carma| [optional] `MoveConverter` is only enabled for r-value references");
#endif
        internal::ArrayView arr(src);
        internal::FitsArmaType().check<armaT>(arr);
        memory_order_policy().template check<armaT>(arr);
        return resolution_policy().template resolve<armaT, convertion_policy>(arr);
    }
};

template <typename, template <typename...> typename>
// struct is_instance_impl : public std::false_type {};
struct is_instance_impl {
    static constexpr bool value = false;
};

template <template <typename...> typename U, typename... Ts>
struct is_instance_impl<U<Ts...>, U> {
    static constexpr bool value = true;
};

template <typename T, template <typename...> typename U>
// using is_instance = is_instance_impl<std::decay_t<T>, U>;
struct isInstance {
    static constexpr bool value = is_instance_impl<std::decay_t<T>, U>::value;
};

template <typename T>
using isConverterConfig = isInstance<T, ConverterConfig>;

template <typename T>
using iffConverterConfig = std::enable_if_t<isConverterConfig<T>::value, int>;

template <typename numpyT, typename armaT, typename config, typename eT = typename armaT::elem_type, iffArma<armaT> = 0,
          iffNumpy<numpyT, eT> = 0, iffConverterConfig<config> = 0>
struct npConverter {
    armaT operator()(numpyT src) {
        return npConverterImpl<py::array_t<eT>, arma::Mat<eT>, typename config::converter, typename config::resolution,
                               typename config::mem_order>()(src);
    };
};

template <typename numpyT, typename armaT, typename convertion_policy,
          typename resolution_policy = CARMA_DEFAULT_RESOLUTION,
          typename memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER, iffArma<armaT> = 0,
          iffNumpy<numpyT, typename armaT::elem_type> = 0>
struct npConverterBase {
    armaT operator()(numpyT src) {
        using eT = typename armaT::elem_type;
        // check template arguments
        static_assert(isConvertionPolicy<convertion_policy>::value,
                      "|carma| `convertion_policy` must be one of: BorrowConverter, CopyConverter, ViewConverter or "
                      "MoveConverter.");
        static_assert(
            isResolutionPolicy<resolution_policy>::value,
            "|carma| `resolution_policy` must be one of: CopyResolution, RaiseResolution, CopySwapResolution.");
        static_assert(isMemoryOrderPolicy<memory_order_policy>::value,
                      "|carma| `memory_order_policy` must be one of: ColumnOrder, TransposedRowOrder.");
        return npConverterImpl<py::array_t<eT>, arma::Mat<eT>, convertion_policy, resolution_policy,
                               memory_order_policy>()(src);
    }
};

template <typename eT, typename numpyT, typename config, iffConverterConfig<config> = 0>
arma::Mat<eT> arr_to_mat(numpyT arr) {
    return npConverter<numpyT, arma::Mat<eT>, config>()(arr);
}

template <typename eT, typename converter = BorrowConverter, iffConvertionPolicy<converter> = 0>
arma::Mat<eT> arr_to_mat(py::array_t<eT>& arr) {
    return npConverterBase<py::array_t<eT>, arma::Mat<eT>, converter>()(arr);
}

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_NPTOARMA_HPP_
