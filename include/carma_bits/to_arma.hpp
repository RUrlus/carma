#ifndef INCLUDE_CARMA_BITS_TO_ARMA_HPP_
#define INCLUDE_CARMA_BITS_TO_ARMA_HPP_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <carma_bits/common.hpp>
#include <carma_bits/numpy.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carma {

namespace py = pybind11;

namespace internal {

template <typename armaT, iff_Row<armaT> = 0>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Row<eT> dest(src.data<eT>(), src.n_elem, src.copy_in, src.strict);
    return dest;
};

template <typename armaT, iff_Col<armaT> = 1>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Col<eT> dest(src.data<eT>(), src.n_elem, src.copy_in, src.strict);
    return dest;
};

template <typename armaT, iff_Mat<armaT> = 2>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Mat<eT> dest(src.data<eT>(), src.n_rows, src.n_cols, src.copy_in, src.strict);
    return dest;
};

template <typename armaT, iff_Cube<armaT> = 3>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    arma::Cube<eT> dest(src.data<eT>(), src.n_rows, src.n_cols, src.n_slices, src.copy_in, src.strict);
    return dest;
};

template <typename armaT, iff_Arma<armaT> = 0>
inline void give_ownership(armaT& dest, ArrayView& src) {
    arma::access::rw(dest.n_alloc) = src.n_elem;
    arma::access::rw(dest.mem_state) = 0;
    src.release_if_copied_in();
}

class FitsArmaType {
    template <typename armaT, iff_Vec<armaT> = 0>
    bool fits_vec(const ArrayView& src) {
        return (src.n_dim == 1) || ((src.n_dim == 2) && (src.shape[1] == 1 || src.shape[0] == 1));
    }

    template <typename armaT, iff_Mat<armaT> = 0>
    bool fits_mat(const ArrayView& src) {
        return (src.n_dim == 2) || ((src.n_dim == 3) && (src.shape[2] == 1 || src.shape[1] == 1 || src.shape[0] == 1));
    }

   public:
    template <typename armaT, iff_Vec<armaT> = 0>
    void check(const ArrayView& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 2) || (!fits_vec<armaT>(src)))) {
            throw std::runtime_error("|carma| cannot convert array to arma::Vec with dimensions: " +
                                     std::to_string(src.n_dim));
        }
    }

    template <typename armaT, iff_Mat<armaT> = 0>
    void check(const ArrayView& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 3) || (!fits_mat<armaT>(src)))) {
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
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(const internal::ArrayView& src) {
        return internal::to_arma<armaT>(src);
    };
};

struct ViewConverter {
    template <typename armaT, iff_Arma<armaT> = 0>
    const armaT get(const internal::ArrayView& src) {
        return internal::to_arma<armaT>(src);
    };
};

struct CopyConverter {
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(internal::ArrayView& src) {
        src.steal_copy();
        auto dest = internal::to_arma<armaT>(src);
        internal::give_ownership(dest, src);
        return dest;
    };
};

struct MoveConverter {
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(internal::ArrayView& src) {
        src.take_ownership();
        auto dest = internal::to_arma<armaT>(src);
        internal::give_ownership(dest, src);
        return dest;
    };
};

template <typename T>
using is_BorrowConverter = std::is_same<T, BorrowConverter>;

template <typename T>
using is_CopyConverter = std::is_same<T, CopyConverter>;

template <typename T>
using is_MoveConverter = std::is_same<T, MoveConverter>;

template <typename T>
using is_ViewConverter = std::is_same<T, ViewConverter>;

template <typename T>
struct is_Converter {
    static constexpr bool value = (is_BorrowConverter<T>::value || is_CopyConverter<T>::value ||
                                   is_ViewConverter<T>::value || is_MoveConverter<T>::value);
};

template <typename T>
using iff_BorrowConverter = std::enable_if_t<is_BorrowConverter<T>::value, int>;

template <typename T>
using iff_CopyConverter = std::enable_if_t<is_CopyConverter<T>::value, int>;

template <typename T>
using iff_MoveConverter = std::enable_if_t<is_MoveConverter<T>::value, int>;

template <typename T>
using iff_ViewConverter = std::enable_if_t<is_ViewConverter<T>::value, int>;

template <typename T>
using iff_Borrow_or_CopyConverter = std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value, int>;

template <typename T>
using iff_Converter = std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value ||
                                           is_MoveConverter<T>::value || is_ViewConverter<T>::value,
                                       int>;

template <typename T>
using iff_mutable_Converter =
    std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value || is_MoveConverter<T>::value, int>;

/* --------------------------------------------------------------
                    Memory order policies
-------------------------------------------------------------- */
struct TransposedRowOrder {
    template <typename aramT, iff_Row<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    };

    template <typename aramT, iff_Col<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    };

    template <typename aramT, iff_Mat<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[1];
        src.n_cols = src.shape[0];
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    };

    template <typename aramT, iff_Cube<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[2];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[0];
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    };
};

struct ColumnOrder {
    template <typename aramT, iff_Row<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    };

    template <typename aramT, iff_Col<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    };
    template <typename aramT, iff_Mat<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    };

    template <typename aramT, iff_Cube<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[2];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    };
};

template <typename T>
struct is_MemoryOrderPolicy {
    static constexpr bool value = (std::is_same_v<T, ColumnOrder> || std::is_same_v<T, TransposedRowOrder>);
};

/* --------------------------------------------------------------
                    Resolution policies
-------------------------------------------------------------- */

struct CopyResolution {
    template <typename armaT, typename converter, iff_BorrowConverter<converter> = 0>
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

    template <typename armaT, typename converter, iff_CopyConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_MoveConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            return CopyConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_ViewConverter<converter> = 0>
    const armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            return CopyConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    };
};

struct RaiseResolution {
    template <typename armaT, typename converter, iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.writeable))) {
            throw std::runtime_error("|carma| Cannot borrow an array that is ill-conditioned");
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_CopyConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_MoveConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            throw std::runtime_error("|carma| Cannot take ownership of an array that is ill-conditioned");
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_ViewConverter<converter> = 0>
    const armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            throw std::runtime_error("|carma| Cannot create view of an array that is ill-conditioned");
        }
        return ViewConverter().get<armaT>(src);
    };
};
struct CopySwapResolution {
    template <typename armaT, typename converter, iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY((!src.writeable) || (!src.owndata))) {
            throw std::runtime_error("|carma| Cannot CopySwap an array that does not own the data or is not writeable");
        } else if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            src.swap_copy();
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_CopyConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_MoveConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            return CopyConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_ViewConverter<converter> = 0>
    const armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            return CopyConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    };
};

template <typename T>
struct is_ResolutionPolicy {
    static constexpr bool value = (std::is_same_v<T, CopyResolution> || std::is_same_v<T, RaiseResolution> ||
                                   std::is_same_v<T, CopySwapResolution>);
};

/* --------------------------------------------------------------
                    ConversionConfig
-------------------------------------------------------------- */
#ifndef CARMA_DEFAULT_LVALUE_CONVERTER
#define CARMA_DEFAULT_LVALUE_CONVERTER carma::BorrowConverter
#endif  // CARMA_DEFAULT_LVALUE_CONVERTER

#ifndef CARMA_DEFAULT_CONST_LVALUE_CONVERTER
#define CARMA_DEFAULT_CONST_LVALUE_CONVERTER carma::ViewConverter
#endif  // CARMA_DEFAULT_CONST_LVALUE_CONVERTER

#ifndef CARMA_DEFAULT_RESOLUTION
#define CARMA_DEFAULT_RESOLUTION carma::CopyResolution
#endif  // CARMA_DEFAULT_RESOLUTION

#ifndef CARMA_DEFAULT_MEMORY_ORDER
#define CARMA_DEFAULT_MEMORY_ORDER carma::ColumnOrder
#endif  // CARMA_DEFAULT_MEMORY_ORDER

/*
Create compile-time configuration object.

converter : [BorrowConverter, CopyConverter, MoveConverter, ViewConverter]
    the converter to be used
resolution_policy : [CopyResolution, RaiseResolution, CopySwapResolution]
    which resolution policy to use when the array cannot be converted directly
memory_order_policy : [ColumnOrder, TransposedRowOrder]
    which memory order policy to use.
*/
template <class converter, class resolution_policy = CARMA_DEFAULT_RESOLUTION,
          class memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct ConversionConfig {
    static_assert(
        is_Converter<converter>::value,
        "|carma| `converter` must be one of: BorrowConverter, CopyConverter, ViewConverter or MoveConverter.");
    using int_converter = converter;
    static_assert(is_ResolutionPolicy<resolution_policy>::value,
                  "|carma| `resolution_policy` must be one of: CopyResolution, RaiseResolution, CopySwapResolution.");
    using resolution = resolution_policy;
    static_assert(is_MemoryOrderPolicy<memory_order_policy>::value,
                  "|carma| `memory_order_policy` must be one of: ColumnOrder, TransposedRowOrder.");
    using mem_order = memory_order_policy;
};

/*

                                ConversionConfig type_traits

*/
template <typename T>
using is_ConversionConfig = internal::is_instance<T, ConversionConfig>;

template <typename T>
using iff_ConversionConfig = std::enable_if_t<is_ConversionConfig<T>::value, int>;

namespace internal {

/*

                                npConverterImpl

*/

template <typename numpyT, typename armaT, typename converter, typename resolution_policy, typename memory_order_policy>
struct npConverterImpl {
    armaT operator()(numpyT src) {
        static_assert(is_Numpy<numpyT, typename armaT::elem_type>::value,
                      "|carma| `numpyT` must be a specialisation of py::array_t.");
        static_assert(is_Arma<armaT>::value, "|carma| `armaT` must be a (subclass) of Row, Col, Mat or Cube.");
        static_assert(not((is_MoveConverter<converter>::value || is_BorrowConverter<converter>::value) &&
                          std::is_const_v<std::remove_reference_t<numpyT>>),
                      "|carma| BorrowConverter and MoveConverter cannot be used with const py::array_t.");
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        static_assert(not(is_MoveConverter<converter>::value && (!std::is_rvalue_reference_v<numpyT>)),
                      "|carma| [optional] `MoveConverter` is only enabled for r-value references");
#endif
        internal::ArrayView arr(src);
        internal::FitsArmaType().check<armaT>(arr);
        memory_order_policy().template check<armaT>(arr);
        return resolution_policy().template resolve<armaT, converter>(arr);
    }
};

/*

                                npConverterBase

*/
template <typename numpyT, typename armaT, typename converter, typename resolution_policy = CARMA_DEFAULT_RESOLUTION,
          typename memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct npConverterBase {
    armaT operator()(numpyT src) {
        using eT = typename armaT::elem_type;
        // check template arguments
        static_assert(is_Converter<converter>::value,
                      "|carma| `converter` must be one of: BorrowConverter, CopyConverter, ViewConverter or "
                      "MoveConverter.");
        static_assert(
            is_ResolutionPolicy<resolution_policy>::value,
            "|carma| `resolution_policy` must be one of: CopyResolution, RaiseResolution, CopySwapResolution.");
        static_assert(is_MemoryOrderPolicy<memory_order_policy>::value,
                      "|carma| `memory_order_policy` must be one of: ColumnOrder, TransposedRowOrder.");
        return internal::npConverterImpl<numpyT, arma::Mat<eT>, converter, resolution_policy, memory_order_policy>()(
            src);
    }
};

}  // namespace internal

/*

                                npConverter

*/
template <typename numpyT, typename armaT, typename config>
struct npConverter {
    armaT operator()(numpyT src) {
        static_assert(is_ConversionConfig<config>::value,
                      "|carma| config must be a specialisation of `ConversionConfig`");
        return internal::npConverterImpl<numpyT, armaT, typename config::int_converter, typename config::resolution,
                                         typename config::mem_order>()(src);
    };
};

/*

                                arr_to_row

*/
template <typename eT, typename numpyT, typename config>
auto arr_to_row(numpyT arr) {
    return npConverter<numpyT, arma::Row<eT>, config>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_LVALUE_CONVERTER>
auto arr_to_row(py::array_t<eT>& arr) {
    return internal::npConverterBase<py::array_t<eT>&, arma::Row<eT>, converter>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_CONST_LVALUE_CONVERTER>
auto arr_to_row(const py::array_t<eT>& arr) {
    return internal::npConverterBase<const py::array_t<eT>&, arma::Row<eT>, converter>()(arr);
}

template <typename eT, typename converter = MoveConverter>
auto arr_to_row(py::array_t<eT>&& arr) {
    return internal::npConverterBase<py::array_t<eT>&&, arma::Row<eT>, converter>()(arr);
}

/*

                                arr_to_col

*/
template <typename eT, typename numpyT, typename config>
auto arr_to_col(numpyT arr) {
    return npConverter<numpyT, arma::Col<eT>, config>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_LVALUE_CONVERTER>
auto arr_to_col(py::array_t<eT>& arr) {
    return internal::npConverterBase<py::array_t<eT>&, arma::Col<eT>, converter>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_CONST_LVALUE_CONVERTER>
auto arr_to_col(const py::array_t<eT>& arr) {
    return internal::npConverterBase<const py::array_t<eT>&, arma::Col<eT>, converter>()(arr);
}

template <typename eT, typename converter = MoveConverter>
auto arr_to_col(py::array_t<eT>&& arr) {
    return internal::npConverterBase<py::array_t<eT>&&, arma::Col<eT>, converter>()(arr);
}

/*

                                arr_to_mat

*/
template <typename eT, typename numpyT, typename config>
auto arr_to_mat(numpyT arr) {
    return npConverter<numpyT, arma::Mat<eT>, config>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_LVALUE_CONVERTER>
auto arr_to_mat(py::array_t<eT>& arr) {
    return internal::npConverterBase<py::array_t<eT>&, arma::Mat<eT>, converter>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_CONST_LVALUE_CONVERTER>
auto arr_to_mat(const py::array_t<eT>& arr) {
    return internal::npConverterBase<const py::array_t<eT>&, arma::Mat<eT>, converter>()(arr);
}

template <typename eT, typename converter = MoveConverter>
auto arr_to_mat(py::array_t<eT>&& arr) {
    return internal::npConverterBase<py::array_t<eT>&&, arma::Mat<eT>, converter>()(arr);
}

/*

                                arr_to_cube

*/
template <typename eT, typename numpyT, typename config>
auto arr_to_cube(numpyT arr) {
    return npConverter<numpyT, arma::Cube<eT>, config>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_LVALUE_CONVERTER>
auto arr_to_cube(py::array_t<eT>& arr) {
    return internal::npConverterBase<py::array_t<eT>&, arma::Cube<eT>, converter>()(arr);
}

template <typename eT, typename converter = CARMA_DEFAULT_CONST_LVALUE_CONVERTER>
auto arr_to_cube(const py::array_t<eT>& arr) {
    return internal::npConverterBase<const py::array_t<eT>&, arma::Cube<eT>, converter>()(arr);
}

template <typename eT, typename converter = MoveConverter>
auto arr_to_cube(py::array_t<eT>&& arr) {
    return internal::npConverterBase<py::array_t<eT>&&, arma::Cube<eT>, converter>()(arr);
}

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_TO_ARMA_HPP_
