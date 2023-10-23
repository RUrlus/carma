#pragma once
#include <armadillo>
#include <carma_bits/base/config.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_container.hpp>
#include <carma_bits/internal/numpy_converters.hpp>
#include <carma_bits/internal/type_traits.hpp>
#include <utility>  // std::forward

namespace carma {
/**
 * \brief Convert by copying the Numpy array's memory
 *
 * \details Convert the Numpy array to `armaT` by copying the
 *          memory in using arma. The resulting arma object
 *          is _not_ strict and owns the data.
 *
 *          The copy in converter requires that be
 *              * aligned and contiguous
 *              * compatible with the specified `memory_order_policy`
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct CopyInConverter {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    armaT get(internal::NumpyContainer& src) {
        src.copy_in = true;
        auto dest = internal::to_arma<armaT>(src);
        return dest;
    };

#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopyInConverter";
#endif
};

/**
 * \brief Convert by copying the Numpy array's memory
 *
 * \details Convert the Numpy array to `armaT` by copying the
 *          memory. The resulting arma object is _not_ strict
 *          and owns the data.
 *
 *          The copy converter does not have any requirements
 *          with regard to the memory
 *
 * if the array is not well-behaved we need to copy with Numpy
 * If we copy in because of the pre-alloc size we need to free the memory again
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct CopyIntoConverter {
    template <typename armaT, internal::iff_MatOrVec<armaT> = 0>
    armaT get(internal::NumpyContainer& src) {
        auto dest = internal::construct_arma<armaT>(src);
        src.copy_into(dest);
        return dest;
    };

    template <typename armaT, internal::iff_Cube<armaT> = 0>
    armaT get(internal::NumpyContainer& src) {
        src.copy_in = true;
        src.make_arma_compatible();
        auto dest = internal::to_arma<armaT>(src);
        src.free();
        return dest;
    };

#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopyIntoConverter";
#endif
};

}  // namespace carma
