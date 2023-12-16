#pragma once

namespace carma {

/* --------------------------------------------------------------
                        Configs
-------------------------------------------------------------- */

/**
 * \brief Create compile-time configuration object for Numpy to Armadillo
 * conversion.
 *
 * \tparam converter the converter to be used options are: BorrowConverter, CopyConverter, MoveConverter, ViewConverter
 * \tparam resolution_policy which resolution policy to use when the array cannot be converted directly, options are:
 * RaiseResolution, CopyResolution, CopySwapResolution
 * \tparam memory_order_policy which memory order policy to use, options are: ColumnOrder, TransposedRowOrder
 */
template <class converter, class resolution_policy, class memory_order_policy>
struct NumpyConversionConfig;

/* --------------------------------------------------------------
                        Converters
-------------------------------------------------------------- */
/**
 * \brief Borrow the memory s.t. the destination object is a mutable view.
 *
 * \details The destination object is a mutable view on the source object's memory.
 *          This requires that the lifetime of the source object is at least as long
 *          as that of the destination object. The source object keeps ownership of
 *          the data and is responsible for the memory management.
 *
 *          Numpy -> Arma:
 *          The destination arma object is strict and does not own the data.
 *          Borrowing is a good choice when you want to set/change
 *          values but the shape of the object will not change.
 *
 *          In order to borrow an array it's memory should
 *          be:
 *              * writeable
 *              * aligned and contiguous
 *              * compatible with the specified `memory_order_policy`
 *
 *          Arma -> Numpy:
 *          The destination array has Fortran order and is writeable
 *          but does not own the data.
 */
struct BorrowConverter;

/**
 * \brief Borrow the memory s.t. the destination object is a const view.
 *
 * \details The destination object is immutable view on the source object's memory.
 *          This requires that the lifetime of the source object is at least as long
 *          as that of the destination object. The source object keeps ownership of
 *          the data and is responsible for the memory management.
 *
 *          Numpy -> Arma:
 *          The destination arma object is strict and does not own the data.
 *          Viewing is a good choice when you only need read access to the data.
 *
 *          In order to create a const view of an array it's memory should
 *          be:
 *                 * aligned and contiguous
 *                 * compatible with the specified `memory_order_policy`
 *
 *          Arma -> Numpy:
 *          The destination array has Fortran order and is not writeable
 *          and does not own the data.
 */
struct ViewConverter;

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
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct CopyConverter;

/**
 * \brief Convert by taking ownership of the Numpy array's memory
 *
 * \details Convert the Numpy array to `armaT` by transfering
 *          ownership of the memory to the armadillo object.
 *          The resulting arma object is _not_ strict
 *          and owns the data.
 *
 *          After conversion the Numpy array will no longer own the
 *          memory, `owndata == false`.
 *
 *          In order to take ownership, the array's memory order should
 *          be:
 *              * owned by the array, aka not a view or alias
 *              * writeable
 *              * aligned and contiguous
 *              * compatible with the specified `memory_order_policy`
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct MoveConverter;

// /**
//  * \brief Convert by copying the Numpy array's memory
//  *
//  * \details Convert the Numpy array to `armaT` by copying the
//  *          memory. The resulting arma object is _not_ strict
//  *          and owns the data.
//  *
//  *          The copy converter does not have any requirements
//  *          with regard to the memory
//  *
//  *          If the array is not well-behaved we need to copy with Numpy
//  *          Note if we copy in because of the pre-alloc size we need to
//  *          free the memory again
//  *
//  * \param[in]   src    the view of the numpy array
//  * \return arma object
//  */
// struct CopyIntoConverter;

/* --------------------------------------------------------------
                    Memory order policies
-------------------------------------------------------------- */

/**
 * \brief Memory order policy that looks for C-order contiguous arrays
 *        and transposes them.
 * \details The TransposedRowOrder memory_order_policy expects
 *          that input arrays are row-major/C-order and converts them
 *          to column-major/F-order by transposing the array.
 *          If the array does not have the right order it is marked
 *          to be copied to the right order.
 */
struct TransposedRowOrder;

/**
 * \brief Memory order policy that looks for F-order contiguous arrays.
 * \details The ColumnOrder memory_order_policy expects
 *          that input arrays are column-major/F-order.
 *          If the array does not have the right order it is marked
 *          to be copied to the right order.
 */
struct ColumnOrder;

/* --------------------------------------------------------------
                    Resolution policies
-------------------------------------------------------------- */

/**
 * \brief Resolution policy that allows (silent) copying to meet the required
 * conditions when required. \details The CopyResolution is the default
 * resolution policy and will copy the input array when needed and possible.
 * CopyResolution policy cannot resolve when the BorrowConverter is used, the
 * CopySwapResolution policy can handle this scenario.
 */
struct CopyResolution;

/**
 * \brief Resolution policy that raises an runtime exception when the required
 * conditions are not met. \details The RaiseResolution is the strictest policy
 * and will raise an exception if any condition is not met, in contrast the
 * CopyResolution will silently copy when it needs and can. This policy should
 * be used when silent copies are undesired or prohibitively expensive.
 */
struct RaiseResolution;

/**
 * \brief Resolution policy that allows (silent) copying to meet the required
 * conditions when required even with BorrowConverter.
 *
 * \details The CopySwapResolution is behaves identically to CopyResolution policy with the
 * exception that it can handle ill conditioned and/or arrays with the wrong
 * memory layout. An exception is raised when the array does not own it's memory
 * or is marked as not writeable.
 *
 * \warning CopySwapResolution handles ill conditioned memory by copying the
 * array's memory to the right state and swapping it in the place of the existing memory.
 * This makes use of an deprecated numpy function to directly interface with the array fields. As
 * such this resolution policy should be considered experimental. This policy
 * will likely not work with Numpy >= v2.0
 */
struct CopySwapResolution;

}  // namespace carma
