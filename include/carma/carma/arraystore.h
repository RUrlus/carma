#include <iostream>
#include  <pybind11/pybind11.h>
#include  <pybind11/numpy.h>
#include  <carma/carma/converters.h>

namespace py = pybind11;

#ifndef CARMA_ARRAYSTORE
#define CARMA_ARRAYSTORE

namespace carma {

template <typename T> class ArrayStore {
    protected:
        constexpr static ssize_t tsize = sizeof(T);
        bool _steal;
        bool _writeable;
        arma::Mat<T> _mat;
        T * _ptr;
        py::capsule _base;

        void _convert_to_arma(py::array_t<T> & arr) {
            if (_steal) {
                _mat = arr_to_mat<T>(arr, false);

                _ptr = _mat.memptr();
                _base = create_capsule(_ptr);

                // inform numpy it no longer owns the data
                set_not_owndata(arr);
                if (!_writeable) {
                    set_not_writeable(arr);
                }
            } else {
                _mat = arr_to_mat<T>(arr, true);
                _ptr = _mat.memptr();
                // create a dummy capsule as armadillo will be repsonsible
                // for descruction of the memory
                // We need a capsule to prevent a copy on the way out.
                _base = py::capsule(_ptr, [](void *f) {
                    #ifndef NDEBUG
                    // if in debug mode let us know what pointer is being freed
                    std::cerr << "freeing memory @ " << f << std::endl;
                    #endif

                });
            }
        }

    public:

        ArrayStore(py::array_t<T> & arr, bool steal, bool writeable) :
        _steal{steal}, _writeable{writeable}
        {
            /* Constructor
             *
             * Takes numpy array and converterts to Armadillo matrix.
             * If the array should be stolen we set owndata false for
             * numpy array.
             *
             * We store a capsule to serve as a reference for the
             * views on the data
             *
             */
            _convert_to_arma(arr);
        }

        void set_data(py::array_t<T> & arr, bool steal, bool writeable) {
            _steal = steal;
            _writeable = writeable;
            _convert_to_arma(arr);
        }

        py::array_t<T> get_view(bool writeable) {
            ssize_t nrows = static_cast<ssize_t>(_mat.n_rows);
            ssize_t ncols = static_cast<ssize_t>(_mat.n_cols);

            // create the array
            py::array_t<T> arr = py::array_t<T>(
                {nrows, ncols}, // shape
                {tsize, nrows * tsize}, // F-style contiguous strides
                _ptr, // the data pointer
                _base // numpy array references this parent
            );

            // inform numpy it does not own the buffer
            set_not_owndata(arr);

            if (!writeable) {
                set_not_writeable(arr);
            } else if (!_writeable) {
                throw std::runtime_error("store is marked non-writeable");
            }
            return arr;
        }
};

} /* namespace carma */

#endif /* CARMA_ARRAYSTORE */
