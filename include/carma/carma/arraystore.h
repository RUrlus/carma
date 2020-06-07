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
    public:
        ArrayStore(py::array_t<T> & arr, bool steal, bool writeable) :
        _steal{steal}, _writeable{writeable}
        {
            if (steal) {
                _mat = arr_to_mat<T>(arr, false);
                // inform numpy it no longer owns the data
                set_not_owndata(arr);
                if (!_writeable) {
                    set_not_writeable(arr);
                }
            } else {
                _mat = arr_to_mat<T>(arr, true);
            }
            _ptr = _mat.memptr();
            // create object to return
            _base = create_capsule(_ptr);
        }

        inline void _update() {
            _ptr = _mat.memptr();
            _base = create_capsule(_ptr);
        }

        void set_data(py::array_t<T> & arr) {
            _mat = arr_to_mat<T>(arr, false);
            _update();
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
