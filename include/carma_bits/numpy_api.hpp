#ifndef INCLUDE_CARMA_BITS_NUMPY_API_HPP_
#define INCLUDE_CARMA_BITS_NUMPY_API_HPP_

#include <Python.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>

namespace carma {

namespace py = pybind11;

namespace internal {

struct npy_api {
    typedef struct {
        Py_intptr_t *ptr;
        int len;
    } PyArray_Dims;

    static npy_api &get() {
        static npy_api api = lookup();
        return api;
    }

    int (*PyArray_Size_)(PyObject *src);
    int (*PyArray_CopyInto_)(PyArrayObject *dest, PyArrayObject *src);
    PyObject *(*PyArray_NewCopy_)(PyArrayObject *, int);
    PyObject *(*PyArray_NewFromDescr_)(PyTypeObject *subtype, PyArray_Descr *descr, int nd, npy_intp const *dims,
                                       npy_intp const *strides, void *data, int flags, PyObject *obj);
    void (*PyArray_Free_)(PyArrayObject *, void *ptr);
    PyObject *(*PyArray_NewLikeArray_)(PyArrayObject *prototype, NPY_ORDER order, PyArray_Descr *descr, int subok);
    void *(*PyDataMem_NEW_)(size_t nbytes);
    void (*PyDataMem_FREE_)(void *ptr);

   private:
    enum functions {
        API_PyArray_Size = 59,
        API_PyArray_CopyInto = 82,
        API_PyArray_NewCopy = 85,
        API_PyArray_NewFromDescr = 94,
        API_PyArray_Free = 165,
        API_PyArray_NewLikeArray = 277,
        API_PyDataMem_NEW = 288,
        API_PyDataMem_FREE = 289,
    };

    static npy_api lookup() {
        py::module m = py::module::import("numpy.core.multiarray");
        auto c = m.attr("_ARRAY_API");
        void **api_ptr = reinterpret_cast<void **>(PyCapsule_GetPointer(c.ptr(), nullptr));
        npy_api api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_))api_ptr[API_##Func];
        DECL_NPY_API(PyArray_Size);
        DECL_NPY_API(PyArray_CopyInto);
        DECL_NPY_API(PyArray_NewCopy);
        DECL_NPY_API(PyArray_NewFromDescr);
        DECL_NPY_API(PyArray_Free);
        DECL_NPY_API(PyArray_NewLikeArray);
        DECL_NPY_API(PyDataMem_NEW);
        DECL_NPY_API(PyDataMem_FREE);
#undef DECL_NPY_API
        return api;
    }
};

}  // namespace internal
}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_NUMPY_API_HPP_
