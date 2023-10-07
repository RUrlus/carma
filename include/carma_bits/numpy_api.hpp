#pragma once

#include <Python.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>

namespace carma {

namespace py = pybind11;

namespace internal {

inline void *get_data(PyArrayObject *src) { return PyArray_DATA(src); }
inline bool is_aligned(const PyArrayObject *src) { return PyArray_CHKFLAGS(src, NPY_ARRAY_ALIGNED); }
inline bool is_f_contiguous(const PyArrayObject *src) { return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS); }
inline bool is_c_contiguous(const PyArrayObject *src) { return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS); }

inline void set_not_writeable(PyArrayObject *src) { PyArray_CLEARFLAGS(src, NPY_ARRAY_WRITEABLE); }
inline void set_not_owndata(PyArrayObject *src) { PyArray_CLEARFLAGS(src, NPY_ARRAY_OWNDATA); }

inline void set_writeable(PyArrayObject *src) { PyArray_ENABLEFLAGS(src, NPY_ARRAY_WRITEABLE); }
inline void set_owndata(PyArrayObject *src) { PyArray_ENABLEFLAGS(src, NPY_ARRAY_OWNDATA); }

struct npy_api {
    using PyArray_Dims = struct {
        Py_intptr_t *ptr;
        int len;
    };

    static npy_api &get() {
        static npy_api api = lookup();
        return api;
    }

    PyArray_Descr *(*PyArray_DescrFromType_)(int typenum);
    int (*PyArray_Size_)(PyObject *src);
    int (*PyArray_CopyInto_)(PyArrayObject *dest, PyArrayObject *src);
    PyObject *(*PyArray_NewCopy_)(PyArrayObject *, int);
    PyObject *(*PyArray_NewFromDescr_)(
        PyTypeObject *subtype,
        PyArray_Descr *descr,
        int nd,
        npy_intp const *dims,
        npy_intp const *strides,
        void *data,
        int flags,
        PyObject *obj
    );
    void (*PyArray_Free_)(PyArrayObject *, void *ptr);
    PyObject *(*PyArray_NewLikeArray_)(PyArrayObject *prototype, NPY_ORDER order, PyArray_Descr *descr, int subok);
    void *(*PyDataMem_NEW_)(size_t nbytes);
    void (*PyDataMem_FREE_)(void *ptr);

   private:
    enum functions {
        API_PyArray_DescrFromType = 45,
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
        DECL_NPY_API(PyArray_DescrFromType);
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
