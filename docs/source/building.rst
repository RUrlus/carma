.. role:: bash(code)
   :language: bash

.. role:: cmake(code)
   :language: cmake

Build Configuration
###################

CARMA v0.5 requires a compiler with support for C++14 and supports:

* Python 3.6 -- 3.9
* Numpy >= 1.14
* Pybind11 v2.6.0 -- v2.6.2
* Armadillo >= 10.5.2

CARMA target
------------

CARMA provides a CMake configuration that can be used to integrate into existing builds.
It is advised to use :bash:`ADD_SUBDIRECTORY`, this provides an interface target, ``carma``, that can be linked to your target.
This target pre-compiles the ``cnalloc.h`` header containing wrappers around Numpy's (de)allocator that are then picked up by Armadillo.
By pre-compiling the header we can ensure that the ``ARMA_ALIEN_MEM_ALLOC`` and ``ARMA_ALIEN_MEM_FREE`` definitions exist when including Armadillo
regardless of the include order.

.. warning:: if you are not using CARMA's cmake target you have to ensure that you include CARMA before Armadillo. Not doing so results in a compile error.

Armadillo
---------

Users can provide a specific Armadillo version by making sure the target ``armadillo`` is set before including CARMA or by setting:

The Armadillo version can be set using:

.. code-block:: bash
    
    -DARMADILLO_ROOT_DIR=/path/to/armadillo/root/directory

If neither is set, CARMA will provide the ``armadillo`` target at build time and store a clone of armadillo in ``carma/extern/armadillo-code``.  The Armadillo version, by default ``10.5.2``, can be set using:

.. code-block:: bash
    
    -DUSE_ARMA_VERSION=10.5.x

Pybind11
--------

Users can provide a specific Pybind11 version by making sure the target ``pybind11`` is set before including CARMA or by setting:

.. code-block:: bash
    
    -DPYBIND11_ROOT_DIR=/path/to/pybind11/root/directory


If neither is set, CARMA will provide the ``pybind11`` target at build time and store a clone in ``carma/extern/pybind11``.  The Pybind11 version, by default ``v2.6.2`` can be set using:

.. code-block:: bash
    
    -DUSE_PYBIND11_VERSION=v2.6.2

Python
------

CARMA needs to link against Python's and Numpy's headers and uses CMake's :bash:`FIND_PYTHON` to locate them.
:bash:`FIND_PYTHON` doesn't always find the right Python, e.g. when using ``pyenv``. When this happens you can set :bash:`Python3_EXECUTABLE`, which is then also passed on to Pybind11 to ensure the same Python versions are found.

.. code-block:: bash

    -DPython3_EXECUTABLE=$(which python3)
    # or
    -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)')
