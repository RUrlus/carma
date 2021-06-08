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
* Armadillo 10.4.x -- 10.5.x

See :ref:`Configuration` for instructions on how to set CARMA's configurable
settings.

Armadillo
---------

Note, at the time of writing CARMA requires a forked version of Armadillo that
uses Numpy's allocator and deallocator, this is required when handing over
memory to Armadillo and support for all major platforms. Although both Numpy and
Armadillo use ``malloc``, memory allocated by one cannot be de-allocated by the
other on Windows. 

CARMA provides forks for 10.4.x and 10.5.x and will provide forks of newer
versions when they are released. If a newer version of Armadillo is not yet
supported and you need it please open a ticket `here <https://github.com/RUrlus/carma/issues>`_. A pull-request will be opened to integrate the changes in Armadillo but in the mean time forked versions are shipped with the library and provided at build time. The fork is cloned and stored in ``carma/extern`` and installed alongside CARMA.

The Armadillo version can be set using:

.. code-block:: bash
    
    -DUSE_ARMA_VERSION=10.5.x

Pybind11
--------

Users can proivde a specific Pybind11 version by making sure the target ``pybind11`` is set before including CARMA or by setting:

.. code-block:: bash
    
    -DPYBIND11_ROOT_DIR=/path/to/pybind11/root/directory


If neither is set, CARMA will provide the ``pybind11`` target at build time and stored in ``carma/extern``.  The Pybind11 version can be set using:

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
