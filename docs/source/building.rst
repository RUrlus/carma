.. role:: bash(code)
   :language: bash

.. role:: cmake(code)
   :language: cmake

Build Configuration
###################

CARMA v0.5 requires a compiler with support for C++14 and supports:

* Python 3.6 -- 3.9
* Numpy >= 1.14
* Pybind11 >= v2.6.0
* Armadillo >= 10.5.2

CMake build
-----------

CARMA provides a CMake configuration that can be used to integrate into existing builds.
You can either use it directly or install it first. To edit the configuration please see 

It is advised to use the ``carma::carma`` interface target that can be linked to your target.
This target pre-compiles the ``cnalloc.h`` header containing wrappers around Numpy's (de)allocator that are then picked up by Armadillo.
By pre-compiling the header we can ensure that the ``ARMA_ALIEN_MEM_ALLOC`` and ``ARMA_ALIEN_MEM_FREE`` definitions exist when including Armadillo
regardless of the include order.

.. warning:: if you are not using CARMA's cmake target you have to ensure that you include CARMA before Armadillo. Not doing so results in a compile error.

Installation
************

Make sure to change the configuration if desired before installing CARMA.

CARMA can be installed using:

.. code-block:: bash
    
    mkdir build
    cd build
    # optionally with -DCMAKE_INSTALL_PREFIX:PATH=
    cmake -DCARMA_INSTALL_LIB=ON ..
    cmake --build . --config Release --target install

You can than include it in a project using:

.. code-block:: cmake

    FIND_PACKAGE(carma CONFIG REQUIRED)
    TARGET_LINK_LIBRARIES(<your_target> PRIVATE carma::carma)

The ``REQUIRED`` state is propagated to the dependencies of CARMA.

Variables
^^^^^^^^^

The :bash:`FIND_PACKAGE` call sets the following variables

- ``carma_FOUND`` -- true if CARMA was found
- ``carma_INCLUDE_DIR`` -- the path to CARMA's include directory
- ``carma_INCLUDE_DIRS`` -- the paths to CARMA's include directory and the paths to the include directories of the dependencies.

Components
^^^^^^^^^^

When including carma using :bash:`FIND_PACKAGE` two targets are created:

- ``carma::carma``

``carma::carma`` has been linked with Python, Numpy, Pybind11 and Armadillo and pre-compiles the ``cnalloc.h`` header which means that there is no required order to includes or carma and armadillo. If you only want this component you can use :bash:`FIND_PACKAGE(carma CONFIG REQUIRED COMPONENTS carma)`

- ``carma::headers``

If you want to have a header-only target that is not linked with the dependencies and does not pre-compile ``cnalloc.h``. You must than make sure to link it to it's dependencies and make sure to always include carma before armadillo.
If you only want this component you can use :bash:`FIND_PACKAGE(carma CONFIG REQUIRED COMPONENTS headers)`

Subdirectory
************

Alternatively, you can use CARMA without installing it by using:

.. code-block:: cmake

    ADD_SUBDIRECTORY(extern/carma)
    TARGET_LINK_LIBRARIES(<your_target> PRIVATE carma::carma)

The same targets and conditions as for the installation hold, however, this build will obtain Armadillo and Pybind11 if they have not been provided.

Armadillo
---------

Users can provide a specific Armadillo version by making sure the target ``armadillo`` is set before including CARMA or by setting:

.. code-block:: bash
    
    -DARMADILLO_ROOT_DIR=/path/to/armadillo/root/directory

When using the subdirectory build, if neither is set, CARMA will provide the ``armadillo`` target at build time and store a clone of armadillo in ``carma/extern/armadillo-code``.  The Armadillo version, by default ``10.5.2``, can be set using:

.. code-block:: bash
    
    -DUSE_ARMA_VERSION=10.5.x

Pybind11
--------

Users can provide a specific Pybind11 version by making sure the target ``pybind11`` is set before including CARMA or by setting:

.. code-block:: bash
    
    -DPYBIND11_ROOT_DIR=/path/to/pybind11/root/directory


When using the subdirectory build, if neither is set, CARMA will provide the ``pybind11`` target at build time and store a clone in ``carma/extern/pybind11``.  The Pybind11 version, by default ``v2.6.2`` can be set using:

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
