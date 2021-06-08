.. role:: bash(code)
   :language: bash

First steps
###########

Carma relies on Pybind11 for the generation of the bindings and casting of the arguments from Python to C++.
Make sure you are familiar with `Pybind11 <https://pybind11.readthedocs.io/en/stable/intro.html>`_ before continuing on.

You can embed CARMA in a Pybind11 project using CMake command

.. code-block:: cmake

    ADD_SUBDIRECTORY(extern/carma)
    TARGET_LINK_LIBRARIES(<your_target> PRIVATE carma)

See `Pybind11's CMake build system documentation <https://pybind11.readthedocs.io/en/stable/compiling.html#building-with-cmake>`_ or `CARMA's examples <https://github.com/RUrlus/carma/blob/stable/examples/CMakeLists.txt>`_ for a start.

.. note:: 
    
    At the time of writing CARMA requires a forked version of Armadillo that
    uses Numpy's allocator and deallocator.
    The forked version is shipped with library and provided at build time, see :ref:`Build Configuration` for details.

The tests and examples can be compiled using CMake.
CMake can be installed with :bash:`pip install cmake`, your package manager or directly from `cmake <http://cmake.org/download/>`__.

.. code-block:: bash

   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. -DBUILD_EXAMPLES=true  -DBUILD_TESTS=true .. && make install

To run the tests you need to install `pytest`:

.. code-block:: bash

   pip install pytest

and run:

.. code-block:: bash

   ctest

To install `carma`, you have to define 

.. code-block:: bash
    
    -DCMAKE_INSTALL_PREFIX=/installation/path/directory

(default value is ``/usr/local``)

The installation directory contains

.. code-block:: bash

    include                  # carma headers
    extern/armadillo-code    # fork of armadillo required by carma
    tests                    # carma tests with python module (if enabled using -DBUILD_TESTS=on)
    examples                 # carma python examples with python module (if enabled using -DBUILD_EXAMPLES=on)

See section :ref:`Examples` for an overview of the conversion approaches.

Design Patterns
###############

CARMA was designed with three patterns in mind: borrow, transfer ownership and view.

Borrow
------

You can borrow the underlying memory of a Numpy array using the ``arr_to_*(py::array_t<T>, copy=false)``. The Armadillo object should not be returned without a copy out. Use this when you want to modify the memory in-place.
If the array is not well behaved, see :ref:`Well behaved`, the data is copied to well-behaved memory and swapped in place of the input array. If ``copy=true`` this is equivalent to the copy approach below.

.. note:: the size of the Armadillo object is not allowed change when you borrow, i.e. ``strict=true``.

Transfer ownership
------------------

You can transfer ownership to Armadillo using steal or copy.
After transferring ownership of the memory, Armadillo behaves as if it has allocated the memory itself, hence it will also free the memory upon destruction using Numpy's deallocator.

Steal
*****

If you want to take ownership of the underlying memory but don't want to copy the
data you can steal the array. The Armadillo object can be safely returned out without a copy. There are multiple compile time definitions on how the memory is stolen, see :doc:`Configuration <configuration>` for details. If the memory of the array is not well-behaved a copy of the memory is stolen.

.. note:: the size of the Armadillo object is allowed change after stealing, ``strict=false``.

Copy
****

If you want to give Armadillo full control of underlying memory but also want to keep Numpy as owner you should copy. The Armadillo object can be safely returned out without a copy. If the memory of the array is not well-behaved a copy of the memory is used instead.

.. note:: the size of the Armadillo object is allowed change after copying.

View
----

If you want to have a read-only view on the underlying memory you can use ``arr_to_*_view``. If the underlying memory is not well-behaved, excluding writeable, it will be copied.
