.. role:: cmake(code)
   :language: cmake

.. role:: bash(code)
   :language: bash


Configuration
#############

CARMA offers a number of compile-time settings that determine when the input array's memory is not considered well-behaved and how to handle stealing the
memory of an incoming array

You can find the configuration file, ``carma_config.cmake``, in the root of the repository where you can enable or disable the various settings.

Array conditions
----------------

As detailed in the section :ref:`Well behaved` input arrays are copied when they don't meet the criteria. Two of these criteria can be turned off.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_DONT_REQUIRE_OWNDATA "Enable CARMA_DONT_REQUIRE_OWNDATA" ON)

.. warning::
    
    When using this option you have make sure that the memory of the input array can be safely freed using Numpy's deallocator (``PyDataMem_FREE``) as this is used to manage the memory after transfering ownership to Armadillo.

Do **not** copy arrays if the data is not owned by Numpy, default behaviour is to copy when OWNDATA is False.
Note this will lead to ``segfaults`` when the array's data is stolen or otherwise aliased by Armadillo.
Is useful when you want to pass back and forth arrays previously converted by CARMA.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_DONT_REQUIRE_F_CONTIGUOUS "Enable CARMA_DONT_REQUIRE_F_CONTIGUOUS" ON)

Do **not** copy C-contiguous arrays, default behaviour is to copy C-contiguous arrays to Fortran order as this is what Armadillo is optimised for.
Note that on the conversion back, it is assumed that the memory of the Armadillo object has Fortran order layout.

Stealing
--------

The default behaviour is to only set ``OWNDATA=false`` when stealing the data of an array. This is fast and leaves the array usable.
However, two additional options exists that make it clearer when an array has been stolen.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_SOFT_STEAL "Enable CARMA_SOFT_STEAL" ON)

When stealing the data of an array replace it with an array containing a single NaN. This is a safer but slower option compared to HARD_STEAL but easier to notice than the default.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_HARD_STEAL "Enable CARMA_HARD_STEAL" ON)

When stealing the data of an array CARMA sets a ``nullptr`` in place of the pointer to the memory.
Note this will cause a ``segfault`` when accessing the original array's data. However, this ensures stolen arrays are not accidentally used later on.

Debugging
---------

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_EXTRA_DEBUG "Enable CARMA_EXTRA_DEBUG" ON)
    OPTION(ENABLE_ARMA_EXTRA_DEBUG "Enable ARMA_EXTRA_DEBUG" ON)

Turn this setting on if you want to debug conversions. Debug prints are generated that specify when arrays are not well-behaved, stolen or swapped in
place. Note that the additional debugging information from Armadillo can be enabled using the second setting.


Release
-------

.. code-block:: cmake
    
    OPTION(ENABLE_ARMA_NO_DEBUG "Enable ENABLE_ARMA_NO_DEBUG" OFF)

This option sets ARMA_NO_DEBUG as part of the release flags.

"Disable all run-time checks, such as bounds checking. This will result in faster code, but you first need to make sure that your code runs correctly! We strongly recommend to have the run-time checks enabled during development, as this greatly aids in finding mistakes in your code, and hence speeds up development. We recommend that run-time checks be disabled only for the shipped version of your program (i.e. final release build)." -- Armadillo documentation

Developer settings
------------------

Two settings exists to faciliate development of CARMA:

.. code-block:: bash

    -DCARMA_DEV_MODE=ON

This enables:

- :bash:`CARMA_BUILD_TESTS=ON`
- :bash:`CARMA_DEV_TARGET=ON`
- :bash:`CMAKE_EXPORT_COMPILE_COMMANDS=1`
- :bash:`CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)`

.. code-block:: bash

    -DCARMA_DEV_DEBUG_MODE=ON

Turns on :bash:`CARMA_DEV_MODE` and
:bash:`ENABLE_CARMA_EXTRA_DEBUG`.
