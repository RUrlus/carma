.. role:: cmake(code)
   :language: cmake

.. role:: bash(code)
   :language: bash


Configuration
#############

CARMA offers a number of compile-time settings that determine when the input
array's memory is not considered well-behaved and how to handle stealing the
memory of an incoming array

You can find the configuration file, ``carma_config.cmake``, in the root of the repository where you can enable or disable the various settings.

Array conditions
----------------

As detailed in the section :ref:`Well behaved` input arrays are copied when they don't meet the criteria. Two of these criteria can be turned off.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_DONT_REQUIRE_OWNDATA "Enable CARMA_DONT_REQUIRE_OWNDATA" ON)

Do **not** copy arrays if the data is not owned by Numpy, default behaviour is to copy when OWNDATA is False. Note this will lead to segfaults when the array's data is stolen or otherwise aliased by Armadillo. Is useful when you want to pass back and forth arrays previously converted by CARMA.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_DONT_REQUIRE_F_CONTIGUOUS "Enable CARMA_DONT_REQUIRE_F_CONTIGUOUS" ON)

Do **not** copy C-contiguous arrays, default behaviour is to copy C-contiguous arrays to Fortran order as this is what Armadillo is optimised for. Note that on the conversion back it is assumed that the memory of a Armadillo object has Fortran order layout.

Stealing
--------

The default behaviour is only to set ``OWNDATA=false`` when stealing
the data of an array. This is fast and leaves the array usable.

However, two additional options exists that make it clearer when an array has been stolen.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_SOFT_STEAL "Enable CARMA_SOFT_STEAL" ON)

When stealing the data of an array replace it with an array containing a single NaN. This is a safer but slower option compared to HARD_STEAL but easier to notice than the default.

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_HARD_STEAL "Enable CARMA_HARD_STEAL" ON)

When stealing the data of an array set nullptr in place of the memory. Note this will cause a segfault when accessing the original array's data. This ensures stolen arrays are not accidently used later on.

Debugging
---------

.. code-block:: cmake
    
    OPTION(ENABLE_CARMA_EXTRA_DEBUG "Enable CARMA_EXTRA_DEBUG" ON)
    OPTION(ENABLE_ARMA_EXTRA_DEBUG "Enable ARMA_EXTRA_DEBUG" ON)

Turn this setting on if you want to debug conversions. Debug prints are
generated that specify when arrays are not well-behaved, stolen or swapped in
place. Note that the additional debugging information from Armadillo can be
enabled using the second setting.


Developer settings
------------------

Two settings exists to faciliate development of CARMA:

.. code-block:: bash

    -DDEV_MODE=ON

This enables:

- :bash:`BUILD_TESTS=ON`
- :bash:`CARMA_DEV_TARGET=ON`
- :bash:`CMAKE_EXPORT_COMPILE_COMMANDS=1`
- :bash:`CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/build)`

.. code-block:: bash

    -DDEV_DEBUG_MODE=ON

Turns on :bash:`DEV_MODE` and
:bash:`ENABLE_CARMA_EXTRA_DEBUG`.
