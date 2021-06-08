CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

# CARMA Configuration
# 
# This is the configuration module for carma set the option to ON in carma_config to enable the setting.
# Please see the documentation for details: https://carma.readthedocs.io/en/latest/configuration.html

# -- Armadillo version --
# Set which version of Armadillo to use
# Supported versions are:
#     - 10.5.x
#     - 10.4.x
# OPTION(USE_ARMA_VERSION "Which version of Armadillo to use" 10.5.x)

# -- ENABLE_CARMA_EXTRA_DEBUG --
# This option enables additional debuggin statements
OPTION(ENABLE_CARMA_EXTRA_DEBUG "Enable CARMA_EXTRA_DEBUG" OFF)
OPTION(ENABLE_ARMA_EXTRA_DEBUG "Enable ARMA_EXTRA_DEBUG" OFF)

# -- ENABLE_CARMA_SOFT_STEAL --
# When stealing the data of an array replace it with
# an array containing a single NaN
# This is a safer option compared to HARD_STEAL
# The default approach when staling is to only set the OWNDATA flag to False
OPTION(ENABLE_CARMA_SOFT_STEAL "Enable CARMA_SOFT_STEAL" OFF)

# -- ENABLE_CARMA_HARD_STEAL --
# When stealing the data of an array set nullptr
# NOTE this will cause a segfault when accessing the original array's data
# The default approach when staling is to only set the OWNDATA flag to False
OPTION(ENABLE_CARMA_HARD_STEAL "Enable CARMA_HARD_STEAL" OFF)

# -- REQUIRE_OWNDATA --
# Do NOT copy arrays if the data is not owned by Numpy, default behaviour
# is to copy when OWNDATA is False
OPTION(ENABLE_CARMA_DONT_REQUIRE_OWNDATA "Enable CARMA_DONT_REQUIRE_OWNDATA" OFF)

# -- REQUIRE_F_CONTIGUOUS --
# Do NOT copy C-contiguous arrays, default behaviour is to copy C-contiguous arrays to Fortran order as this is what Armadillo is optimised for.
# Note that on the conversion back it is assumed that the memory of a Armadillo object has Fortran order layout.
OPTION(ENABLE_CARMA_DONT_REQUIRE_F_CONTIGUOUS "Enable CARMA_DONT_REQUIRE_F_CONTIGUOUS" OFF)
