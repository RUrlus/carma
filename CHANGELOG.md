# Changelog

## [0.6.2] - 2021-10-28

### Fixed

* Bug where ill-conditioned arrays that don't own the data were copy-swapped -- #93

### Changes

* Generate config.h in build directory rather than in source -- #88
* Add MRE test suite for easier debugging
* Add support for Pybind11 2.8.1 and Armadillo 10.7.x

## [0.6.1] - 2021-09-04

### Changes

* include Pybind11 header in cnalloc to prevent linking issue on Windows in debug mode #87 

## [0.6.0] - 2021-08-23

### Fixed

- Bug wrt const return type path in type-caster (#83)
- Bug where TC input was copied (#83)
- Bug wrt handling of input arrays smaller than pre-allocation limit (#85)
- Remove unused variables (#84)

### Enhancements

- CARMA is now compatible with ``FIND_PACKAGE(carma CONFIG)`` (#81, #82)
- introduces ``to_numpy_view`` a read-only view on ArmaT as non-writeable array

## [0.5.2] - 2021-07-19

### Fixed

- Fixes bug where internal header had wrong relative include path after installation.

## [0.5.1] -  2021-07-08

### Fixed

- Fixes issue where CARMA version was not set if not installed or build through CMake

## [0.5.0] - 2021-06-14

New release of CARMA that has better interoperability with Armadillo.

### Changed

- Wrappers around Numpy (de)allocator included

- The wrappers round Numpy's (de)allocator that were used in the forked version of Armadillo have been moved to CARMA.
This header is pre-compiled to remove any include order dependency between CARMA and Armadillo

- Armadillo version check during compilation

- Added separate definitions for major, minor and patch version of CARMA

- Updated documentation

## [0.4.0] - 2021-06-08

Release 0.4.0 features almost complete rewrite of CARMA and can be considered the release candidate for 1.0.0.
Note this release is breaking but fixes a number of underlying issues in the old versions

### Breaking changes

- Include is now carma
- Default branch renamed to stable
- The nested structure in the include directory has been flattened. Users only require #include carma rather than carma/carma.h

- Minimum requirements updated

Pybind11 version is 2.6.x
Armadillo version is 10.4.x
Default copy behaviour for Cube, Col and Row is now identical to Mat

Previous versions were not always able to correctly take ownership of the memory underlying Armadillo objects.
This has been resolved with the new version.

- Update functions have been removed

The update functions have been removed as they no longer fit in the design pattern and were quite fragile.
Borrowed arrays no longer need update as they have strict parameter enforced. Stolen or copied arrays can be safely returned.

- Strict parameter to `arr_to_*` has been removed

The new behaviour sets the auxiliary memory for Armadillo objects to strict when borrowing and not when copying or stealing the memory.

- In-place swap when borrowing and ill-behaved arrays

On the conversion from Numpy to Armadillo we copy arrays that are not well-behaved to a well-behaved array and swap it in the place of the input array

## Changed

- Requires fork of Armadillo

v0.4.0 requires a fork of Armadillo that provides the Numpy allocator/de-allocator, we support Armadillo v1.14.x and v1.15.x
See build documentation for details.

- Armadillo and Pybind11 are no longer submodules but are provided at runtime when not present

- Introduced `arr_to_*_view`

The view functions returns a constant Armadillo object based on a const input that does not require writeable memory.

- new overloads for `arr_to_*`

We now provide const &, &, && overloads for `arr_to_*` and to_arma

- new overloads to `*_to_arr and to_numpy`

We now provide const &, &, && overloads for `*_to_arr` and `to_numpy`

- Moved CI/CD to Github actions

## [0.3.0] - 2020-07-13

### Fixed

- Fix deallocation bug

A bug existed where the memory was deallocated with `free` rather than the deallocator matching the allocator.

### Added

- CI/CD support for Windows, MacOS
- Test support with ctest

### Changed

- Armadillo and Pybind11 as submodules

Armadillo and Pybind11 are no longer shipped with CARMA in the test directory but have been included as submodules.

- Enable use of Armadillo and Pybind11 out of carma repository

This enables CARMA to be used in an existing project with different versions that included as submodules.

- Clang format

All source files have been formatted using clang format.

- Typos

Multiple typos have been correct in the comments and tests.
This change has no influence on API.

## [0.2.0] - 2020-06-28

### Changed 

- Fix spelling of writeable
- Restructure include directory

### Added

- ArrayStore

A class for holding the memory of a Numpy array as an Armadillo matrix in C++ and creating views on the memory as Numpy arrays.
An example use-case would be a C++ class that does not return all data computed, say a Hessian, but should do so on request.
The memory of the views is tied to lifetime of the class.


Functions to edit Numpy flags (OWNDATA, WRITEABLE)
Documentation example on how to take ownership of Numpy array

## [0.1.2] - 2020-05-31

### Added

- Functions to edit Numpy flags (OWNDATA, WRITEABLE)

Functions and documentation example on how to take ownership of Numpy array

## [0.1.2] - 2020-05-22

### Changed

- Fix in CMakelists as interface lib
- Fix non-template type in carma.h
