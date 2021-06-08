<div align="center">
  <img src="docs/source/carma_logo_small.png" alt="carma_logo"/>
</div>

<br/>

<p align="center">
  A C++ header only library providing conversions between Numpy arrays and Armadillo matrices.
</p>
<p align="center">
  |
  <a href="https://carma.readthedocs.io/en/latest/">Documentation</a>
  |
</p>

[![Linux Build Status](https://github.com/RUrlus/carma/actions/workflows/linux.yml/badge.svg?branch=stable)](https://github.com/RUrlus/carma/actions/workflows/linux.yml)
[![MacOS Build Status](https://github.com/RUrlus/carma/actions/workflows/macos.yml/badge.svg?branch=stable)](https://github.com/RUrlus/carma/actions/workflows/macos.yml)
[![Windows Build Status](https://github.com/RUrlus/carma/actions/workflows/windows.yml/badge.svg?branch=stable)](https://github.com/RUrlus/carma/actions/workflows/windows.yml)
[![Coverage Status](https://coveralls.io/repos/github/RUrlus/carma/badge.svg?branch=master)](https://coveralls.io/github/RUrlus/carma?branch=master)
[![Documentation Status](https://readthedocs.org/projects/carma/badge/?version=latest)](https://carma.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/RUrlus/carma)](https://github.com/RUrlus/carma/blob/stable/LICENSE)
[![Release](https://img.shields.io/github/v/release/rurlus/carma)](https://github.com/RUrlus/carma/releases)

## Introduction

CARMA provides fast bidirectional conversions between [Numpy](https://numpy.org) arrays and [Armadillo](http://arma.sourceforge.net/docs.html) matrices, vectors and cubes, much like [RcppArmadillo](https://github.com/RcppCore/RcppArmadillo) does for R and Armadillo.

The library extends the impressive [pybind11](https://pybind11.readthedocs.io/en/stable/intro.html) library with support for Armadillo.
For details on Pybind11 and Armadillo refer to their respective documentation [1](https://pybind11.readthedocs.io/en/stable/intro.html), [2](http://arma.sourceforge.net/docs.html).

## Installation
CARMA is a header only library that relies on two other header only libraries, Armadillo and Pybind11.

CARMA can be integrated in a CMake build using `ADD_SUBDIRECTORY(<path_to_carma>)` which provides an interface library target `carma`
that has been linked with Python, Numpy, Pybind11 and Armadillo.
**Note, at the time of writing CARMA requires a forked version of Armadillo that
uses Numpy's allocator and deallocator.** This fork is provided as an interface libary target
`carma_armadillo`.

To link with CARMA:
```cmake
ADD_SUBDIRECTORY(extern/carma)
TARGET_LINK_LIBRARIES(<your_target> PRIVATE carma)
```
CARMA and Armadillo can then be included using:
```C++
#include <carma>
#include <armadillo>
```

CARMA provides a number of configurations that can be set in the `carma_config.cmake` file at the root of the directory or passed to CMake, see [Configuration](https://carma.readthedocs.io/en/stable/configuration.html) and [Build configuration](https://carma.readthedocs.io/en/stable/building.html) documentation sections for details.

## Requirements

CARMA v0.5 requires a compiler with support for C++14 and supports:

* Python 3.6 -- 3.9
* Numpy >= 1.14
* Pybind11 v2.6.0 -- v2.6.2
* Armadillo 10.4.x -- 10.5.x

**Note, at the time of writing CARMA requires a forked version of Armadillo that
uses Numpy's allocator and deallocator.**
The forked version is shipped with the library and provided at build time.
For details see the [Build configuration](https://carma.readthedocs.io/en/stable/building.html).

### Considerations

In order to achieve fast conversions the default behaviour is avoid copying both from and to Numpy whenever possible and reasonable.
This allows very low overhead conversions but it impacts memory safety and requires user vigilance.

If you intend to return the memory of the input array back as another array, you must make sure to either copy or steal the memory on the conversion in or copy the memory out.
If you don't the memory will be aliased by the two Numpy arrays and bad things will happen.

A second consideration is memory layout. Armadillo is optimised for column-major (Fortran order) memory whereas Numpy defaults to row-major (C order).
The default behaviour is to automatically convert, read copy, C-order arrays to F-order arrays upon conversion to Armadillo. Users should note that the library will not convert back to C-order when returning.

For details see the documentation section [Memory Management](https://carma.readthedocs.io/en/latest/memory_management.html).

### Example

On a high level CARMA provides two ways to work Numpy arrays in Armadillo:
Automatic conversion saves a bit on code but provides less flexibility with
regards to when to copy and when not.
Manual conversion should be used when you need fine grained control.

Combining the two; we use automatic conversion on the conversion in and manual when
creating the tuple for the way out.

```cpp

#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

py::tuple ols(arma::mat& X, arma::colvec& y) {
    // We borrow the data underlying the numpy arrays
    int n = X.n_rows, k = X.n_cols;

    arma::colvec coeffs = arma::solve(X, y);
    arma::colvec resid = y - X * coeffs;

    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n-k));
    arma::colvec std_errs = arma::sqrt(sig2 * arma::diagvec( arma::inv(arma::trans(X)*X)) );

    // We take ownership of the memory from the armadillo objects and
    // return to python as a tuple containing two Numpy arrays.
    return py::make_tuple(
        carma::col_to_arr(coeffs),
        carma::col_to_arr(std_errs)
    );
}

// adapted from https://gallery.rcpp.org/articles/fast-linear-model-with-armadillo/
```

Which can be called using:

```python
y = np.linspace(1, 100, num=100) + np.random.normal(0, 0.5, 100)
X = np.hstack((np.ones(100)[:, None], np.arange(1, 101)[:, None]))
coeff, std_err = carma.ols(X, y)
```

The repository contains tests, examples and CMake build instructions that can be used as an reference.

### About

This project was created by Ralph Urlus. Significant improvements to the project have been contributed by [Pascal H.](https://github.com/hpwxf)

### License

CARMA is provided under a Apache 2.0 license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.
