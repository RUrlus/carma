"""Conversion examples python side."""
import numpy as np

import example_carma as carma

sample = np.asarray(
    np.random.random(size=(10, 2)),
    dtype=np.float64,
    order='F'
)

print(carma.manual_example(sample))
print(carma.automatic_example(sample))


sample2 = np.asarray(
    np.random.random(size=(10, 2)),
    dtype=np.float64,
    order='F'
)

example_class = carma.ExampleClass(sample, sample2)
arr = example_class.member_func()
print(arr)

print('\n\n OLS Example \n\n')
y = np.linspace(1, 100, num=100) + np.random.normal(0, 0.5, 100)
X = np.hstack((
    np.ones(100)[:, None],
    np.arange(1, 101)[:, None]
))
coeff, std_err = carma.ols(y, X)
print('coefficients: ', coeff.flatten())
print('std errors: ', std_err.flatten())
