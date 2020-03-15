"""Conversion examples python side."""
import numpy as np

import carma_examples as carma

sample = np.asarray(
    np.random.random(size=(10, 2)),
    dtype=np.float64,
    order='F'
)

carma.manual_example(sample)
carma.update_example(sample)
carma.automatic_example(sample)
