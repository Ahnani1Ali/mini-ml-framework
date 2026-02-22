from .mnist_loader import load_mnist, read_idx
from .synthetic import (make_spirals, make_gaussians, make_moons,
                        make_regression, make_mnist_like,
                        train_test_split, standardize)

__all__ = [
    'load_mnist', 'read_idx',
    'make_spirals', 'make_gaussians', 'make_moons',
    'make_regression', 'make_mnist_like',
    'train_test_split', 'standardize',
]
