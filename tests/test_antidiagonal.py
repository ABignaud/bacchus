#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Module to test functions from antidiagonal module."""


import bacchus.antidiagonal as bca
import numpy as np


def test_compute_antidiagonal():
    M1 = np.array(
        [
            [1.0, 0.97, 0.94, 0.06, 0.58, 0.61],
            [0.25, 1.0, 0.29, 0.32, 0.81, 0.7],
            [0.11, 0.07, 1.0, 0.73, 0.93, 0.87],
            [0.7, 0.96, 0.76, 1.0, 0.46, 0.27],
            [0.71, 0.89, 0.51, 0.48, 1.0, 0.98],
            [0.86, 0.14, 0.46, 0.66, 0.24, 1.0],
        ]
    )
    values1 = bca.compute_antidiagonal(M1,)
    val1_true = np.array(
        [
            1.82905,
            1.49145,
            1.70085,
            1.00000,
            1.95299,
            1.99145,
            1.82905,
            1.49145,
            1.70085,
            1.00000,
            1.95299,
            1.99145,
        ]
    )
    np.testing.assert_allclose(values1, val1_true, rtol=1e-05)

    M2 = np.array(
        [
            [1.0, 0.47, 0.07, 0.07, 0.4, 0.74, 0.27],
            [0.47, 1.0, 0.97, 0.63, 0.93, 0.89, 0.07],
            [0.16, 0.85, 1.0, 0.72, 0.3, 0.44, 0.81],
            [0.48, 0.62, 0.9, 1.0, 0.83, 0.68, 0.67],
            [0.89, 0.17, 0.67, 0.1, 1.0, 0.28, 0.57],
            [0.88, 0.95, 0.67, 0.26, 0.96, 1.0, 0.81],
            [0.32, 0.99, 0.23, 0.68, 0.11, 0.24, 1.0],
        ]
    )
    values2 = bca.compute_antidiagonal(M2, full=False)
    val2_true = np.array([
        3.24818,
        1.48218, 
        1.68716, 
        2.86975,
        2.57016,
        2.55439,
        2.04982,
        1.46641,
        2.64900,
        1.95521,
        2.47556,
        1.65562,
        4.13118,
        0.93030,
    ])
    np.testing.assert_allclose(values2, val2_true, rtol=1e-05)


def test_antidiagonal_scalogram():
    ...
