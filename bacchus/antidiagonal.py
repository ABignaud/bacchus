#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""General functions for antidiagonal analysis.

Functions:
    - compute_antidiagonal
    - compute_antidiagonal_scalogram
    - compute_partial_antidiagonal
"""


import numpy as np
from typing import List


def compute_antidiagonal(
    M: "numpy.ndarray",
    full: bool = True,
    ignore_diags: int = 2,
) -> "numpy.ndarray":
    """Function to compute all the antidiagonal strength centered on the origin
    to be able to evaluate the strength of the antidiagonal passing through the
    origin of replication.

    Parameters
    ----------
    M : numpy.ndarray
        Dense matrix.
    full : bool
        Either to compute the antidiagonals strength on their full length or
        only on half the antidiagonal. The first one will give you a "symetric"
        profile as all antidiagonals pass through two genomic coordinates, the
        second one will not be symmetric.
    ignore_diags : int
        Number of main diagonals to ignore.

    Returns
    -------
    numpy.ndarray:
        Vector of the strength of all the antidiagonals.
    """

    # Compute the size of the matrix and rotate it to have the antidiagonal as
    # the primary diagonal.
    n = np.shape(M)[0]
    for i in range(n):
        for j in range(i - ignore_diags + 1, i + ignore_diags):
            if (j >= 0) and (j < n):
                M[i, j] = 0
    N = np.rot90(M)

    # If full compute all the antidiagonals strength and duplicate it for the
    # symetric, if not will compute the half partial antidiagonals strength.
    if full:
        # Add all the diagonal mean values for one bacteria in a vector. As the
        # genome is circular we can considered any antidiagonal by adding two
        # antidiagonals a poistion n-i and i.
        values = np.zeros((n))
        for i in range(-n + 1, 0, 1):
            values[n + i - 1] = np.nanmean(
                np.concatenate((np.diag(N, k=i), np.diag(N, k=n + i)))
            )
        values[n - 1] = np.nanmean(np.diag(N, k=0))

        # Concatenate the vector too cover all the genomic positions as we take
        # the whole antidiagonals, each of them are equivalent to the opposite
        # point on the genome.
        values = np.concatenate((values, values))

    else:
        values = compute_partial_antidiagonal(N, n // 4)

    # Normalize by the first percentile (not the minimum as there may is an
    # outsider) to have a basal value of 1 far away from the antidaiagonal.
    values = values / np.nanpercentile(values, 5)

    return values


def compute_antidiagonal_scalogram(
    M: "numpy.ndarray",
    binning: int,
    windows: List[int],
    ignore_diags: int = 2,
) -> "numpy.ndarray":
    """Function to compute all the antidiagonal strength centered on the origin
    to be able to evaluate the strength of the antidiagonal passing through the
    origin of replication. It will automatically compute the half antidaigonal
    to be able to compare with it at a last point.

    Parameters
    ----------
    M : numpy.ndarrayormat "région", a
        Origin of replication position in base pair.
    binning : int
        Binning size in base pair.
    windows : List of int
        List of the windows size to takes in base pairs.
    ignore_diags : int
        Number of main diagonals to ignore.

    Returns
    -------
    numpy.ndarray:
        Vector of the strength of all the antidiagonals.
    """

    # Compute the size of the matrix and rotate it to have the antidiagonal as
    # the primary diagonal.
    n = np.shape(M)[0]
    for i in range(n):
        for j in range(i - ignore_diags + 1, i + ignore_diags):
            if (j >= 0) and (j < n):
                M[i, j] = 0
    N = np.rot90(M)

    # Transform windows and origin position in bin coordinates.
    w_binned = [round(s / (2 * binning)) for s in windows]
    w_binned.append(n // 4)

    # Add all the diagonal mean values for one bacteria in a vector. As the
    # genome is circular we can considered any antidiagonal by adding two
    # antidiagonals a poistion n-i and i.
    values = np.zeros((len(w_binned), 2 * n))

    # Iterates on the different length.
    for i, s in enumerate(w_binned):
        values[i] = compute_partial_antidiagonal(N, s)

    for i in range(len(w_binned)):
        # Normalize by the first percentile (not the minimum as there may is an
        # outsider) to have a basal value of 1 far away from the antidaiagonal.
        values[i] = values[i] / np.nanpercentile(values[-1], 5)

    return values


def compute_partial_antidiagonal(N: "numpy.ndarray", s: int) -> "numpy.ndarray":
    """Function to compute from the rotated matrix the strength of a partial
    antidiagonal (all values takes between the main diagonal and the given
    genomic distance s).

    Parameters
    ----------
    N : numpy.ndarray
        Rotated matrice (pi/2 rotation in the trigonometric sens).
    s : int
        Genomic distance away from the main diagonal to consider in matrice
        bin scale to compute the strength of the partial antidiagonal.

    Returns
    -------
    numpy.ndarray
        Vector of the strength of the partial antidaigonal at the given
        genomic distance s.
    """

    # Defines length of the vector.
    n = np.shape(N)[0]
    values = np.zeros((2 * n))

    # Compute main antidiagonal
    diag0 = np.diag(N, k=0)
    # Iterates on left half of the antidiagonals (negative coordinates).
    for i in range(-n + 1, 0, 1):
        diag1 = np.diag(N, k=i)
        diag2 = np.diag(N, k=n + i)
        l1 = len(diag1)
        l2 = len(diag2)
        if l1 >= 2 * s:
            values[n + i - 1] = np.nanmean(diag1[l1 // 2 - s : l1 // 2 + s])
        else:
            values[n + i - 1] = np.nanmean(
                np.concatenate(
                    (diag1, diag2[: s - l1 // 2], diag2[l2 - s + l1 // 2 :])
                )
            )
    values[n - 1] = np.nanmean(diag0[n // 2 - s : n // 2 + s])

    # Iterates on right half of the antidiagonals (positive coordinates).
    for i in range(1, n, 1):
        diag1 = np.diag(N, k=i)
        diag2 = np.diag(N, k=-n + i)
        l1 = len(diag1)
        l2 = len(diag2)
        if l1 >= 2 * s:
            values[n - 1 + i] = np.nanmean(diag1[l1 // 2 - s : l1 // 2 + s])
        else:
            values[n - 1 + i] = np.nanmean(
                np.concatenate(
                    (diag1, diag2[: s - l1 // 2], diag2[l2 - s + l1 // 2 :])
                )
            )
    values[2 * n - 1] = np.nanmean(np.concatenate((diag0[:s], diag0[n - s :])))
    return values
