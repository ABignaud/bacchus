#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to compute borders and insulation score.

Functions:
    - get_relative_insulation
    - get_ri_score
    - get_local_score
    - detect_final_borders
    - get_insulation_score
"""


import bacchus.hic as bch
import copy
import numpy as np
import os
import scipy.signal as ss
import scipy.sparse as sp
from typing import List, Optional, Tuple


def get_relative_insulation(M: "numpy.ndarray", window: int, s: int) -> float:
    """Function to compute the relative insulation score depending on the window
    size and of one position. The method is based on the method described in
    https://doi.org/10.1093/nar/gky789 and implemented in R:
    https://github.com/ChenFengling/RHiCDB.

    Parameters
    ----------
    M : numpy.ndarray
        Extended dense matrix with size w.
    window : int
        Size of the half window (Look for contacts at a genomic range of window)
        used to compute the insulation score.
    s : int
        Genomic coordiates where to compute the relative insulation score.

    Returns
    -------
    float:
        Relative insulation score at the genomic position between s and s + 1
        with the range window.
    """
    # Extract the matrix of interest. No problematic edge case are possible are
    # the matrix have been extended.
    N = M[s : s + 2 * window, s : s + 2 * window]
    # Compute the mean for each part of the matrix
    up = np.nanmean(np.triu(N[:window, :window]))
    down = np.nanmean(np.triu(N[window:, window:]))
    between = np.nanmean(N[:window, window:])
    # Compute the relative index
    return (up + down - between) / (up + down + between)


def get_ri_score(M: "numpy.ndarray", list_w: List[int]) -> "numpy.ndarray":
    """Function to get the relative insulation score on the whole matrix. It
    supposed the matrix is circular.

    Parameters
    ----------
    M : numpy.ndarray
        Matrix dense which should be circular to manage edge case.
    list_w : list of int
        List of half window size (Look for contacts at a genomic range of size
        window) to use in bin scale.

    Returns
    -------
    numpy.ndarray
        Vector of the relative insulation score for each bin of the matrix.
    """
    # Compute size of M and prepare the vector.
    n = len(M)
    ri_score = np.zeros(n)
    max_w = np.max(list_w)
    len_w = len(list_w)

    # Extend the matrix to avoid edge case using circular matrix property.
    M = bch.map_extend(M, max_w)
    # Compute the relative index score.
    for s in range(n):
        for w in list_w:
            # Correct s to correspond at the current w (correction for extended
            # matrix coordinates)
            pos = s + max_w - w
            # Compute the average relative insulation score
            ri_score[s] += get_relative_insulation(M, w, pos) / len_w
    return ri_score


def get_local_score(
    ri_score: "numpy.ndarray",
) -> Tuple["numpy.ndarray", List["numpy.ndarray"]]:
    """As the relative insulation score depends on the matrix and insulation
    score of the closed regions. It's necessary to defined it locally. This
    function allows to do it by removing the second enveloppe to the signal. The
    method is based on the method described in
    https://doi.org/10.1093/nar/gky789 and implemented in R:
    https://github.com/ChenFengling/RHiCDB.

    Parameters
    ----------
    ri_score : numpy.ndarray
        Vector of the relative insulation score along the whole genome.

    Returns
    -------
    numpy.ndarray:
        Vector of the local relative insulation score along the whole genome.
    List of numpy.ndarray:
        List of the vector of the first enveloppe, its index, the second
        enveloppe and its index.
    """

    # Compute the first and second envelope to compute the local LRI
    # First enveloppe
    first_enveloppe_index = ss.find_peaks([-x for x in ri_score])[0]
    first_enveloppe_index = np.concatenate(
        ([0], first_enveloppe_index, [len(ri_score) - 1])
    )
    first_enveloppe = [ri_score[int(i)] for i in first_enveloppe_index]
    # Possible as the genome is circular
    a = np.mean([first_enveloppe[1], first_enveloppe[-2]])
    first_enveloppe[0] = a
    first_enveloppe[-1] = a

    # Second enveloppe
    second_enveloppe_index = ss.find_peaks([-x for x in first_enveloppe])[0]
    second_enveloppe_index = [
        first_enveloppe_index[i] for i in second_enveloppe_index
    ]
    second_enveloppe_index = np.concatenate(
        ([0], second_enveloppe_index, [len(ri_score) - 1])
    )
    second_enveloppe = [ri_score[int(i)] for i in second_enveloppe_index]

    # Possible as the genome is circular
    a = np.mean([second_enveloppe[1], second_enveloppe[-2]])
    second_enveloppe[0] = a
    second_enveloppe[-1] = a

    # Remove second enveloppe to the relative insulation score to have a local
    # relative insulation score.
    i = 0
    lri_score = []
    for j, k in enumerate(second_enveloppe_index):
        while i < k:
            a = (second_enveloppe[j] - second_enveloppe[j - 1]) / (
                k - second_enveloppe_index[j - 1]
            )
            b = second_enveloppe[j] - a * k
            lri_score.append(ri_score[i] - (a * i + b))
            i += 1
        if i == k:
            lri_score.append(ri_score[i] - second_enveloppe[j])
            i += 1
    return (
        lri_score,
        [
            first_enveloppe,
            first_enveloppe_index,
            second_enveloppe,
            second_enveloppe_index,
        ],
    )


def detect_final_borders(
    lri_score: "numpy.ndarray", min_dist: int = 10
) -> List[int]:
    """Function to detect the borders based on the peaks on local relative
    insulation score.

    Parameters
    ----------
    lri_score : numpy.ndarray
        Vector of the local relative insulation score.
    min_dist : int
        Size in bin which have to separate two different borders. [Default: 10]

    Returns
    -------
    list of int:
        Position in bin coordinates of the final borders detected by the local
        relative insulation score.
    """
    # Detect the peaks.
    peaks = ss.find_peaks(lri_score)[0]

    # Define a cutoff as the median plus one time the standard deviation.
    cutoff = np.nanmedian(lri_score) + np.nanstd(lri_score)

    # Append peaks in final borders if there are bigger than the cutoff and far
    # enough from each other.
    final_borders = []
    previous_border = -np.inf
    n = len(lri_score)
    for i in peaks:
        # Check if the peak is relevant.
        k1 = i - 2 if i - 2 >= 0 else n - i - 2
        k2 = i + 2 if i + 2 < n else i + 2 - n
        if (
            lri_score[i] >= cutoff
            and lri_score[i] > lri_score[k1]
            and lri_score[i] > lri_score[k2]
        ):
            # Check if the peak is not part of a bigger peak. Keep only the
            # biggest peak if there are two closed peaks.
            if i - previous_border < min_dist:
                if lri_score[i] > lri_score[previous_border]:
                    final_borders[-1] = i
            else:
                final_borders.append(i)
                previous_border = i

    return final_borders, peaks


def get_insulation_score(
    M: "numpy.ndarray", list_w: List[int]
) -> Tuple["numpy.ndarray"]:
    """Main function to return the peak border position from a HiC matrix using
    the insulation score. The insulation score is computed on the correlation
    matrix. The method is based on the method described in
    https://doi.org/10.1093/nar/gky789 and implemented in R:
    https://github.com/ChenFengling/RHiCDB.

    Parameters
    ----------
    M : numpy.ndarray
        Extended dense matrix with size w.
    list_w : List of int
        List of half window size (Look for contacts at a genomic range of size
        window) to use in bin scale.

    Returns
    -------
    numpy.ndarray:
        Final borders detected.
    numpy.ndarray:
        Local relative insulation score list a the genomic position.
    """
    # Copy the matrix
    matrix = copy.copy(M)

    # Detect the white lines on the matrix.
    mask = bch.mask_white_line(matrix)

    # Change white lines in values close to 0 fro the correlation.
    matrix[mask, :] = np.random.random_sample(matrix.shape[0]) / 10**6
    matrix[:, mask] = np.random.random_sample(matrix.shape[0])[0] / 10**6
    matrix = np.corrcoef(matrix)

    # Transform them in nan to do not take them into account later.
    matrix[mask, :] = np.nan
    matrix[:, mask] = np.nan

    # Get the realtive insulation curev on the correlation matrix.
    ri = get_ri_score(matrix, list_w)
    lri, _ = get_local_score(ri)

    # Compute the final borders.
    final_borders, _ = detect_final_borders(lri, list_w[1])
    return final_borders, lri
