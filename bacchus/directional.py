#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to compute directional index on an HiC contact map. Methods
from Lioy et al. 2018, Cell (https://doi.org/10.1016/j.cell.2017.12.027).

Functions:
    - directional_index
    - di_borders
"""


import bacchus.hic as bch
import numpy as np
import scipy.stats as st
from typing import List, Optional


def directional_index(
    M: "scipy.sparse.csr_matrix",
    window_size: int,
    corr: bool = True,
    normalize: bool = False,
    plot_dir: Optional[str] = None,
) -> List[float]:
    """Function to compute the directional index from a sparse matrix. The
    directional index is defined for one bin as the t-value between the left and
    right vectors of contacts of the bin until a given range.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        The input, normalized contact map. Must be a single chromosome. Values
        are assumed to be only the upper triangle of a symmetric matrix.
    window_size : int
        Size of the window to consider to compute the directional index.
    corr : bool
        Whether to compute the correlation matrix or not. If False you should
        have given the correlation matrix as input.
    normalize : bool
        If enables normalize the matrix first. [Default: False].
    plot_dir : directory
        Directory to save plot if one givens.

    Returns
    -------
    numpy.ndarray:
        The directional index for each position of the bins.
    """
    # Compute the correlation coefficient matrix.
    if corr:
        M = bch.corr_matrix_sparse(
            M,
            detrend=False,
            normalize=normalize,
            plot_dir=plot_dir,
        )

    # Compute thse size of the matrix.
    n = np.shape(M)[0]
    di = np.zeros((n))
    vec_left = np.zeros((window_size))
    vec_right = np.zeros((window_size))

    # Extend the matrix base on circularity to avoid borders issue
    M = bch.map_extend(M.todense(), window_size)

    # Iterates on the matrix to build the diretcional index for each bin.
    for i in range(n):
        j = i + window_size
        # Iterates on the window size to build the two vectors considers to
        # compute the directional index.
        for k in range(window_size):
            # Build vectors put the log at 0 for non positive values
            if M[j, j + k] > 0:
                vec_right[k] = np.log(M[j, j + k])
            else:
                vec_right[k] = 0
            if M[j, j - k] > 0:
                vec_left[k] = np.log(M[j, j - k])
            else:
                vec_left[k] = 0
        # Compute the directional index, i.e. the t-value between both vectors.
        if np.sum(vec_left) != 0 and np.sum(vec_right) != 0:
            di[i] = st.ttest_rel(vec_right, vec_left)[0]
        else:
            di[i] = 0
    return di


def di_borders(di: List[float], threshold=1.96) -> List[float]:
    """Function to return the list of borders based on the directional index
    vector. Threshold chosen at 1.96 to have a p-value below 0.05. As the
    directional index is a t-value, a value below -1.96 or above 1.96 will have
    a p-value below 0.05. Like that we took only significative changes.

    Parameters
    ----------
    di : numpy.ndarray
        List of the directional index computed for each bin.
    threshold : int
        t-value threshold to consider a border.

    Returns
    -------
    list of int:
        Positions in bins of the detected borders.

    Example
    -------
        >>> di = [0.5, 2., 3., 4., 0.1, -3.2, -3.5, 0.]
        >>> print(di_borders(di))
        []
        >>> di = [0.5, 2., 3., 4., 0.1, -3.2, 4., -2.]
        >>> print(di_borders(di))
        [1, 6]
    """
    # Initiation use last value as previous one as the genome is considered as
    # circular.
    borders = []
    if di[-1] < -threshold:
        negative = True
    else:
        negative = False

    # Iterates on the DI values
    for i, curr_di in enumerate(di):
        if curr_di >= threshold and negative:
            borders.append(i)
            negative = False
        if curr_di <= -threshold:
            negative = True
    return borders
