#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to normalize an HiC sparse matrices.

Functions:
    - 
"""


import numpy as np
import scipy as sp
from typing import Optional


def _zero_diags(
    data: "scipy.sparse.csr_matrix", n_diags: int = 2
) -> "scipy.sparse.csr_matrix":
    """
    Functions to add zeros on the main diagonals to avoid issue during
    normalization.

    Parameters
    ----------
    data : scipy.sparse.csr_matrix
        Input matrix.
    n_diags : int
        Number of main diagonals to remove.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix with O in the n main diagonals.

    Examples
    --------
        >>> import scipy as sp
        >>> data = sp.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> r = _zero_diags(data, n_diags=2)
        >>> r.eliminate_zeros()
        >>> r2 = np.array([[0, 0, 3], [0, 0, 0], [7, 0, 0]])
        >>> (r == r2).all()
        True
    """
    r = sp.sparse.csr_matrix(data)
    row_indices, col_indices = r.nonzero()
    mask = np.abs(row_indices - col_indices) < n_diags
    r.data[mask] = 0
    return r


def mad(M: "scipy.sparse.coo_matrix", axis: Optional[int] = None) -> float:
    """
    Computes median absolute deviation of matrix bins sums.

    Parameters
    ----------
    M : scipy.sparse.coo_matrix
        Sparse matrix in COO format.

    axis: int
        Compute MAD on rows if 0, on columns if 1 or on all pixels if None. If
        axis is None, MAD is computed only on nonzero pixels.

    Returns
    -------
    float:
        MAD estimator of matrix bin sums
    """
    # Compute median on nonzero data values
    # otherwise, median is 0 if sufficiently sparse
    if axis is None:
        if sp.sparse.issparse(M):
            r = M.tocoo()
            dist = r.data
        else:
            dist = M

    else:
        if axis < 0:
            axis += 2
        dist = np.array(M.sum(axis=axis, dtype=float)).flatten()

    return np.median(np.absolute(dist - np.median(dist)))


def _sum_mat_bins(mat: "scipy.sparse.csr_matrix") -> "numpy.array":
    """
    Compute the sum of matrices bins (i.e. rows or columns) using only the upper
    triangle, assuming symmetrical matrices.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        Contact map in sparse format, either in upper triangle or full matrix.

    Returns
    -------
    numpy.array :
        1D array of bin sums.
    """
    # Equivalaent to row or col sum on a full matrix
    # Note: mat.sum returns a 'matrix' object. A1 extracts the 1D flat array
    # from the matrix
    return mat.sum(axis=0).A1 + mat.sum(axis=1).A1 - mat.diagonal(0)


def _get_good_bins(
    M: "scipy.sparse.coo_matrix",
    n_mad: float = 3.0,
    s_min: Optional[float] = None,
    s_max: Optional[float] = None,
    symmetric: bool = False,
) -> "numpy.array ":
    """
    Filters out bins with outstanding sums using median and MAD of the log
    transformed distribution of bin sums. Only filters weak outlier bins unless
    `symmetric` is set to True.

    Parameters
    ----------
    M : scipy.sparse.coo_matrix
        Input sparse matrix representing the Hi-C contact map.
    n_mad : float
        Minimum number of median absolut deviations around median in the bin
        sums distribution at which bins will be filtered out.
    s_min : float
        Optional fixed threshold value for bin sum below which bins should be
        filtered out.
    s_max: float
        Optional fixed threshold value for bin sum above which bins should be
        filtered out.
    symmetric : bool
        If set to true, filters out outliers on both sides of the distribution.
        Otherwise, only filters out bins on the left side (weak bins).

    Returns
    -------
    numpy.array of bool :
        A 1D numpy array whose length is the number of bins in the matrix and
        values indicate if bins values are within the acceptable range (1) or
        considered outliers (0).
    """
    r = M.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        bins = _sum_mat_bins(r)
        bins[bins == 0] = 1
        norm = np.log10(bins)
        median = np.median(norm)
        sigma = 1.4826 * mad(norm)

    if s_min is None:
        s_min = median - n_mad * sigma
    if s_max is None:
        s_max = median + n_mad * sigma

    if symmetric:
        filter_bins = (norm > s_min) * (norm < s_max)
    else:
        filter_bins = norm > s_min

    return filter_bins


def normalize_sparse(
    M: "scipy.sparse.csr_matrix",
    iterations: int = 200,
    n_mad: float = 3.0,
    tol: float = 1e-05,
    ignore_diags: int = 2,
) -> "scipy.sparse.csr_matrix":
    """
    Applies a normalization type to a sparse matrix. Adapated from
    hicstuff.hicstuff

    Parameters
    ----------
    M : scipy.sparse.csr_matrix of floats
        Input matrix to normalize.
    iterations : int
        Iterations parameter when using an iterative normalization
        procedure. [Default: 200]
    n_mad : float
        Maximum number of median absolute deviations of bin sums to allow for
        including bins in the normalization procedure. Bins more than `n_mad`
        mads below the median are excluded. Bins excluded from normalisation
        are set to 0. [Default: 3.]
    tol : float
        Convergence criterion is the variance of the marginal (row/col) sum
        vector. [Default: 1e-5]
    ignore_diags : int
        Number of main diagonals to ignore for normalization. [Default: 2]

    Returns
    -------
    scipy.sparse.csr_matrix of floats :
        Normalized sparse matrix.
    """
    # Making full symmetric matrix if not symmetric already (e.g. upper triangle)
    r = M.astype(np.float64)
    # Removing diagonal
    if ignore_diags > 0:
        r = _zero_diags(r, n_diags=ignore_diags)

    good_bins = _get_good_bins(r, n_mad=n_mad)
    # Set values in non detectable bins to 0
    # For faster masking of bins, mask bins using dot product with an identity
    # matrix where bad bins have been masked on the diagonal
    # E.g. if removing the second bin (row and column):
    # 1 0 0     9 6 5     1 0 0     9 0 5
    # 0 0 0  X  6 8 7  X  0 0 0  =  0 0 0
    # 0 0 1     6 7 8     0 0 1     6 0 8
    mask_mat = sp.sparse.eye(r.shape[0])
    mask_mat.data[0][~good_bins] = 0
    r = mask_mat.dot(r).dot(mask_mat)
    r = sp.sparse.coo_matrix(r)
    r.eliminate_zeros()

    # Row and col indices of each nonzero value in matrix
    bias = np.zeros(np.shape(r)[0])
    bias[good_bins] = 1.0
    row_indices, col_indices = r.nonzero()
    for k in range(iterations):
        # Symmetric matrix: rows and cols have identical sums
        bin_sums_it = _sum_mat_bins(r)
        # Normalize bin sums by the median sum of detectable bins for stability
        bin_sums = bin_sums_it / np.median(bin_sums_it[good_bins])
        bin_sums[bin_sums == 0] = 1
        # Divide each nonzero value by the product of the sums of
        # their respective rows and columns.
        bias[good_bins] /= bin_sums[good_bins]
        r.data = r.data / (bin_sums[row_indices] * bin_sums[col_indices])
        var = bin_sums_it[good_bins].var()
        print(f"The variance is {var}")
        if var < tol:
            break
    # Scale to 1 (The factor 1/2 is because we have row and columns)
    bias /= np.sqrt(1 / 2 * np.median(_sum_mat_bins(r)[good_bins]))
    return bias
