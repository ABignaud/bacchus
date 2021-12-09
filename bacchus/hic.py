#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to analyze HiC experiments data and extract information
from the HiC contact map.

Functions:
    - compute_hic_signal
    - get_win_density
    - is_symmetric
    - map_extend
    - mask_white_line
    - get_symmetric
"""


import chromosight.utils.detection as cud
import chromosight.utils.preprocessing as cup
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from typing import Optional


def compute_hic_signal(
    M: "numpy.ndarray",
    binning: Optional[int] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
) -> "numpy.ndarray":
    """Compute the Hic signal of a dense matrix.

    Parameters
    ----------
    M : numpy.ndarray
        Matrix to compute the HiC signal. The matrix has to be circular.
    binning : int
        Binning size of the matrix in base pair.
    start : int
        Position in base pair to start consider the signal (genomic distance).
    stop : int
        Position in base pair to stop consider the signal (genomic distance).

    Returns
    -------
    numpy.ndarray:
        Vector of HiC signal values.

    TODO: Transform it to use map extend instead.
    """
    # Compute the size of the matrix and change positions in bin coordonates.
    n = len(M)
    values = np.zeros((n))

    # Define start and stop if none given it will use the second diagonal.
    if binning is None:
        start = 1
        stop = 2
    else:
        start = start // binning
        stop = stop // binning

    # Compute the HiC signal and correct values depending on the circular
    # signal.
    for i in range(n):
        if i < stop:
            if i < start:
                values[i] = np.nansum(
                    M[n + i - stop : n + i - start, i + start : i + stop]
                )
            else:
                values[i] = np.nansum(
                    np.concatenate(
                        (
                            M[n + i - stop :, i + start : i + stop],
                            M[: i - start, i + start : i + stop],
                        ),
                        axis=0,
                    )
                )
        elif n - i < stop:
            if n - i < start:
                values[i] = np.nansum(
                    M[i - stop : i - start, start - n + i : stop - n + i]
                )
            else:
                values[i] = np.nansum(
                    np.concatenate(
                        (
                            M[i - stop : i - start, i + start :],
                            M[i - stop : i - start, : stop - n + i],
                        ),
                        axis=1,
                    )
                )
        else:
            values[i] = np.nansum(M[i - stop : i - start, i + start : i + stop])
    return values


def get_win_density(
    mat: "scipy.sparse.csr_matrix", win_size: int = 3, sym_upper: bool = False
) -> "scipy.sparse.csr_matrix":
    """Compute local pixel density in sparse matrices using convolution. The
    convolution is performed in 'full' mode: Computations are performed all the
    way to the edges by trimming the kernel.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        The input sparse matrix to convolve.
    win_size : int
        The size of the area in which to compute the proportion of nonzero
        pixels. This will be the kernel size used in convolution.
    sym_upper : bool
        Whether the input matrix is symmetric upper, in which case only the
        upper triangle is returned.

    Returns
    -------
    scipy.sparse.csr_matrix:
        The result of the convolution of the uniform kernel (win_size x
        win_size) on the binarized input matrix. Each value represents the
        proportion of nonzero pixels in the neighbourhood.
    """
    ms, ns = mat.shape
    # Generate a binary matrix (pixels are either empty or full)
    bin_mat = mat.copy()
    if sym_upper:
        bin_mat = sp.triu(bin_mat)
    bin_mat.data = bin_mat.data.astype(bool)
    # Adding a frame of zeros around the signal
    tmp = sp.csr_matrix((win_size - 1, ns), dtype=bool)
    bin_mat = sp.vstack([tmp, bin_mat, tmp], format=mat.format)
    tmp = sp.csr_matrix((ms + 2 * (win_size - 1), win_size - 1), dtype=bool)
    bin_mat = sp.hstack([tmp, bin_mat, tmp], format=mat.format)
    # Convolve the uniform kernel with this matrix to get the proportion of
    # nonzero pixels in each neighbourhood
    kernel = np.ones((win_size, win_size))
    win_area = win_size ** 2
    density = cud.xcorr2(bin_mat, kernel / win_area)

    # Compute convolution of uniform kernel with a frame of ones to get number
    # of missing pixels in each window.
    frame = cup.frame_missing_mask(
        sp.csr_matrix(mat.shape, dtype=bool), kernel.shape, sym_upper=sym_upper
    )
    frame = cud.xcorr2(frame, kernel).tocoo()
    # From now on, frame.data contains the number of 'present' samples. (where
    # there is at least one missing pixel)
    frame.data = win_area - frame.data

    # Adjust the proportion for values close to the border (lower denominator)
    density[frame.row, frame.col] = (
        density[frame.row, frame.col].A1 * win_area / frame.data
    )
    # Trim the frame out from the signal
    density = density[
        win_size - 1 : -win_size + 1, win_size - 1 : -win_size + 1
    ]
    if sym_upper:
        density = sp.triu(density)
    return density


def is_symmetric(M: "scipy.sparse.csr_matrix") -> bool:
    """Test if a matrix is symmetric, i.e. is the transposed matrix is the same.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        matrix to test if symetric.

    Returns
    -------
    bool:
        Either the matrix is symetric or not
    """
    return (abs(M - M.T) > 1e-10).nnz == 0


def map_extend(M: "numpy.ndarray", s: int) -> "numpy.ndarray":
    """Function to extend a circular matrix at all the edges to easily managed
    edge cases when we do computation using.

    Parameters
    ----------
    M : numpy.ndarray
        Matrix to extend.
    s : int
        Numbers of bin to extend.

    Returns
    -------
    numpy.ndarray
        Extended matrix.
    """
    # Comput the size of the initial matrix.
    n = len(M)
    # Concatenate the borders as the matrix to extend it playing on the circular
    # relations on both axis.
    M = np.concatenate(
        (
            M[
                n - s :,
            ],
            M,
            M[
                :s,
            ],
        ),
        axis=0,
    )
    M = np.concatenate((M[:, n - s :], M, M[:, :s]), axis=1)
    return M


def mask_white_line(
    matrix: "numpy.ndarray", n_mads: int = 3
) -> "numpy.ndarray":
    """Function to put nan in the row/column where there are too much zeros to
    mask them in further analysis.

    Parameters
    ----------
    matrix : numpy.ndarray
        Dense matrix.
    n_mads : int
        Number of median absolute deviation used as threshold.

    Returns
    -------
    numpy.ndarray
        List of positions the bins with poor coverage (white lines).
    """

    def mad(x):
        return st.median_absolute_deviation(x, nan_policy="omit")

    # Compute number of nonzero values in each bin
    sum_bins = (matrix == 0).sum(axis=0)
    # Compute variation in the number of nonzero pixels
    sum_mad = mad(sum_bins)
    # Find poor interacting rows and columns
    sum_med = np.median(sum_bins)
    detect_threshold = max(1, sum_med + sum_mad * n_mads)
    # Removal of poor interacting rows and columns
    bad_bins = np.flatnonzero(sum_bins >= detect_threshold)

    return bad_bins


def get_symmetric(M: "scipy.sparse.csr_matrix") -> "scipy.sparse.csr_matrix":
    """Function to set the symmetric of a triangular matrix. Do nothing if the
    matrix is already triangular.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        matrix to symetrize.

    Returns
    -------
    scipy.sparse.csr_matrix:
        matrix symetrized
    """
    # Test whether it's already symetric or not.
    if not is_symmetric(M):
        M = M + M.T
        # Divide the diagonal by 2 as we add it twice.
        M.setdiag(M.diagonal() / 2)
        # Remove zeros from teh sparse matrix
        M.eliminate_zeros()
    return M
