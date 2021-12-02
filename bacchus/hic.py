#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to analyze HiC experiments data and extract information
from the HiC contact map.

Functions:
    - get_win_density
    - is_sym
    - sym
"""


import chromosight.utils.detection as cud
import chromosight.utils.preprocessing as cup
import scipy.sparse as sp


def get_win_density(
    mat: sp.csr_matrix, win_size: int = 3, sym_upper: bool = False
) -> sp.csr_matrix:
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


## TODO
# def compute_hicreppy(matrices: List[numpy.ndarray], fragment):


def is_sym(M: scipy.sparse.csr_matrix) -> bool:
    """Test if a matrix is symmetric, i.e. is the transposed matrix is the same.
    
    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        matrix to symetrize.

    Returns
    -------
    bool:
        Either the matrix is symetric or not
    """
    return np.all(M == M.T)


def sym(M: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
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
    if not is_sym(M):
        M = M + M.T
        # Divide the diagonal by 2 as we add it twice.
        M.setdiag(M.diagonal() / 2)
        # Remove zeros from teh sparse matrix
        M.eliminate_zeros()
    return M
