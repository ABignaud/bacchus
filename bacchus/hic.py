#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to analyze HiC experiments data and extract information
from the HiC contact map.

Functions:
    - compartments_sparse
    - compute_hic_signal
    - corr_matrix_sparse
    - detrend_matrix_sparse
    - fourc_like
    - get_hicreppy
    - get_symmetric
    - get_win_density
    - interpolate_white_lines
    - is_symmetric
    - map_extend
    - mask_white_line
"""


import bacchus.plot as bcp
import chromosight.utils.detection as cud
import chromosight.utils.preprocessing as cup
import cooler
import copy
import hicreppy.hicrep as hicrep
import hicstuff.commands as hcc
import hicstuff.hicstuff as hcs
import math
import numpy as np
import os
import scipy as sp
import scipy.stats as st
from typing import List, Optional, Tuple


def compartments_sparse(
    M: "scipy.sparse.csr_matrix",
    normalize: bool = True,
    plot_dir: Optional[str] = None,
    circular: bool = True,
    antidiagonal: bool = False,
) -> Tuple["numpy.ndarray"]:
    """A/B compartment analysis

    Performs a detrending of the power law followed by a PCA-based A/B
    compartment analysis on a sparse, normalized, single chromosome contact map.
    The results are two vectors whose values (negative or positive) should
    presumably correlate with the presence of 'active' vs. 'inert' chromatin.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        The input, normalized contact map. Must be a single chromosome. Values
        are assumed to be only the upper triangle of a symmetric matrix.
    normalize : bool
        Whether to normalize the matrix beforehand.
    plot_dir : directory
        Directory to save plot if one given.
    circular : bool
        Either the matrix is circualr or not.
    antidiagonal : bool
        Either there is an antidiagonal on the matrix or not. This is still in
        development.

    Returns
    -------
    numpy.ndarray:
        An array containing the first principal component.
    numpy.ndarray:
        An array containing the second principal component.

    TODO: Adapt detrending to circular matrix ? Is it really necessary ?
    TODO: Detrending antidiagonal, make sure to center on the antidiagonal.
    """
    # Detrend and compute correlation matrix on full matrix
    M = corr_matrix_sparse(
        M,
        detrend=True,
        normalize=normalize,
        antidiagonal=antidiagonal,
        plot_dir=plot_dir,
    )

    # Extract eigen vectors and eigen values
    [eigen_vals, pr_comp] = sp.linalg.eig(M)

    return pr_comp[:, 0], pr_comp[:, 1]


def compute_hic_signal(
    M: "numpy.ndarray",
    binning: Optional[int] = None,
    start: int = 1,
    stop: int = 2,
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


def corr_matrix_sparse(
    M: "scipy.sparse.csr_matrix",
    detrend: bool = False,
    normalize: bool = False,
    antidiagonal: bool = False,
    plot_dir: Optional[str] = None,
) -> "scipy.sparse.csr_matrix":
    """Function to compute the correlation matrix from a sparse matrix.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        The input, normalized contact map. Must be a single chromosome. Values
        are assumed to be only the upper triangle of a symmetric matrix.
    normalize : bool
        If enables normalize the matrix first. [Default: False].
    antidiagonal : bool
        Either there is an antidiagonal on the matrix or not. This is still in
        development. If enables, it will also detrend the antidiagonal.
    plot_dir : directory
        Directory to save plot if one given.s

    Returns
    -------
    scipy.sparse.csr_matrix:
        The correlation matrix from the detrended matrix.
    """
    # Make matrix symetric (in case of upper triangle)
    M = get_symmetric(M)

    # Remove nan values.
    np.nan_to_num(M.data, copy=False)

    # Normalize the matrix
    if normalize:
        M = hcs.normalize_sparse(M, norm="SCN")

    # Detrend the matrix.
    if detrend:
        M = detrend_matrix_sparse(
            M, normalize=False, antidiagonal=antidiagonal, plot_dir=plot_dir
        )

    # Compute the corelation coeficient matrix
    M = hcs.corrcoef_sparse(M)
    M[np.isnan(M)] = 0.0

    # Plot correlation matrix.
    if plot_dir is not None:
        correlation_map_file = os.path.join(plot_dir, "correlation_map.png")
        bcp.contact_map_ratio(
            M,
            dpi=200,
            cmap="seismic",
            lim=1,
            out_file=correlation_map_file,
            ratio=True,
            title="Correlation contact map",
        )
    return M


def detrend_matrix_sparse(
    M: "scipy.sparse.csr_matrix",
    normalize: bool = False,
    antidiagonal: bool = False,
    plot_dir: Optional[str] = None,
) -> "scipy.sparse.csr_matrix":
    """Function to compute the correlation matrix from a sparse matrix.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        The input, normalized contact map. Must be a single chromosome. Values
        are assumed to be only the upper triangle of a symmetric matrix.
    normalize : bool
        If enables normalize the matrix first. [Default: False].
    antidiagonal : bool
        Either there is an antidiagonal on the matrix or not. This is still in
        development. If enables, it will also detrend the antidiagonal.
    plot_dir : directory
        Directory to save plot if one given.

    Returns
    -------
    scipy.sparse.csr_matrix:
        The detrended matrix.
    """
    # Make matrix symetric (in case of upper triangle)
    M = get_symmetric(M)

    # Normalize the matrix
    if normalize:
        M = hcs.normalize_sparse(M, norm="SCN")
    M = M.tocoo()
    n = M.shape[0]

    # Detrend by the distance law
    dist_vals = np.array([np.average(M.diagonal(j)) for j in range(n)])
    M.data = M.data / dist_vals[abs(M.row - M.col)]

    # Detrend the antidiagonal
    if antidiagonal:
        M = np.rot90(M.todense())
        dist_vals = np.array([np.average(M.diagonal(j)) for j in range(n)])
        M = sp.sparse.coo_matrix(M)
        M.data = np.log2(M.data / dist_vals[abs(M.row - M.col)])
        M = np.rot90(M.todense()).T
        M = sp.sparse.coo_matrix(M)

    # Plot detrend matrix
    if plot_dir is not None:
        detrend_map_file = os.path.join(plot_dir, "detrend_map.png")
        bcp.contact_map(
            M.toarray(),
            dpi=200,
            cmap="Reds",
            out_file=detrend_map_file,
            title="Detrend contact map",
        )
    return M.tocsr()


def fourc_like(
    M: "scipy.sparse.csr.csr_matrix",
    frags: "pandas.DataFrame",
    reg1: str,
    reg2: str,
    bin_size: int,
    window: int = 10_000,
    stride: int = 1_000,
    out_file: Optional[str] = None,
) -> "pandas.DataFrame":
    """Compute a 4C-like vector of the region 2 contacts against the region 1.

    Parameters
    ----------
    M : scipy.sparse.csr.csr_matrix
        Sparse or dense symetric matrix.
    frags : pandas.DataFrame
        Index tables of the different fragments and chromosomes.
    reg1 : str
        UCSC position of the region to sum contacts.
    reg2 : str
        UCSC position of the region of interest.
    bin_size : int
        Size of bin in the matrix in base pair.
    window : int
        Size of one window in base pair. [Default 10000].
    stride : int
        Size of step to do on output genome in base pair. [Default 1000].
    out_file : str
        Output file to write the track in tsv format.

    Returns
    -------
    pandas.DataFrame:
        4C-like normalized vector.
    """
    # Define bin_size as the greatest common divisor between the stride and the
    # resolution.
    window = int((window / bin_size) / 2)
    stride = int(stride / bin_size)

    # Parse UCSC regions.
    bins = frags.iloc[:, 0:2]
    reg1 = hcc.parse_ucsc(reg1, bins)
    reg2 = hcc.parse_ucsc(reg2, bins)
    fourc = frags.iloc[reg1[0] : reg1[1], 0:3]

    # Compute the 4C like vector.
    fourc["val"] = 0

    # Check the symetry of the matrix.
    if not is_symmetric(M):
        M = get_symmetric(M)

    # Proportion of contacts to normalize.
    proportion = np.nansum(
        M[reg2[0] : reg2[1], reg2[0] : reg2[1]].toarray()
    ) / np.nansum(M[reg1[0] : reg1[1], reg1[0] : reg1[1]].toarray())

    reg = [0, 0]
    for i in range(reg1[0], reg1[1], stride):
        reg[0] = max(reg1[0], i - window)
        reg[1] = min(reg1[1] + 1, i + window + 1)
        size = reg[1] - reg[0]
        # Inter contacts to take into account.
        count = np.nansum(M[reg[0] : reg[1], reg2[0] : reg2[1]].toarray())
        # Intra contacts in the reference same region to normalize.
        count_ref = np.nansum(M[reg[0] : reg[1], reg1[0] : reg1[1]].toarray())
        # Normalize the value by the size, the number of intra-contacts and the
        # proportion of reads between the region 2 and 1.
        fourc.loc[i, "val"] = ((count / size) / (count_ref / size)) / proportion

    # Write it in a CSV object.
    if out_file is not None:
        fourc.to_csv(out_file, sep="\t", header=True, index=False)
    return fourc


def get_hicreppy(
    matrix_list: List[str],
    max_dist: int = 100_000,
    subsample: int = 0,
    h: Optional[int] = None,
) -> "numpy.ndarray":
    """Compute a correlation matrix using HiCreppy between HiC matrix. It needs
    cooler files as input.

    Parameters
    ----------
    matrix_list : list of str
        List of path to cooler matrices.
    max_dist : int
        Maximum distance at which to compute the SCC, in basepairs.
        [Default: 100000]
    subsample : int
        Subsample values of the matrices. If 0 is given it will give a subsample
        value of the smallest numbers of contacts of all matrices.
    h : int
        Value of the smoothing parameter h to use. Should be a positive integer.
        By default use hicrep.htrain function to find the optimal value.

    Returns
    -------
    numpy.ndarray:
        Correlation matrix between HiC matrices based on HiCreppy correlation.
    """
    # Initiate the table.
    N = len(matrix_list)
    data = np.zeros((N, N))

    # Search for minimal contacts map if no subsample value are given.
    if subsample == 0:
        subsample = np.inf
        for cool_file in matrix_list:
            subsample = int(
                min(cooler.Cooler(cool_file).info["sum"], subsample)
            )

    # Compute the smoothing optimal parameters to used if none given.
    if h is None:
        h = 0
        for i in range(N):
            for j in range(i + 1, N):
                M1 = cooler.Cooler(matrix_list[i])
                M2 = cooler.Cooler(matrix_list[j])
                h = max(h, hicrep.h_train(M1, M2, max_dist=max_dist, h_max=10))

    # Compute Hicrep
    for i in range(N):
        data[i, i] = 1
        for j in range(i + 1, N):
            M1 = cooler.Cooler(matrix_list[i])
            M2 = cooler.Cooler(matrix_list[j])
            scc = hicrep.genome_scc(
                M1, M2, max_dist=max_dist, h=h, subsample=subsample
            )
            data[i, j] = scc
            data[j, i] = scc
    return data


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

    Example
    -------
        >>> import numpy as np
        >>> import scipy as sp
        >>> M = sp.sparse.csr_matrix(np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]]))
        >>> M2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
        >>> (get_symmetric(M) == M2).all()
        True
        >>> M = sp.sparse.csr_matrix(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
        >>> (get_symmetric(M) == M.todense()).all()
        True
    """
    # Test whether it's already symetric or not.
    if not is_symmetric(M):
        M = M + M.T
        # Divide the diagonal by 2 as we add it twice.
        M.setdiag(M.diagonal() / 2)
        # Remove zeros from teh sparse matrix
        M.eliminate_zeros()
    return M


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
        bin_mat = sp.sparse.triu(bin_mat)
    bin_mat.data = bin_mat.data.astype(bool)
    # Adding a frame of zeros around the signal
    tmp = sp.sparse.csr_matrix((win_size - 1, ns), dtype=bool)
    bin_mat = sp.sparse.vstack([tmp, bin_mat, tmp], format=mat.format)
    tmp = sp.sparse.csr_matrix(
        (ms + 2 * (win_size - 1), win_size - 1), dtype=bool
    )
    bin_mat = sp.sparse.hstack([tmp, bin_mat, tmp], format=mat.format)
    # Convolve the uniform kernel with this matrix to get the proportion of
    # nonzero pixels in each neighbourhood
    kernel = np.ones((win_size, win_size))
    win_area = win_size**2
    density = cud.xcorr2(bin_mat, kernel / win_area)

    # Compute convolution of uniform kernel with a frame of ones to get number
    # of missing pixels in each window.
    frame = cup.frame_missing_mask(
        sp.sparse.csr_matrix(mat.shape, dtype=bool),
        kernel.shape,
        sym_upper=sym_upper,
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
        density = sp.sparse.triu(density)
    return density


def interpolate_white_lines(M: "numpy.ndarray") -> "numpy.ndarray":
    """Function to interpolate the white lines in a matrix. It will interpolate
    only single white lines. The interpolation is based on the local p(s). (The
    four closest points on the same diagonals).

    Parameters
    ----------
    M : numpy.array
        Matrix where the white lines needs tob interpolate.

    Returns
    -------
    numpy.array:
        Matrix with interpolated white lines.
    """
    # Size of the matrix.
    N = copy.copy(M)
    N_tmp = copy.copy(M)
    n = len(N)
    # Detect white lines or not well covered lines.
    zeros = mask_white_line(N)
    N[zeros] = 0
    N[:, zeros] = 0

    # Detect white lines shifted of one to keep only single white lines.
    N2 = map_extend(N, 1)
    mask = np.sum(
        np.logical_or(
            np.logical_and((N2 == 0)[:-2, :-2], N == 0),
            np.logical_and((N2 == 0)[2:, 2:], N == 0),
        ),
        axis=1,
    ) == len(N)
    zeros = np.sum(N, axis=1) == 0
    print(zeros)

    # Put values to nan to avoid to use them as mean.
    N_tmp[zeros] = np.nan
    N_tmp[:, zeros] = np.nan

    for k in zeros:
        # Make the interpolation for rows.
        for j in range(len(N)):
            i = k
            # Border case not took into account...
            if i <= n - 3 and j <= n - 3 and i >= 2 and j >= 2:
                N[i, j] = np.nanmean(
                    [
                        N_tmp[i - 2, j - 2],
                        N_tmp[i - 1, j - 1],
                        N_tmp[i + 1, j + 1],
                        N_tmp[i + 2, j + 2],
                    ]
                )
        # Make the interpolation for columns.
        for i in range(len(N)):
            j = k
            # Border case not took into account...
            if i <= n - 3 and j <= n - 3 and i >= 2 and j >= 2:
                N[i, j] = np.nanmean(
                    [
                        N_tmp[i - 2, j - 2],
                        N_tmp[i - 1, j - 1],
                        N_tmp[i + 1, j + 1],
                        N_tmp[i + 2, j + 2],
                    ]
                )
    # Put back the mask on the values which have multiple white lines.
    # N[mask] = 0
    # N[:, mask] = 0
    return N


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

    Example
    -------
        >>> import numpy as np
        >>> import scipy as sp
        >>> M = sp.sparse.csr_matrix(np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]]))
        >>> is_symmetric(M)
        False
        >>> M = sp.sparse.csr_matrix(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
        >>> is_symmetric(M)
        True
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

    Example
    -------
        >>> import numpy as np
        >>> M = np.array([[1, 2], [3, 4]])
        >>> print(map_extend(M, 1))
        [[4 3 4 3]
         [2 1 2 1]
         [4 3 4 3]
         [2 1 2 1]]
    """
    # Comput the size of the initial matrix.
    n = len(M)
    # Concatenate the borders as the matrix to extend it playing on the circular
    # relations on both axis.
    M = np.concatenate(
        (
            M[n - s :,],
            M,
            M[:s,],
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
        return sp.stats.median_abs_deviation(x, nan_policy="omit")

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


def ratio_inter_cool(clr):
    """Compute the ratio of inter/intra contacts of non balanced cooler object.

    Parameters
    ----------
    clr : cooler.api.Cooler
        Cooler matrix stored in cooler object. It doesn't need to be normalized.

    Return
    ------
    int:
        Number of intra contacts.
    int:
        Number of inter contacts.
    """

    # Import binning size and matrix.
    binning = clr.info["bin-size"]
    mat = np.triu(clr.matrix(balance=False)[:])
    total = clr.info["sum"]

    # Initialize.
    intra = 0
    cumul_length = 0
    # Compute intra sum for each DNA sequence (chromosome).
    for length in clr.chroms()[:].length:
        length = math.ceil(length / binning)
        intra += np.nansum(
            mat[
                cumul_length : cumul_length + length,
                cumul_length : cumul_length + length,
            ]
        )
        cumul_length += length

    # Compute the inter numbers of contact
    inter = total - intra

    return intra, inter
