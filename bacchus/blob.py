#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions for blob detections. These functions have been develop to 
detect blob on 500bp resolution HiC contact map of Escherichia coli and have not
been tested on others organisms or others resolutions.

Class:
    - Blob:
        - refine_borders
        - set_score
        - set_rna_score

Function:
    - build_kernel
    - compute_blob_score
    - compute_convolution_score
    - get_blob_borders
"""


import bacchus.hic as bch
import bacchus.insulation as bci
import copy
import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Tuple


class Blob:
    """Class to handle blob objects. The idea is to access easily to the data of
    one blob.

    start : int
        Left border of the blob in bin coordinate.
    end : int
        Right border of the blob in bin coordinate.
    score : numpy.ndarray
        Local blob score vector.
    rna : numpy.ndarray
        Vector of rna counts.
    """

    def __init__(
        self,
        start: int,
        end: int,
        score: "numpy.ndarray",
        rna: Optional["numpy.ndarray"] = None,
    ):
        """Initiate Blob instance."""
        self.start = start
        self.end = end
        self.size = end - start
        self.set_score(score)
        # Build an RNA score if rna counts given. RNA score not sure it's good.
        if rna is not None:
            self.set_rna_score(rna)

    def refine_borders(
        self,
        score: "numpy.ndarray",
        alpha: float,
        rna: Optional["numpy.ndarray"] = None,
    ) -> List["Blob"]:
        """Refine a broad border of one blob based on a ratio to exclude.

        Parameters
        ----------
        self : Blob
            Blob to refine.
        score : numpy.ndarray
            Local blob score vector.
        alpha : float
            Ratio to exclude from the blob.
        rna : numpy.ndarray
            Vector of rna counts.

        Returns
        -------
        List of Blob:
            List of new instance(s) of Blob refined.
        """
        start = max(0, self.start - 5)
        end = min(self.end + 5, len(score))
        # Extract part of the vector of interest.
        blob_values = score[start:end]
        # Defined a new cutoff
        cutoff = alpha * (max(blob_values) - min(blob_values)) + min(
            blob_values
        )
        return get_blob_borders(score, start, end, cutoff, rna)

    def set_score(self, score: "numpy.ndarray"):
        # Set blob score.
        self.score = np.nanmean(score[self.start : self.end])

    def set_rna_score(self, rna: "numpy.ndarray"):
        # Set rna score of the region of the blob.
        self.rna_score = np.nanmean(rna[self.start : self.end])


def build_kernel(size: int) -> "numpy.ndarray":
    """Function to build a kernel to screen the diagonal for blob borders
    detection.

    Parameters
    ----------
    size : int
        Half size of the kernel built.

    Returns
    -------
    numpy.ndarray:
        Kernel.
    """
    # Compute size of the kernel and create an empty kernel.
    n_kernel = (2 * size) + 1
    kernel = np.zeros((n_kernel, n_kernel))
    # Build the kernel.
    for i in range(n_kernel):
        for j in range(n_kernel):
            # Create two axis ax0 is the distance to the diagonal, and ax1 is
            # the distance to the antidiagonal.
            ax0 = abs(i - j) / (2 * size)
            ax1 = abs(2 * size - i - j) / (2 * size)
            # Let 0 in the diagonal and do a pseudo gaussian kernel based on
            # these axis.
            # if j != i:
            kernel[i, j] = (1 / np.sqrt(2)) * np.exp(
                -1 / 2 * (1 / 2 * (ax0 + ax1)) ** 2
            )
    return kernel


def compute_blob_score(
    M: "numpy.ndarray", size: int, n_mads: int = 3
) -> "numpy.ndarray":
    """Function to compute score for the blobs based a kernel of half size size.

    Parameters
    ----------
    M : numpy.ndarray
        Extended matrix used to compute the correlation score.
    size : int
        Half size of the kernel built.
    n_mads : int
        Number of median absolute deviation used as threshold.

    Returns
    -------
    numpy.ndarray:
        Convolution local score vector.
    """ 
    # Extend matrix for border effects and remove white lines.
    N = bch.map_extend(M, size)
    N[np.isnan(N)] = 0
    mask = bch.mask_white_line(N, n_mads)
    
    # Transform into white lines into nan.
    N[mask] = np.nan
    N[:, mask] = np.nan

    # Build kernel.
    kernel = build_kernel(size)

    # Compute the convulotion score.
    score = compute_convolution_score(N, kernel)

    # Remove the second lower enveloppe to avoid local high values for a whole
    # region.compute_convolution_score
    local_score, _ = bci.get_local_score(score)

    return local_score


def compute_convolution_score(
    M: "numpy.ndarray", kernel: "numpy.ndarray"
) -> "numpy.ndarray":
    """Function to do the convolution product along the diagonal. To work the
    matrix needs to have been extended by the size of the kernel.

    Parameters
    ----------
    M : numpy.ndarray
        Extended matrix used to compute the correlation score.
    kernel : numpy.ndarray
        Convolution kernel to use.

    Returns
    -------
    numpy.ndarray:
        Convolution score vector.
    """
    size = len(kernel)
    n = len(M) - size + 1
    score = np.zeros((n))
    for i in range(n):
        # Extract the matrix.
        L = M[i : i + size, i : i + size]
        # Do the convolution product.
        score[i] = np.nanmean(L * kernel)
    return score


def find_blobs(
    M: "numpy.ndarray",
    size: int,
    n_mads: int = 3,
    refine: Optional[float] = None,
    rna: Optional["numpy.ndarray"] = None,
) -> Tuple[List["Blob"], "numpy.ndarray"]:
    """Function to find blobs from a matrix and compute their scores.

    Parameters
    ----------
    M : numpy.ndarray
        Extended matrix used to compute the correlation score.
    size : int
        Half size of the kernel built.
    n_mads : int
        Number of median absolute deviation used as threshold.
    refine : float
        Ratio to exclude part of the blobs. If it's low it could increase a
        little bit the size of the blobs. By default, it will not be done.
    rna : numpy.ndarray
        Vector of rna counts.

    Returns
    -------
    List of blobs:
        List of blob object with their start and end positions refined or not.
    """
    # Compute blob score.
    blob_score = compute_blob_score(M, size, n_mads)

    # Get blob_borders
    blobs = get_blob_borders(blob_score)

    # Refine blob borders if necessary.
    if refine is not None:
        blobs = refine_borders(blobs, blob_score, refine, rna)

    return blobs, blob_score


def get_blob_borders(
    blob_score: "numpy.ndarray",
    start: int = 0,
    end: Optional[int] = None,
    cutoff: Optional[float] = None,
    rna: Optional["numpy.ndarray"] = None,
) -> List["Blob"]:
    """Function to get the blob borders from a vector of a blob score. Positions
    should be given in bin coordinates. If an RNA vector is given a rna score
    can be attribute to the blobs.

    Paramters
    ---------
    blob_score : numpy.ndarray
        Local blob score.
    start : int
        Start position in bin coordinates (0-based) to get the blobs.
        [Default: 0]
    end : int
        End position in bin coordinates (0-based) to get the blobs. By default
        it will search until the end of teh vector.
    cutoff : float
        Cutoff to use as threshold to determine a border. By default it will use
        the median plus one quarter of the standard deviation of the blob score
        vector.
    rna : numpy.ndarray
        Vector of rna counts.

    Returns
    -------
    List of Blob:
        List of new instance(s) of Blob.
    """
    # Define end and cutoff if None given:
    if end is None:
        end = len(blob_score)
    if cutoff is None:
        cutoff = np.nanmedian(blob_score) + 0.25 * np.nanstd(blob_score)
    # Detect peak if currently no peak look for start.
    peak = False
    blobs = []
    # Iterates on all the range given.
    for i in range(start, end):
        v = blob_score[i]
        # Find start.
        if not peak:
            if v >= cutoff:
                start = i
                peak = True
        # Find end.
        else:
            if v < cutoff:
                end = i
                peak = False
                if end - start > 1:
                    blobs.append(Blob(start, end, blob_score, rna))
    # Case of an unfinsihed blob at the end.
    end = i
    if peak and start - end > 1:
        blobs.append(Blob(start, end, blob_score, rna))
    # TODO: Do the circularity case with a blob starting at the end and
    # finishing at teh start.
    return blobs


def refine_borders(
    blobs: List["Blob"],
    blob_score: "numpy.ndarray",
    alpha: float,
    rna: Optional["numpy.ndarray"],
) -> List["Blob"]:
    """Function to reform the borders of a list of blob objects.

    Paramters
    ---------
    blobs : list of Blob
        List of blob objects.
    blob_score : numpy.ndarray
        Local blob score.
    alpha : float
        Ratio to exclude part of the blobs. If it's low it could increase a
        little bit the size of the blobs.
    rna : numpy.ndarray
        Vector of rna counts.

    Returns
    -------
    List of Blob:
        List of new instance(s) of Blob.
    """
    new_blobs = []
    # Refine borders for each blobs.
    for blob in blobs:
        blobs_local = blob.refine_borders(blob_score, alpha, rna)
        # Flatten the list
        for blob_local in blobs_local:
            new_blobs.append(blob_local)
    return new_blobs
