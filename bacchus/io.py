#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Input output functions to import or export genomics data from files.

Functions:
    - binned_map
    - build_map
    - extract_big_wig
    - generates_frags
    - generate_track
"""


import bacchus.hic as bch
import cooler
import hicstuff.hicstuff as hcs
import hicstuff.io as hio
import math
import numpy as np
import pandas as pd
import pyBigWig
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional, Tuple
from bacchus.genomes import Track


def binned_map(
    matrix_file: str, fragment_file: str, bin_size: int
) -> Tuple["scipy.sparse.csr_matrix", "pandas.DataFrame"]:
    """Function to bin a sparse matrix at a bin size value from a given graal
    file matrix.

    Parameters
    ----------
    matrix_file : str
        Path to the graal matrix file.
    fragment_file : str
        Path to the graal fragment file.
    bin_size : int
        Binning size of the final matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binned sparse matrix.
    pandas.DataFrame
        Fragments position information.
    """
    # Load the matrix
    sparse_map, frags, _ = hio.flexible_hic_loader(
        matrix_file, fragments_file=fragment_file, quiet=True
    )

    # Bin the matrix
    pos = frags.iloc[:, 2]
    binned_map, binned_pos = hcs.bin_bp_sparse(
        M=sparse_map, positions=pos, bin_len=bin_size
    )

    # Get bin numbers of chromosome starts
    binned_start = np.append(np.where(binned_pos == 0)[0], len(binned_pos))

    # Get bin length of each chromosome
    num_binned = binned_start[1:] - binned_start[:-1]

    # Get unique chromosome names without losing original order
    # (numpy.unique sorts output)
    chr_names_idx = np.unique(frags.iloc[:, 1], return_index=True)[1]
    chr_names = [frags.iloc[index, 1] for index in sorted(chr_names_idx)]
    binned_chrom = np.repeat(chr_names, num_binned)
    binned_frags = pd.DataFrame(
        {"chrom": binned_chrom, "start_pos": binned_pos[:, 0]}
    )
    binned_frags["end_pos"] = binned_frags.groupby("chrom")["start_pos"].shift(
        -1
    )
    chrom_ends = frags.groupby("chrom").end_pos.max()
    # Fill ends of chromosome bins with actual chromosome length
    for cn in chrom_ends.index:
        binned_frags.end_pos[
            np.isnan(binned_frags.end_pos) & (binned_frags.chrom == cn)
        ] = chrom_ends[cn]
    binned_frags.start_pos = binned_frags.start_pos.astype(int)
    binned_frags.end_pos = binned_frags.end_pos.astype(int)
    return binned_map, binned_frags


def build_map(
    matrix_files: List[str],
    fragment_file: str,
    bin_size: int = 1,
    mat_format: str = "graal",
    normalize: bool = True,
    subsample: int = 0,
) -> "numpy.ndarray":
    """Function to bin, normalize and subsample if necessary a given matrix. If
    more than a matrix is given it will concatenate them.

    Parameters
    ----------
    matrix_files : list of str
        List of the path of graal contact map files to build the final matrix.
        All the matrices needs to have been made using the same fragments
        references.
    fragment_file : str
        Path to the graal fragment file.
    bin_size : int
        Binning size of the final matrix.
    mat_format : str
        Format of the matrix file, either graal or cool. Default graal.
    normalize : bool
        Either the matrix needs to be normalized or not.
    subsample : int
        Numbers of subsampled contacts. If zero, no subsamples are done.

    Returns
    -------
    numpy.ndarray
        Final dense matrix.
    """
    if mat_format == "graal":
        # Iterates on the sparse matrices given and sum them if more than one is
        # given.
        for i, matrix_file in enumerate(matrix_files):
            m, _ = binned_map(matrix_file, fragment_file, bin_size)
            if i == 0:
                M = m
            else:
                M += m
        # Subsample the contacts if necessary.
        M = M.tocoo()
        if subsample != 0:
            M = hcs.subsample_contacts(M, int(subsample))
        # Normalize the matrix if necessary.
        if normalize:
            M = hcs.normalize_sparse(M, norm="ICE", n_mad=10)
        # Do the symetrics.
        M = bch.get_symmetric(M)
        # Transform to dense matrix.
        M = M.toarray()

    if mat_format == "cool":
        # If mcool file, bin size is required.
        if bin_size != 1:
            M = cooler.Cooler(
                f"{matrix_files[0]}::/resolutions/{bin_size}"
            ).matrix(balance=normalize)[:]
        else:
            M = cooler.Cooler(matrix_files[0]).matrix(balance=normalize)[:]
    return M


def extract_big_wig(
    file: str,
    binning: int = 1,
    circular: bool = True,
    sigma: Optional[int] = None,
    ztransform: bool = False,
    chroms: Optional[List[str]] = None,
) -> "numpy.ndarray":
    """Function to extract big wig information. It considered that the file is
    bin at 1 base pair and it has only one chromosome. If binning is set it will
    bin the tracks.

    Parameters
    ----------
    file : str
        Path to the BigWig file
    binning : int
        Binning size for the output tracks in bp.
    circular : bool
        If the genome is circular or not.
    sigma : int
        If one given will do a gaussian filter with this sigma value.
    ztransform : bool
        Whether to Z-transformed the track or not.
    chrom : list of str
        Chromosome names to extract.

    Returns
    -------
    numpy.ndarray:
        Vector of the tracks values binned and z-transformed if asked.
    dictionnary:
        Dictionnary with the names of the chromosome as key and the start
        positions as value.
    """
    # Create a dictionnary with starting position of chromosomes.
    dict_chr = {}
    length = 0

    # Open BigWig file
    tab = pyBigWig.open(file)

    chroms = chroms if chroms is not None else tab.chroms()
    for name in chroms:
        dict_chr[name] = length
        length += math.ceil(tab.chroms()[name] / binning)

    # Defined final vector of values.
    binned_values = np.zeros((length + 1))

    for name in chroms:
        # Defined start position of the chromosome in the final binned vector.
        start = dict_chr[name]

        # Extract values for one chromosomes.
        values = tab.values(name, 0, tab.chroms()[name])
        values = np.array([0 if np.isnan(x) else x for x in values])

        # Do a gaussian blur if sigma is given.
        if sigma is not None:
            # Define relevant mode according to the geometry of the genome.
            if circular:
                mode = "wrap"  # abcd|abcd|abcd
            else:
                mode = "nearest"  # aaaa|abcd|dddd
            values = gaussian_filter1d(values, sigma=sigma, mode=mode)

        # Binned values if values different of one.
        if binning == 1:
            binned_values[start : start + len(values)] = values
        else:
            for i in range((len(values) // binning) + 1):
                binned_values[start + i] = np.nanmean(
                    values[binning * i : binning * (i + 1)]
                )

    # Ztransform
    if ztransform:
        binned_values = (binned_values - np.nanmean(binned_values)) / np.nanstd(
            binned_values
        )
    return binned_values, dict_chr


def generates_frags(n: int, binning: int) -> "pandas.DataFrame":
    """Generates fragments pandas dataframe for a binned matrix of size n.

    Parameters
    ----------
    n : int
        Size of the binned matrix.
    binning : int
        Size in base pair of one bin.

    Returns
    -------
    pandas.DataFrame:
        Table of the binned fragments.
    """
    # Create the dataframe.
    frags = {
        "id": np.arange(1, n + 1),
        "chrom": np.repeat("chr01", n),
        "start_pos": np.arange(0, n * binning, binning),
        "end_pos": np.arange(binning, (n + 1) * binning, binning),
        "size": np.repeat(binning, n),
        "gc_content": np.repeat(0, n),
    }
    frags = pd.DataFrame(frags)
    return frags


def generate_track(
    file: str,
    circular: Optional[bool] = True,
) -> Track:
    """Function to extract a track from a bigwig binned at 1bp.

    Parameters
    ----------
    file : str
        Path to the bigwig file.
    circular : bool
        If the genome is circular or not.
    """
    bigwig = pyBigWig.open(file)
    return Track(bigwig, circular)
