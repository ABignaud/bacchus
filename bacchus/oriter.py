#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""General functions for origin of replication and terminaison detection.

Functions:
    - compute_oriter_ratio
    - detect_ori_ter
    - frags_center
    - gc_skew_shift_detection
    - get_window
    - main_oriter_detection
    - interpolate_nmad
    - remove_nmad
"""


import bacchus.io as bcio
import pandas as pd
import numpy as np
import pyfastx
from typing import List, Optional, Tuple
from bacchus.genomes import Genome, Fragment, Position, Track
from scipy.interpolate import splrep, BSpline


def compute_oriter_ratio(
    cov: Track,
    oriter: List[Position],
    window: int = 100_000,
    circular: bool = True,
) -> dict:
    """Function to compute the ratio of the coverage between the ori and the
    ter.

    Parameters:
    -----------
    cov : Track
        Table with 4 columns : "chr", "start", "end", "value".
    ori : list of Position
        Positions of the ori and ter of each chromosomes.
    window : int
        Size of the window to consider around the ori or ter to get the mean
        coverage.
    circular : bool
        Either the chromosome is circular or not.

    Returns:
    --------
    dict:
        Values of coverage at the ori and ter positions for each chromosome.
        Chromosome are the keys of the dictionnary.
    """

    # Ratio init
    ratio = {}

    # Iterates on chromosomes.
    for chrom in cov.values:
        ori_pos, ter_pos = None, None
        for p in oriter:
            # Search for ori and ter positions.
            if (p.chrom == chrom) & (p.description == "Ori"):
                ori_pos = int(p.coord)
            elif (p.chrom == chrom) & (p.description == "Ter"):
                ter_pos = int(p.coord)

        if ori_pos is not None:  # Don't do anything if ori or ter is None.
            val = remove_nmad(cov.values[chrom])
            ori_cov = np.nanmean(get_window(val, ori_pos, window, circular))
            ter_cov = np.nanmean(get_window(val, ter_pos, window, circular))
            ratio[chrom] = {
                "ori_cov": ori_cov,
                "ter_cov": ter_cov,
                "ratio": ori_cov / ter_cov,
            }
    return ratio


def detect_ori_ter(
    gc_shift: List[Position],
    pars: List[Fragment],
    genome: Genome,
    mats: Optional[List[Fragment]],
    cov_file: Optional[str] = None,
    window: Optional[int] = 50_000,
    circular: bool = True,
) -> List[Position]:
    """Function to find the ori and ter thanks to GC skew and parS sites. If no
    parS sites have been found (empty DataFrame) we detect Ori and Ter positions
    based on the coverage.

    Parameters
    ----------
    gc_shift : list of Positions
        List of Position of the GC shift.
    pars : list of Fragments
        List of Fragments corresponding to the parS sites.
    genome : Genome
        Genome of the sample.
    mats : list of Fragments
        List of Fragments corresponding to the matS sites.
    cov_file : str
        Path to the coverage bigwig file.
    window : int
        Size of the window to average coverage at a GC shift.
    circular : bool
        Either the chromosomes are circular or not.

    Returns:
    --------
    list of Positions:
        Ori and ter positions.
    """

    pos_list = []
    stride = window // 2

    # Iterates on each chromosome.
    for chrom in genome.chroms:
        chrom_size = len(genome.chroms[chrom])
        pars_chrom = []
        for pos in pars:
            if pos.chrom == chrom:
                pars_chrom.append(pos)
        mats_chrom = []
        for pos in mats:
            if pos.chrom == chrom:
                mats_chrom.append(pos)
        # Separate case where we have parS or not.
        ori, ter = None, None
        # Case with matS only if at least 10 matS sites.
        if len(mats_chrom) > 10:
            ter_pos = frags_center(mats_chrom, chrom_size, circular)
            ori_pos = ter_pos + chrom_size / 2
            # Take the circularity into account
            if ori_pos > chrom_size:
                ori_pos = ori_pos - chrom_size
        elif len(pars_chrom) > 0:  # Case with parS sites.
            ori_pos = frags_center(pars_chrom, chrom_size, circular)
            ter_pos = ori_pos + chrom_size / 2
            # Take the circularity into account
            if ter_pos > chrom_size:
                ter_pos = ter_pos - chrom_size
        else:  # Cas with no parS sites
            # Import cov data
            if cov_file is not None:
                cov_chrom, _ = bcio.extract_big_wig(
                    cov_file, binning=stride, chroms=[chrom]
                )
                x = np.arange(0, len(cov_chrom) * stride, stride)
                cov_chrom = interpolate_nmad(cov_chrom)
                cov_chrom = [i - min(cov_chrom) for i in cov_chrom]
                cov_chrom = [i / max(cov_chrom) for i in cov_chrom]
                s = 10
                c = []
                while (len(c) != 2) and (s < 50):
                    tck_s2 = splrep(x, cov_chrom, s=s, k=3, per=True)
                    min_derivate = min(abs(BSpline(*tck_s2).derivative()(x)))
                    mins = [
                        i
                        for i in np.arange(0, x[-1], stride / 100)
                        if abs(BSpline(*tck_s2).derivative()(i)) < min_derivate
                    ]
                    oriter = np.unique([k // stride * stride for k in mins])
                    s += 1
                if s == 50:
                    pos_list.append(Position(chrom, ori, description="Ori"))
                    pos_list.append(Position(chrom, ter, description="Ter"))
                    continue
                if BSpline(*tck_s2)(oriter[0]) < BSpline(*tck_s2)(oriter[1]):
                    ori_pos = oriter[1]
                    ter_pos = oriter[0]
                else:
                    ori_pos = oriter[0]
                    ter_pos = oriter[1]
            else:
                pos_list.append(Position(chrom, ori, description="Ori"))
                pos_list.append(Position(chrom, ter, description="Ter"))
                continue
        # Refined coordinates using GC shift.
        diff_ori = 10**12
        diff_ter = 10**12
        if len(gc_shift) > 1:  # Case with no GC shift.
            ori = gc_shift[0].coord
            ter = gc_shift[1].coord
        else:
            pos_list.append(Position(chrom, ori, description="Ori"))
            pos_list.append(Position(chrom, ter, description="Ter"))
            continue
        for pos in gc_shift:
            if pos.chrom == chrom:
                curr_diff = abs(pos.coord - ori_pos)
                # Take the circularity into account
                if curr_diff > chrom_size / 2:
                    curr_diff = chrom_size - curr_diff
                if diff_ori > curr_diff:
                    ori = pos.coord
                    diff_ori = curr_diff
                curr_diff = abs(pos.coord - ter_pos)
                # Take the circularity into account
                if curr_diff > chrom_size / 2:
                    curr_diff = chrom_size - curr_diff
                if diff_ter > curr_diff:
                    ter = pos.coord
                    diff_ter = curr_diff
        pos_list.append(Position(chrom, ori, description="Ori"))
        pos_list.append(Position(chrom, ter, description="Ter"))
    return pos_list


def frags_center(
    frags: List[Fragment], chrom_size: int, circular: bool = True
) -> int:
    """Function to compute the middle positions of a list of fragments (parS or
    matS positions). The function handle the case of circular where the middle
    position could be different than the mean of the coordinates.

    Parameters
    ----------
    frags : list of Fragment
        Coordinates of the positions of the fragments.
    chrom_size : int
        Size of the given chromosome.
    circular : bool
        Either the genome is circular or not.

    Return
    ------
    int:
        Position of the middle of the cluster of frags.

    Example
    -------
        >>> from bacchus.genomes import Fragment
        >>> frags = [Fragment('chr', 10, 30), Fragment('chr', 980, 1000)]
        >>> frags_center(frags, 1000, True)
        5
        >>> frags_center(frags, 1000, False)
        505
        >>> frags = [Fragment('chr', 10, 30), Fragment('chr', 40, 60)]
        >>> frags_center(frags, 1000, True)
        35
    """
    middles = [site.middle() for site in frags]
    if circular:
        # Middle in the middle of the chromosome.
        posA = np.mean(middles)
        diffA = np.sum([np.abs(posA - pos) for pos in middles])
        # Middle passing by the edges.
        middles_edges = [
            pos if pos > chrom_size // 2 else pos + chrom_size
            for pos in middles
        ]
        posB = np.mean(middles_edges)
        diffB = np.sum([np.abs(posB - pos) for pos in middles_edges])
        # Smaller difference is the best.
        if diffA <= diffB:
            pos = posA
        else:
            pos = posB if posB < chrom_size else posB - chrom_size
    else:
        pos = np.mean(middles)
    return int(pos)


def gc_skew_shift_detection(
    data: "pandas.DataFrame",
    genome: Genome,
    circular: Optional[bool] = True,
) -> List[Position]:
    """Function to detect GC skew shift.

    Parameters:
    -----------
    data : pandas.DataFrame
        For columns table with the GC skew.
    genome : Genome
        Genome object of the reference.
    Circular : bool
        Either the chromosomes are circular or not.

    Returns:
    --------
    List of Position:
        List of the GC shift position.
    """
    data.columns = ["chr", "start", "end", "val"]
    alpha = 100  # use 100 windows to average the inversion.

    # Look for step and window_size
    step = data.start[1] - data.start[0]
    window = data.end[0] - data.start[0] + 1

    # Iterates on the chromosome.
    pos_shift = []
    chromsizes = genome.chromsizes

    for chrom in genome.chroms:
        # Initialization
        diff = 1000
        pos = 0
        # Subset data by chromosome.
        data_chrom = data.loc[data["chr"] == chrom]
        n = data_chrom.shape[0]
        if n < alpha:  # Remove small chromosome (plasmides).
            continue

        # If circular count a shift if one over the start/end of the chromosome.
        if circular:
            previous_value = data_chrom.val.iloc[n - 1]
        else:
            previous_value = 0

        # Search for GC skew shift on the GC skew
        for i in range(n):
            current_value = data_chrom.val.iloc[i]
            # Search for local inversion of GC skew
            if previous_value * current_value < 0:
                # Check if global inversion or just a small one.
                if i < alpha:  # Edge case too close to chrom start.
                    a = np.sum(
                        [
                            x / (abs(x) + 10**-12)
                            for x in data_chrom.val.iloc[0:i]
                        ]
                    )
                    if circular:
                        a += np.sum(
                            [
                                x / (abs(x) + 10**-12)
                                for x in data_chrom.val.iloc[n - alpha + i : n]
                            ]
                        )
                    b = np.sum(
                        [
                            x / (abs(x) + 10**-12)
                            for x in data_chrom.val.iloc[i : i + alpha]
                        ]
                    )
                elif i + alpha > n - 1:  # Edge case too close to chrom end.
                    a = np.sum(
                        [
                            x / (abs(x) + 10**-12)
                            for x in data_chrom.val.iloc[i - alpha : i]
                        ]
                    )
                    b = np.sum(
                        [
                            x / (abs(x) + 10**-12)
                            for x in data_chrom.val.iloc[i:n]
                        ]
                    )
                    if circular:
                        b += np.sum(
                            [
                                x / (abs(x) + 10**-12)
                                for x in data_chrom.val.iloc[0 : alpha - n + i]
                            ]
                        )
                else:
                    a = np.sum(
                        [
                            x / (abs(x) + 10**-12)
                            for x in data_chrom.val.iloc[i - alpha : i]
                        ]
                    )
                    b = np.sum(
                        [
                            x / (abs(x) + 10**-12)
                            for x in data_chrom.val.iloc[i : i + alpha]
                        ]
                    )
                if a * b < 0:
                    # Look for the middle of the inversion.A new shift is
                    # consider if it's at more than 200kb from the previous one.
                    curr_diff = abs(a + b)
                    curr_pos = int(
                        data_chrom.start.iloc[i] + int(window / 2) - 1
                    )  # -1 for the 0-based.
                    if curr_pos > pos + 200000 and pos != 0:
                        pos_shift.append(Position(chrom, pos))
                        diff = curr_diff
                        pos = curr_pos
                    if curr_diff < diff:  # The closer to 0 the better.
                        diff = curr_diff
                        pos = curr_pos
            previous_value = current_value
        pos_shift.append(Position(chrom, pos))
    return pos_shift


def get_window(val: List, pos: int, wind: int, circular: bool = True) -> List:
    """Function to extract a window from list. it can wrap it if circular track.

    Parameters
    ----------
    val : list of values
        Vector to subset.
    pos : int
        index of central position.
    wind : int
        size of the window.
    circular : bool
        Either the vector is circular or not to handle edge cases.

    Return
    ------
    list:
        Subset vector.
    """
    wind = wind // 2
    n = len(val)
    if circular:
        if pos < wind:
            val_window = np.concatenate((val[wind - pos :], val[: pos + wind]))
        elif pos > n - wind:
            val_window = np.concatenate(
                (val[pos - wind :], val[: pos - n - wind])
            )
        else:
            val_window = val[pos - wind : pos + wind]
    else:
        if pos < wind:
            val_window = val[: pos + wind]
        elif pos > n - wind:
            val_window = val[pos - wind :]
        else:
            val_window = val[pos - wind : pos + wind]
    return val_window


def main_oriter_detection(
    fasta_file: str,
    gc_skew_file: str,
    pars_file: str,
    outfile: str,
    mats_file: Optional[str] = None,
    cov_file: Optional[str] = None,
    window: Optional[int] = 50_000,
    circular: Optional[bool] = True,
):
    # Import genome.
    fasta = pyfastx.Fasta(fasta_file, build_index=False)
    genome = Genome(fasta)

    # Import GC_skew data.
    gc_skew_data = pd.read_csv(
        gc_skew_file, sep="\t", header=0, names=["chr", "start", "end", "val"]
    )

    # Import parS data.
    pars_tmp = pd.read_csv(
        pars_file,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "type", "seq", "err"],
    )
    pars_data = []
    for i in pars_tmp.index:
        pars_data.append(
            Fragment(
                pars_tmp.loc[i, "chr"],
                pars_tmp.loc[i, "start"],
                pars_tmp.loc[i, "end"],
                description="parS",
            )
        )
    # Import matS data.
    if mats_file is not None:
        mats_tmp = pd.read_csv(
            mats_file,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "type", "seq", "err"],
        )
        mats_data = []
        for i in mats_tmp.index:
            mats_data.append(
                Fragment(
                    mats_tmp.loc[i, "chr"],
                    mats_tmp.loc[i, "start"],
                    mats_tmp.loc[i, "end"],
                    description="matS",
                )
            )

    # Compute GC_shift.
    gc_shift = gc_skew_shift_detection(gc_skew_data, genome, circular)

    # Detect ori and ter.
    pos = detect_ori_ter(
        gc_shift,
        pars_data,
        genome,
        mats_data,
        cov_file,
        window,
        circular,
    )

    # Write in a bed file.
    with open(outfile, "w") as out:
        for p in pos:
            out.write(f"{p.chrom}\t{p.coord}\t{p.description}\n")


def interpolate_nmad(values: "numpy.ndarray", n: int = 3) -> "numpy.ndarray":
    """Function to interpolate values which are at different from the median
    more than n times the mad.
    """
    mad = np.nanmedian(np.absolute(values - np.nanmedian(values)))
    med = np.nanmedian(values)
    values = [x if abs(x - med) <= mad * n else np.nan for x in values]
    values = pd.DataFrame(values).interpolate().values.ravel().tolist()
    values[0] = values[1]
    return values


def remove_nmad(values: "numpy.ndarray", n: int = 3) -> "numpy.ndarray":
    """Function to remove values which are at different from the median more
    than n times the mad.
    """
    mad = np.nanmedian(np.absolute(values - np.nanmedian(values)))
    med = np.nanmedian(values)
    return [x for x in values if abs(x - med) <= mad * n]
