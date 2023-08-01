#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""General functions for origin of replication and terminaison detection.

Functions:
    - compute_oriter_ratio
    - detect_ori_ter
    - gc_skew_shift_detection
    - oriter_main
    - remove_nmad
"""


import bacchus.io as bcio
import pandas as pd
import numpy as np
import pyfastx
from typing import List, Optional, Tuple
from bacchus.genomes import Genome, Fragment, Position, Track


def compute_oriter_ratio(
    data: "pandas.DataFrame", ori: int, ter: int, window: int = 100_000
) -> Tuple[float]:
    """Function to compute the ratio of the coverage between the ori and the
    ter.

    Parameters:
    -----------
    data : pandas.DataFrame
        Table with 4 columns : "chr", "start", "end", "value".
    ori : int
        Position of the ori.
    ter : int
        Position of the ter.
    window : int
        Size of the window to consider around the ori or ter to get the mean
        coverage.

    Returns:
    --------
    float:
        Ratio of the coverage between ori and ter region.
    float:
        Mean coverage around the ori.
    float:
        Mean coverage around the ter.
    """
    # Rename columns and initialize values.
    data.columns = ["chr", "start", "end", "val"]
    step = data.start[1] - data.start[0]
    ori_values = []
    ter_values = []

    # Extract values vector and add 50kb at each extremities to have a
    # pseudo-circular array to manage easily the edge cases.
    val = np.array(data.val)
    extend = round((window / 2) / step)
    values = np.concatenate(
        (np.concatenate((val[-extend:], val)), val[:extend])
    )

    # Transform ori and ter in the same coordinates as the bed indexes.
    ori = ori // step
    ter = ter // step
    ori_values = values[ori : ori + 2 * extend]
    ter_values = values[ter : ter + 2 * extend]

    # Remove outliers values and do the mean coverage at each size.
    ori_value = np.mean(remove_nmad(ori_values))
    ter_value = np.mean(remove_nmad(ter_values))
    # Compute ratio.
    ratio = ori_value / ter_value
    return ratio, ori_value, ter_value


def detect_ori_ter(
    gc_shift: List[Position],
    pars: List[Fragment],
    genome: Genome,
    cov_data: Optional[Track] = None,
    window: Optional[int] = 50_000,
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
    cov_data : Track
        Path to the coverage bed or bigwig file.
    window : int
        Size of the window to average coverage at a GC shift.

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
        # Separate case where we have parS or not.
        if len(pars_chrom) > 0:  # Case with parS sites.
            pars_cluster_pos = np.mean([site.middle() for site in pars_chrom])
            pars_opp_pos = pars_cluster_pos + chrom_size / 2
            # Take the circularity into account
            if pars_opp_pos > chrom_size:
                pars_opp_pos = pars_opp_pos - chrom_size
            diff_ori = 10 ** 12
            diff_ter = 10 ** 12
            ori = gc_shift[0].coord
            ter = gc_shift[1].coord
            for pos in gc_shift:
                if pos.chrom == chrom:
                    curr_diff = abs(pos.coord - pars_cluster_pos)
                    # Take the circularity into account
                    if curr_diff > chrom_size / 2:
                        curr_diff = chrom_size - curr_diff
                    if diff_ori > curr_diff:
                        ori = pos.coord
                        diff_ori = curr_diff
                    curr_diff = abs(pos.coord - pars_opp_pos)
                    # Take the circularity into account
                    if curr_diff > chrom_size / 2:
                        curr_diff = chrom_size - curr_diff
                    if diff_ter > curr_diff:
                        ter = pos.coord
                        diff_ter = curr_diff
            pos_list.append(Position(chrom, ori, description="Ori"))
            pos_list.append(Position(chrom, ter, description="Ter"))
        else:  # Cas with no parS sites
            # Import cov data
            if cov_data is not None:
                min_cov, max_cov = 0, 0
                ori, ter = None, None
                try:
                    cov_data.values[chrom]
                    for pos in gc_shift:
                        if pos.chrom == chrom:
                            local_cov = np.nanmean(
                                remove_nmad(
                                    cov_data.values[pos.chrom][
                                        pos.coord - stride : pos.coord + stride
                                    ]
                                )
                            )
                            if local_cov < min_cov or min_cov == 0:
                                ter = pos.coord
                                min_cov = local_cov
                            elif local_cov > max_cov or max_cov == 0:
                                ori = pos.coord
                                max_cov = local_cov
                    if ori is None or ter is None:
                        ori, ter = None, None
                except KeyError:  # Case no coverage on chrom.
                    pass
                pos_list.append(Position(chrom, ori, description="Ori"))
                pos_list.append(Position(chrom, ter, description="Ter"))
    return pos_list


def gc_skew_shift_detection(
    data: "pandas.DataFrame", genome: Genome, circular: Optional[bool] = True,
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
                            x / (abs(x) + 10 ** -12)
                            for x in data_chrom.val.iloc[0:i]
                        ]
                    )
                    if circular:
                        a += np.sum(
                            [
                                x / (abs(x) + 10 ** -12)
                                for x in data_chrom.val.iloc[n - alpha + i : n]
                            ]
                        )
                    b = np.sum(
                        [
                            x / (abs(x) + 10 ** -12)
                            for x in data_chrom.val.iloc[i : i + alpha]
                        ]
                    )
                elif i + alpha > n - 1:  # Edge case too close to chrom end.
                    a = np.sum(
                        [
                            x / (abs(x) + 10 ** -12)
                            for x in data_chrom.val.iloc[i - alpha : i]
                        ]
                    )
                    b = np.sum(
                        [
                            x / (abs(x) + 10 ** -12)
                            for x in data_chrom.val.iloc[i:n]
                        ]
                    )
                    if circular:
                        b += np.sum(
                            [
                                x / (abs(x) + 10 ** -12)
                                for x in data_chrom.val.iloc[0 : alpha - n + i]
                            ]
                        )
                else:
                    a = np.sum(
                        [
                            x / (abs(x) + 10 ** -12)
                            for x in data_chrom.val.iloc[i - alpha : i]
                        ]
                    )
                    b = np.sum(
                        [
                            x / (abs(x) + 10 ** -12)
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


def main_oriter_detection(
    fasta_file: str,
    gc_skew_file: str,
    pars_file: str,
    outfile: str,
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

    # Import coverage track.
    if cov_file is not None:
        cov_data = bcio.generate_track(cov_file, circular)
    else:
        cov_data = None

    # Compute GC_shift.
    gc_shift = gc_skew_shift_detection(gc_skew_data, genome, circular)

    # Detect ori and ter.
    pos = detect_ori_ter(gc_shift, pars_data, genome, cov_data, window)

    # Write in a bed file.
    with open(outfile, "w") as out:
        for p in pos:
            out.write(f"{p.chrom}\t{p.coord}\t{p.description}\n")


def remove_nmad(values: "numpy.ndarray", n: int = 3) -> "numpy.ndarray":
    """Function to remove values which are at different from the median more
    than n times the mad.
    """
    mad = np.median(np.absolute(values - np.median(values)))
    med = np.median(values)
    return [x for x in values if abs(x - med) <= mad * n]
