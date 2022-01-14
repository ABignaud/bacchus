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
from typing import List, Optional, Tuple


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
    gcskew_data: "pandas.DataFrame", pars_data: "pandas.DataFrame"
) -> Tuple[int]:
    """Function to find the ori and ter thanks to GC skew and parS sites.

    Parameters:
    -----------
    gcskew_data : pandas.DataFrame
        Four columns table with the GC skew.
    pars_data : pandas.DataFrame
        Table with parS sites informations.

    Returns:
    --------
    tuple of int:
        Ori and ter positions.
    """
    pos_shift, genome_size = gc_skew_shift_detection(gcskew_data)
    pars_cluster_pos = np.mean(pars_data.start)
    pars_opp_pos = pars_cluster_pos + genome_size / 2
    # Take the circularity into account
    if pars_opp_pos > genome_size:
        pars_opp_pos - genome_size
    diff_ori = 10 ** 12
    diff_ter = 10 ** 12
    ori = pos_shift[0]
    ter = pos_shift[1]
    for pos in pos_shift:
        curr_diff = abs(pos - pars_cluster_pos)
        # Take the circularity into account
        if curr_diff > genome_size / 2:
            curr_diff = genome_size - curr_diff
        if diff_ori > curr_diff:
            ori = pos
            diff_ori = curr_diff
        curr_diff = abs(pos - pars_opp_pos)
        # Take the circularity into account
        if curr_diff > genome_size / 2:
            curr_diff = genome_size - curr_diff
        if diff_ter > curr_diff:
            ter = pos
            diff_ter = curr_diff
    return ori, ter


def gc_skew_shift_detection(
    data: "pandas.DataFrame",
) -> Tuple[List[int], int]:
    """Function to detect GC skew shift.

    Parameters:
    -----------
    data : pandas.DataFrame
        For columns table with the GC skew.

    Returns:
    --------
    List of int:
        List of the GC shift position.
    int:
        Genome size
    """
    data.columns = ["chr", "start", "end", "val"]
    # Look for step and window_size
    step = data.start[1] - data.start[0]
    window = data.end[0] - data.start[0] + 1
    # Initialization
    genome_size = data.end[data.shape[0] - 1]
    previous_value = data.val[data.shape[0] - 1]
    diff = 1000
    pos_shift = []
    pos = 0
    # Search for GC skew shift on the GC skew
    n = data.shape[0]
    for i in range(n):
        current_value = data.val[i]
        # Search for local inversion of GC skew
        if previous_value * current_value < 0:
            # Check if global inversion or just a small one.
            alpha = 100
            if i < alpha:
                a = np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[0:i]]
                ) + np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[n - i : n]]
                )
                b = np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[i : i + alpha]]
                )
            elif i + alpha > n - 1:
                alpha = n - i - 1
                a = np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[i - alpha : i]]
                )
                b = np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[i:n]]
                ) + np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[0 : n - i]]
                )
            else:
                a = np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[i - alpha : i]]
                )
                b = np.sum(
                    [x / (abs(x) + 10 ** -12) for x in data.val[i : i + alpha]]
                )
            if a * b < 0:
                # Look for the middle of the inversion.
                # A new shift is consider if it's at more than
                # 300kb from the previous one
                curr_diff = abs(a + b)
                curr_pos = int(data.start[i] + int(window / 2) - 1)
                if curr_pos > pos + 200000 and pos != 0:
                    pos_shift.append(pos)
                    diff = curr_diff
                    pos = curr_pos
                if curr_diff < diff:
                    diff = curr_diff
                    pos = curr_pos
        previous_value = current_value
    pos_shift.append(pos)
    return pos_shift, genome_size


def oriter_main(
    gc_skew_file: str, pars_file: str, cov_file: Optional[str] = None
) -> Tuple[int, int, float]:
    """Main function to detect ori and ter and compute the ratio of coverage if
    a coverage file is provided. The position is based on the parS sites and the
    GC skew shift.

    Parameters
    ----------
    gc_skew_file : str
        Path to the gcskew bed file.
    pars_file : str
        Path to the pars bed file.
    cov_file : str
        Path to the coverage bed file.

    Returns
    -------
    int:
        Position of ori.
    int:
        Position of ter.
    float:
        Ratio of coverage between ori and ter.
    """
    # Extract data
    gc_skew_data = pd.read_csv(
        gc_skew_file, sep="\t", header=0, names=["chr", "start", "end", "val"]
    )
    pars_data = pd.read_csv(
        pars_file,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "type", "seq", "err"],
    )
    # Detect ori and ter
    ori, ter = detect_ori_ter(gc_skew_data, pars_data)
    # Compute ori/ter ratio if coverage provided.
    if cov_file is not None:
        cov_data = pd.read_csv(
            cov_file,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "val"],
        )
        ratio = compute_oriter_ratio(cov_data, ori, ter)

    return ori, ter, ratio


def remove_nmad(values: "numpy.ndarray", n: int = 3) -> "numpy.ndarray":
    """Function to remove values which are at different from the median more
    than n times the mad.
    """
    mad = np.median(np.absolute(values - np.median(values)))
    med = np.median(values)
    return [x for x in values if abs(x - med) <= mad * n]
