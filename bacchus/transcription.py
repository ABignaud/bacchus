#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to do transcriptional analysis based on HiC contac map.

Functions:
    - extract_window_matrix
    - extract_window_track
    - pileup_genes
"""


import bacchus.hic as bch
import numpy as np
import random
from typing import List, Optional


def extract_window_matrix(
    clr: "cooler.api.Cooler",
    chrom: str,
    pos: int,
    binning: int,
    window: int,
    circular: bool = True,
    flip: bool = False,
) -> "numpy.ndarray":
    """Fetch contact map around a given position on a chromosome.

    Parameters
    ----------
    clr : cooler.api.Cooler
        Cooler object of the contact map.
    chrom : str
        Chromosome name of the region to extract.
    pos : int
        Position in base pair of the region to extract.
    binning: int
        Size of the bins in base pair.
    window : int
        Window size to extract in base pair. (it would take the size of the
        window from each border of the postion).
    circular : bool
        Either the chromosome is circular or not.
    flip : bool
        Either to flip the matrix or not.

    Returns
    -------
    np.ndarray:
        The extracted window.
    """
    # Fetch chrom size.
    chrom_size = clr.chromsizes[chrom]
    # Sanity check of the size of the chromosomes.
    if chrom_size < 2 * window + binning:
        mat_win = np.zeros(
            (2 * window // binning + 1, 2 * window // binning + 1)
        )
        mat_win[:] = np.nan
        return mat_win

    # Compute start and end of the region to fetch.
    start = round((pos - window) / binning - 1 / (2 * binning)) * binning
    end = round((pos + window) / binning + 1 - 1 / (2 * binning)) * binning
    if (start < 0) or (end > chrom_size):
        border = True
    else:
        border = False

    # Fetch the region
    if border:
        # If cicrcular concatenate the borders.
        if circular:
            end1 = (chrom_size // binning) * binning
            start2 = 0
            if start < 0:
                start1 = chrom_size + start
                end2 = end
            else:
                start1 = start
                end2 = end - chrom_size
            # Concatenate both extremities of the matrix.
            mat_win = np.concatenate(
                (
                    np.concatenate(
                        (
                            clr.matrix(balance=True).fetch(
                                f"{chrom}:{start1}-{end1}",
                                f"{chrom}:{start1}-{end1}",
                            ),
                            clr.matrix(balance=True).fetch(
                                f"{chrom}:{start1}-{end1}",
                                f"{chrom}:{start2}-{end2}",
                            ),
                        ),
                        axis=1,
                    ),
                    np.concatenate(
                        (
                            clr.matrix(balance=True).fetch(
                                f"{chrom}:{start2}-{end2}",
                                f"{chrom}:{start1}-{end1}",
                            ),
                            clr.matrix(balance=True).fetch(
                                f"{chrom}:{start2}-{end2}",
                                f"{chrom}:{start2}-{end2}",
                            ),
                        ),
                        axis=1,
                    ),
                ),
                axis=0,
            )

        # Else nan padding.
        else:
            if start < 0:
                pad_with = (-start, 0)
                start = 0
            else:
                pad_with = (0, end - chrom_size)
                end = chrom_size
            mat_win = np.pad(
                clr.matrix(balance=True).fetch(f"{chrom}:{start}-{end}"),
                pad_width=pad_with,
                mode="constant",
                constant_values=np.nan,
            )
    else:
        mat_win = clr.matrix(balance=True).fetch(f"{chrom}:{start}-{end}")

    # Flip the region if required.
    if flip:
        mat_win = np.flip(mat_win)

    return mat_win


def extract_window_track(
    track: "numpy.ndarray",
    chrom_start_size: dict,
    chrom: str,
    pos: int,
    window: int,
    circular: bool = True,
    flip: bool = False,
) -> "numpy.ndarray":
    """Fetch contact map around a given position on a chromosome.

    Parameters
    ----------
    track : np.ndarray
        Track of genomic assay.
    chrom_start_size : dict
        Start positions of each chromosome from the previous array and their
        length.
    chrom : str
        Chromosome name of the region to extract.
    pos : int
        Position in bin of the region to extract.
    window : int
        Window size to extract in bin. (it would take the size of the window
        from each border of the postion).
    circular : bool
        Either the chromosome is circular or not.
    flip : bool
        Either to flip the matrix or not.

    Returns
    -------
    np.ndarray:
        The extracted window.
    """
    # Fetch chrom size.
    track_start = chrom_start_size[chrom]["start"]
    track_end = track_start + chrom_start_size[chrom]["length"]
    track_chrom = track[track_start:track_end]
    # Sanity check of the size of the chromosomes.
    if track_end - track_start < 2 * window + 1:
        track_win = np.zeros((2 * window + 1))
        track_win[:] = np.nan
        return track_win

    if circular:
        track_chrom = np.pad(
            track_chrom, pad_width=(window, window), mode="wrap"
        )
    else:
        track_chrom = np.pad(
            track_chrom,
            pad_width=(window, window),
            mode="constant",
            constant_values=np.nan,
        )

    # Compute start and end of the region to fetch.
    start = pos
    end = pos + 2 * window + 1

    # Fetch the region
    track_win = track_chrom[start:end]

    # Flip the region if required.
    if flip:
        track_win = np.flip(track_win)

    return track_win


def pileup_genes(
    clr: "cooler.api.Cooler",
    annotation: "pd.DataFrame",
    rna: "np.ndarray",
    chrom_start_size: dict,
    window_size: int = 20000,
    binning: int = 500,
    threshold: float = 20.0,
    neg: str = "non-transcribed",
    tu_length: int = 3000,
    operation: str = "mean",
    circular: bool = True,
) -> List["np.ndarray"]:
    """Function to do the pileup of the genes based on a rna threshold category.

    Parameters
    ----------
    clr : cooler.api.Cooler
        Contact map stored in a cooler object.
    annotation : pd.DataFrame
        Table with gene annotation and rna rpkm values.
    rna : np.ndarray
        Vector of the RNA values in CPM binned at the same size as the matrix.
    chrom_start_size : dict
        Start positions of each chromosome from the previous array and their
        length.
    window_size : int
        Size of the pileup in base pair. [Default: 20000].
    binning : int
        Binning size of the matrix. [Default: 500].
    threshold : float
        RPKM percentage of genes to considered as expressed. [Default: 20]
    neg : str
        Which type of data to use as control. Either 'non-transcribed' (pileup
        of genes below the threshold), 'detrend' (average p(s) matrix),
        'random-neighbor' (random window close to the one taken).
        [Default: 'non-transcribed'].
    tu_length : int
        Size in base pair between two transcribed genes to consider them from
        different transcription units. It will only take first gene in each
        transcription unit.
    operation : str
        Operation to pileup either 'mean' or 'median'. [Default: 'mean'].
    circular : bool
        Either the chromosomes are circular or not. [Default: True]

    Returns
    -------
    np.array:
        Final pileup ratio of the genes.
    """

    # Compute the window_size in bins
    w_bin = window_size // binning

    # Compute threshold abse on genes percentile.
    threshold_value = np.nanpercentile(annotation.rpkm, 100 - threshold)
    print(
        f"{threshold}% of the genes have a RPKM value superior to {threshold_value}."
    )

    # Prepare tables for output, two tables each for gene superior or inferior
    # to the theshold
    pattern_windows_neg = np.zeros(
        ((len(annotation), w_bin * 2 + 1, w_bin * 2 + 1))
    )
    rna_windows_neg = np.zeros(((len(annotation), w_bin * 2 + 1)))
    pattern_windows_pos = np.zeros(
        ((len(annotation), w_bin * 2 + 1, w_bin * 2 + 1))
    )
    rna_windows_pos = np.zeros(((len(annotation), w_bin * 2 + 1)))
    n = len(annotation)
    n_pos = 0
    state = "None"
    pos_state = 0

    # Iterates on genes
    for i in range(len(annotation)):

        # Extract genes data
        chrom = annotation.loc[i, "chr"]
        pos = annotation.loc[i, "tss"]
        flip = True if annotation.loc[i, "strand"] == "-" else False
        rpkm = annotation.loc[i, "rpkm"]

        if (neg == "random-neighbor") and (rpkm >= threshold_value):
            mat_win_neg_all = np.zeros((10, w_bin * 2 + 1, w_bin * 2 + 1))
            rna_win_neg_all = np.zeros((10, w_bin * 2 + 1))
            for j in range(10):
                if pos < window_size:
                    pos_neg = random.randint(0, pos + 2 * window_size)
                elif pos + window_size > clr.chromsizes[chrom]:
                    pos_neg = random.randint(
                        clr.chromsizes[chrom] - 2 * window_size,
                        clr.chromsizes[chrom],
                    )
                else:
                    pos_neg = random.randint(
                        pos - window_size, pos + window_size
                    )
                mat_win_neg_all[j] = extract_window_matrix(
                    clr=clr,
                    chrom=chrom,
                    pos=pos_neg,
                    binning=binning,
                    window=window_size,
                    circular=circular,
                    flip=flip,
                )
                rna_win_neg_all[j] = extract_window_track(
                    track=rna,
                    chrom_start_size=chrom_start_size,
                    chrom=chrom,
                    pos=pos_neg // binning,
                    window=w_bin,
                    circular=circular,
                    flip=flip,
                )
            mat_win_neg = np.apply_along_axis(np.nanmean, 0, mat_win_neg_all)
            rna_win_neg = np.apply_along_axis(np.nanmean, 0, rna_win_neg_all)

        # Extract matrix_window
        if ((neg == "random-neighbor") and (rpkm >= threshold_value)) or (
            neg == "non-transcribed"
        ):
            mat_win = extract_window_matrix(
                clr=clr,
                chrom=chrom,
                pos=pos,
                binning=binning,
                window=window_size,
                circular=circular,
                flip=flip,
            )
            rna_win = extract_window_track(
                track=rna,
                chrom_start_size=chrom_start_size,
                chrom=chrom,
                pos=pos // binning,
                window=w_bin,
                circular=circular,
                flip=flip,
            )

        # Add the matrix in the positive pileup if first gene expressed.
        if rpkm >= threshold_value:
            if neg == "non-transcribed":
                pattern_windows_neg[i][:] = np.nan
                rna_windows_neg[i][:] = np.nan
            if flip:
                # Reverse case: If in the same transcription unit, remove the
                # previous one and write the new one instead.
                if state == "reverse" and pos - pos_state <= tu_length:
                    pattern_windows_pos[i - 1][:] = np.nan
                    rna_windows_pos[i - 1][:] = np.nan
                    pattern_windows_pos[i][:] = mat_win
                    rna_windows_pos[i][:] = rna_win
                    if neg == "random-neighbor":
                        pattern_windows_neg[i - 1][:] = np.nan
                        rna_windows_neg[i - 1][:] = np.nan
                        pattern_windows_neg[i][:] = mat_win_neg
                        rna_windows_neg[i][:] = rna_win_neg
                else:
                    n_pos += 1
                    pattern_windows_pos[i][:] = mat_win
                    rna_windows_pos[i][:] = rna_win
                    if neg == "random-neighbor":
                        pattern_windows_neg[i][:] = mat_win_neg
                        rna_windows_neg[i][:] = rna_win_neg
                state = "reverse"

            else:
                # Forward case: Do not write anything if one gene before.
                if state != "forward" or pos - pos_state > tu_length:
                    n_pos += 1
                    pattern_windows_pos[i][:] = mat_win
                    rna_windows_pos[i][:] = rna_win
                    if neg == "random-neighbor":
                        pattern_windows_neg[i][:] = mat_win_neg
                        rna_windows_neg[i][:] = rna_win_neg
                else:
                    pattern_windows_pos[i][:] = np.nan
                    rna_windows_pos[i][:] = np.nan
                    if neg == "random-neighbor":
                        pattern_windows_neg[i][:] = np.nan
                        rna_windows_neg[i][:] = np.nan
                state = "forward"
            pos_state = pos

        # If no expression add it in the negative pileup.
        elif neg == "non-transcribed":
            pattern_windows_neg[i] = mat_win
            rna_windows_neg[i] = rna_win
            pattern_windows_pos[i][:] = np.nan
            rna_windows_pos[i][:] = np.nan
        else:
            pattern_windows_neg[i][:] = np.nan
            rna_windows_neg[i][:] = np.nan
            pattern_windows_pos[i][:] = np.nan
            rna_windows_pos[i][:] = np.nan

    print(f"Number of genes transcribed: {n_pos}")
    print(f"Total genes: {n}")

    # Apply either mean or median on the windows
    if operation == "mean":
        pileup_pos = np.apply_along_axis(np.nanmean, 0, pattern_windows_pos)
        rna_pileup_pos = np.apply_along_axis(np.nanmean, 0, rna_windows_pos)
        if neg == "detrend":
            pileup_neg = np.nan
            rna_pileup_neg = np.nanmean(rna)
        else:
            pileup_neg = np.apply_along_axis(np.nanmean, 0, pattern_windows_neg)
            rna_pileup_neg = np.apply_along_axis(np.nanmean, 0, rna_windows_neg)

    elif operation == "median":
        pileup_pos = np.apply_along_axis(np.nanmedian, 0, pattern_windows_pos)
        rna_pileup_pos = np.apply_along_axis(np.nanmedian, 0, rna_windows_pos)
        if neg == "detrend":
            pileup_neg = np.nan
            rna_pileup_neg = np.nanmedian(rna)
        else:
            pileup_neg = np.apply_along_axis(
                np.nanmedian, 0, pattern_windows_neg
            )
            rna_pileup_neg = np.apply_along_axis(
                np.nanmedian, 0, rna_windows_neg
            )

    return rna_pileup_pos, rna_pileup_neg, pileup_pos, pileup_neg
