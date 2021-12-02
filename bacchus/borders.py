#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to compute borders and insulation score.

Functions:
    - compartments_sparse
"""


import bacchus.hic as bch
import bacchus.plot as bcp
import hicstuff.hicstuff as hcs
import numpy as np
import os
import scipy.signal as ss
import scipy.sparse as sp
import scipy.linalg as sl
from typing import List, Optional


def compartments_sparse(
    M: "scipy.sparse.cr_matrix",
    normalize: bool = True,
    plot_dir: Optional[str] = None,
    circular: bool = True,
    antidiagonal: bool = False,
):
    """A/B compartment analysis

    Performs a detrending of the power law followed by a PCA-based A/B
    compartment analysis on a sparse, normalized, single chromosome contact map.
    The results are two vectors whose values (negative or positive) should
    presumably correlate with the presence of 'active' vs. 'inert' chromatin.

    Parameters
    ----------
    M : array_like
        The input, normalized contact map. Must be a single chromosome. Values
        are assumed to be only the upper triangle of a symmetrix matrix.
    normalize : bool
        Whether to normalize the matrix beforehand.
    plot_dir : directory
        Directory to save plot. Required if none given, do not build plot.
    circular : bool
        Either the matrix is circualr or not.
    antidiagonal : True
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

    # Make matrix symetric (in case of upper triangle)
    M = M.tocsr()
    M = bch.sym(M)

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
        M = sp.coo_matrix(M)
        M.data = np.log2(M.data / dist_vals[abs(M.row - M.col)])
        M = np.rot90(M.todense()).T
        M = sp.coo_matrix(M)

    # Plot detrend matrix
    if plot_dir is not None:
        detrend_map_file = os.path.join(plot_dir, "detrend_map.png")
        bcp.map(
            M.toarray(),
            dpi=200,
            cmap="Reds",
            out_file=detrend_map_file,
            title="Detrend contact map",
        )

    # Compute correlation matrix on full matrix
    M = M.tocsr()
    M = hcs.corrcoef_sparse(M)
    M[np.isnan(M)] = 0.0

    # Plot correlation matrix
    if plot_dir is not None:
        correlation_map_file = os.path.join(plot_dir, "correlation_map.png")
        bcp.map_ratio(
            M,
            dpi=200,
            cmap="seismic",
            lim=1,
            out_file=correlation_map_file,
            ratio=True,
            title="Correlation contact map",
        )

    # Extract eigen vectors and eigen values
    [eigen_vals, pr_comp] = sl.eig(M)

    return pr_comp[:, 0], pr_comp[:, 1]


def get_relative_insulation(M: "numpy.ndarray", window: int, s: int):
    """Function to compute the relative insulation score depending on the window
    size and of one position. The method is based on the method described in
    https://doi.org/10.1093/nar/gky789 and implemented in R:
    https://github.com/ChenFengling/RHiCDB.

    Parameters
    ----------
    M : numpy.ndarray
        Extended dense matrix with size w.
    window : int
        Size of the half window (Look for contacts at a genomic range of window)
        used to compute the insulation score. 
    s : int
        Genomic coordiates where to compute the relative insulation score.

    Returns
    -------
    float:
        Relative insulation score at the genomic position between s and s + 1
        with the range window.
    """
    # Extract the matrix of interest. No problematic edge case are possible are
    # the matrix have been extended.
    N = M[s : s + 2 * window, s : s + 2 * window]
    # Compute the mean for each part of the matrix
    up = np.nanmean(np.triu(N[:window, :window]))
    down = np.nanmean(np.triu(N[window:, window:]))
    between = np.nanmean(N[:window, window:])
    # Compute the relative index
    return (up + down - between) / (up + down + between)


def get_ri_score(M: "numpy.ndarray", list_w: List[int]):
    """Function to get the relative insulation score on the whole matrix. It
    supposed the matrix is circular.

    Parameters
    ----------
    M : numpy.ndarray
        Matrix dense which should be circular to manage edge case.
    list_w : list of int
        List of half window size (Look for contacts at a genomic range of size
        window) to use in bin scale.

    Returns
    -------
    numpy.ndarray
        Vector of the relative insulation score for each bin of the matrix.
    """
    # Compute size of M and prepare the vector.
    n = len(M)
    relative_insulation_score = np.zeros(n)
    max_w = np.max(list_w)
    len_w = len(list_w)

    # Extend the matrix to avoid edge case using circular matrix property.
    M = bch.map_extend(M, max_w)
    # Compute the relative index score.
    for s in range(n):
        for w in list_w:
            # Correct s to correspond at the current w (correction for extended
            # matrix coordinates)
            pos = s + max_w - w
            # Compute the average relative insulation score
            relative_insulation_score[s] += (
                get_relative_insulation(M, w, pos) / len_w
            )
    return ri_score


def get_lri_score(ri_score: "numpy.ndarray"):
    """As the relative insulation score depends on the matrix and insulation
    score of the closed regions. It's necessary to defined it locally. This
    function allows to do it by removing the second enveloppe to the signal. The
    method is based on the method described in
    https://doi.org/10.1093/nar/gky789 and implemented in R:
    https://github.com/ChenFengling/RHiCDB.

    Parameters
    ----------
    ri_score : numpy.ndarray
        Vector of the relative insulation score along the whole genome.

    Returns
    -------
    numpy.ndarray:
        Vector of the local relative insulation score along the whole genome.
    List of numpy.ndarray:
        List of the vector of the first enveloppe, its index, the second
        enveloppe and its index.
    """

    # Compute the first and second envelope to compute the local LRI
    # First enveloppe
    first_enveloppe_index = ss.find_peaks([-x for x in ri_score])[0]
    first_enveloppe_index = np.concatenate(
        ([0], first_enveloppe_index, [len(ri_score) - 1])
    )
    first_enveloppe = [ri_score[i] for i in first_enveloppe_index]
    # Possible as the genome is circular
    a = np.mean([first_enveloppe[1], first_enveloppe[-2]])
    first_enveloppe[0] = a
    first_enveloppe[-1] = a

    # Second enveloppe
    second_enveloppe_index = ss.find_peaks([-x for x in first_enveloppe])[0]
    second_enveloppe_index = [
        first_enveloppe_index[i] for i in second_enveloppe_index
    ]
    second_enveloppe_index = np.concatenate(
        ([0], second_enveloppe_index, [len(ri_score) - 1])
    )
    second_enveloppe = [ri_score[i] for i in second_enveloppe_index]

    # Possible as the genome is circular
    a = np.mean([second_enveloppe[1], second_enveloppe[-2]])
    second_enveloppe[0] = a
    second_enveloppe[-1] = a

    # Remove second enveloppe to the relative insulation score to have a local
    # relative insulation score.
    i = 0
    lri_score = []
    for j, k in enumerate(second_enveloppe_index):
        while i < k:
            a = (second_enveloppe[j] - second_enveloppe[j - 1]) / (
                k - second_enveloppe_index[j - 1]
            )
            b = second_enveloppe[j] - a * k
            lri_score.append(ri_score[i] - (a * i + b))
            i += 1
        if i == k:
            lri_score.append(ri_score[i] - second_enveloppe[j])
            i += 1
    return (
        lri_score,
        [
            first_enveloppe,
            first_enveloppe_index,
            second_enveloppe,
            second_enveloppe_index,
        ],
    )


## TODO


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import hicstuff.commands as hcc
import hicstuff.io as hio
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import numpy as np
import scipy.signal


def plot(M, lri, final_borders, outfile):
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 14),
        gridspec_kw={"height_ratios": [4, 1], "width_ratios": [1]},
    )
    ax[0].imshow(M, cmap="Reds", vmax=np.percentile(M, 99))
    ax[0].scatter(
        final_borders,
        final_borders,
        edgecolors=None,
        color="yellow",
        s=15,
        marker="x",
    )
    # plt.colorbar()
    # ax[1].plot(ri)
    # cutoff = np.median(lri) + np.std(lri)
    # ax[1].axhline(y=cutoff, color="b", linestyle="dashed")
    # ax[1].plot(first_enveloppe_index, first_enveloppe)
    # ax[1].plot(second_enveloppe_index, second_enveloppe)
    ax[1].plot(np.arange(0, len(lri)) * 0.002, lri, c="r")
    ax[1].set_xlim([0, len(lri) * 0.002])
    for i in final_borders:
        ax[1].axvline(
            x=i * 0.002, color="black", linestyle="dashed", linewidth=0.5
        )
    plt.savefig(outfile)


def detect_final_borders(lri, list_w):
    peaks = scipy.signal.find_peaks(lri)[0]
    cutoff = np.median(lri) + np.std(lri)
    cutoff_id = []
    for i, v in enumerate(lri):
        if v > cutoff:
            cutoff_id.append(i)
    final_borders = []
    previous_border = -1000
    for i in peaks:
        if i in cutoff_id:
            if i - previous_border < list_w[0]:
                i = (i + previous_border) / 2
                final_borders[-1] = i
            else:
                final_borders.append(i)
            previous_border = i

    return final_borders


def main(M, list_w, out):
    matrix = copy.copy(M)
    mask = mask_white_line(matrix)
    matrix[mask, :] = np.random.sample(matrix.shape[0]) / 10 ** 6
    matrix[:, mask] = np.random.sample(matrix.shape[0])[0] / 10 ** 6
    matrix = np.corrcoef(matrix)
    matrix[mask, :] = np.nan
    matrix[:, mask] = np.nan
    ri = get_relative_insulation_curve(matrix, list_w)
    (
        lri,
        first_enveloppe_index,
        second_enveloppe,
        second_enveloppe_index,
    ) = compute_lri(ri)
    final_borders = detect_final_borders(lri, list_w)
    outfile = join(out, "in_vitro_R1_insulation_score.pdf")
    out_lri = join(out, "in_vitro_R1_insulation_score.bed")
    out_borders = join(out, "in_vitro_R1_borders.bed")
    plot(M, lri, final_borders, outfile)
    with open(out_lri, "w") as out:
        out.write("position\tscore\n")
        for i, value in enumerate(lri):
            out.write("{0}\t{1}\n".format(i * 2000, value))
    with open(out_borders, "w") as out:
        out.write("position\tscore\n")
        for i in final_borders:
            out.write("{0}\t{1}\n".format(i * 2000, lri[int(i)]))
    return lri


@click.command()
@click.argument("fragment_file", type=click.Path(exists=True))
@click.argument("matrix_file", type=click.Path(exists=True))
@click.option(
    "-b",
    "--binning",
    default="10kb",
    help="Size of the bins in bp, kb, Mb, Gb.",
)
@click.option(
    "-o", "--out_dir", help="output directory where to write the output fikes"
)
@click.option("-c", "--chrom-name", help="Name of the chromosome.")
@click.option(
    "-C",
    "--circular",
    default=False,
    is_flag=True,
    help="If enable compute compartments as a circular chromosome.",
)
@click.option(
    "-p",
    "--plot",
    default=False,
    is_flag=True,
    help="If enable display some plots.",
)
def main(
    fragment_file, matrix_file, binning, out_dir, chrom_name, circular, plot
):
    """ Main function to detect A/B compartment in a genome."""
    # Bin the matrix to the desire size
    bp_unit = False
    bin_str = binning.upper()
    try:
        # Subsample binning
        bin_size = int(bin_str)
    except ValueError:
        if re.match(r"^[0-9]+[KMG]?B[P]?$", bin_str):
            # Load positions from fragments list
            bin_size = hcc.parse_bin_str(bin_str)
            bp_unit = True
        else:
            print("Please provide an integer or basepair value for binning.")
            raise ValueError
    sparse_map, frags, _ = hio.flexible_hic_loader(
        matrix_file, fragments_file=fragment_file, quiet=True
    )
    # BINNING
    if bin_size > 1:
        if bp_unit:
            pos = frags.iloc[:, 2]
            binned_map, binned_pos = hcs.bin_bp_sparse(
                M=sparse_map, positions=pos, bin_len=bin_size
            )
            # Get bin numbers of chromosome starts
            binned_start = np.append(
                np.where(binned_pos == 0)[0], len(binned_pos)
            )
            # Get bin length of each chromosome
            num_binned = binned_start[1:] - binned_start[:-1]
            # Get unique chromosome names without losing original order
            # (numpy.unique sorts output)
            chr_names_idx = np.unique(frags.iloc[:, 1], return_index=True)[1]
            chr_names = [
                frags.iloc[index, 1] for index in sorted(chr_names_idx)
            ]
            binned_chrom = np.repeat(chr_names, num_binned)
            binned_frags = pd.DataFrame(
                {"chrom": binned_chrom, "start_pos": binned_pos[:, 0]}
            )
            binned_frags["end_pos"] = binned_frags.groupby("chrom")[
                "start_pos"
            ].shift(-1)
            chrom_ends = frags.groupby("chrom").end_pos.max()
            # Fill ends of chromosome bins with actual chromosome length
            for cn in chrom_ends.index:
                binned_frags.end_pos[
                    np.isnan(binned_frags.end_pos) & (binned_frags.chrom == cn)
                ] = chrom_ends[cn]

        else:
            # Note this is a basic binning procedure, chromosomes are
            # not taken into account -> last few fragments of a chrom
            # are merged with the first few of the next
            binned_map = hcs.bin_sparse(
                M=sparse_map, subsampling_factor=bin_size
            )
            if frags:
                binned_frags = frags.iloc[::bin_size, :]
                binned_frags = binned_frags.reset_index(drop=True)
                # Since matrix binning ignores chromosomes, we
                # have to do the same procedure with fragments
                # we just correct the coordinates to start at 0
                def shift_min(x):
                    try:
                        x[x == min(x)] = 0
                    except ValueError:
                        pass
                    return x

                binned_frags.start_pos = binned_frags.groupby(
                    "chrom", sort=False
                ).start_pos.apply(shift_min)
            else:
                binned_frags = frags

    else:
        binned_map = sparse_map
        binned_frags = frags

    # Build output files.
    os.makedirs(out_dir, exist_ok=True)
    cool_file = join(out_dir, "matrix_" + binning + ".cool")
    pca1_file = join(out_dir, "pca1.bed")
    pca2_file = join(out_dir, "pca2.bed")

    if plot:
        plot_dir = join(out_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = None

    # Save binned matrix in cool format
    hio.save_cool(cool_file, binned_map, binned_frags)

    pca1, pca2 = compartments_sparse(
        M=binned_map,
        normalize=True,
        plot=plot,
        plot_dir=plot_dir,
        circular=circular,
    )

    with open(pca1_file, "w") as out_1, open(pca2_file, "w") as out_2:
        for i in range(len(pca1)):
            out_1.write(
                "\t".join(
                    [
                        chrom_name,
                        str(i * bin_size),
                        str((i + 1) * bin_size),
                        str(pca1[i]),
                    ]
                )
                + "\n"
            )
            out_2.write(
                "\t".join(
                    [
                        chrom_name,
                        str(i * bin_size),
                        str((i + 1) * bin_size),
                        str(pca2[i]),
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
