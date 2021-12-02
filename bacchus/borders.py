#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General functions to compute borders and insulation score.

Functions:
    - insulation_score
    - borders
    - blob_score
    - ab_compartments
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import copy
import re
import hicstuff.hicstuff as hcs
import hicstuff.commands as hcc
import hicstuff.io as hio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from os.path import join
from scipy.linalg import eig


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def get_relative_insulation(M, w, s):
    """
    Function to compute the relative insulation score
    depending on the window size and of one position.
    The method is based on the method described in
    https://doi.org/10.1093/nar/gky789 and implemented
    in R: https://github.com/ChenFengling/RHiCDB.
    """
    n = len(M)
    up = []
    down = []
    between = []
    if s < w:
        N11 = M[n - w + s :, n - w + s :]
        N22 = M[: s + w, : s + w]
        N12 = M[n - w + s :, : s + w]
        N21 = M[: s + w, n - w + s :]
        N = np.concatenate(
            (
                np.concatenate((N11, N12), axis=1),
                np.concatenate((N21, N22), axis=1),
            ),
            axis=0,
        )

    elif n - s < w:
        N11 = M[s - w :, s - w :]
        N22 = M[: w - n + s, : w - n + s]
        N12 = M[s - w :, : w - n + s]
        N21 = M[: w - n + s, s - w :]
        N = np.concatenate(
            (
                np.concatenate((N11, N12), axis=1),
                np.concatenate((N21, N22), axis=1),
            ),
            axis=0,
        )
    else:
        N = M[s - w : s + w, s - w : s + w]

    for i in range(0, 2 * w):
        for j in range(i, 2 * w):
            if i < w:
                if j < w:
                    up.append(N[i, j])
                # elif j - i <= w:
                else:
                    between.append(N[i, j])
            else:
                down.append(N[i, j])
    up = np.nanmean(up)
    down = np.nanmean(down)
    between = np.nanmean(between)
    ri = (up + down - between) / (2 * (up + down + between))
    return ri


def get_average_relative_insulation(M, list_w, s):
    ari = 0
    for w in list_w:
        ari += get_relative_insulation(M, w, s)
    return ari / len(list_w)


def get_relative_insulation_curve(M, list_w):
    ri = []
    for s in range(len(M)):
        ri.append(get_average_relative_insulation(M, list_w, s))
    return ri


def get_relative_insulation_curve_simple(M, w):
    ri = []
    for s in range(len(M)):
        ri.append(get_relative_insulation(M, w, s))
    return ri


def mask_white_line(M, n_mads=3):
    """Function to put nan in the row/column where there are too much zeros."""
    matrix = M.copy()

    def mad(x):
        return ss.median_absolute_deviation(x, nan_policy="omit")

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


def compute_lri(ri):
    # Compute the first and second envelope to compute the local LRI
    # First enveloppe
    first_enveloppe_index = scipy.signal.find_peaks([-x for x in ri])[0]
    first_enveloppe_index = np.concatenate(
        ([0], first_enveloppe_index, [len(ri) - 1])
    )
    first_enveloppe = [ri[i] for i in first_enveloppe_index]
    # Possible as the genome is circular
    a = np.mean([first_enveloppe[1], first_enveloppe[-2]])
    first_enveloppe[0] = a
    first_enveloppe[-1] = a

    # Second enveloppe
    second_enveloppe_index = scipy.signal.find_peaks(
        [-x for x in first_enveloppe]
    )[0]
    second_enveloppe_index = [
        first_enveloppe_index[i] for i in second_enveloppe_index
    ]
    second_enveloppe_index = np.concatenate(
        ([0], second_enveloppe_index, [len(ri) - 1])
    )
    second_enveloppe = [ri[i] for i in second_enveloppe_index]
    # Possible as the genome is circular
    a = np.mean([second_enveloppe[1], second_enveloppe[-2]])
    second_enveloppe[0] = a
    second_enveloppe[-1] = a

    # Remove second enveloppe to ri to have a local ri
    i = 0
    lri = []
    for j, k in enumerate(second_enveloppe_index):
        while i < k:
            a = (second_enveloppe[j] - second_enveloppe[j - 1]) / (
                k - second_enveloppe_index[j - 1]
            )
            b = second_enveloppe[j] - a * k
            lri.append(ri[i] - (a * i + b))
            i += 1
        if i == k:
            lri.append(ri[i] - second_enveloppe[j])
            i += 1
    return lri, first_enveloppe_index, second_enveloppe, second_enveloppe_index


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


def compartments_sparse(
    M, normalize=True, plot=False, plot_dir=None, circular=False
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
    plot : bool
        An optionale boolean to display plot or not.
    plot_dir : directory
        Directory to save plot. Required if plot enabled.

    Returns
    -------
    numpy.ndarray
        An array containing the first principal component.
    numpy.ndarray
        An array containing the second principal component.
    """
    # Normalize the matrix
    if normalize:
        N = hcs.normalize_sparse(M, norm="SCN")
    else:
        N = copy.copy(M)
    N = N.tocoo()
    n = N.shape[0]

    # Detrend by the distance law
    dist_vals = np.array([np.average(N.diagonal(j)) for j in range(n)])
    N.data = N.data / dist_vals[abs(N.row - N.col)]

    # Detrend secondary diagonal
    L = np.rot90(N.todense())
    dist_vals = np.array([np.average(L.diagonal(j)) for j in range(n)])
    L = scipy.sparse.coo_matrix(L)
    L.data = np.log2(L.data / dist_vals[abs(L.row - L.col)])
    N = np.rot90(L.todense()).T
    N = scipy.sparse.coo_matrix(N)

    # Make matrix symmetric (in case of upper triangle)
    N = N.tocsr()
    if (abs(N - N.T) > 1e-10).nnz != 0:
        N = N + N.T
        N.setdiag(N.diagonal() / 2)
        N.eliminate_zeros()

    # Plot detrend matrix
    if plot:
        detrend_map_file = join(plot_dir, "detrend_map.png")
        M = N.toarray()
        plt.imshow(M, cmap="seismic")
        plt.colorbar()
        plt.savefig(detrend_map_file)
        plt.clf()

    # Compute correlation matrix on full matrix
    N = N.tocsr()
    N = hcs.corrcoef_sparse(N)
    N[np.isnan(N)] = 0.0

    # Plot correlation matrix
    if plot:
        correlation_map_file = join(plot_dir, "correlation_map.png")
        plt.imshow(N, cmap="seismic", vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(correlation_map_file)
        plt.clf()

    # Extract eigen vectors and eigen values
    [eigen_vals, pr_comp] = eig(N)

    return pr_comp[:, 0], pr_comp[:, 1]


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
