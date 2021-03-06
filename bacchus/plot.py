#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General plot functions to display contact maps and others genomics data
around bacterial HiC experiments analysis.

Functions:
    - antidiagonal_plot
    - antidiagonal_scalogram
    - contact_map
    - contact_map_ratio
    - hicreppy_plot
    - parse_axis_str
    - pileup_plot
"""


import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from typing import List, Optional


def antidiagonal_plot(
    values: "numpy.ndarray",
    axis: str = "Mb",
    binning: int = 1,
    dpi: int = 100,
    ori_pos: Optional[int] = None,
    out_file: Optional[str] = None,
    pars_pos: Optional[List[int]] = None,
    title: Optional[str] = None,
):
    """Function to plot the antidiagonal strength of one bacteria.

    Parameters
    ----------
    values : numpy.ndarray
        Vector of the strength of the antidaigonals.
    axis : str
        Unit to use for x axis. 'bp', 'kb', 'Mb'.
    binning : int
        Size in base pair of the bins.
    dpi : int
        Final dpi of the plot.
    ori_pos : int
        Position of the ori in base pair. Binning is necessary.
    out_file : str
        Path to save the plot if one given.
    pars_pos : list of int
        List of the positions of the parS sites in base pair. Ori position is
        necessary.
    title : str
        Name of the plot if one given.
    """
    # Defines the scaling factor for x axis.
    if binning > 1:
        scaling_factor = parse_axis_str(axis)
    else:
        scaling_factor = 1.0
        axis = "bin"

    # Define figures.
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=dpi)

    # Define size of the vector and x axis. The 0.5 is added as we have twice
    # the numbers of bins.
    size = len(values)

    if ori_pos is not None and binning != 1:
        # Build x axis centered at the origin
        x = np.arange(-size // 2, size // 2) * binning * scaling_factor * 0.5
        ax.axvline(
            x=0, linewidth=1, color="black", linestyle="dashed", label="ori"
        )
        # Defined ori position at zero and reorder values accordingly.
        ori_pos_val_coor = 2 * ori_pos // binning
        if ori_pos_val_coor > size // 2:
            start = ori_pos_val_coor - size // 2
        else:
            start = ori_pos_val_coor + size // 2
        values = np.concatenate((values[start:], values[:start]))
        # If parS positions given, correct them and display thier positions.
        if pars_pos is not None:
            for pars in pars_pos:
                pars = (pars - ori_pos) * scaling_factor
                if pars > (size // 4) * binning * scaling_factor:
                    pars -= (size // 2) * binning * scaling_factor
                elif pars < (-size // 4) * binning * scaling_factor:
                    pars += (size // 2) * binning * scaling_factor
                pars_line = ax.axvline(
                    x=pars,
                    linewidth=1,
                    color="red",
                    linestyle="dashed",
                )
            pars_line.set_label("parS")

    else:
        x = np.arange(size) * binning * scaling_factor * 0.5
        # Display parS sites.
        if pars_pos is not None:
            for pars in pars_pos:
                pars_line = ax.axvline(
                    x=pars * scaling_factor,
                    linewidth=1,
                    color="red",
                    linestyle="dashed",
                )
            pars_line.set_label("parS")

    # Plot the antidiagonals strength.
    ax.plot(x, values, linewidth=1, color="#1f78b4")

    # Legend
    ax.set_xlabel(f"Genomic coordinates {axis:s}", fontsize=16)
    ax.set_ylabel("Antidiagonal strength", fontsize=16)
    ax.tick_params(size=16)
    if ori_pos is not None or pars_pos is not None:
        ax.legend()
    if title is not None:
        ax.set_title(title, size=18)

    # Savefig if necessary
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)


def antidiagonal_scalogram(
    values: "numpy.ndarray",
    axis: str = "Mb",
    binning: int = 1,
    dpi: int = 100,
    labels: Optional[List[str]] = None,
    ori_pos: Optional[int] = None,
    out_file: Optional[str] = None,
    pars_pos: Optional[List[int]] = None,
    title: Optional[str] = None,
):
    """Function to plot the antidiagonal scalogram strength of one bacteria.

    Parameters
    ----------
    values : numpy.ndarray
        List of vector of the strength of the antidaigonals.
    axis : str
        Unit to use for x axis. 'bp', 'kb', 'Mb'.
    binning : int
        Size in base pair of the bins.
    dpi : int
        Final dpi of the plot.
    labels : list of str
        Labels to give to each curve.
    ori_pos : int
        Position of the ori in base pair.
    out_file : str
        Path to save th eplot if one given.
    pars_pos : list of int
        List of the positions of the parS sites in base pair.
    title : str
        Name of the plot if one given.
    """
    # Defines the scaling factor for x axis.
    if binning > 1:
        scaling_factor = parse_axis_str(axis)
    else:
        scaling_factor = 1.0
        axis = "bin"

    # Define color palette
    color = [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ]
    if len(values) > 12:
        color = color * (len(values) // 12 + 1)

    # Define figures.
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=dpi)

    # Define size of the vector and x axis. The 0.5 is added as we have twice
    # the numbers of bins.
    size = len(values[0])

    if ori_pos is not None and binning != 1:
        # Build x axis centered at the origin
        x = np.arange(-size // 2, size // 2) * binning * scaling_factor * 0.5
        ax.axvline(x=0, linewidth=1, color="black", linestyle="dashed")
        # Defined ori position at zero and reorder values accordingly.
        ori_pos_val_coor = 2 * ori_pos // binning
        if ori_pos_val_coor > size // 2:
            start = ori_pos_val_coor - size // 2
        else:
            start = ori_pos_val_coor + size // 2
        # If parS positions given, correct them and display thier positions.
        if pars_pos is not None:
            for pars in pars_pos:
                pars = (pars - ori_pos) * scaling_factor
                if pars > (size // 4) * binning * scaling_factor:
                    pars -= (size // 2) * binning * scaling_factor
                elif pars < (-size // 4) * binning * scaling_factor:
                    pars += (size // 2) * binning * scaling_factor
                pars_line = ax.axvline(
                    x=pars,
                    linewidth=1,
                    color="red",
                    linestyle="dashed",
                )
            # pars_line.set_label("parS")

    else:
        x = np.arange(size) * binning * scaling_factor * 0.5
        start = 0
        # Display parS sites.
        if pars_pos is not None:
            for pars in pars_pos:
                pars_line = ax.axvline(
                    x=pars * scaling_factor,
                    linewidth=1,
                    color="red",
                    linestyle="dashed",
                )
            # pars_line.set_label("parS")

    # Check labels:
    if labels is None:
        labels = np.zeros((len(values)))
        legend = False
    else:
        legend = True

    # Plot the antidiagonals strength.
    for i in range(len(values)):
        y = np.concatenate((values[i][start:], values[i][:start]))
        ax.plot(x, y, linewidth=0.7, color=color[i], label=labels[i])

    # Legend
    ax.set_xlabel(f"Genomic coordinates {axis:s}", fontsize=16)
    ax.set_ylabel("Antidiagonal strength", fontsize=16)
    ax.tick_params(size=16)
    if title is not None:
        ax.set_title(title, size=18)
    if legend:
        ax.legend(loc="upper right", fontsize=8)

    # Savefig if necessary
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)


def contact_map(
    mat: "numpy.ndarray",
    axis: str = "kb",
    binning: int = 1,
    cmap: str = "Reds",
    dpi: int = 300,
    end: int = 0,
    out_file: Optional[str] = None,
    start: int = 0,
    title: Optional[str] = None,
    vmax: float = 99,
):
    """Function to plot a matrix or a region of this matrix.

    Parameters
    ----------

    mat : numpy.ndarray
        Matrix to use for the plot.
    axis : str
        Either 'bin', 'bp', 'kb' or 'Mb'. Graduation on the x and y axis. The
        default is 'bin' binning is set, then it's 'kb'.
    binning : int
        Binning size in bp of the matrix. If none given, do not translate bins
        in bp genomic coordiantes.
    cmap : str
        Colormap used. [Default: 'Reds']
    dpi : int
        Dpi used. [Default: 300]
    end : int
        End position in bins of the region to plot. If none given, it will plot
        until the end of the matrix.
    out_file : str
        Path of the output file to save the plot. The extension have to match
        matplotlib.pyplot.savefig extensions. If none given, don't save the
        figure.
    start : int
        Start position in bins of the region plot. If none given, it will start
        with the beginning of the matrix. [Default: 0]
    title : str
        Title to put on the plot. If none given, do not put any title.
    vmax : float
        Value of the percentile used for the clorscale.

    TODO: adapt it for multiple chromosomes maps
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=dpi)

    # Axis values
    if binning > 1:
        scaling_factor = parse_axis_str(axis) * binning
    else:
        scaling_factor = 1.0
        axis = "bin"

    # No end values given.
    if end == 0:
        end = len(mat)

    # Display plots
    im = ax.imshow(
        mat[start:end, start:end],
        cmap=cmap,
        vmin=0,
        vmax=np.nanpercentile(mat, vmax),
        extent=(
            start * scaling_factor,
            end * scaling_factor,
            end * scaling_factor,
            start * scaling_factor,
        ),
    )

    # Legend
    ax.set_xlabel(f"Genomic coordinates ({axis:s})", fontsize=16)
    ax.set_ylabel(f"Genomic coordinates ({axis:s})", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)

    # Title
    if title is not None:
        ax.set_title(title, size=18)

    # Colorbar
    cbar = plt.colorbar(im, shrink=0.33, anchor=(0, 0.5))
    cbar.ax.tick_params(labelsize=16)

    # Savefig
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")


def contact_map_ratio(
    mat1: "numpy.ndarray",
    mat2: Optional["numpy.ndarray"] = None,
    axis: str = "kb",
    binning: int = 1,
    cmap: str = "seismic",
    dpi: int = 300,
    end: int = 0,
    lim: float = 2,
    out_file: Optional[str] = None,
    ratio: bool = False,
    start: int = 0,
    title: Optional[str] = None,
):
    """Function to plot a log2 ratio of two matrices or a region of it. A
    already computed serpentine log ratio could be given instead and displayed.
    If two matrices are given with the serpentine option, it will displayed the
    first one in the upper trinagle and the second if the lower triangle.

    Parameters
    ----------

    mat1 : numpy.ndarray
        First matrix to use for the log ratio plot. Positive value on the
        colorscale. If ratio set to True it will be the final matrix (upper
        right if a second is given).
    mat2 : numpy.ndarray
        Second matrix to use for the log ratio plot. Negative value on the
        colorscale. If ratio set to True it will be the lower down matrix if
        one is given.
    axis : str
        Either 'bin', 'bp', 'kb' or 'Mb'. Graduation on the x and y axis. The
        default is 'bin' binning is set, then it's 'kb'.
    binning : int
        Binning size in bp of the matrix. If none given, do not translate bins
        in bp genomic coordiantes.
    cmap : str
        Colormap used. [Default: 'seismic']
    dpi : int
        Dpi used. [Default: 300]
    end : int
        End position in bins of the region to plot. If none given, it will plot
        until the end of the matrix.
    lim : float
        Limits of the colorscale (max and minus will be the negative).
    ratio : bool
        Either final ratios are given or not.
    out_file : str
        Path of the output file to save the plot. The extension have to match
        matplotlib.pyplot.savefig extensions. If none given, don't save the
        figure.
    start : int
        Start position in bins of the region plot. If none given, it will start
        with the beginning of the matrix. [Default: 0]
    title : str
        Title to put on the plot. If none given, do not put any title.

    TODO: adapt it for multiple chromosomes maps
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)

    # Axis values
    if binning > 1:
        scaling_factor = parse_axis_str(axis) * binning
    else:
        scaling_factor = 1.0
        axis = "bin"

    # No end values given.
    if end == 0:
        end = len(mat1)

    # Compute the matrices
    if ratio:
        if mat2 is not None:
            mat = np.tril(mat2, k=-1) + np.triu(mat1)
        else:
            mat = mat1
    else:
        mat = np.log2(mat1) - np.log2(mat2)

    # Display plots
    im = ax.imshow(
        mat[start:end, start:end],
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        extent=(
            start * scaling_factor,
            end * scaling_factor,
            end * scaling_factor,
            start * scaling_factor,
        ),
    )

    # Legend
    ax.set_xlabel(f"Genomic coordinates {axis:s}", fontsize=16)
    ax.set_ylabel(f"Genomic coordinates {axis:s}", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)

    # Title
    if title is not None:
        ax.set_title(title, size=18)

    # Colorbar
    cbar = plt.colorbar(im, shrink=0.33, anchor=(0, 0.5))
    cbar.ax.tick_params(labelsize=16)

    # Savefig
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)


def hicreppy_plot(
    data: "numpy.ndarray", labels: Optional[List[str]], out_file: Optional[str]
):
    """Function to plot the correlation matrix from hicreppy.

    Parameters
    ----------
    data : numpy.ndarray
        Matrix of the stratum corelation coefficient from hicreppy.
    labels : list of str
        List of the string to use as labels.
    out_file : str
        Path were to write the output plots. Extension should be compatible with
        savefig.
    """
    # If no labels were given, just create a list of index (1-based).
    if labels is None:
        labels = [str(x) for x in np.arange(1, len(data) + 1)]

    # Plot the correlation matrix with seaborn.
    sns.set(font_scale=1)
    ax = sns.clustermap(
        data=data,
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        row_cluster=True,
        col_cluster=True,
        tree_kws={"linewidths": 3},
        annot=True,
    )
    plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    fig = ax.fig
    fig.tight_layout(rect=[0, 0, 1, 1], pad=2)

    # Save figure if an outfile is given.
    if out_file is not None:
        plt.savefig(out_file, dpi=100)


def parse_axis_str(axis: str) -> float:
    """Axis string parsing

    Take a basepair unit string as input and converts it into corresponding
    basepair values.

    Parameters
    ----------
    axis : str
        A basepair unit (e.g. 150KB). Unit can be bp, BP, kB, Mb, GB...

    Returns
    -------
    float:
        The scaling factor to use for the plot to convert in the right unit.

    Example
    -------
    >>> parse_axis_str("KB")
    0.001
    >>> parse_axis_str("mb")
    1e-06

    """
    axis = axis.upper()
    binsuffix = {"B": 1, "K": 1000, "M": 1e6, "G": 1e9}
    unit_pos = re.search(r"[KMG]?B[P]?$", axis).start()
    bp_unit = axis[unit_pos:]
    # Extract unit and use the according bin at the scaling factor.
    scaling_factor = 1 / binsuffix[bp_unit[0]]
    return scaling_factor


def pileup_plot(
    pileup: "numpy.ndarray",
    pileup_control: Optional["numpy_ndarray"] = None,
    gen_tracks: Optional[List["numpy_ndarray"]] = None,
    gen_tracks_control: Optional[List["numpy_ndarray"]] = None,
    binning: int = 1,
    window: int = 0,
    ratio: str = "diff",
    out_file: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = 100,
):
    """Function to plot pileup of genes.

    Parameters
    ----------
    pileup : numpy.ndarray
        Pileup contact map.
    pileup_control : numpy.ndarray
        Control for the pileup contact map. Should have the same dimension as
        the pileup.
    gen_tracks : list of numpy.ndarray
        List of genomic tracks to plot under the pileup.
    gen_tracks_control :
        List of control genomic tracks to plot under the pileup. Should have the
        same dimension as the genomics tracks.
    binning : int
        Size of the bins in base pair. [Default: 1]
    windwow : int
        Size of the window to plot in base pair. Need the binning value. If 0
        plot the whole matrix. [Default: 0]
    ratio : str
        Either diff or log. The ratio done between pileup of interest and
        control pileup (difference or log ratio).
    out_file : file
        Path were to write the output plots. Extension should be compatible with
        savefig.
    title : str
        Title of the plot.
    dpi : int
        Dpi to plot the figure.
    """

    # If one control is given make the the difference.
    if pileup_control is not None:
        if ratio == "diff":
            pileup = pileup - pileup_control
        elif ratio == "log":
            pileup = np.log10(pileup) - np.log10(pileup_control)
    if gen_tracks is not None and gen_tracks_control is not None:
        if ratio == "diff" or ratio == "log":
            for i in range(len(gen_tracks)):
                gen_tracks[i] = gen_tracks[i] - gen_tracks_control[i]
        # elif ratio == "log":
        #     for i in range(len(gen_tracks)):
        #         gen_tracks[i] = np.log10(gen_tracks[i]) - np.log10(
        #             gen_tracks_control[i]
        #         )

    # Set the window size
    n = np.shape(pileup)[0]
    w_bin = window // binning
    if binning != 1:
        ax_kb = 1000
    else:
        ax_kb = 1
    if window == 0 or w_bin >= n // 2:
        window_plot = (n // 2) * binning / ax_kb
    elif w_bin < n // 2:
        window_plot = window / ax_kb
        start = n // 2 - w_bin
        end = n // 2 + w_bin + 1
        pileup = pileup[start:end, start:end]
        if gen_tracks is not None:
            for i in range(len(gen_tracks)):
                gen_tracks[i] = gen_tracks[i][start:end]

    # Plot gen tracks if one given.
    if gen_tracks is not None:
        fig, ax = plt.subplots(
            2, 1, figsize=(8, 13), gridspec_kw={"height_ratios": [7, 3]}
        )
        ax[1].axvline(
            0, color="black", linestyle="dashed", linewidth=1.5, alpha=0.4
        )
        ax[1].tick_params(axis="both", labelsize=14)
        for i in range(len(gen_tracks)):
            ax[1].plot(
                np.arange(
                    -window_plot,
                    window_plot + (1 * binning / ax_kb),
                    binning / ax_kb,
                ),
                gen_tracks[i],
            )
        ax[1].set_ylabel("Transcription (CPM)", fontsize=15)
        if binning != 1:
            ax[1].set_xlabel("Genomic distance (kb)", fontsize=15)
        else:
            ax[1].set_xlabel("Genomic distance (bin)", fontsize=15)
        pax = ax[0]
        pax.get_xaxis().set_visible(False)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pax = ax
        if binning != 1:
            pax.set_xlabel("Genomic distance (kb)", fontsize=15)
        else:
            pax.set_xlabel("Genomic distance (bin)", fontsize=15)

    # Plot the pileup contact map.
    im = pax.imshow(
        pileup,
        cmap="seismic",
        vmin=-np.max(
            np.abs(pileup)
        ),  # min(-0.0005, -np.nanpercentile(pileup, 99.5)),
        vmax=np.max(
            np.abs(pileup)
        ),  # max(0.0005, np.nanpercentile(pileup, 99.5)),
        extent=[-window_plot, window_plot, window_plot, -window_plot],
    )
    pax.axvline(0, color="black", linestyle="dashed", linewidth=1.5, alpha=0.4)
    pax.axhline(0, color="black", linestyle="dashed", linewidth=1.5, alpha=0.4)
    pax.tick_params(axis="both", labelsize=14)
    if binning != 1:
        pax.set_ylabel("Genomic distance (kb)", fontsize=15)
    else:
        pax.set_ylabel("Genomic distance (bin)", fontsize=15)

    # Add colorbar.
    if gen_tracks is not None:
        fig.colorbar(
            im, ax=ax.ravel().tolist(), shrink=0.33, anchor=(1.3, 0.75)
        )
        plt.subplots_adjust(hspace=0.1)
    else:
        fig.colorbar(im, fontsize=5)

    # Add title if one given.
    if title is not None:
        pax.set_title(title, fontsize=20)

    # Save figure if an outfile is given.
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)
