#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""General plot functions to display contact maps and others genomics data
around bacterial HiC experiments analysis.

Functions:
    - antidagonal
    - map
    - map_ratio
"""


import matplotlib.pyplot as plt
import numpy as np


def antidiagonal(
    values: "numpy.ndarray",
    ori_pos: int = -1,
    outfile: str = "none",
    pars_pos: List[int] = None,
    
)

def map(
    mat: "numpy.ndarray",
    axis: str = "kb",
    binning: int = 1,
    cmap: str = "Reds",
    dpi: int = 300,
    end: int = 0,
    out_file: str = "none",
    start: int = 0,
    title: str = "none",
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
    if binning == 1:
        axis = "bin"
    elif axis == "kb" and binning > 1:
        binning = binning / 1000
    elif axis == "Mp" and binning > 1:
        binning = binning / 1000000

    # No end values given.
    if end == 0:
        end = len(mat)

    # Display plots
    im = ax.imshow(
        mat[start:end, start:end],
        cmap=cmap,
        vmin=0,
        vmax=np.percentile(mat, vmax),
        extent=(start * binning, end * binning, end * binning, start * binning),
    )

    # Legend
    ax.set_xlabel("Genomic coordinates ({0})".format(axis), fontsize=16)
    ax.set_ylabel("Genomic coordinates ({0})".format(axis), fontsize=16)
    ax.tick_params(size=16)

    # Title
    if title != "none":
        ax.set_title(title, size=18)

    # Colorbar
    cbar = plt.colorbar(im, shrink=0.33, anchor=(0, 0.5))
    cbar.ax.tick_params(labelsize=16)

    # Savefig
    if out_file != "none":
        plt.savefig(out_file, dpi=dpi)


def map_ratio(
    mat1: "numpy.ndarray",
    mat2: "numpy.ndarray",
    axis: str = "kb",
    binning: int = 1,
    cmap: str = "seisimic",
    dpi: int = 300,
    end: int = 0,
    lim: float = 2,
    out_file: str = "none",
    start: int = 0,
    serpentine: bool = False,
    title: str = "none",
):
    """Function to plot a log2 ratio of two matrices or a region of it. A
    already computed serpentine log ratio could be given instead and displayed.
    If two matrices are given with the serpentine option, it will displayed the
    first one in the upper trinagle and the second if the lower triangle.
    
    Parameters
    ----------

    mat1 : numpy.ndarray
        First matrix to use for the log ratio plot. Positive value on the
        colorscale. If serpentine set to True it will be the serpentine matrix.
    mat2 : numpy.ndarray
        Second matrix to use for the log ratio plot. Negative value on the
        colorscale. If serpentine set to True it will be the log ratio matrix if
        one is given.
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
    lim : float
        Limits of the colorscale (max and minus will be the negative).
    out_file : str
        Path of the output file to save the plot. The extension have to match 
        matplotlib.pyplot.savefig extensions. If none given, don't save the
        figure.
    serpentine : bool
        Either a serpentine ratio is given or not.
    start : int
        Start position in bins of the region plot. If none given, it will start 
        with the beginning of the matrix. [Default: 0]
    title : str
        Title to put on the plot. If none given, do not put any title.

    TODO: adapt it for multiple chromosomes maps
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)

    # Axis values
    if binning == 1:
        axis = "bin"
    elif axis == "kb" and binning > 1:
        binning = binning / 1000
    elif axis == "Mp" and binning > 1:
        binning = binning / 1000000

    # No end values given.
    if end == 0:
        end = len(mat)

    # Compute the matrices
    if serpentine:
        if mat2 != "none":
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
        extent=(start * binning, end * binning, end * binning, start * binning),
    )

    # Legend
    ax.set_xlabel("Genomic coordinates (%s)", axis, fontsize=16)
    ax.set_ylabel("Genomic coordinates (%s)", axis, fontsize=16)
    ax.tick_params(size=16)

    # Title
    if title != "none":
        ax.set_title(title, size=18)

    # Colorbar
    cbar = plt.colorbar(im, shrink=0.33, anchor=(0, 0.5))
    cbar.ax.tick_params(labelsize=16)

    # Savefig
    if outfile != "none":
        plt.savefig(out_file, dpi=dpi)
