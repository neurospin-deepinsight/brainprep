# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to spatialy normalize the data.
"""

# Imports
import os
import nibabel
import numpy as np
from .utils import check_version, check_command, execute_command


def scale(imfile, scaledfile, scale, check_pkg_version=False):
    """ Scale the MRI image.

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: str
        the input image.
    scaledfile: str
        the path to the scaled input image.
    scale: int
        the scale factor in all directions.
    check_pkg_version: bool, default False
        optionally check the package version using dpkg.

    Returns
    -------
    scaledfile, trffile: str
        the generated files.
    """
    check_version("fsl", check_pkg_version)
    check_command("flirt")
    trffile = scaledfile.split(".")[0] + ".txt"
    cmd = ["flirt", "-in", imfile, "-ref", imfile, "-out",
           scaledfile, "-applyisoxfm", str(scale), "-omat", trffile]
    execute_command(cmd)
    return scaledfile, trffile


def bet2(imfile, brainfile, frac=0.5, cleanup=True, check_pkg_version=False):
    """ Skull stripped the MRI image.

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: str
        the input image.
    brainfile: str
        the path to the brain image file.
    frac: float, default 0.5
        fractional intensity threshold (0->1);smaller values give larger brain
        outline estimates
    cleanup: bool, default True
        optionnally add bias field & neck cleanup.
    check_pkg_version: bool, default False
        optionally check the package version using dpkg.

    Returns
    -------
    brainfile, maskfile: str
        the generated files.
    """
    check_version("fsl", check_pkg_version)
    check_command("bet")
    maskfile = brainfile.split(".")[0] + "_mask.nii.gz"
    cmd = ["bet", imfile, brainfile, "-f", str(frac), "-R", "-m"]
    if cleanup:
        cmd.append("-B")
    execute_command(cmd)
    return brainfile, maskfile


def reorient2std(imfile, stdfile, check_pkg_version=False):
    """ Reorient the MRI image to match the approximate orientation of the
    standard template images (MNI152).

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: str
        the input image.
    stdfile: str
        the reoriented image file.
    check_pkg_version: bool, default False
        optionally check the package version using dpkg.

    Returns
    -------
    stdfile: str
        the generated file.
    """
    check_version("fsl", check_pkg_version)
    check_command("fslreorient2std")
    cmd = ["fslreorient2std", imfile, stdfile]
    execute_command(cmd)
    return stdfile


def biasfield(imfile, bfcfile, maskfile=None, nb_iterations=50,
              convergence_threshold=0.001, bspline_grid=(1, 1, 1),
              shrink_factor=1, bspline_order=3,
              histogram_sharpening=(0.15, 0.01, 200), check_pkg_version=False):
    """ Perform MRI bias field correction using N4 algorithm.

    .. note:: This function is based on ANTS.

    Parameters
    ----------
    imfile: str
        the input image.
    bfcfile: str
        the bias fieled corrected file.
    maskfile: str, default None
        the brain mask image.
    nb_iterations: int, default 50
        Maximum number of iterations at each level of resolution. Larger
        values will increase execution time, but may lead to better results.
    convergence_threshold: float, default 0.001
        Stopping criterion for the iterative bias estimation. Larger values
        will lead to smaller execution time.
    bspline_grid: int, default (1, 1, 1)
        Resolution of the initial bspline grid defined as a sequence of three
        numbers. The actual resolution will be defined by adding the bspline
        order (default is 3) to the resolution in each dimension specified
        here. For example, 1,1,1 will result in a 4x4x4 grid of control points.
        This parameter may need to be adjusted based on your input image.
        In the multi-resolution N4 framework, the resolution of the bspline
        grid at subsequent iterations will be doubled. The number of
        resolutions is implicitly defined by Number of iterations parameter
        (the size of this list is the number of resolutions).
    shrink_factor: int, default 1
        Defines how much the image should be upsampled before estimating the
        inhomogeneity field. Increase if you want to reduce the execution
        time. 1 corresponds to the original resolution. Larger values will
        significantly reduce the computation time.
    bspline_order: int, default 3
        Order of B-spline used in the approximation. Larger values will lead
        to longer execution times, may result in overfitting and poor result.
    histogram_sharpening: 3-uplate, default (0.15, 0.01, 200)
        A vector of up to three values. Non-zero values correspond to Bias
        Field Full Width at Half Maximum, Wiener filter noise, and Number of
        histogram bins.
    check_pkg_version: bool, default False
        optionally check the package version using dpkg.

    Returns
    -------
    bfcfile, bffile: str
        the generatedd files.
    """
    check_version("ants", check_pkg_version)
    check_command("N4BiasFieldCorrection")
    ndim = 3
    bspline_grid = [str(e) for e in bspline_grid]
    histogram_sharpening = [str(e) for e in histogram_sharpening]
    bffile = bfcfile.split(".")[0] + "_field.nii.gz"
    cmd = [
        "N4BiasFieldCorrection",
        "-d", str(ndim),
        "-i", imfile,
        "-s", str(shrink_factor),
        "-b", "[{0}, {1}]".format("x".join(bspline_grid), bspline_order),
        "-c", "[{0}, {1}]".format(
            "x".join([str(nb_iterations)] * 4), convergence_threshold),
        "-t", "[{0}]".format(", ".join(histogram_sharpening)),
        "-o", "[{0}, {1}]".format(bfcfile, bffile),
        "-v"]
    if maskfile is not None:
        cmd += ["-x", maskfile]
    execute_command(cmd)
    return bfcfile, bffile


def register_affine(imfile, targetfile, regfile, mask=None, cost="normmi",
                    bins=256, interp="spline", dof=9, check_pkg_version=False):
    """ Register the MRI image to a target image using an affine transform
    with 9 dofs.

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: str
        the input image.
    targetfile: str
        the target image.
    regfile: str
        the registered file.
    mask: str, default None
        the white matter mask image needed by the bbr cost function.
    cost: str, default 'normmi'
        Choose the most appropriate metric: 'mutualinfo', 'corratio',
        'normcorr', 'normmi', 'leastsq', 'labeldiff', 'bbr'.
    bins: int, default 256
        Number of histogram bins
    interp: str, default 'spline'
        Choose the most appropriate interpolation method: 'trilinear',
        'nearestneighbour', 'sinc', 'spline'.
    dof: int, default 9
        Number of affine transform dofs.
    check_pkg_version: bool, default False
        optionally check the package version using dpkg.

    Returns
    -------
    regfile, trffile: str
        the generated files.
    """
    check_version("fsl", check_pkg_version)
    check_command("flirt")
    trffile = regfile.split(".")[0] + ".txt"
    cmd = ["flirt",
           "-in", imfile,
           "-ref", targetfile,
           "-cost", cost,
           "-searchcost", cost,
           "-anglerep", "euler",
           "-bins", str(bins),
           "-interp", interp,
           "-dof", str(dof),
           "-out", regfile,
           "-omat", trffile,
           "-verbose", "1"]
    if cost == "bbr":
        if mask is None:
            raise ValueError("A white matter mask image is needed by the "
                             "bbr cost function.")
        cmd += ["-wmseg", mask]
    execute_command(cmd)
    return regfile, trffile


def apply_affine(imfile, targetfile, regfile, affines, interp="spline",
                 check_pkg_version=False):
    """ Apply affine transformations to an image.

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: nibabel.Nifti1Image
        the input image.
    targetfile: nibabel.Nifti1Image
        the target image.
    regfile: str
        the registered file.
    affines: str or list of str
        the affine transforms to be applied. If multiple transforms are
        specified, they are first composed.
    interp: str, default 'spline'
        Choose the most appropriate interpolation method: 'trilinear',
        'nearestneighbour', 'sinc', 'spline'.
    check_pkg_version: bool, default False
        optionally check the package version using dpkg.

    Returns
    -------
    regfile, trffile: str
        the generated files.
    """
    check_version("fsl", check_pkg_version)
    check_command("flirt")
    if not isinstance(affines, list):
        affines = [affines]
    elif len(affines) == 0:
        raise ValueError("No transform specified.")
    trffile = regfile.split(".")[0] + ".txt"
    affines = [np.loadtxt(path) for path in affines][::-1]
    affine = affines[0]
    for matrix in affines[1:]:
        affine = np.dot(matrix, affine)
    np.savetxt(trffile, affine)
    cmd = ["flirt",
           "-in", imfile,
           "-ref", targetfile,
           "-init", trffile,
           "-interp", interp,
           "-applyxfm",
           "-out", regfile]
    execute_command(cmd)
    return regfile, trffile


def apply_mask(imfile, maskfile, genfile):
    """ Apply brain mask.

    Parameters
    ----------
    imfile: str
        the input image.
    maskfile: str
        the mask image.
    genfile: str
        the input masked file.

    Returns
    -------
    genfile: str
        the generated file.
    """
    im = nibabel.load(imfile)
    mask_im = nibabel.load(maskfile)
    arr = im.get_fdata()
    arr[mask_im.get_fdata() == 0] = 0
    gen_im = nibabel.Nifti1Image(arr, im.affine)
    nibabel.save(gen_im, genfile)
    return genfile
