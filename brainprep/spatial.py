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
import SimpleITK as sitk
from .utils import check_version, check_command, execute_command, print_error, print_command


def scale(imfile, scaledfile, scale, check_pkg_version=False):
    """ Resample the MRI image to a new isotropic voxel size.

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: str
        the input image.
    scaledfile: str
        the path to the scaled input image.
    scale: int
        Desired isotropic voxel size in mm.
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


def bet2(imfile, brainfile, frac=0.5, cleanup=True, save_brain_mask=True, check_pkg_version=False):
    """ Skull stripped the MRI image.

    .. note:: This function is based on FSL.

    Parameters
    ----------
    imfile: str
        the input image.
    brainfile: str
        the path to the output brain image file (with masked applied).
    frac: float, default=0.5
        fractional intensity threshold (0->1);smaller values give larger brain
        outline estimates
    cleanup: bool, default=True
        optionnally add bias field & neck cleanup.
    save_brain_mask: bool, default=True
        optionnally save the brain mask with suffix "_mask.nii.gz".
    check_pkg_version: bool, default=False
        optionally check the package version using dpkg.

    Returns
    -------
    brainfile, maskfile: str, str or None
        the generated files. 
        If `save_brain_mask` is False, the maskfile will be None
    """
    check_version("fsl", check_pkg_version)
    check_command("bet")
    cmd = ["bet", imfile, brainfile, "-f", str(frac), "-R"]
    maskfile = None
    if save_brain_mask:
        cmd.append("-m")
        maskfile = brainfile.split(".")[0] + "_mask.nii.gz"
    if cleanup:
        cmd.append("-B")
    execute_command(cmd)
    return brainfile, maskfile

def synthstrip(imfile, brainfile, save_brain_mask=True):
    """
    Skull strip the MRI image using SynthStrip (FreeSurfer).

    Parameters
    ----------
    imfile: str
        Input image (T1, T2, FLAIR, etc.)
    brainfile: str
        Output skull-stripped brain image file path.
    save_brain_mask: bool, default=True
        Optionally save the brain mask with suffix '_mask.nii.gz'.

    Returns
    -------
    brainfile, maskfile: str, str or None
        The skull-stripped image and mask file paths.
        If `save_brain_mask` is False, maskfile is None.
    """
    check_command("mri_synthstrip")
    
    cmd = [
        "mri_synthstrip",
        "-i", imfile,
        "-o", brainfile,
        "--no-csf"
    ]
    maskfile = None
    if save_brain_mask:
        maskfile = brainfile.split(".")[0] + "_mask.nii.gz"
        cmd.extend(["-m", maskfile])
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
        "-b", "[{0},{1}]".format("x".join(bspline_grid), bspline_order),
        "-c", "[{0},{1}]".format(
            "x".join([str(nb_iterations)] * 4), convergence_threshold),
        "-t", "[{0}]".format(", ".join(histogram_sharpening)),
        "-o", "[{0},{1}]".format(bfcfile, bffile),
        "-v"]
    if maskfile is not None:
        cmd += ["-x", maskfile]
    try:
        execute_command(cmd)
    except Exception as e:
        print(f"Error occurred while executing command: {e}")
        cmd = ["cp", imfile, bfcfile]
        execute_command(cmd)
        print("Using the original image as the bias field corrected image.")
    return bfcfile, bffile

def biasfield_sitk(imfile, bfcfile, maskfile=None, nb_iterations=50,
                   convergence_threshold=0.001,
                   shrink_factor=1, bspline_order=3,
                   histogram_sharpening=(0.15, 0.01, 200)):
    """
    Perform MRI bias field correction using SimpleITK's N4 algorithm.
    It is the SITK-equivalent of ANTS' N4 algorithm.

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

    """
    # Read input image
    image = sitk.ReadImage(imfile, sitk.sitkFloat32)

    # Read mask
    if maskfile is not None:
        mask = sitk.ReadImage(maskfile, sitk.sitkUInt8)
    else:
        mask = None

    # Optionally shrink image and mask for faster processing
    if shrink_factor > 1:
        image = sitk.Shrink(image, [shrink_factor]*image.GetDimension())
        if mask is not None:
            mask = sitk.Shrink(mask, [shrink_factor]*mask.GetDimension())

    # Set up the corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([nb_iterations]*4)
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrector.SetSplineOrder(bspline_order)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(histogram_sharpening[0])
    corrector.SetWienerFilterNoise(histogram_sharpening[1])
    corrector.SetNumberOfHistogramBins(histogram_sharpening[2])

    # Run correction
    print_command([str(corrector)])
    if mask is None:
        corrected = corrector.Execute(image)
    else:
        corrected = corrector.Execute(image, mask)

    # Write output
    sitk.WriteImage(corrected, bfcfile)

    # Get bias field as image
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    bias_field = sitk.Exp(log_bias_field)
    bffile = bfcfile.replace(".nii", "_field.nii")
    sitk.WriteImage(bias_field, bffile)
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


def register_affine_sitk(imfile, targetfile, regfile, maskfile=None, regmaskfile=None, 
                         cost="mattesmi", bins=50, interp="spline"):
    """
    Register MRI image to target using an affine transform via SimpleITK.

    Parameters
    ----------
    imfile: str
        the input image.
    targetfile: str
        the target image.
    regfile: str
        the registered file.
    maskfile: str, default=None
        Brain mask to eventually register using the same transformation.
    regmaskfile: str, default=None
        Registered brain mask file name (only used if maskfile is given).
    cost: str, default 'mattesmi'
        Choose the most appropriate metric: 'mi', 'mattesmi', 'correlation',
        'mse'.
    bins: int, default 50
        Number of histogram bins
    interp: str, default 'spline'
        Choose the most appropriate interpolation method: 'trilinear',
        'nearestneighbour', 'spline'.
    
    Returns
    -------
    regfile, trffile: str
        the generated files.
    """

    # Read fixed (target) and moving (input) images
    fixed = sitk.ReadImage(targetfile, sitk.sitkFloat32)
    moving = sitk.ReadImage(imfile, sitk.sitkFloat32)

    # Set up initial transform: Affine with 12 DOF
    transform = sitk.CenteredTransformInitializer(
        fixed, 
        moving, 
        sitk.AffineTransform(fixed.GetDimension()), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Configure registration method
    registration = sitk.ImageRegistrationMethod()

    # Cost SimpleITK metrics
    if cost.lower() == "mi":
        registration.SetMetricAsJointHistogramMutualInformation(
            numberOfHistogramBins=bins, varianceForJointPDFSmoothing=1.5)
    elif cost.lower() == "mattesmi":
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    elif cost.lower() == "correlation":
        registration.SetMetricAsCorrelation()
    elif cost.lower() == "mse":
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Unsupported cost function '{cost}' for SimpleITK.")
    
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)

    # Map interpolation methods
    interp = interp.lower()
    if interp == "trilinear":
        registration.SetInterpolator(sitk.sitkLinear)
    elif interp == "nearestneighbour":
        registration.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interp == "spline":
        registration.SetInterpolator(sitk.sitkBSpline)
    else:
        raise ValueError(f"Unsupported interpolation method '{interp}'.")

    # Optimizer settings
    registration.SetOptimizerAsGradientDescent(
                learningRate=1.0, 
                numberOfIterations=100, 
                convergenceMinimumValue=1e-6, 
                convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(transform)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Execute registration
    print_command([str(registration)])
    final_transform = registration.Execute(fixed, moving)

    # Resample and write output image
    resampled = sitk.Resample(moving, fixed, final_transform,
                              sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    sitk.WriteImage(resampled, regfile)

    # Save transform to text file (ITK format)
    trffile = regfile.rsplit(".", 1)[0] + ".txt"
    sitk.WriteTransform(final_transform, trffile)

    # Eventuall apply the transformation to mask
    if maskfile is not None:
        mask_img = sitk.ReadImage(maskfile, sitk.sitkUInt8)
        mask_resampled = sitk.Resample(mask_img, fixed, final_transform,
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(mask_resampled, regmaskfile)

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
