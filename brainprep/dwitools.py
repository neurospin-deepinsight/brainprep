##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Utility Python functions to load diffusion metadata.
"""

# System import
import numpy
import glob
import os
import numpy as np
import nibabel
from nilearn import plotting
import matplotlib.pyplot as plt

from .utils import check_command, execute_command


def extract_dwi_shells(dwi_nii_path, bvals_path, bvecs_path, outdir):
    """ Convert a multi-shell serie to multiple single shell series.
    Parameters
    ----------
    dwi_nii_path: str
        path to the diffusion volume file.
    bvals_path: str
        path to the diffusion b-values file.
    bvecs_path: str
        path to the diffusion b-vectors file.
    outdir: str
        path to the destination folder.
    Returns
    -------
    nodiff_file: str
        path to the mean b0 file.
    dwis: dict
        path the each shell 'dwi', 'bvals' and 'bvecs' files: b-values are
        the keys of this dictionary.
    """
    # Load input data
    dwi = nibabel.load(dwi_nii_path)
    dwi_array = dwi.get_data()
    real_bvals, real_bvecs, nb_shells, nb_nodiff = read_bvals_bvecs(
        bvals_path, bvecs_path, min_bval=100.)

    # Detect shell indices
    bvals = real_bvals.copy()
    # bvecs = real_bvecs.copy()
    b0_indices = numpy.where(bvals < 50)[0].tolist()
    bvals[b0_indices] = 0
    bvals = [int(round(bval, -2)) for bval in bvals]
    bvals_set = set(bvals) - {0}

    # Create mean b0 image
    b0 = numpy.mean(dwi_array[..., b0_indices], axis=3)
    im = nibabel.Nifti1Image(b0, affine=dwi.get_affine())
    nodiff_file = os.path.join(outdir, "nodiff.nii.gz")
    nibabel.save(im, nodiff_file)
    b0.shape += (1, )

    # Create dwi for each shell
    dwis = {}
    bvals = numpy.asarray(bvals)
    for bval in bvals_set:

        # Get shell indices
        bval_outdir = os.path.join(outdir, str(bval))
        if not os.path.isdir(bval_outdir):
            os.mkdir(bval_outdir)
        shell_indices = numpy.where(bvals == bval)[0].tolist()

        # Create single shell dwi
        shell_dwi = dwi_array[..., shell_indices]
        shell_dwi = numpy.concatenate((b0, shell_dwi), axis=3)
        im = nibabel.Nifti1Image(shell_dwi, affine=dwi.get_affine())
        dwi_file = os.path.join(bval_outdir, "dwi.nii.gz")
        nibabel.save(im, dwi_file)

        # Create associated bvecs/bvals
        shell_bvecs = real_bvecs[shell_indices]
        shell_bvecs = numpy.concatenate((numpy.zeros((1, 3)), shell_bvecs),
                                        axis=0)
        bvecs_file = os.path.join(bval_outdir, "bvecs")
        numpy.savetxt(bvecs_file, shell_bvecs)
        shell_bvals = real_bvals[shell_indices]
        shell_bvals = numpy.concatenate((numpy.zeros((1, )), shell_bvals))
        bvals_file = os.path.join(bval_outdir, "bvals")
        numpy.savetxt(bvals_file, shell_bvals)

        # Update output structure
        dwis[bval] = {
            "bvals": bvals_file,
            "bvecs": bvecs_file,
            "dwi": dwi_file
        }

    return nodiff_file, dwis


def read_bvals_bvecs(bvals_path, bvecs_path, min_bval=200.):
    """ Read b-values and associated b-vectors.
    Parameters
    ----------
    bvals_path: str or list of str
        path to the diffusion b-values file(s).
    bvecs_path: str or list of str
        path to the diffusion b-vectors file(s).
    min_bval: float, optional
        if a b-value under this threshold is detected raise an ValueError.
    Returns
    -------
    bvals: array (N, )
        array containing the diffusion b-values.
    bvecs: array (N, 3)
        array containing the diffusion b-vectors.
    nb_shells: int
        the number of shells.
    nb_nodiff: int
        the number of no diffusion weighted images.
    Raises
    ------
    ValueError: if the b-values or the corresponding b-vectors have not
        matching sizes this exception is raised.
    """
    # Format input path
    if not isinstance(bvals_path, list):
        bvals_path = [bvals_path]
    if not isinstance(bvecs_path, list):
        bvecs_path = [bvecs_path]

    # Read .bval & .bvecs files
    bvals = None
    bvecs = None
    for bvalfile, bvecfile in zip(bvals_path, bvecs_path):
        if bvals is None:
            bvals = np.loadtxt(bvalfile)
        else:
            bvals = np.concatenate((bvals, np.loadtxt(bvalfile)))
        if bvecs is None:
            bvecs = np.loadtxt(bvecfile)
        else:
            axis = bvecs.shape.index(max(bvecs.shape))
            bvecs = np.concatenate((bvecs, np.loadtxt(bvecfile)), axis=axis)

    # Check consistency between bvals and associated bvecs
    if bvecs.ndim != 2:
        raise ValueError("b-vectors file should be saved as a two dimensional "
                         "array: '{0}'.".format(bvecs_path))
    if bvals.ndim != 1:
        raise ValueError("b-values file should be saved as a one dimensional "
                         "array: '{0}'.".format(bvals_path))
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    if bvals.shape[0] != bvecs.shape[0]:
        raise ValueError("b-values and b-vectors shapes do not correspond.")

    # Infer nb of T2 and nb of shells.
    nb_nodiff = np.sum(bvals <= 50)  # nb of volumes where bvalue<50
    b0_set = set(bvals[bvals <= 50])
    bvals_set = set(bvals) - b0_set    # set of non-zero bvalues
    bvals_set = set([int(round(bval, -2)) for bval in list(bvals_set)])
    nb_shells = len(bvals_set)
    if min(bvals_set) < min_bval:
        raise ValueError("Small b-values detected (<{0}) in '{1}'.".format(
            min_bval, bvals_path))

    return bvals, bvecs, nb_shells, nb_nodiff


def topup(
        b0s,
        phase_enc_dirs,
        readout_time,
        outroot,
        apply_to=None):
    """ FSL topup tool to estimate the susceptibility induced
    off-resonance field.

    Parameters
    ----------
    b0s: list of str
        path to b0 file acquired in opposite phase enc. directions.
    phase_enc_dirs: list of str
        the phase enc. directions.
    readout_time: float
        the readout time.
    outroot: str
        fileroot name for output.
    apply_to: 2-uplet, default None
        apply the topup correction to the volumes acquired in the blip up and
        blip down acquisition settings respectivelly. Will take the first
        and last indices of the acquisiton parameters file during the
        correction.

    Returns
    -------
    fieldmap: str
        path to the fieldmap in Hz.
    corrected_b0s: str
        path to the unwarped b0 images.
    mean_corrected_b0s: str
        path to the mean unwarped b0 images.
    """

    # Write topup acqp
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide
    if len(b0s) != len(phase_enc_dirs):
        raise ValueError("Please specify properly the topup input data.")
    acqp_file = os.path.join(outroot, "acqp.txt")
    data = []
    affine = None
    tot_nvol = 0
    with open(acqp_file, "wt") as open_file:
        for enc_dir, path in zip(phase_enc_dirs, b0s):
            print("b0 enc_dir : ", enc_dir, path)
            im = nibabel.load(path)
            if affine is None:
                affine = im.affine
            else:
                assert numpy.allclose(affine, im.affine)
            arr = im.get_data()
            for cnt, size in enumerate(arr.shape[:3]):
                if size % 2 == 1:
                    print("[warn] reducing TOPUP B0 image size.")
                    arr = numpy.delete(arr, -1, axis=cnt)
            if arr.ndim == 3:
                arr.shape += (1, )
            data.append(arr)
            nvol = arr.shape[-1]
            tot_nvol += nvol
            print("tot nvol : ", tot_nvol)
            if enc_dir == "i":
                row = "1 0 0 {0}".format(readout_time)
            elif enc_dir == "i-":
                row = "-1 0 0 {0}".format(readout_time)
            elif enc_dir == "j":
                row = "0 1 0 {0}".format(readout_time)
            elif enc_dir == "j-":
                row = "0 -1 0 {0}".format(readout_time)
            else:
                raise ValueError("Unknown encode phase direction : "
                                 "{0}...".format(enc_dir))
            for indx in range(nvol):
                open_file.write(row + "\n")
    concatenated_b0s_file = os.path.join(outroot, "concatenated_b0s.nii.gz")
    concatenated_b0s = numpy.concatenate(data, axis=-1)
    concatenated_b0s_im = nibabel.Nifti1Image(concatenated_b0s, affine)
    nibabel.save(concatenated_b0s_im, concatenated_b0s_file)

    # The topup command
    fieldmap = os.path.join(outroot, "fieldmap.nii.gz")
    corrected_b0s = os.path.join(outroot, "unwarped_b0s.nii.gz")
    cmd = [
        "topup",
        "--imain={0}".format(concatenated_b0s_file),
        "--datain={0}".format(acqp_file),
        "--config=b02b0.cnf",
        "--out={0}".format(os.path.join(outroot, "topup")),
        "--fout={0}".format(fieldmap),
        "--iout={0}".format(corrected_b0s),
        "-v"]
    check_command(cmd[0])
    execute_command(cmd)

    # Apply topup correction
    if apply_to is not None:
        if len(apply_to) != 2:
            raise ValueError(
                "Need two volumes acquired in the blip up and blip down "
                "acquisition settings to apply topup.")
        cmd = [
            "applytopup",
            "--imain={0}".format(",".join(apply_to)),
            "--inindex=1,{0}".format(tot_nvol),
            "--datain={0}".format(acqp_file),
            "--topup={0}".format(os.path.join(outroot, "topup")),
            "--out={0}".format(os.path.join(outroot, "applytopup"))]
        check_command(cmd)
        execute_command(cmd)

    # Average b0s
    mean_corrected_b0s = os.path.join(outroot, "mean_unwarped_b0s.nii.gz")
    cmd = [
        "fslmaths",
        corrected_b0s,
        "-Tmean", mean_corrected_b0s]
    check_command(cmd[0])
    execute_command(cmd)
    return fieldmap, corrected_b0s, mean_corrected_b0s, acqp_file


def epi_reg(
        epi_file, structural_file, brain_structural_file, output_fileroot,
        fieldmap_file=None, effective_echo_spacing=None, magnitude_file=None,
        brain_magnitude_file=None, phase_encode_dir=None, wmseg_file=None):
    """ Register EPI images (typically functional or diffusion) to structural
    (e.g. T1-weighted) images. The pre-requisites to use this method are:

    1) a structural image that can be segmented to give a good white matter
    boundary.
    2) an EPI that contains some intensity contrast between white matter and
    grey matter (though it does not have to be enough to get a segmentation).

    It is also capable of using fieldmaps to perform simultaneous registration
    and EPI distortion-correction. The fieldmap must be in rad/s format.

    Parameters
    ----------
    epi_file: str
        The EPI images.
    structural_file: str
        The structural image.
    brain_structural_file
        The brain extracted structural image.
    output_fileroot: str
        The corrected EPI file root.
    fieldmap_file: str, default None
        The fieldmap image (in rad/s)
    effective_echo_spacing: float, default None
        If parallel acceleration is used in the EPI acquisition then the
        effective echo spacing is the actual echo spacing between acquired
        lines in k-space divided by the acceleration factor.
    magnitude_file: str, default None
        The magnitude image.
    brain_magnitude_file: str
        The brain extracted magnitude image: should only contains brain
        tissues.
    phase_encode_dir: str, default None
         The phase encoding direction x/y/z/-x/-y/-z.
    wmseg_file: str, default None
        The white matter segmentatiion of structural image. If provided do not
        execute FAST.
    fsl_sh: str, default DEFAULT_FSL_PATH
        The FSL configuration batch.

    Returns
    -------
    corrected_epi_file: str
        The corrected EPI image.
    warp_file: str
        The deformation field (in mm).
    distortion_file: str
        The distortion correction only field (in voxels).
    """
    # Check the input parameter
    for path in (epi_file, structural_file, brain_structural_file,
                 fieldmap_file, magnitude_file, brain_magnitude_file,
                 wmseg_file):
        if path is not None and not os.path.isfile(path):
            raise ValueError("'{0}' is not a valid input file.".format(path))

    # Define the FSL command
    cmd = [
        "epi_reg",
        "--epi={0}".format(epi_file),
        "--t1={0}".format(structural_file),
        "--t1brain={0}".format(brain_structural_file),
        "--out={0}".format(output_fileroot),
        "-v"]
    if fieldmap_file is not None:
        cmd.extend([
            "--fmap={0}".format(fieldmap_file),
            "--echospacing={0}".format(effective_echo_spacing),
            "--fmapmag={0}".format(magnitude_file),
            "--fmapmagbrain={0}".format(brain_magnitude_file),
            "--pedir={0}".format(phase_encode_dir)])
    if wmseg_file is not None:
        cmd.append("--wmseg={0}".format(wmseg_file))

    # Call epi_reg
    check_command(cmd[0])
    execute_command(cmd)

    # Get outputs
    corrected_epi_file = glob.glob(output_fileroot + ".*")[0]
    if fieldmap_file is not None:
        warp_file = glob.glob(output_fileroot + "_warp.*")[0]
        distortion_file = glob.glob(
            output_fileroot + "_fieldmaprads2epi_shift.*")[0]
    else:
        warp_file = None
        distortion_file = None

    return corrected_epi_file, warp_file, distortion_file


def eddy(
        dwi,
        dwi_brain_mask,
        acqp,
        index,
        bvecs,
        bvals,
        outroot,
        field=None,
        no_qspace_interpolation=False,
        no_slice_correction=True,
        strategy="openmp"):
    """ FSL eddy tool to correct eddy currents and movements in
    diffusion data:

    * 'eddy_cuda' runs on multiple GPUs. For more information, refer to:
      https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--mask. You may
      need to install nvidia-cuda-toolkit'.
    * 'eddy_openmp' runs on multiple CPUs. The outlier replacement step is
      not available with this precessing strategy.

    Note that this code is working with FSL >= 5.0.11.

    Parameters
    ----------
    dwi: str
        path to dwi volume.
    dwi_brain_mask: str
        path to dwi brain mask segmentation.
    acqp: str
        path to the required eddy acqp file. Refer to:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq#
        How_do_I_know_what_to_put_into_my_--acqp_file
    index: str
        path to the required eddy index file. Refer to:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--imain
    bvecs: str
        path to the bvecs file.
    bvals: str
        path to the bvals file.
    outroot: str
        fileroot name for output.
    field: str, default None
        path to the field map in Hz.
    no_qspace_interpolation: bool, default False
        if set do not remove any slices deemed as outliers and replace them
        with predictions made by the Gaussian Process.
    no_slice_correction: bool, True
        if set do not perform the slice to volume correction.
    strategy: str, default 'openmp'
        the execution strategy: 'openmp' or 'cuda'.
    fsl_sh: str, optional default 'DEFAULT_FSL_PATH'
        path to fsl setup sh file.

    Returns
    -------
    corrected_dwi: str
        path to the corrected DWI.
    corrected_bvec: str
        path to the rotated b-vectors.
    """
    # The Eddy command
    cmd = [
        "eddy_{0}".format(strategy),
        "--imain={0}".format(dwi),
        "--mask={0}".format(dwi_brain_mask),
        "--acqp={0}".format(acqp),
        "--index={0}".format(index),
        "--bvecs={0}".format(bvecs),
        "--bvals={0}".format(bvals),
        "--out={0}".format(outroot),
        "-v"]
    if field is not None:
        cmd += ["--field={0}".format(field)]
    if not no_qspace_interpolation:
        cmd += ["--repol"]
    else:
        cmd += ["--data_is_shelled"]
    if not no_slice_correction:
        cmd += [
            "--mporder=6",
            "--s2v_niter=5",
            "--s2v_lambda=1",
            "--s2v_interp=trilinear"]

    # Run the Eddy correction
    check_command(cmd[0])
    execute_command(cmd)

    # Get the outputs
    corrected_dwi = "{0}.nii.gz".format(outroot)
    corrected_bvec = "{0}.eddy_rotated_bvecs".format(outroot)

    return corrected_dwi, corrected_bvec


def extract_image(in_file, index, out_file=None):
    """ Extract the image at 'index' position.

    Parameters
    ----------
    in_file: str (mandatory)
        the input image.
    index: int (mandatory)
        the index of last image dimention to extract.
    out_file: str (optional, default None)
        the name of the extracted image file.

    Returns
    -------
    out_file: str
        the name of the extracted image file.
    """
    # Set default output if necessary
    dirname = os.path.dirname(in_file)
    basename = os.path.basename(in_file).split(".")[0]
    if out_file is None:
        out_file = os.path.join(
            dirname, "extract{0}_{1}.nii.gz".format(index, basename))

    # Extract the image of interest
    image = nibabel.load(in_file)
    affine = image.get_affine()
    extracted_array = image.get_data()[..., index]
    extracted_image = nibabel.Nifti1Image(extracted_array, affine)
    nibabel.save(extracted_image, out_file)

    return out_file


def flirt(in_file, ref_file, omat=None, out=None, init=None, cost="corratio",
          usesqform=False, displayinit=False, anglerep="euler", bins=256,
          interp="trilinear", dof=12, applyxfm=False, applyisoxfm=None,
          nosearch=False, wmseg=None, verbose=0):
    """ Command flirt.

    The basic usage is:
    flirt [options] -in <inputvol> -ref <refvol> -out <outputvol>
    flirt [options] -in <inputvol> -ref <refvol> -omat <outputmatrix>
    flirt [options] -in <inputvol> -ref <refvol> -applyxfm -init <matrix>
    -out <outputvol>

    Parameters
    ----------
    in_file: str (mandatory)
        Input volume.
    ref_file: str (mandatory)
        Reference volume.
    omat: str (optional, default None)
        Matrix filename. Output in 4x4 ascii format.
    out: str (optional, default None)
        Output volume.
    init: (optional, default None)
        Input 4x4 affine matrix
    cost: str (optional, default "corratio")
        Choose the most appropriate option: "mutualinfo", "corratio",
        "normcorr", "normmi", "leastsq", "labeldiff", "bbr".
    usesqform: bool (optional, default False)
        Initialise using appropriate sform or qform.
    displayinit: bool
        Display initial matrix.
    anglerep: str (optional default "euler")
        Choose the most appropriate option: "quaternion", "euler".
    bins: int (optional, default 256)
        Number of histogram bins
    interp: str (optional, default "trilinear")
        Choose the most appropriate option: "trilinear", "nearestneighbour",
        "sinc", "spline". (final interpolation: def - trilinear)
    dof: int (optional, default 12)
        Number of transform dofs.
    applyxfm: bool
        Applies transform (no optimisation) - requires -init.
    applyisoxfm: float (optional)
        The integer defines the scale. As applyxfm but forces isotropic
        resampling.
    verbose: int (optional)
        0 is least and default.
    nosearch: bool (optional, default False)
        if set perform no search to initializa the optimization.
    wmseg: str (optional)
        White matter segmentation volume needed by BBR cost function.
    shfile: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.

    Returns
    -------
    out: str
        Output volume.
    omat: str
        Output matrix filename. Output in 4x4 ascii format.
    """
    # Check the input parameters
    for filename in (in_file, ref_file):
        if not os.path.isfile(filename):
            raise ValueError(
                "'{0}' is not a valid input file.".format(filename))

    # Define the FSL command
    cmd = ["flirt",
           "-in", in_file,
           "-ref", ref_file,
           "-cost", cost,
           "-searchcost", cost,
           "-anglerep", anglerep,
           "-bins", str(bins),
           "-interp", interp,
           "-dof", str(dof),
           "-verbose", str(verbose)]

    # Set default parameters
    if usesqform:
        cmd += ["-usesqform"]
    if displayinit:
        cmd += ["-displayinit"]
    if applyxfm:
        cmd += ["-applyxfm"]
    if nosearch:
        cmd += ["-nosearch"]
    if init is not None:
        cmd += ["-init", init]
    if applyisoxfm is not None:
        cmd += ["-applyisoxfm", str(applyisoxfm)]
    if cost == "bbr":
        cmd += ["-wmseg", wmseg]

    dirname = os.path.dirname(in_file)
    basename = os.path.basename(in_file).split(".")[0]
    if out is None:
        out = os.path.join(dirname, "flirt_out_{0}.nii.gz".format(basename))
        cmd += ["-out", out]
    else:
        cmd += ["-out", out]

    if omat is None:
        if not applyxfm:
            omat = os.path.join(dirname, "flirt_omat_{0}.txt".format(basename))
            cmd += ["-omat", omat]
    else:
        cmd += ["-omat", omat]

    # Call flirt
    check_command(cmd[0])
    execute_command(cmd)

    return out, omat


def triplanar(input_file, output_fileroot, title=None, nb_slices=1,
              overlays=None, overlays_colors=None, overlay_opacities=None,
              contours=False, edges=False, marker_coords=None,
              resolution=300):
    """ Snap an image with edge/overlay/contour on top (useful for checking
    registration).

    Parameters
    ----------
    input_file: str
        the input image.
    output_fileroot: str
        output fileroot.
    title: str, default None
        the snap title.
    nb_slices: int
        number of slices outputted.
    overlays: array of str
        array of paths to overlay images
    overlays_colors: list of str or int
        overlay images color index.
    overlay_opacities: list of float
        overlay images opacities.
    contours: bool
        if set to True, add overlays as contours.
    edges: bool
        if set to True, add overlays as edges.
    marker_coords: 3-uplet
        Coordinates of the markers to plot.
    resolution: int
        png outputs resolution.

    Returns
    -------
    output_png_file: str
        the generated output snap.
    """
    # Load files
    input_img = nibabel.load(input_file)

    # Create the display
    if input_img.get_data().ndim == 3:
        display = plotting.plot_anat(
            input_img,
            vmin=0,
            vmax=numpy.percentile(input_img.get_data(), 98),
            display_mode="ortho",
            title=title,
            draw_cross=False if marker_coords is None else True,
            cut_coords=marker_coords)
    else:
        display = plotting.plot_epi(
            input_img,
            vmin=0,
            vmax=numpy.percentile(input_img.get_data(), 98),
            display_mode="ortho",
            draw_cross=False if marker_coords is None else True,
            cut_coords=marker_coords)

    # Add overlays
    if overlays is not None and len(overlays) > 0:

        # Get all available cmaps
        maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
        colors = "bgrcmy"

        # Add overlays
        if overlays_colors is None:
            overlays_colors = [None] * len(overlays)
        if overlay_opacities is None:
            overlay_opacities = [None] * len(overlays)
        for overlay, alpha, color in zip(overlays, overlay_opacities,
                                         overlays_colors):
            if isinstance(color, int):
                color = colors[color % len(colors)]
            elif color is not None and color not in maps:
                raise ValueError("Available cmap are: {0}.".format(maps))
            if contours:
                display.add_contours(
                    overlay,
                    threshold=1e-06,
                    colorbar=False,
                    alpha=alpha,
                    cmap=color)
            elif edges:
                display.add_edges(
                    overlay,
                    color=color or "r")
            else:
                display.add_overlay(
                    overlay,
                    threshold=1e-06,
                    colorbar=False,
                    alpha=alpha,
                    cmap=color)

        # Add markers
        if marker_coords is not None:
            marker_coords = numpy.asarray([marker_coords])
            display.add_markers(marker_coords, marker_color="y",
                                marker_size=30)

    # Save image
    output_png_file = output_fileroot + "_ortho.png"
    display.savefig(output_png_file, dpi=resolution)
    display.close()

    return output_png_file


def register_mask_from_t1(t1, t1_mask, nodiff, outdir, name):
    """ Get nodiff_mask from t1_mask, t1 and nodiff ,
    Compute the transformation from nodiff to t1,
    Inverse the mat to get t1 to nodiff,
    Apply the inverse to t1_mask to get nodiff_mask,
    Apply nodiff brain mask to extract nodiff brain.

    Parameters
    ----------
    t1: str
        the input t1 image.
    outdir: str
        output fileroot.
    name: str
        basename for the generated files
    t1_mask: str
        the input t1_mask image corresponding to the t1.
    nodiff: str
        the input b0 image.

    Returns
    -------
    nodiff_brain: str
        the extracted brain from the nodiff with the new nodiff brain mask.
    nodiff_brain_mask: str
        the new nodiff brain mask
    """
    # Get nodiff_mask from t1_mask, t1 and nodiff
    # Compute the transformation from nodiff to t1
    nodiff_to_t1 = os.path.join(
        outdir, "{NAME}_to_t1.nii.gz".format(NAME=name))
    omat1 = os.path.join(outdir, "{NAME}_to_t1.mat".format(NAME=name))
    omat2 = os.path.join(outdir, "t1_to_{NAME}.mat".format(NAME=name))

    flirt(
        in_file=nodiff,
        ref_file=t1,
        out=nodiff_to_t1,
        dof=6,
        omat=omat1)

    # Inverse the mat to get t1 to nodiff
    fsl_cmd = ["convert_xfm", "-omat", omat2, "-inverse", omat1]
    check_command(fsl_cmd[0])
    execute_command(fsl_cmd)

    # Apply the inverse to t1_mask to get nodiff_mask
    nodiff_mask = os.path.join(outdir, "{NAME}_brain_mask.nii.gz"
                                       .format(NAME=name))
    flirt(
        in_file=t1_mask,
        ref_file=nodiff,
        applyxfm=True,
        init=omat2,
        out=nodiff_mask,
        interp="nearestneighbour")

    # Extract
    # Apply nodiff brain mask to extract nodiff brain
    output_fileroot_fslmaths = os.path.join(outdir, "{NAME}_brain")
    fsl_cmd = ["fslmaths", nodiff,
               "-mas", nodiff_mask,
               output_fileroot_fslmaths]
    check_command(fsl_cmd[0])
    execute_command(fsl_cmd)
    nodiff_brain = glob.glob(output_fileroot_fslmaths + ".*")[0]
    nodiff_brain_mask = nodiff_mask
    return nodiff_brain, nodiff_brain_mask
