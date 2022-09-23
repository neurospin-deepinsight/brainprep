#!/usr/bin/env python3
##########################################################################
# NSAp - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import shutil


from .utils import check_command, execute_command
from .color_utils import print_subtitle
from .dwitools import read_bvals_bvecs, topup, epi_reg, eddy, extract_image,\
                      flirt, triplanar, register_mask_from_t1

# Third-Party imports
import nibabel
import numpy


"""
1 - Reshape input data.
"""


def reshape_input_data(subject,
                       outdir,
                       t1,
                       dwi,
                       bvec,
                       bval,
                       t1_mask,
                       nodiff_mask=None,
                       mag_mask=None):
    outputs = {}
    outdir = os.path.join(outdir, "sub-"+subject)
    # Create preproc output dir
    reshape_outdir = os.path.join(outdir, "1-Reshape")
    if not os.path.isdir(reshape_outdir):
        cmd = ["mkdir", "-p", reshape_outdir]
        execute_command(cmd)

    # create a binary mask from the t1_mask
    mask_im = nibabel.load(t1_mask)
    arr = mask_im.get_fdata()
    arr[arr != 0] = 1
    gen_im = nibabel.Nifti1Image(arr, mask_im.affine)
    nibabel.save(gen_im, reshape_outdir+"/t1_mask.nii.gz")
    t1_mask = reshape_outdir+"/t1_mask.nii.gz"

    # Reorient input files
    files_to_reorient = {"dwi": dwi, "t1": t1, "t1_mask": t1_mask}
    if nodiff_mask is not None:
        files_to_reorient["nodiff_mask"] = nodiff_mask
    if mag_mask is not None:
        files_to_reorient["mag_mask"] = mag_mask
    for key in files_to_reorient:
        outfile = os.path.join(reshape_outdir, key + ".nii.gz")
        input_image = files_to_reorient[key]
        fsl_cmd = ["fslreorient2std", input_image, outfile]
        check_command(fsl_cmd[0])
        execute_command(fsl_cmd)

        files_to_reorient[key] = outfile
    dwi = files_to_reorient["dwi"]
    t1 = files_to_reorient["t1"]
    t1_mask = files_to_reorient["t1_mask"]
    if nodiff_mask is not None:
        nodiff_mask = files_to_reorient["nodiff_mask"]
    if mag_mask is not None:
        mag_mask = files_to_reorient["mag_mask"]

    outputs["t1"] = t1

    bval_array, _, nshell, nb_dif = read_bvals_bvecs(bval, bvec)
    print("Number of shell : ", nshell)
    print("Number of nodif : ", nb_dif)
    for ind in numpy.argwhere(bval_array == 0):
        nodiffb0 = extract_image(dwi, ind)
    outputs["nodif"] = nodiffb0

    # Apply input mask on t1
    brain_img_fileroot = os.path.join(
        reshape_outdir, os.path.basename(t1).replace(".nii.gz",
                                                     "_brain.nii.gz"))

    fsl_cmd = ["fslmaths", t1, "-mas", t1_mask, brain_img_fileroot]
    check_command(fsl_cmd[0])
    execute_command(fsl_cmd)
    brain_img = brain_img_fileroot
    brain_mask = t1_mask
    outputs["t1_brain"] = brain_img
    outputs["t1_brain_mask"] = brain_mask

    nodiff_brain, nodiff_brain_mask = register_mask_from_t1(
                                                        t1=t1,
                                                        t1_mask=t1_mask,
                                                        nodiff=nodiffb0,
                                                        outdir=reshape_outdir,
                                                        name="nodiff")

    outputs["nodif_brain"] = nodiff_brain
    outputs["nodif_brain_mask"] = nodiff_brain_mask
    return outputs


"""
2.1- Compute susceptibility correction.
"""


def Compute_and_Apply_susceptibility_correction(subject,
                                                t1,
                                                dwi,
                                                outdir,
                                                outputs,
                                                license_fs=None,
                                                topup_b0_dir=None,
                                                readout_time=None,
                                                topup_b0=None):
    outdir = os.path.join(outdir, "sub-"+subject)
    susceptibility_dir = os.path.join(outdir, "2-Susceptibility")
    if not os.path.isdir(susceptibility_dir):
        cmd = ["mkdir", "-p", susceptibility_dir]
        execute_command(cmd)
    susceptibility_method = None
    # Perform susceptibility correction: check carefully the result

    # FSL topup

    if topup_b0 is not None and \
       topup_b0_dir is not None and \
       readout_time is not None:
        print(topup_b0, type(topup_b0))
        print(topup_b0_dir, type(topup_b0_dir))
        # Select case
        susceptibility_method = "topup"
        print_subtitle("susceptibility_method : {sm}"
                       .format(sm=susceptibility_method))

        # Topup
        fieldmap_hz_to_diff_file, corrected_b0s,\
            mean_corrected_b0s, acqp = topup(
                                             b0s=topup_b0,
                                             phase_enc_dirs=topup_b0_dir,
                                             readout_time=readout_time,
                                             outroot=susceptibility_dir)
        # outputs["fieldmap_hz_to_diff"] = fieldmap_hz_to_diff_file
        outputs["fieldmap_hz"] = fieldmap_hz_to_diff_file
        outputs["mean_corrected_b0s"] = mean_corrected_b0s
        outputs["acqp"] = acqp

        # Check diffusion image size
        im = nibabel.load(dwi)
        arr = im.get_data()
        for cnt, size in enumerate(arr.shape[:3]):
            if size % 2 == 1:
                print("[warn] reducing DWI image size.")
                arr = numpy.delete(arr, -1, axis=cnt)
        im = nibabel.Nifti1Image(arr, im.affine)
        nibabel.save(im, dwi)

        nodifftopup_brain, nodifftopup_brain_mask = register_mask_from_t1(
                                            t1=t1,
                                            t1_mask=outputs["t1_brain_mask"],
                                            nodiff=mean_corrected_b0s,
                                            outdir=susceptibility_dir,
                                            name="nodifftopup")

        # Coregistration only
        dwi_corrected_fileroot = os.path.join(susceptibility_dir, "epi2struct")
        corrected_epi_file, _, _ = epi_reg(
            epi_file=dwi,
            structural_file=t1,
            brain_structural_file=outputs["t1_brain"],
            output_fileroot=dwi_corrected_fileroot,
            wmseg_file=None)
        nodif_to_t1_mat = os.path.join(susceptibility_dir, "epi2struct.mat")
        t1_to_nodif_mat = os.path.join(
            susceptibility_dir, "epi2struct_inv.mat")
        aff = numpy.loadtxt(nodif_to_t1_mat)
        aff_inv = numpy.linalg.inv(aff)
        numpy.savetxt(t1_to_nodif_mat, aff_inv)

    # synb0-Disco
    elif type(topup_b0_dir) is str and\
            topup_b0_dir in ["i", "i-", "j", "j-", "k", "k-"] and\
            topup_b0 is None:

        #  Select case
        susceptibility_method = "synb0"
        print_subtitle("susceptibility_method : {sm}"
                       .format(sm=susceptibility_method))

        # synb0_input_dir = os.path.join(susceptibility_dir, "synb0_inputs")
        # synb0_output_dir = os.path.join(susceptibility_dir, "synb0_outputs")
        synb0_input_dir = "/INPUTS"
        synb0_output_dir = "/OUTPUTS"
        # os.mkdir(synb0_input_dir)
        # os.mkdir(synb0_output_dir)
        print(synb0_input_dir)
        # Prepare t1 for synb0
        cmd = ["cp", t1, 
              os.path.join(synb0_input_dir, "T1.nii.gz")]
        check_command(cmd[0])
        execute_command(cmd)
        # Prepare b0 for synb0
        cmd = ["cp", outputs["nodif"],
              os.path.join(synb0_input_dir, "b0.nii.gz")]
        check_command(cmd[0])
        execute_command(cmd)

        # Prepare acqp file for synb0
        data = []
        affine = None
        enc_dir = topup_b0_dir
        print("PhaseEncodingDirection : ", enc_dir)
        acqp_file = os.path.join(synb0_input_dir, "acqparams.txt")
        with open(acqp_file, "wt") as open_file:
            im = nibabel.load(dwi)
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
            if enc_dir == "i":
                row1 = "1 0 0 {0}".format(readout_time)
                row2 = "1 0 0 0.000"
            elif enc_dir == "i-":
                row1 = "-1 0 0 {0}".format(readout_time)
                row2 = "-1 0 0 0.000"
            elif enc_dir == "j":
                row1 = "0 1 0 {0}".format(readout_time)
                row2 = "0 1 0 0.000"
            elif enc_dir == "j-":
                row1 = "0 -1 0 {0}".format(readout_time)
                row2 = "0 -1 0 0.000"
            else:
                raise ValueError("Unknown encode phase direction : "
                                 "{0}...".format(enc_dir))
            open_file.write(row1 + "\n")
            open_file.write(row2 + "\n")
        outputs["acqp"] = acqp_file
        # Prepare freesurfer licence for synb0
        cmd = ["ln", license_fs, "/extra/freesurfer/license.txt"]
        check_command(cmd[0])
        execute_command(cmd)
        fs = license_fs

        singularity run -e 
        -B /home/ld265905/Documents/synb0_inputs/:/INPUTS:ro 
        -B /home/ld265905/Documents/synb0_outputs/:/OUTPUTS 
        -B /home/ld265905/Documents/license.txt:/extra/freesurfer/license.txt 
        synb0-disco_v3.0.sif

        # cmd = ["singularity", "run", "-e",
        #        "-B",
        #        "{INPUTS}/:/INPUTS".format(INPUTS=synb0_input_dir),
        #        "-B",
        #        "{OUTPUTS}/:/OUTPUTS".format(OUTPUTS=synb0_output_dir),
        #        "-B",
        #        "{LICENSE}:/extra/freesurfer/license.txt".format(LICENSE=fs),
        #        "/home/ld265905/synb0-disco_v3.0.sif"]

        cmd = ["bash /extra/pipeline.sh"]
        check_command(cmd[0])
        execute_command(cmd)
        susceptibility_dir_synb0 = os.path.join(susceptibility_dir,
                                                "topup-synb0")
        if not os.path.isdir(susceptibility_dir_synb0):
            cmd = ["mkdir", "-p", susceptibility_dir_synb0]
            execute_command(cmd)
        outputs["eddy_file_from_topup"] = \
                                        os.path.join(susceptibility_dir_synb0,
                                        "topup")

        # Check diffusion image size
        im = nibabel.load(dwi)
        arr = im.get_data()
        for cnt, size in enumerate(arr.shape[:3]):
            if size % 2 == 1:
                print("[warn] reducing DWI image size.")
                arr = numpy.delete(arr, -1, axis=cnt)
        im = nibabel.Nifti1Image(arr, im.affine)
        nibabel.save(im, dwi)

        # Coregistration only
        dwi_corrected_fileroot = os.path.join(susceptibility_dir, "epi2struct")
        corrected_epi_file, _, _ = epi_reg(
            epi_file=dwi,
            structural_file=t1,
            brain_structural_file=outputs["t1_brain"],
            output_fileroot=dwi_corrected_fileroot,
            wmseg_file=None)
        nodif_to_t1_mat = os.path.join(susceptibility_dir, "epi2struct.mat")
        t1_to_nodif_mat = os.path.join(
            susceptibility_dir, "epi2struct_inv.mat")
        aff = numpy.loadtxt(nodif_to_t1_mat)
        aff_inv = numpy.linalg.inv(aff)
        numpy.savetxt(t1_to_nodif_mat, aff_inv)

    # Create an zero deformation field
    else:
        # Coregistration only
        print("No susceptibility correction")
        dwi_corrected_fileroot = os.path.join(susceptibility_dir, "epi2struct")
        corrected_epi_file, _, _ = epi_reg(
            epi_file=dwi,
            structural_file=t1,
            brain_structural_file=outputs["t1_brain"],
            output_fileroot=dwi_corrected_fileroot,
            wmseg_file=None)
        nodif_to_t1_mat = os.path.join(susceptibility_dir,
                                       "epi2struct.mat")
        t1_to_nodif_mat = os.path.join(susceptibility_dir,
                                       "epi2struct_inv.mat")
        aff = numpy.loadtxt(nodif_to_t1_mat)
        aff_inv = numpy.linalg.inv(aff)
        numpy.savetxt(t1_to_nodif_mat, aff_inv)

        # Create the null deformation field
        im = nibabel.load(dwi)
        zero_field = numpy.zeros(im.get_data().shape[:3] + (3, ))
        deformation_field_im = nibabel.Nifti1Image(zero_field, im.affine)
        deformation_field_file = os.path.join(susceptibility_dir,
                                              "field.nii.gz")
        nibabel.save(deformation_field_im, deformation_field_file)
        outputs["warp_file"] = deformation_field_file

        # Create a null field map
        outputs["fieldmap"] = None
    outputs["t1_to_nodif_mat"] = t1_to_nodif_mat

    """
    2.2- Apply susceptibility correction.
    """

    # Apply susceptibility correction to diffusion volumes
    if susceptibility_method == "topup" or susceptibility_method == "synb0":

        # Susceptibility will be corrected directly by eddy
        eddy_dwi_input = dwi
    else:

        # For now, except for topup sequences, susceptibility is corrected by
        # applying a voxel shift map generated from either a registration based
        # method or a an equivalent fieldmap.
        # TODO: Apply directly an Hz fieldmap directly into eddy to avoid a
        # double interpolation.
        susceptibility_corrected_dwi = os.path.join(
            susceptibility_dir, "susceptibility_corrected_dwi.nii.gz")

        in_file = dwi
        ref_file = dwi
        out_file = susceptibility_corrected_dwi
        warp_file = deformation_field_file
        interp = "spline"
        fsl_cmd = ["applywarp",
                   "-i", in_file,
                   "-r", ref_file,
                   "-o", out_file,
                   "-w", warp_file,
                   "--interp={0}".format(interp),
                   "--verbose=2"]
        check_command(fsl_cmd)
        execute_command(fsl_cmd)

        eddy_dwi_input = susceptibility_corrected_dwi
        fieldmap_hz_to_diff_file = None
        outputs["fieldmap_hz"] = fieldmap_hz_to_diff_file
    outputs["eddy_dwi_input"] = eddy_dwi_input

    nodif_brain_mask = outputs["nodif_brain_mask"]

    if susceptibility_method == "topup":
        outputs["nodif_brain_mask_undistorted"] = nodifftopup_brain_mask
    else:
        outputs["nodif_brain_mask_undistorted"] = nodif_brain_mask
    outputs["2_method"] = susceptibility_method
    return outputs


"""
3- Eddy current and motion correction.
"""


def eddy_and_motion_correction(subject,
                               t1,
                               index,
                               bvec,
                               bval,
                               outdir,
                               outputs):

    # Correct eddy current and motion

    # Outdir
    outdir = os.path.join(outdir, "sub-"+subject)
    eddy_outputdir = os.path.join(outdir, "3-Eddy")
    if not os.path.isdir(eddy_outputdir):
        os.mkdir(eddy_outputdir)
    eddy_outroot = os.path.join(
        eddy_outputdir, "{0}_dwi_eddy_corrected".format(subject))

    # eddy
    if outputs["2_method"] == "synb0":
        # corrected_dwi, corrected_bvec = eddy(
        #     dwi=outputs["eddy_dwi_input"],
        #     dwi_brain_mask=outputs["nodif_brain_mask_undistorted"],
        #     acqp=outputs["acqp"],
        #     index=index,
        #     bvecs=bvec,
        #     bvals=bval,
        #     outroot=eddy_outroot,
        #     field=field,
        #     strategy="openmp")
        fsl_cmd = ["eddy_openmp",
                   "--imain="+outputs["eddy_dwi_input"],
                   "--mask="+outputs["nodif_brain_mask_undistorted"],
                   "--acqp="+outputs["acqp"],
                   "--index="+index,
                   "--bvecs="+bvec,
                   "--bvals="+bval,
                   "--topup="+outputs["eddy_file_from_topup"],
                   "--out="+eddy_outroot]
        check_command(fsl_cmd[0])
        execute_command(fsl_cmd)
        corrected_dwi = "{0}.nii.gz".format(eddy_outroot)
        corrected_bvec = "{0}.eddy_rotated_bvecs".format(eddy_outroot)
    elif outputs["2_method"] == "topup":

        # field
        field = None
        if outputs["fieldmap_hz"] is not None:
            field = outputs["fieldmap_hz"].replace(".nii.gz", "")

        corrected_dwi, corrected_bvec = eddy(
            dwi=outputs["eddy_dwi_input"],
            dwi_brain_mask=outputs["nodif_brain_mask_undistorted"],
            acqp=outputs["acqp"],
            index=index,
            bvecs=bvec,
            bvals=bval,
            outroot=eddy_outroot,
            field=field,
            strategy="openmp")

    # Copy eddy corrected outputs
    corrected_dwi_undistorted = os.path.join(
        outdir, "dwi_corrected.nii.gz")
    shutil.copy2(corrected_dwi, corrected_dwi_undistorted)
    corrected_bvec_undistorted = os.path.join(
        outdir, "dwi_corrected.bvec")
    shutil.copy2(corrected_bvec, corrected_bvec_undistorted)

    # Extract eddy corrected b0
    corrected_nodiff_undistorted = os.path.join(
        outdir, "nodiff_corrected.nii.gz")
    extract_image(
        in_file=corrected_dwi_undistorted,
        index=0,
        out_file=corrected_nodiff_undistorted)

    # Register t1 to dwi
    t1_to_dwi = os.path.join(outdir, "t1_to_dwi.nii.gz")

    flirt(
        in_file=t1,
        ref_file=corrected_nodiff_undistorted,
        applyxfm=True,
        init=outputs["t1_to_nodif_mat"],
        out=t1_to_dwi,
        interp="spline")

    outputs["corrected_dwi"] = corrected_dwi_undistorted
    outputs["corrected_bvec"] = corrected_bvec_undistorted
    outputs["t1_to_diff"] = t1_to_dwi
    outputs["corrected_nodiff"] = corrected_nodiff_undistorted

    return outputs


"""
4- Create QC report
"""


def create_qc_report(subject, t1, outdir, outputs):
    # Create PDF comparison susceptibility correction/ no susceptibility
    # correction
    outdir = os.path.join(outdir, "sub-"+subject)
    qc_dir = os.path.join(outdir, "4-QC")
    if not os.path.isdir(qc_dir):
        os.mkdir(qc_dir)

    # T1 mask snap
    t1_mask_snap = triplanar(
        input_file=t1,
        output_fileroot=os.path.join(qc_dir, "t1_mask"),
        overlays=[outputs["t1_brain_mask"]],
        overlays_colors=None,
        overlay_opacities=[0.7],
        contours=True,
        edges=False,
        marker_coords=(5, -9, 7),
        resolution=300)

    # DWI mask snap
    nodif_mask_snap = triplanar(
        input_file=outputs["nodif"],
        output_fileroot=os.path.join(qc_dir, "nodif_mask"),
        overlays=[outputs["nodif_brain_mask"]],
        overlays_colors=None,
        overlay_opacities=[0.7],
        contours=True,
        edges=False,
        marker_coords=(5, -9, 7),
        resolution=300)

    # T1 overlayed with nodif snap
    t1_nodif_snap = triplanar(
        input_file=outputs["nodif"],
        output_fileroot=os.path.join(qc_dir, "t1_nodif"),
        overlays=[outputs["t1_to_diff"]],
        overlays_colors=None,
        overlay_opacities=[0.7],
        contours=True,
        edges=False,
        marker_coords=(5, -9, 7),
        resolution=300)

    # Corrected nodif overlayed with T1 snap
    t1_corrected_nodif_snap = triplanar(
        input_file=outputs["corrected_nodiff"],
        output_fileroot=os.path.join(qc_dir, "t1_corrected_nodif"),
        overlays=[outputs["t1_to_diff"]],
        overlays_colors=None,
        overlay_opacities=[0.7],
        contours=True,
        edges=False,
        marker_coords=(5, -9, 7),
        resolution=300)

    outputs["triplanar"] = [t1_mask_snap,
                            nodif_mask_snap,
                            t1_nodif_snap,
                            t1_corrected_nodif_snap]

    return outputs
