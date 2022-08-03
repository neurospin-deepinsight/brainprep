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
import re
import json
import shutil
import argparse
import textwrap
import subprocess
from pprint import pprint
from datetime import datetime
from packaging import version
from collections import OrderedDict
from argparse import RawTextHelpFormatter


from .utils import check_command, execute_command
from .dwitools import extract_dwi_shells, read_bvals_bvecs, topup, epi_reg, eddy, extract_image,\
                      flirt, triplanar

# Third-Party imports
import nibabel
import numpy
import glob



# Pyconnectomist imports
from pyconnectomist.preproc.susceptibility import (
    susceptibility_correction_wo_fieldmap)
from pyconnectomist.utils.pdftools import generate_pdf

# Pyconnectome imports
import pyconnectome
# from pyconnectome.plotting.slicer import triplanar
# from pyconnectome.utils.regtools import flirt
from pyconnectome.utils.regtools import applywarp
from pyconnectome.utils.segtools import bet2
from pyconnectome.utils.segtools import robustfov
# from pyconnectome.utils.preproctools import eddy
# from pyconnectome.utils.preproctools import epi_reg
from pyconnectome.utils.preproctools import fsl_prepare_fieldmap
from pyconnectome.utils.preproctools import smooth_fieldmap
from pyconnectome.utils.preproctools import pixel_shift_to_fieldmap
from pyconnectome.utils.preproctools import fieldmap_reflect
# from pyconnectome.utils.preproctools import topup
# from pyconnectome.models.tensor import dtifit
from pyconnectome.utils.filetools import fslreorient2std
from pyconnectome.utils.filetools import apply_mask
from pyconnectome.utils.filetools import erode
# from pyconnectome.utils.filetools import extract_image
from pyconnectome.wrapper import FSLWrapper
from pyconnectome import DEFAULT_FSL_PATH


# Script documentation
DOC = """
dMRI preprocessing steps
------------------------
Correct with brainsuite susceptibility artifact without fieldmap function and
eddy current/movements/outliers with FSL.
Requirements:
    - T1 image file (required).
    - DWI file (required).
    - bval file (required).
    - bvec file (required).
    - phase encode direction (required).
    - subject id (required).
    - index file (see fsl eddy) (required).
    - acqp file (see fsl eddy) (required).
Steps:
1- Reshape input data.
2- Susceptibility correction.
3- Eddy current and motion correction.
4- Tensor fit.
Command example on the MAPT data:
python $HOME/git/pyconnectome/pyconnectome/scripts/pyconnectome_dmri_preproc \
    -s 03990185BAI \
    -d /tmp/mapt/03990185BAI/dwi.nii.gz \
    -b /tmp/mapt/03990185BAI/dwi.bval \
    -r /tmp/mapt/03990185BAI/dwi.bvec \
    -t /neurospin/cati/cati_shared/MAPT/CONVERTED/0399/03990185BAI/M0/MRI/3DT1/03990185BAI_M0_3DT1_S003_PN_DIS2D.nii.gz \
    -c /tmp/mapt/03990185BAI/acqp.txt \
    -i /tmp/mapt/03990185BAI/index.txt \
    -m /tmp/mapt/03990185BAI/info.json \
    -o /tmp/mapt/03990185BAI \
    -T 4 \
    -V 2 \
    -F /etc/fsl/5.0.11/fsl.sh \
    -S
Command example on the SENIOR data:
python $HOME/git/pyconnectome/pyconnectome/scripts/pyconnectome_dmri_preproc \
    -s ag160127 \
    -d /tmp/senior/ag160127/dwi.nii.gz \
    -b /tmp/senior/ag160127/dwi.bval \
    -r /tmp/senior/ag160127/dwi.bvec \
    -t /neurospin/senior/nsap/data/V0/nifti/ag160127/000002_3DT1/000002_3DT1.nii.gz \
    -c /tmp/senior/ag160127/acqp.txt \
    -i /tmp/senior/ag160127/index.txt \
    -m SIEMENS \
    -p j \
    -o /tmp/senior/ag160127 \
    -w 0.00027 \
    -Z 45 \
    -V 2 \
    -F /etc/fsl/5.0.11/fsl.sh
python $HOME/git/pyconnectome/pyconnectome/scripts/pyconnectome_dmri_preproc \
    -s ag160127 \
    -d /tmp/senior/ag160127_/dwi.nii.gz \
    -b /tmp/senior/ag160127_/dwi.bval \
    -r /tmp/senior/ag160127_/dwi.bvec \
    -t /neurospin/senior/nsap/data/V0/nifti/ag160127/000002_3DT1/000002_3DT1.nii.gz \
    -c /tmp/senior/ag160127_/acqp.txt \
    -i /tmp/senior/ag160127_/index.txt \
    -m SIEMENS \
    -p j \
    -o /tmp/senior/ag160127_ \
    -w 0.00027 \
    -Z 45 \
    -V 2 \
    -F /etc/fsl/5.0.11/fsl.sh \
    -P /neurospin/senior/nsap/data/V0/nifti/ag160127/000018_B0MAP/000018_B0MAP.nii.gz \
    -M /neurospin/senior/nsap/data/V0/nifti/ag160127/000017_B0MAP/000017_B0MAP.nii.gz \
    -T 2.46 \
    -A 2 \
    -Q 0.55 \
    -B 0.4
python $HOME/git/pyconnectome/pyconnectome/scripts/pyconnectome_dmri_preproc \
    -s ag160127 \
    -d /tmp/senior/ag160127_/dwi.nii.gz \
    -b /tmp/senior/ag160127_/dwi.bval \
    -r /tmp/senior/ag160127_/dwi.bvec \
    -t /neurospin/senior/nsap/data/V0/nifti/ag160127/000002_3DT1/000002_3DT1.nii.gz \
    -c /tmp/senior/ag160127_/acqp.txt \
    -i /tmp/senior/ag160127_/index.txt \
    -m SIEMENS \
    -p j \
    -o /tmp/senior/ag160127_ \
    -w 0.00027 \
    -Z 45 \
    -V 2 \
    -F /etc/fsl/5.0.11/fsl.sh \
    -P /neurospin/senior/nsap/data/V0/nifti/ag160127/000018_B0MAP/000018_B0MAP.nii.gz \
    -M /neurospin/senior/nsap/data/V0/nifti/ag160127/000017_B0MAP/000017_B0MAP.nii.gz \
    -T 2.46 \
    -A 2 \
    -W /tmp/senior/ag160127_/2-Susceptibility/epi2struct_fast_wmseg.nii.gz \
    -L /tmp/senior/ag160127_/test_nodiff_mask.nii.gz \
    -Q 0.55 \
    -B 0.4
"""


def is_file(filepath):
    """ Check file's existence - argparse 'type' argument.
    """
    if not os.path.isfile(filepath):
        raise argparse.ArgumentError("File does not exist: %s" % filepath)
    return filepath


def is_directory(dirarg):
    """ Type for argparse - checks that directory exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The directory '{0}' does not exist!".format(dirarg))
    return dirarg


# # Parse input arguments
# def get_cmd_line_args():
#     """
#     Create a command line argument parser and return a dict mapping
#     <argument name> -> <argument value>.
#     """
#     parser = argparse.ArgumentParser(
#         prog="python pyconnectome_preproc_steps",
#         description=textwrap.dedent(DOC),
#         formatter_class=RawTextHelpFormatter)

#     # Required arguments
#     required = parser.add_argument_group("required arguments")
#     required.add_argument(
#         "-s", "--subject",
#         required=True,
#         help="Subject ID.")
#     required.add_argument(
#         "-d", "--dwi",
#         type=is_file, required=True, metavar="<path>",
#         help="Path to the DWI image file.")
#     required.add_argument(
#         "-b", "--bval",
#         type=is_file, required=True, metavar="<path>",
#         help="Path to the bval file.")
#     required.add_argument(
#         "-r", "--bvec",
#         type=is_file, required=True, metavar="<path>",
#         help="Path to the bvec file.")
#     required.add_argument(
#         "-t", "--t1",
#         type=is_file, required=True, metavar="<path>",
#         help="Path to the T1 image file.")
#     required.add_argument(
#         "-c", "--acqp",
#         type=is_file, required=True, metavar="<path>",
#         help="Path to the FSL eddy acqp file.")
#     required.add_argument(
#         "-i", "--index",
#         type=is_file, required=True, metavar="<path>",
#         help="Path to the FSL eddy index file.")
#     required.add_argument(
#         "-w", "--dwell-time",
#         type=float, required=True, metavar="<dwell time>",
#         help="The dwell time or effective echo spacing.")
#     required.add_argument(
#         "-x", "--readout-time",
#         type=float, required=True, metavar="<readout time>",
#         help="The readout time.")
#     required.add_argument(
#         "-p", "--phase-encode-dir",
#         type=str, required=True,
#         help="Phase encoding direction.")
#     required.add_argument(
#         "-m", "--manufacturer",
#         type=str, required=True,
#         help="Scanner manufacturer.")
#     required.add_argument(
#         "-o", "--outdir",
#         type=is_directory, required=True, metavar="<path>",
#         help="Path to the output directory.")

#     # Optional argument
#     parser.add_argument(
#         "-O", "--topup-b0",
#         type=is_file, metavar="<path>", nargs="+",
#         help="The b0 data acquired in opposite phase enc. direction.")
#     parser.add_argument(
#         "-D", "--topup-b0-dir",
#         metavar="<path>", nargs="+", choices=("i", "i-", "j", "j-"),
#         help="The b0 data enc.directions.")
#     parser.add_argument(
#         "-K", "--t1-mask",
#         type=is_file, metavar="<path>",
#         help="Path to the t1 brain mask image.")
#     parser.add_argument(
#         "-J", "--nodiff-mask",
#         type=is_file, metavar="<path>",
#         help="Path to the t1 brain mask image.")
#     parser.add_argument(
#         "-L", "--mag-mask",
#         type=is_file, metavar="<path>",
#         help="Path to the magnitude mask image.")
#     parser.add_argument(
#         "-W", "--wm-seg",
#         type=is_file, metavar="<path>",
#         help="Path to the t1 white matter segmentation image.")
#     parser.add_argument(
#         "-C", "--clean", action="store_true",
#         help="Delete brain suite susceptibility correction generated "
#              "intermediate files.")
#     required.add_argument(
#         "-Z", "--nthread",
#         type=int, default=1,
#         help="Number of thread for brainsuite.")
#     parser.add_argument(
#         "-S", "--skip-susceptibility", action="store_true",
#         help="Do not perform susceptibility correction.")
#     parser.add_argument(
#         "-M", "--magnitude",
#         type=is_file, metavar="<path>",
#         help="Two magnitude fieldmap image from a SIEMENS scanner (one for "
#              "each echo time).")
#     parser.add_argument(
#         "-P", "--phase",
#         type=is_file, metavar="<path>",
#         help="Phase difference fieldmap image from a SIEMENS scanner.")
#     parser.add_argument(
#         "-F", "--fsl-config",
#         type=is_file, metavar="<path>",
#         help="Bash script initializing FSL's environment.")
#     # parser.add_argument(
#     #     "-Q", "--echo-spacing",
#     #     type=float,
#     #     help=("the acquisition time in msec between 2 centers of 2 "
#     #           "consecutively acquired lines in k-space."))
#     # parser.add_argument(
#     #     "-A", "--parallel-acceleration-factor",
#     #     type=float, default=1.,
#     #     help="the number of parallel acquisition in the k-space plane.")
#     parser.add_argument(
#         "-T", "--delta-te",
#         type=float,
#         help=("the difference in msec between the 2 echoes of the B0 magnitude"
#               " map."))
#     parser.add_argument(
#         "-B", "--bet-threshold-t1",
#         type=float, default=0.5,
#         help="bet threshold for t1 brain extraction.")
#     parser.add_argument(
#         "-N", "--bet-threshold-nodiff",
#         type=float, default=0.25,
#         help="bet threshold for nodiff brain extraction.")
#     parser.add_argument(
#         "-G", "--bet-threshold-magnitude",
#         type=float, default=0.65,
#         help="bet threshold for magnitude brain extraction.")
#     parser.add_argument(
#         "-V", "--verbose",
#         type=int, choices=[0, 1, 2], default=2,
#         help="Increase the verbosity level: 0 silent, [1, 2] verbose.")

#     # Create a dict of arguments to pass to the 'main' function
#     args = parser.parse_args()
#     kwargs = vars(args)
#     verbose = kwargs.pop("verbose")
#     if kwargs["fsl_config"] is None:
#         kwargs["fsl_config"] = DEFAULT_FSL_PATH
#     return kwargs, verbose


# """
# Parse the command line.
# """
# inputs, verbose = get_cmd_line_args()
# runtime = {
#     "tool": "pyconnectome_dmri_preproc",
#     "tool_version": pyconnectome.__version__,
#     "timestamp": datetime.now().isoformat(),
#     "fsl_version": FSLWrapper([], shfile=inputs["fsl_config"]).version}
# outputs = {}
# outdir = os.path.join(inputs["outdir"], inputs["subject"])
# if not os.path.isdir(outdir):
#     raise ValueError("Please call first 'pyconnectome_get_eddy_data'.")
# if verbose > 0:
#     pprint("[info] Starting dMRI preprocessings...")
#     pprint("[info] Runtime:")
#     pprint(runtime)
#     pprint("[info] Inputs:")
#     pprint(inputs)
# if version.parse(runtime["fsl_version"]) < version.parse("5.0.11"):
#     raise ValueError("This script need FSL version >= 5.0.11 in order to "
#                      "work properly.")

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
        # os.mkdir(reshape_outdir)
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
    # # Crop neck with FSL robust fov
    # if inputs["t1_mask"] is None:
    #     t1 = os.path.join(reshape_outdir, "robust_fov")
    #     cropped_trf = t1 + ".txt"
    #     robustfov(
    #         input_file=inputs["t1"],
    #         output_file=t1,
    #         brain_size=170,
    #         matrix_file=cropped_trf,
    #         fsl_sh=inputs["fsl_config"])
    #     cropped_und_file = t1 + "_und.nii.gz"
    #     cropped_und_file, _ = flirt(
    #         in_file=t1 + ".nii.gz",
    #         ref_file=inputs["t1"],
    #         out=cropped_und_file,
    #         init=cropped_trf,
    #         applyxfm=True,
    #         verbose=verbose,
    #         shfile=inputs["fsl_config"])
    #     t1 = cropped_und_file
    # else:
    outputs["t1"] = t1

    # Split nodiff from diffusion weighted: one shell expected
    # TODO: use template b0 image
    # nodiff_mean, dwis = extract_dwi_shells(
    #     dwi_nii_path=dwi,
    #     bvals_path=bval,
    #     bvecs_path=bvec,
    #     outdir=reshape_outdir)
    # if len(dwis) > 1:
    #     print("[warn] '{0}' shells detected: do not use FA map."
    #           .format(len(dwis)))
    # outputs["nodif"] = nodiff_mean

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
    # Get brain from nodiff with bet or by registering t1 mask to nodiff space
    # NOW WE FORCE T1 MASK
    # if inputs["nodiff_mask"] is not None:
    #     # Apply nodiff brain mask to extract nodiff brain
    #     nodif_brain = apply_mask(
    #         input_file=nodiff_mean,
    #         output_fileroot=os.path.join(reshape_outdir, "nodiff_brain"),
    #         mask_file=inputs["nodiff_mask"],
    #         fslconfig=inputs["fsl_config"])
    #     nodif_brain_mask = inputs["nodiff_mask"]
    # else:
    #     # Extract brain with bet2 from nodiff dwi volume
    #     nodif_brain, nodif_brain_mask, _, _, _, _, _, _, _, _, _ = bet2(
    #         input_file=nodiff_mean,
    #         output_fileroot=reshape_outdir,
    #         mask=True,
    #         skull=False,
    #         f=inputs["bet_threshold_nodiff"],
    #         shfile=inputs["fsl_config"])

    # Get nodiff_mask from t1_mask, t1 and nodiff
    # Compute the transformation from nodiff to t1
    nodiff_to_t1 = os.path.join(reshape_outdir, "nodiff_to_t1.nii.gz")
    omat1 = os.path.join(reshape_outdir, "nodiff_to_t1.mat")
    omat2 = os.path.join(reshape_outdir, "t1_to_nodiff.mat")
    # fsl_cmd = ["flirt", "-in", nodiff_mean, "-ref", t1, "-out", nodiff_to_t1,
    #            "-dof", "6", "-omat", omat1]
    # check_command(fsl_cmd[0])
    # execute_command(fsl_cmd)
    flirt(
        in_file=outputs["nodif"],
        ref_file=t1,
        out=nodiff_to_t1,
        dof=6,
        omat=omat1)

    # Inverse the mat to get t1 to nodiff
    fsl_cmd = ["convert_xfm", "-omat", omat2, "-inverse", omat1]
    check_command(fsl_cmd[0])
    execute_command(fsl_cmd)

    # Apply the inverse to t1_mask to get nodiff_mask
    nodiff_mask = os.path.join(reshape_outdir, "nodiff_brain_mask.nii.gz")
    # fsl_cmd = ["flirt", "-in", t1_mask, "-ref", nodiff_mean, "-out",
    #            nodiff_mask, "-interp", "nearestneighbour",
    #            "-init", omat1, "-applyxfm"]
    # check_command(fsl_cmd[0])
    # execute_command(fsl_cmd)
    flirt(
        in_file=t1_mask,
        ref_file=outputs["nodif"],
        applyxfm=True,
        init=omat2,
        out=nodiff_mask,
        interp="nearestneighbour")

    # Extract
    # Apply nodiff brain mask to extract nodiff brain
    # flirt entre nodiff et t1 avec dof 6
    # flirt avec mask t1 -applyxfm
    input_file = outputs["nodif"]
    mask_file = nodiff_mask
    output_fileroot_fslmaths = os.path.join(reshape_outdir, "nodiff_brain")
    fsl_cmd = ["fslmaths", input_file,
               "-mas", mask_file,
               output_fileroot_fslmaths]
    check_command(fsl_cmd[0])
    execute_command(fsl_cmd)
    nodiff_brain = glob.glob(output_fileroot_fslmaths + ".*")[0]
    nodiff_brain_mask = nodiff_mask

    # nodif_brain = apply_mask(
    #     input_file=nodiff_mean,
    #     output_fileroot=os.path.join(reshape_outdir, "nodiff_brain"),
    #     mask_file=inputs["nodiff_mask"],
    #     fslconfig=inputs["fsl_config"])
    # else:
    #     # Extract brain with bet2 from nodiff dwi volume
    #     nodif_brain, nodif_brain_mask, _, _, _, _, _, _, _, _, _ = bet2(
    #         input_file=nodiff_mean,
    #         output_fileroot=reshape_outdir,
    #         mask=True,
    #         skull=False,
    #         f=inputs["bet_threshold_nodiff"],
    #         shfile=inputs["fsl_config"])

    outputs["nodif_brain"] = nodiff_brain
    outputs["nodif_brain_mask"] = nodiff_brain_mask
    return outputs


"""
2.1- Compute susceptibility correction.
"""
# ###########ADD choice
# -arbre de décision:
# --1) top up
# --2) synb0
# --3) recalage T1
# Run susceptibility correction


def Compute_and_Apply_susceptibility_correction(subject,
                                                t1,
                                                dwi,
                                                outdir,
                                                outputs,
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

    # #####################################""
    #
    # if not inputs["skip_susceptibility"]:

    # # FSL with B0 maps
    # if inputs["magnitude"] is not None and inputs["phase"] is not None:

    #     # Select case
    #     susceptibility_method = "b0map"

    #     # Deal with phase encoding direction
    #     if inputs["phase_encode_dir"] == "i":
    #         phase_enc_dir = "x"
    #     elif inputs["phase_encode_dir"] == "i-":
    #         phase_enc_dir = "-x"
    #     elif inputs["phase_encode_dir"] == "j":
    #         phase_enc_dir = "y"
    #     elif inputs["phase_encode_dir"] == "j-":
    #         phase_enc_dir = "-y"
    #     else:
    #         raise ValueError("Incorrect phase encode direction: {0}...".format(
    #                         inputs["phase_encode_dir"]))

    #     if inputs["mag_mask"] is None:
    #         # Create a mask for the magnitude image: strict threshold in order
    #         # to avoid border outliers
    #         mag_brain, mag_brain_mask, _, _, _, _, _, _, _, _, _ = bet2(
    #             input_file=inputs["magnitude"],
    #             output_fileroot=susceptibility_dir,
    #             mask=True,
    #             skull=False,
    #             f=inputs["bet_threshold_magnitude"],
    #             shfile=inputs["fsl_config"])
    #         # erode_mask_file = os.path.join(
    #         #    susceptibility_dir, "ero_" + os.path.basename(mag_brain_mask))
    #         # im = nibabel.load(inputs["magnitude"])
    #         # min_spacing = min(im.header.get_zooms()[:3])
    #         # erode(
    #         #    input_file=mag_brain_mask,
    #         #    output_file=erode_mask_file,
    #         #    radius=min_spacing * 2,
    #         #    fslconfig=inputs["fsl_config"])
    #         # erode_fileroot = os.path.join(
    #         #    susceptibility_dir, "ero_" + os.path.basename(
    #         #        mag_brain).split(".")[0])
    #         # erode_file = apply_mask(
    #         #    input_file=mag_brain,
    #         #    output_fileroot=erode_fileroot,
    #         #    mask_file=erode_mask_file,
    #         #    fslconfig=inputs["fsl_config"])

    #     else:
    #         mag_brain_mask = inputs["mag_mask"]
    #         mag_brain = os.path.join(
    #             susceptibility_dir, os.path.basename(
    #                 inputs["magnitude"]).replace(".nii.gz", "_brain.nii.gz"))
    #         apply_mask(
    #             input_file=inputs["magnitude"],
    #             output_fileroot=mag_brain.replace(".nii.gz", ""),
    #             mask_file=mag_brain_mask,
    #             fslconfig=inputs["fsl_config"])
    #         # erode_file = None
    #     outputs["mag_brain"] = mag_brain
    #     outputs["mag_brain_mask"] = mag_brain_mask
    #     # outputs["mag_brain_eroded"] = erode_file

    #     # Prepare the fieldmap
    #     # > Prepare a fieldmap from SIEMENS scanner into a rad/s fieldmap
    #     fieldmap_file = os.path.join(susceptibility_dir, "fieldmap.nii.gz")
    #     fieldmap_file, fieldmap_hz_file = fsl_prepare_fieldmap(
    #         manufacturer=inputs["manufacturer"].upper(),
    #         phase_file=inputs["phase"],
    #         brain_magnitude_file=mag_brain,
    #         output_file=fieldmap_file,
    #         delta_te=str(inputs["delta_te"]),
    #         fsl_sh=inputs["fsl_config"])

    #     # > Smooth fieldmap
    #     im = nibabel.load(fieldmap_hz_file)
    #     sigma = min(im.header.get_zooms()[:3]) / 2.
    #     smooth_fieldmap(
    #         fieldmap_hz_file,
    #         inputs["dwell_time"],
    #         fieldmap_hz_file,
    #         sigma=sigma,
    #         fsl_sh=inputs["fsl_config"])

    #     # > Replace the zero values in the fieldmap by the last non null value
    #     #   in the phase encoding direction.
    #     # fieldmap_reflect(
    #     #     fieldmap=fieldmap_hz_file,
    #     #     phase_enc_dir=inputs["phase_encode_dir"],
    #     #     output_file=fieldmap_hz_file)
    #     fieldmap_hz_to_diff_file = fieldmap_hz_file.replace(
    #         ".nii.gz", "_to_diff.nii.gz")
    #     flirt(
    #         in_file=fieldmap_hz_file,
    #         ref_file=nodif_brain,
    #         out=fieldmap_hz_to_diff_file,
    #         interp="nearestneighbour",
    #         applyxfm=True,
    #         shfile=inputs["fsl_config"])
    #     outputs["fieldmap"] = fieldmap_file
    #     outputs["fieldmap_smooth_hz"] = fieldmap_hz_file
    #     outputs["sigma"] = sigma
    #     outputs["fieldmap_smooth_hz_to_diff"] = fieldmap_hz_to_diff_file

    #     # Simultaneous coregistration and fieldmap unwarping
    #     # The shift image contains a value that represents the amount of
    #     # translation (shift), in units of voxels, at each voxel that would
    #     # need to be applied in the direction specified by the shiftdir.
    #     # TODO: check effective echo spacing computation
    #     dwi_corrected_fileroot = os.path.join(susceptibility_dir, "epi2struct")
    #     # echo_spacing = inputs["echo_spacing"] / (
    #     #     1000 * inputs["parallel_acceleration_factor"])
    #     corrected_epi_file, warp_file, distortion_map = epi_reg(
    #         epi_file=inputs["dwi"],
    #         structural_file=inputs["t1"],
    #         brain_structural_file=brain_img,
    #         output_fileroot=dwi_corrected_fileroot,
    #         fieldmap_file=fieldmap_file,
    #         effective_echo_spacing=inputs["dwell_time"],
    #         magnitude_file=inputs["magnitude"],
    #         brain_magnitude_file=mag_brain,
    #         phase_encode_dir=phase_enc_dir,
    #         wmseg_file=inputs["wm_seg"],
    #         fsl_sh=inputs["fsl_config"])
    #     nodif_to_t1_mat = os.path.join(susceptibility_dir, "epi2struct.mat")
    #     t1_to_nodif_mat = os.path.join(
    #         susceptibility_dir, "epi2struct_inv.mat")
    #     outputs["corrected_epi_file"] = corrected_epi_file
    #     outputs["pixel_shift_map"] = distortion_map

    #     # Reorganize deformation field volume
    #     #if not os.path.isfile(distortion_map):
    #     #    raise ValueError("Unavailable warp file: {0}".format(
    #     #        distortion_map))
    #     #distortion_img = nibabel.load(distortion_map)
    #     #distortion_img_data = distortion_img.get_data()
    #     #distortion_img_data.shape += (1,)
    #     #spacing = distortion_img.header.get_zooms()
    #     #zero_field = numpy.zeros(distortion_img_data.shape)
    #     #if phase_enc_dir in ("x", "x-"):
    #     #    deformation_field = numpy.concatenate(
    #     #        (distortion_img_data * spacing[0], zero_field, zero_field),
    #     #        axis=3)
    #     #elif phase_enc_dir in ("y", "y-"):
    #     #    deformation_field = numpy.concatenate(
    #     #        (zero_field, distortion_img_data * spacing[1], zero_field),
    #     #        axis=3)
    #     #else:
    #     #    raise ValueError("Incorrect phase encode direction: {0}".format(
    #     #                     phase_enc_dir))
    #     #deformation_field_img = nibabel.Nifti1Image(
    #     #    deformation_field, distortion_img.affine)
    #     ##deformation_field_file = os.path.join(
    #     #    susceptibility_dir, "field.nii.gz")
    #     #nibabel.save(deformation_field_img, deformation_field_file)
    #     outputs["warp_file"] = warp_file # deformation_field_file

    synB0 = "odo"
    # FSL topup

    #########""to remove ##########"
    print("topup_b0 : ", topup_b0)
    print("topup_b0_dir : ", topup_b0_dir)
    for i in topup_b0_dir:
        print(i)
    print("readout_time : ", readout_time)
    # #################################"

    if topup_b0 is not None and \
       topup_b0_dir is not None and \
       readout_time is not None:
        #########""to remove ##########"
        topup_b0 = topup_b0.split(",")
        topup_b0_dir = topup_b0_dir.split(",")
        print("topup_b0 : ", topup_b0)
        print("topup_b0_dir : ", topup_b0_dir)
        print("readout_time : ", readout_time)
        print("IN")
        # #################################"
        # Select case
        susceptibility_method = "topup"

        # Topup
        fieldmap_hz_to_diff_file, corrected_b0s, mean_corrected_b0s = topup(
            b0s=topup_b0,
            phase_enc_dirs=topup_b0_dir,
            readout_time=readout_time,
            outroot=susceptibility_dir)
        # outputs["fieldmap_hz_to_diff"] = fieldmap_hz_to_diff_file
        outputs["fieldmap_hz"] = fieldmap_hz_to_diff_file
        outputs["mean_corrected_b0s"] = mean_corrected_b0s

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

    # # Brainsuite
    # else:

    #     # Select case
    #     susceptibility_method = "reg"

    #     # Deal with phase encoding direction
    #     if inputs["phase_encode_dir"] == "i":
    #         phase_enc_dir = "x"
    #     elif inputs["phase_encode_dir"] == "i-":
    #         phase_enc_dir = "x-"
    #     elif inputs["phase_encode_dir"] == "j":
    #         phase_enc_dir = "y"
    #     elif inputs["phase_encode_dir"] == "j-":
    #         phase_enc_dir = "y-"
    #     else:
    #         raise ValueError("Incorrect phase encode direction: {0}...".format(
    #                         inputs["phase-encode-dir"]))

    #     warp_file = os.path.join(
    #         susceptibility_dir,
    #         "{0}.dwi.RAS.correct.distortion.map.nii.gz".format(
    #             inputs["subject"]))
    #     dwi_wo_susceptibility = os.path.join(
    #         susceptibility_dir, "{0}.dwi.RAS.correct.nii.gz".format(
    #                         inputs["subject"]))
    #     t1_in_dwi_space = os.path.join(
    #         susceptibility_dir, "{0}.bfc.D_coord.nii.gz".format(
    #             inputs["subject"]))

    #     # Start correction
    #     if (not os.path.isfile(warp_file) and not
    #             os.path.isfile(dwi_wo_susceptibility) and not
    #             os.path.isfile(t1_in_dwi_space)):
    #         (dwi_wo_susceptibility, bval, bvec, t1_in_dwi_space,
    #         bo_in_t1_space, t1_brain) = susceptibility_correction_wo_fieldmap(
    #             outdir=susceptibility_dir,
    #             t1=inputs["t1"],
    #             dwi=inputs["dwi"],
    #             bval=inputs["bval"],
    #             bvec=inputs["bvec"],
    #             subject_id=inputs["subject"],
    #             phase_enc_dir=phase_enc_dir,
    #             t1_mask=brain_mask,
    #             nodif_mask=nodif_brain_mask,
    #             fsl_sh=inputs["fsl_config"],
    #             nthread=inputs["nthread"])
    #     else:
    #         print("[Warnings]: Using already existing BrainSuite outputs."
    #             "To re-run BrainSuite analysis, please clean '{0}'".format(
    #                 susceptibility_dir))

    #     # If necessary, clean intermediate outputs
    #     if inputs["clean"]:
    #         for basnename in [
    #                 "{0}.bfc.biasfield.nii.gz",
    #                 "{0}.bfc.D_coord.nii.gz",
    #                 "{0}.bfc.D_coord.rigid_registration_result.mat",
    #                 "{0}.bfc.nii.gz",
    #                 "{0}.dwi.bmat",
    #                 "{0}.dwi.RAS.bmat",
    #                 "{0}.dwi.RAS.bvec",
    #                 "{0}.dwi.RAS.less_csf.mask.nii.gz",
    #                 "{0}.dwi.RAS.nii.gz",
    #                 "{0}.BDPSummary.txt",
    #                 "{0}.dwi_fov.D_coord.mask.nii.gz",
    #                 "{0}.dwi_fov.T1_coord.mask.nii.gz",
    #                 "{0}.dwi.RAS.correct.0_diffusion.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.axial.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.bfield.nii.gz",
    #                 "{0}.dwi.RAS.correct.FA.color.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.FA.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.L2.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.L3.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.mADC.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.MD.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.nii.gz",
    #                 "{0}.dwi.RAS.correct.radial.T1_coord.nii.gz",
    #                 "{0}.dwi.RAS.correct.T1_coord.eig.nii.gz",
    #                 "{0}.tensor.T1_coord.bst"]:
    #             to_rm_file = os.path.join(
    #                 susceptibility_dir, basename.format(inputs["subject"]))
    #             os.remove(to_rm_file)
    #     outputs["pixel_shift_map"] = warp_file

    #     # Reorganize deformation field volume
    #     if not os.path.isfile(warp_file):
    #         raise ValueError("Unavailable warp file: {0}".format(warp_file))
    #     distortion_img = nibabel.load(warp_file)
    #     distortion_img_data = distortion_img.get_data()
    #     distortion_img_data.shape += (1,)
    #     spacing = distortion_img.header.get_zooms()
    #     zero_field = numpy.zeros(distortion_img_data.shape)
    #     if phase_enc_dir in ("x", "x-"):
    #         deformation_field = numpy.concatenate(
    #             (distortion_img_data * spacing[0], zero_field, zero_field),
    #             axis=3)
    #     elif phase_enc_dir in ("y", "y-"):
    #         deformation_field = numpy.concatenate(
    #             (zero_field, distortion_img_data * spacing[1], zero_field),
    #             axis=3)
    #     else:
    #         raise ValueError("Incorrect phase encode direction: {0}".format(
    #                         phase_enc_dir))
    #     deformation_field_img = nibabel.Nifti1Image(
    #         deformation_field, distortion_img.affine)
    #     deformation_field_file = os.path.join(
    #         susceptibility_dir, "field.nii.gz")
    #     nibabel.save(deformation_field_img, deformation_field_file)
    #     outputs["warp_file"] = deformation_field_file

    #     # Create a field map from the pixel shift warp
    #     # fieldmap_file = os.path.join(brainsuite_dir, "fieldmap.nii.gz")
    #     # fieldmap_file, fieldmap_hz_to_diff_file = pixel_shift_to_fieldmap(
    #     #     pixel_shift_file=warp_file,
    #     #     dwell_time=inputs["dwell_time"],
    #     #     output_file=fieldmap_file,
    #     #     fsl_sh=inputs["fsl_config"])
    #     outputs["fieldmap"] = None

    elif synB0 == "odo":
        pass

    # Create an zero deformation field
    else:
        # Coregistration only
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
    if susceptibility_method == "topup":

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
    # Apply susceptibility correction to nodiff brain mask
    nodif_brain_mask = outputs["nodif_brain_mask"]
    # nodif_brain_mask_undistorted = os.path.join(
    #     outdir,
    #     os.path.basename(nodif_brain_mask).replace(".nii.gz",
    #                                                "_corrected.nii.gz"))
    # if susceptibility_method == "topup":
    #     nodif_brain_, nodif_brain_mask_, _, _, _, _, _, _, _, _, _ = bet2(
    #         input_file=mean_corrected_b0s,
    #         output_fileroot=susceptibility_dir,
    #         mask=True,
    #         skull=False,
    #         f=inputs["bet_threshold_nodiff"],
    #         shfile=inputs["fsl_config"])
    #     shutil.copy2(nodif_brain_mask_, nodif_brain_mask_undistorted)
    # else:
    #     in_file = nodif_brain_mask
    #     ref_file = nodif_brain_mask
    #     out_file = nodif_brain_mask_undistorted
    #     warp_file = deformation_field_file
    #     interp = "nn"
    #     fsl_cmd = ["applywarp",
    #                "-i", in_file,
    #                "-r", ref_file,
    #                "-o", out_file,
    #                "-w", warp_file,
    #                "--interp={0}".format(interp)]
    #     check_command(fsl_cmd)
    #     execute_command(fsl_cmd)
    outputs["nodif_brain_mask_undistorted"] = nodif_brain_mask
    outputs["2_method"] = susceptibility_method
    return outputs


"""
3- Eddy current and motion correction.
"""


def eddy_current_and_motion_correction(subject,
                                       t1,
                                       acqp,
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

    # field
    field = None
    if outputs["fieldmap_hz"] is not None:
        field = outputs["fieldmap_hz"].replace(".nii.gz", "")

    # eddy
    corrected_dwi, corrected_bvec = eddy(
        dwi=outputs["eddy_dwi_input"],
        dwi_brain_mask=outputs["nodif_brain_mask_undistorted"],
        acqp=acqp,
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

    # #########################################################################
    # if susceptibility_method == "reg":

    #     # t1 has already been registered to dwi by BrainSuite
    #     shutil.copy(t1_in_dwi_space, t1_to_dwi)

    # else:
    # #########################################################################

    # For BrainSuite correction, topup correction or no correction, T1 to DWI
    # transformation matrix has been saved by epi_reg
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
5- Create QC report
"""


def create_qc_report(subject, t1, outdir, outputs):
    # Create PDF comparison susceptibility correction/ no susceptibility 
    # correction
    outdir = os.path.join(outdir, "sub-"+subject)
    qc_dir = os.path.join(outdir, "5-QC")
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

    outputs["QC_susceptibility_pdf"] = filename

    """
    Update the outputs and save them and the inputs in a 'logs' directory.
    """

    # logdir = os.path.join(outdir, "logs")
    # if not os.path.isdir(logdir):
    #     os.mkdir(logdir)
    # for name, final_struct in [("inputs", inputs), ("outputs", outputs),
    #                         ("runtime", runtime)]:
    #     log_file = os.path.join(logdir, "{0}.json".format(name))
    #     with open(log_file, "wt") as open_file:
    #         json.dump(final_struct, open_file, sort_keys=True, check_circular=True,
    #                 indent=4)
    # if verbose > 1:
    print("[final]")
    pprint(outputs)
    return 0

