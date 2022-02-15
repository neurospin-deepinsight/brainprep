#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under
# the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""usefull functions for cat12vbm correlation based automatic QC.

@author: benoit.dufumier
@author: julie.victor


"""

import re
import xml.etree.ElementTree as ET
# import csv
import traceback
from collections import OrderedDict
from click import FileError

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import nibabel
from nilearn import plotting
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfFileMerger
import os
import scipy
import nilearn
import nilearn.masking
import subprocess

# initialization for get_keys() function
participant_re = re.compile("sub-([^_/]+)")
session_re = re.compile("ses-([^_/]+)")
run_re = re.compile("run-([a-zA-Z0-9]+)")


def launch_cat12_qc(img_filenames,
                    mask_filenames,
                    root_cat12vbm,
                    output_dir,
                    inputscores):
    """ call qc functions """
    # concat qcscores
    scores = parse_xml_files_scoresQC(inputscores)
    fieldnames = ['participant_id', 'session', 'run', 'NCR', 'ICR', 'IQR']
    # scores_qccat = pd.DataFrame.from_dict(scores, orient='index')
    scores_qccat = pd.DataFrame(columns=fieldnames)
    for participant_id in scores:
        for session in scores[participant_id].keys():
            for (run, measures) in scores[participant_id][session].items():
                row = dict(participant_id=participant_id,
                           session=session,
                           run=run, **measures)
                scores_qccat = scores_qccat.append(row, ignore_index=True)

    # correlation
    imgs_arr, df, ref_img = img_to_array(img_filenames, sesrun_default=1)
    if mask_filenames is None:
        mask_img = compute_brain_mask(imgs_arr, ref_img)
        mask_arr = mask_img.get_fdata() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) == 1:
        mask_img = nibabel.load(mask_filenames[0])
        mask_arr = mask_img.get_fdata() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) > 1:
        assert len(mask_filenames) == len(imgs_arr), "The list of .nii masks \
                                                      must have the same \
                                                      length as the " \
                                                     "list of .nii input files"
        mask_glob = [nibabel.load(mask_filename).get_fdata() > 0
                     for mask_filename in mask_filenames]
        imgs_arr = imgs_arr.squeeze()[mask_glob]

    # create PCA file
    print("OUTPUT DIR:")
    print(output_dir)
    plot_pca(imgs_arr, df, output_dir)
    print("pca done")
    # create MEAN CORR file
    mean_corr = compute_mean_correlation(imgs_arr, df, output_dir)
    print("corr done")
    # Create qc.tsv file : concat MEAN CORR AND QC SCORES
    # mean_corr = pd.read_csv("PATH/cat12-12.6_vbm_qc/qc.tsv", sep= "\t")

    qc_table = concat_tsv(mean_corr, scores_qccat)
    path_qc = os.path.join(output_dir, "qc.tsv")
    qc_table = qc_table.sort_values(by=["IQR"])
    qc_table.to_csv(path_qc, index=False, sep='\t')

    # create nii brain images pdf ordored by mean correlation
    mean_corr = mean_corr.values
    niipdf = os.path.join(output_dir, 'nii_plottings.pdf')
    nii_filenames_sorted = [df[df['participant_id'].eq(id)].ni_path.values[0]
                            for (id, _, _, _) in mean_corr]
    pdf_plottings(nii_filenames_sorted, mean_corr, niipdf, limit=None)
    print("nii image pdf done")

    # cat12vbm reports pdf ordored by mean correlation
    reportpdf = os.path.join(output_dir, 'cat12_reports.pdf')
    nii_filenames_pdf = mwp1toreport(nii_filenames_sorted, root_cat12vbm)
    if len(nii_filenames_pdf) <= 100:
        pdf_cat(nii_filenames_pdf, reportpdf)
    else:
        pdf_cat2(nii_filenames_pdf, reportpdf, 100)
    print("concat report done")

    return 0


def launch_qr_qc(img_filenames, mask_filenames, output_dir):
    """ call qc functions """

    # correlation
    imgs_arr, df, ref_img = img_to_array(img_filenames, sesrun_default=1)
    if mask_filenames is None:
        mask_img = compute_brain_mask(imgs_arr, ref_img)
        mask_arr = mask_img.get_fdata() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) == 1:
        mask_img = nibabel.load(mask_filenames[0])
        mask_arr = mask_img.get_fdata() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) > 1:
        assert len(mask_filenames) == len(imgs_arr), \
                "The list of .nii masks must have the same length as the " \
                "list of .nii input files"
        mask_glob = [nibabel.load(mask_filename).get_fdata() > 0
                     for mask_filename in mask_filenames]
        imgs_arr = imgs_arr.squeeze()[mask_glob]

    # # create PCA file
    plot_pca(imgs_arr, df, output_dir)
    print("pca done")
    # Create MEAN CORR file
    mean_corr = compute_mean_correlation(imgs_arr, df, output_dir)
    print("corr done")
    # Create qc.tsv file : MEAN CORR
    path_qc = os.path.join(output_dir, "qc.tsv")
    mean_corr.to_csv(path_qc, index=False, sep='\t')
    # create nii brain images pdf ordored by mean correlation
    mean_corr = mean_corr.values
    niipdf = os.path.join(output_dir, 'nii_plottings.pdf')
    nii_filenames_sorted = [df[df['participant_id'].eq(id)].ni_path.values[0]
                            for (id, _, _, _) in mean_corr]
    pdf_plottings_qr(nii_filenames_sorted, mean_corr, niipdf, limit=None)
    print("nii image pdf done")

    return 0


def plot_pca(X, df_description, output_dir):
    """Plot nii image PCA of a specified cohort.

    Parameters
    ----------
    X: array
        the input data.
    df_description: pandas Dataframe
        descriptor of input data

    Saving
    -------
    PCA graph result save in pdf format.
    """
    # Assume that X has dimension (n_samples, ...)
    pca = PCA(n_components=2)
    # Do the SVD
    pca.fit(X.reshape(len(X), -1))
    # Apply the reduction
    PC = pca.transform(X.reshape(len(X), -1))
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.scatter(PC[:, 0], PC[:, 1])
    # Put an annotation on each data point
    for i, participant_id in enumerate(df_description['participant_id']):
        ax.annotate(participant_id, xy=(PC[i, 0], PC[i, 1]),
                    xytext=(4, 4), textcoords='offset pixels')

    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    plt.axis('equal')
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "pca.pdf")
    plt.savefig(pca_path)
    # plt.show()


def compute_mean_correlation(X, df_description, output_dir):
    """Compute mean correlation of a specified cohort.

    Parameters
    ----------
    X: array
        the input data.
    df_description: pandas Dataframe
        descriptor of input data

    Returns
    -------
    cor: pandas DataFrame
        Sorted input data description based on mean correlation.
        columns : 'participant_id', 'session', 'run', 'corr_mean'

    Saving
    -------
    Heatmap of mean correlation saved in pdf format.

    """

    # Compute the correlation matrix
    corr = np.corrcoef(X.reshape(len(X), -1))
    # if nan because of variance 0, put corr to 0,
    # problem encountered with runs of the same session probably
    for j in range(len(corr)):
        if np.isnan(corr[j]).any():
            corr[j] = np.nan_to_num(corr[j])
            print("nan in corr : \n", df_description['ni_path'][j])
    # Compute the Z-transformation of the correlation
    F = 0.5 * np.log((1. + corr) / (1. - corr))
    # Compute the mean value for each sample by masking the diagonal
    np.fill_diagonal(F, 0)
    # if inf because of corr =1 (except diag), replace inf by a big value
    # print if it is the case
    for i in range(len(F)):
        if np.isinf(F[i]).any():
            F[i] = np.nan_to_num(F[i])
            print("inf in F : \n", df_description['ni_path'][i])
    # average of F
    F_mean = F.sum(axis=1)/(len(F)-1)
    # check point
    if np.isnan(F_mean).any() or np.isnan(F).any():
        raise ValueError("F_mean contains nan {0},{1}, {2}"
                         .format(F, F_mean, np.isnan(F).any()))
    # reintroduce diagonal values for plot
    np.fill_diagonal(F, 1)
    # Get the index sorted by descending Z-corrected mean correlation values
    sort_idx = np.argsort(F_mean)
    # Get the corresponding ID
    participant_ids = df_description['participant_id'][sort_idx]
    sessions_ids = df_description['session'][sort_idx]
    run_ids = df_description['run'][sort_idx]
    Freorder = F[np.ix_(sort_idx, sort_idx)]
    plt.subplots(figsize=(10, 10))
    cmap = sns.color_palette("RdBu_r", 110)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(Freorder, mask=None, cmap=cmap, vmin=-1, vmax=1, center=0)
    corr_path = os.path.join(output_dir, "corr_mat.pdf")
    plt.savefig(corr_path)
    # plt.show()
    cor = pd.DataFrame(dict(participant_id=participant_ids,
                            session=sessions_ids, run=run_ids,
                            corr_mean=F_mean[sort_idx]))
    cor = cor.reindex(['participant_id', 'session', 'run', 'corr_mean'],
                      axis='columns')
    return cor


def pdf_plottings(nii_filenames, mean_corr, output_pdf, limit=None):
    """Plot sorted nii images of a specified cohort based on mean correlation.

    Parameters
    ----------
    nii_filenames: pandas Dataframe
        descriptor of input data.
    mean_corr: array
        the mean correlation data.
    output_pdf: filename of the output pdf.
    limit : maximal number of diapo in the pdf
        default=None

    Saving
    -------
    Nii images Diaporama sorted by mean correlation saved in pdf format.
    columns: wm cat12vbm NII image (1 slice), mwp1 NII images (6 slices)
    rows:sagittal, coronal, axial mri.


    """
    # initialise size
    max_range = limit or len(nii_filenames)
    # create pdf output
    pdf = PdfPages(output_pdf)
    # plot slices
    for i, nii_file in list(enumerate(nii_filenames))[:max_range]:
        fig = plt.figure(figsize=(30, 20))
        gs = GridSpec(3, 7, figure=fig)
        # mw
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        # wmp1
        ax4 = fig.add_subplot(gs[0, 1:])
        ax5 = fig.add_subplot(gs[1, 1:])
        ax6 = fig.add_subplot(gs[2, 1:])
        nii = nibabel.load(nii_file)
        nii_ref = nii_file.replace("mwp1", "wm")
        nii_ref = nibabel.load(nii_ref)
        plt.suptitle('Subject %s, session %s, run %s with mean '
                     'correlation %.3f'
                     % (mean_corr[i][0], mean_corr[i][1],
                        mean_corr[i][2], mean_corr[i][3]),
                     fontsize=40)
        # mw
        plotting.plot_anat(nii_ref, figure=fig, axes=ax1, dim=0, cut_coords=1,
                           display_mode='x')
        plotting.plot_anat(nii_ref, figure=fig, axes=ax2, dim=0, cut_coords=1,
                           display_mode='y')
        plotting.plot_anat(nii_ref, figure=fig, axes=ax3, dim=0, cut_coords=1,
                           display_mode='z')
        # wmp1
        plotting.plot_anat(nii, figure=fig, axes=ax4, dim=-1,
                           cut_coords=6, display_mode='x')
        plotting.plot_anat(nii, figure=fig, axes=ax5, dim=-1,
                           cut_coords=6, display_mode='y')
        plotting.plot_anat(nii, figure=fig, axes=ax6, dim=-1,
                           cut_coords=6, display_mode='z')
        # resize
        plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
        # save
        pdf.savefig()
        plt.close(fig)
    pdf.close()


def pdf_plottings_qr(nii_filenames, mean_corr, output_pdf, limit=None):
    """Plot sorted nii images of a specified cohort based on mean correlation.

    Parameters
    ----------
    nii_filenames: pandas Dataframe
        descriptor of input data.
    mean_corr: array
        the mean correlation data.
    output_pdf: filename of the output pdf.
    limit : maximal number of diapo in the pdf
        default=None

    Saving
    -------
    Nii images Diaporama sorted by mean correlation saved in pdf format.
    columns: wm cat12vbm NII image (1 slice), mwp1 NII images (6 slices)
    rows:sagittal, coronal, axial mri.


    """
    # initialise size
    max_range = limit or len(nii_filenames)
    # create pdf output
    pdf = PdfPages(output_pdf)
    # plot slices
    for i, nii_file in list(enumerate(nii_filenames))[:max_range]:
        fig = plt.figure(figsize=(30, 20))
        gs = GridSpec(3, 7, figure=fig)
        # t1w ori
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        # qr
        ax4 = fig.add_subplot(gs[0, 1:])
        ax5 = fig.add_subplot(gs[1, 1:])
        ax6 = fig.add_subplot(gs[2, 1:])
        nii = nibabel.load(nii_file)
        nii_ref = nii_file.replace("_desc-6apply", "")
        nii_ref = nii_ref.replace("/derivatives/quasi-raw/", "/rawdata/")
        if os.path.exists(nii_ref):
            pass
        else:
            nii_ref = nii_ref.replace("/psy_sbox/", "/psy/")
        assert os.path.exists(nii_ref), nii_ref
        nii_ref = nibabel.load(nii_ref)
        plt.suptitle('Subject %s, session %s, run %s with mean '
                     'correlation %.3f'
                     % (mean_corr[i][0], mean_corr[i][1],
                        mean_corr[i][2], mean_corr[i][3]),
                     fontsize=40)
        # t1w ori
        plotting.plot_anat(nii_ref, figure=fig, axes=ax1, dim=0, cut_coords=1,
                           display_mode='x')
        plotting.plot_anat(nii_ref, figure=fig, axes=ax2, dim=0, cut_coords=1,
                           display_mode='y')
        plotting.plot_anat(nii_ref, figure=fig, axes=ax3, dim=0, cut_coords=1,
                           display_mode='z')
        # qr
        plotting.plot_anat(nii, figure=fig, axes=ax4, dim=1,
                           cut_coords=6, display_mode='x')
        plotting.plot_anat(nii, figure=fig, axes=ax5, dim=1,
                           cut_coords=6, display_mode='y')
        plotting.plot_anat(nii, figure=fig, axes=ax6, dim=1,
                           cut_coords=6, display_mode='z')
        # resize
        plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
        # save
        pdf.savefig()
        plt.close(fig)
    pdf.close()


def pdf_cat(pdf_filenames, output_pdf):
    """Concatenation of pdf files in one big pdf.

    Parameters
    ----------
    pdf_filenames: list
        filenames of input data.
    output_pdf: filename of the output pdf.

    Saving
    -------
    Big cat12vbm pdf reports sorted by mean correlation.


    """
    pdfWriter = PdfFileWriter()
    for file in pdf_filenames:
        pdfFileObj = open(file, 'rb')
        pdfReader = PdfFileReader(pdfFileObj)
        for pageNum in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(pageNum)
            pdfWriter.addPage(pageObj)
    pdfOutput = open(output_pdf, 'wb')
    pdfWriter.write(pdfOutput)
    pdfOutput.close()


def pdf_cat2(pdf_filenames, output_pdf, batchsize):
    """Concatenation of pdf files in one big pdf.
    Usefull, in case of a large amount of data.

    Parameters
    ----------
    pdf_filenames: list
        filenames of input data.
    output_pdf: string
        filename of the output pdf.
    batchsize: int
        number of dia in one batch

    Saving
    -------
    Big cat12vbm pdf reports sorted by mean correlation.


    """
    folder_pdf = output_pdf.split(os.sep)[0:-1]
    folder_pdf = os.sep.join(folder_pdf)
    # holds a batch of pdf names
    batch_pdfs = []
    # collects all batches
    list_of_batches = []
    # make batches after this number of files
    print("Batchsize: {0}".format(str(batchsize)))

    # Loop over the list with all pdf names
    # and split them into batches (=batch_pdfs).
    # Collect each batch in list "list_of_batches".
    for count, pdf in enumerate(pdf_filenames, 1):
        batch_pdfs.append(pdf)
        if count % batchsize == 0:
            list_of_batches.append(batch_pdfs)
            batch_pdfs = []
        # if you loop longer than the number of pdfs, something is wrong. exit.
        if count > len(pdf_filenames) + 2:
            print('List count larger than number of PDFs.')
            os.sys.exit(1)

    list_of_batches.append(batch_pdfs)
    print("Number of batches: {0}".format(str(len(list_of_batches))))

    i = 1
    # loop over all batches (=list_of_batches)
    for batchlist in list_of_batches:
        print("Processing Batch: {0} with length: {1}".format(str(i),
              str(len(batchlist))))
        if len(batchlist) > 0:
            # Start the PDF merger for each batch and close it after the batch.
            #  If you try to merge too many pdfs at once, you get a "too many
            # open files" error.
            merger = PdfFileMerger()
            for pdf in batchlist:
                try:
                    with open(pdf, "rb") as file:
                        merger.append(PdfFileReader(file))

                except FileError:
                    print("error merging: " + pdf)

            # Close merger after the batch!
            merger.write("{0}/Batch-{1}.pdf".format(folder_pdf, str(i)))
            merger.close()
        i += 1

    # Create list of batch filenames
    list_batch_filenames = os.listdir(folder_pdf)
    list_batch_filenames = [i for i in list_batch_filenames if
                            re.search("Batch-[0-9]*.pdf", i)]
    list_batch_filenames = sorted(list_batch_filenames)
    print("batches to merge : {0}".format(list_batch_filenames))

    # Merge batches
    pdfWriter = PdfFileWriter()
    for file in list_batch_filenames:
        pdfFileObj = open(file, 'rb')
        pdfReader = PdfFileReader(pdfFileObj)
        for pageNum in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(pageNum)
            pdfWriter.addPage(pageObj)
    pdfOutput = open(output_pdf, 'wb')
    pdfWriter.write(pdfOutput)
    pdfOutput.close()

    # Delete batches
    for file in list_batch_filenames:
        subprocess.check_call(['rm', file])

    print('Check folder: \" {0} \" for PDFs.'.format(output_pdf))
    return 0


def mwp1toreport(nii_filenames, root_cat12vbm):
    """Generate report filenames from mwp1 image filenames.

    Parameters
    ----------
    nii_filenames: list
        descriptor of input data.
    root_cat12vbm: root of the cat12vbm report files

    Returns
    -------
    reports_list: list of the cat12vbm reports sorted by mean correlation


    """
    reports_list = []
    for i in nii_filenames:
        dico = get_keys(i, default="1")
        subject = dico['participant_id']
        session = dico['session']
        filename = os.path.basename(i)[4:]
        if re.search(".gz", i):
            filename = filename.replace(".nii.gz", ".pdf")
        else:
            filename = filename.replace(".nii", ".pdf")
        report_filename1 = "sub-{0}/ses-{1}/anat/report/catreport_{2}"\
                           .format(subject, session, filename)
        report_filename2 = "sub-{0}/anat/report/catreport_{1}"\
                           .format(subject, filename)
        report_filename3 = "sub-{0}/ses-{1}/anat/report/catreport_r{2}"\
                           .format(subject, session, filename)
        if os.path.exists(os.path.join(root_cat12vbm, report_filename1)):
            pathreport = os.path.join(root_cat12vbm, report_filename1)
            reports_list.append(pathreport)
        elif os.path.exists(os.path.join(root_cat12vbm, report_filename2)):
            pathreport = os.path.join(root_cat12vbm, report_filename2)
            reports_list.append(pathreport)
        elif os.path.exists(os.path.join(root_cat12vbm, report_filename3)):
            pathreport = os.path.join(root_cat12vbm, report_filename3)
            reports_list.append(pathreport)
        else:
            print("no reports for : {0} {1}".format(subject, session))

    return reports_list


def concat_tsv(mean_corr, score):
    """Merge mean correlation and cat12 scores tsv.

    Parameters
    ----------
    mean_corr: pandas Dataframe
        descriptor of input data.
    path_score: path of the scoresQC.tsv file

    Returns
    -------
    res: pandas Dataframe
        columns: 'participant_id', 'session', 'run', 'corr_mean',
                 'NCR', 'ICR', 'IQR'.


    """
    res = mean_corr.merge(score,
                          how='outer',
                          on=['participant_id', 'session', 'run'])
    print(mean_corr.shape, res.shape)
    assert mean_corr.shape[0] == res.shape[0]

    return res


def reconstruct_ordored_list(img_filenames, qc_filename):
    """Reconstruct ordored mwp1 nii images filenames from qc.tsv.

    Parameters
    ----------
    img_filenames: list
        mwp1 nii filenames
    qc_filename: string
        path to the qc.tsv file

    Returns
    -------
    ordored_list: list
        Ordored list of mwp1 nii image filenames by mean correlation.


    """
    ordored_list = [0 for i in range(len(img_filenames))]
    qc = pd.read_csv(qc_filename, sep='\t')
    for index, row in qc.iterrows():
        sub = str(row['participant_id'])
        ses = str(row['session'])
        run = str(row['run'])
        for filename in img_filenames:
            if sub in filename and ses in filename and run in filename:
                ordored_list[index] = filename
    return ordored_list


def parse_xml_files_scoresQC(xml_filenames):
    """Create scoresQC.tsv file.

    Parameters
    ----------
    xml_filenames: list
        list of xml_filenames

    Returns
    -------
    output: dict
        dictionary of cat12vbm scores

    Saving
    -------
    scoresQC.tsv : NCR, ICR, IQR of cat12vbm reports.
    """
    # organized as /participant_id/sess_id/[TIV, GM, WM, CSF, ROIs]
    output = dict()
    for xml_file in xml_filenames:

        xml_file_keys = get_keys(xml_file, default="1")
        participant_id = xml_file_keys['participant_id']
        session = xml_file_keys['session'] or 'V1'
        run = xml_file_keys['run'] or '1'

        # Parse the CAT12 report to find the TIV and CGW volumes
        if re.match('.*report/cat_.*\.xml', xml_file):
            tree = ET.parse(xml_file)
            try:
                NCR = float(tree.find('qualityratings').find('NCR').text)
                ICR = float(tree.find('qualityratings').find('ICR').text)
                IQR = float(tree.find('qualityratings').find('IQR').text)

            except ValueError:
                print('Parsing error for %s:\n%s' %
                      (xml_file, traceback.format_exc()))
            else:
                if participant_id not in output:
                    output[participant_id] = {session: {run: dict()}}
                elif session not in output[participant_id]:
                    output[participant_id][session] = {run: dict()}
                elif run not in output[participant_id][session]:
                    output[participant_id][session][run] = dict()

                output[participant_id][session][run]['NCR'] = float(NCR)
                output[participant_id][session][run]['ICR'] = float(ICR)
                output[participant_id][session][run]['IQR'] = float(IQR)

    return output


def compute_brain_mask(imgs,
                       target_img=None,
                       mask_thres_mean=0.1,
                       mask_thres_std=1e-6,
                       clust_size_thres=10,
                       verbose=1):
    """
    Compute brain mask:
    (1) Implicit mask threshold `mean >= mask_thres_mean` and `
        std >= mask_thres_std`
    (2) Use brain mask from
        `nilearn.masking.compute_gray_matter_mask(target_img)`
    (3) mask = Implicit mask & brain mask
    (4) Remove small branches with `scipy.ndimage.binary_opening`
    (5) Avoid isolated clusters: remove clusters (of connected voxels) smaller
        that `clust_size_thres`

    Parameters
    ----------
    imgs : [str] path to images
        or array (n_subjects, 1, , image_axis0, image_axis1, ...) in this case
        target_img must be provided.

    target_img : nii image
        Image defining the referential.

    mask_thres_mean : float (default 0.1)
        Implicit mask threshold `mean >= mask_thres_mean`

    mask_thres_std : float (default 1e-6)
        Implicit mask threshold `std >= mask_thres_std`

    clust_size_thres : float (clust_size_thres 10)
        Remove clusters (of connected voxels) smaller that `clust_size_thres`

    verbose : int (default 1)
        verbosity level

    expected : dict
        optional dictionary of parameters to check,
        ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
         nii image:
             In referencial of target_img or the first imgs

    Example
    -------
    Parameters
    ----------
    NI_arr :  ndarray, of shape (n_subjects, 1, image_shape).
    target_img : image.
    mask_thres_mean : Implicit mask threshold `mean >= mask_thres_mean`
    mask_thres_std : Implicit mask threshold `std >= mask_thres_std`
    clust_size_thres : remove clusters (of connected voxels) smaller that
                       `clust_size_thres`
    verbose : int. verbosity level

    Returns
    -------
    image of mask
    """

    if isinstance(imgs, list) and len(imgs) >= 1 and isinstance(imgs[0], str):
        imgs_arr, df, target_img = img_to_array(imgs, sesrun_default=1)

    elif isinstance(imgs, np.ndarray) and imgs.ndim >= 5:
        imgs_arr = imgs
        assert isinstance(target_img, nibabel.nifti1.Nifti1Image)

    # (1) Implicit mask
    mask_arr = np.ones(imgs_arr.shape[1:], dtype=bool).squeeze()
    if mask_thres_mean is not None:
        mask_arr = mask_arr & (np.abs(np.mean(imgs_arr, axis=0)) >=
                               mask_thres_mean).squeeze()
    if mask_thres_std is not None:
        mask_arr = mask_arr & (np.std(imgs_arr, axis=0) >=
                               mask_thres_std).squeeze()

    # (2) Brain mask: Compute a mask corresponding to the gray matter part of
    # the brain.
    # The gray matter part is calculated through the resampling of MNI152
    # template
    # gray matter mask onto the target image
    # In reality in is a brain mask
    mask_img = nilearn.masking.compute_gray_matter_mask(target_img)

    # (3) mask = Implicit mask & brain mask
    mask_arr = (mask_img.get_fdata() == 1) & mask_arr

    # (4) Remove small branches
    mask_arr = scipy.ndimage.binary_opening(mask_arr)

    # (5) Avoid isolated clusters: remove all cluster smaller that
    # clust_size_thres
    mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)

    labels = np.unique(mask_clustlabels_arr)[1:]
    for lab in labels:
        clust_size = np.sum(mask_clustlabels_arr == lab)
        if clust_size <= clust_size_thres:
            mask_arr[mask_clustlabels_arr == lab] = False

    if verbose >= 1:
        mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
        labels = np.unique(mask_clustlabels_arr)[1:]
        print("Clusters of connected voxels #%i, sizes=" % len(labels),
              [np.sum(mask_clustlabels_arr == lab) for lab in labels])

    return nilearn.image.new_img_like(target_img, mask_arr)


def img_to_array(img_filenames,
                 check_same_referential=True,
                 expected=dict(),
                 sesrun_default=0):
    """
    Convert nii images to array (n_subjects, 1, , image_axis0, image_axis1, ..)
    Assume BIDS organisation of file to retrive participant_id, session and run

    Parameters
    ----------
    img_filenames : [str]
        path to images

    check_same_referential : bool
        if True (default) check that all image have the same referential.

    expected : dict
        optional dictionary of parameters to check,
        ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
        imgs_arr : array (n_subjects, 1, , image_axis0, image_axis1, ...)
            The array data structure
            (n_subjects, n_channels, image_axis0, image_axis1, ...)

        df : DataFrame
            With column: 'participant_id', 'session', 'run', 'path'

        ref_img : nii image
            The first image used to store referential and all information
            relative to the images.

    Example
    -------
    >>> from  nitk.image import img_to_array
    >>> import glob
    >>> img_filenames = glob.glob("PATH/sub-*/ses-*/mri/mwp1sub*.nii")
    >>> imgs_arr, df, ref_img = img_to_array(img_filenames)
    >>> print(imgs_arr.shape)
    (171, 1, 121, 145, 121)
    >>> print(df.shape)
    (171, 3)
    >>> print(df.head())
      participant_id session                                            ni_path
    0       ICAAR017      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    3  STARTLB160534      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    4       ICAAR048      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...

    """
    # usefull if there is "run" only for few subjects of a cohort
    if sesrun_default:
        df = pd.DataFrame([pd.Series(get_keys(filename, default="1"))
                           for filename in img_filenames])
    else:
        df = pd.DataFrame([pd.Series(get_keys(filename))
                           for filename in img_filenames])
    imgs_nii = [nibabel.load(filename) for filename in df.ni_path]

    ref_img = imgs_nii[0]

    # Check expected dimension
    if 'shape' in expected:
        assert ref_img.get_fdata().shape == expected['shape']
    if 'zooms' in expected:
        assert ref_img.header.get_zooms() == expected['zooms']

    if check_same_referential:  # Check all images have the same transformation
        assert np.all([np.all(img.affine == ref_img.affine)
               for img in imgs_nii])
        assert np.all([np.all(img.get_fdata().shape == ref_img.get_fdata()
               .shape) for img in imgs_nii])

    imgs_arr = np.stack([np.expand_dims(img.get_fdata(), axis=0)
                        for img in imgs_nii])

    return imgs_arr, df, ref_img


def get_keys(filename, default=''):
    """
    Extract keys from bids filename. Check consistency of filename.

    Parameters
    ----------
    filename : str
        bids path

    Returns
    -------
    dict
        The minimum returned value is dict(participant_id=<match>,
                             session=<match, '' if empty>,
                             path=filename)

    Raises
    ------
    ValueError
        if match failed or inconsistent match.
    """
    keys = OrderedDict()

    participant_id = participant_re.findall(filename)
    if len(set(participant_id)) != 1:
        raise ValueError('Found several or no participant id',
                         participant_id,
                         'in path',
                         filename)
    keys["participant_id"] = participant_id[0]

    session = session_re.findall(filename)
    if len(set(session)) > 1:
        raise ValueError('Found several sessions',
                         session,
                         'in path',
                         filename)

    elif len(set(session)) == 1:
        keys["session"] = session[0]

    else:
        keys["session"] = default

    run = run_re.findall(filename)
    if len(set(run)) == 1:
        keys["run"] = run[0]

    else:
        keys["run"] = default

    keys["ni_path"] = filename

    return keys
