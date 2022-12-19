import os.path

import fmrius.info
# from .modeling.nilearn import first_level


from sklearn.utils import Bunch
import numpy as np
import os
import pathlib
from os import listdir
from os.path import join as opj
from os.path import isdir
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, KFold
from nilearn import plotting
from nilearn.glm import threshold_stats_img
from nilearn import image
from nilearn import decoding
from nilearn import signal
from nilearn.interfaces import fmriprep
import sys
from scipy import ndimage
import nibabel as nb
import _pickle as cPickle
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker
import bisect
import collections
from joblib import Parallel, delayed
from nilearn.decoding import Decoder
import random
import matplotlib.patches as mpatches
import re
import nibabel
import warnings



sys.path.append( sys.path.append('/home/marius/ownCloud/development/python/atlasreader/atlasreader') )
import atlasreader
sys.path.append('/home/marius/ownCloud/development/python/nideconv')
from nideconv import GroupResponseFitter


class Subject:
    def __init__(self, sub_id, sub_type, session, dirs):
        self.ID = sub_id
        self.subject_type = sub_type
        self.session = session
        self.dirs = dirs
        self.tasks = list(np.unique(
            [fn.split('_')[2][5:] for fn in
             listdir(opj(dirs.fmriprep_dir, sub_id, session, 'func')) if 'task' in fn.split('_')[2]]))   # get available tasks
        self.info = info.get_subject_data(sub_id, session, dirs)
        self.info.ROIs = Bunch()
        self.univariate = Bunch()
        self.univariate.model = Bunch()
        self.univariate.results = Bunch()
        self.multivariate = Bunch()
        self.multivariate.searchlight = Bunch()
        self.multivariate.roi_decoding = Bunch()
        self.multivariate.searchlight.results = []
        self.multivariate.roi_decoding.results = []
        self.hrf_estimation = Bunch()
        self.hrf_estimation.roi_signals = []
        self.hrf_estimation.model = Bunch()

        self.load_contrast_images()

        results_loaded_ = self.load_contrast_images()
        print(results_loaded_)
        if results_loaded_:
            print('Results found for:', self.ID)
        else:
            print('No results found for:', self.ID)

        self.load_rois()

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def load_contrast_images(
            self):

        results_dir = opj(
            info.data_paths.results_dir,
            self.subject_type,
            'activations/univariate_contrasts',
            self.ID)

        if os.path.isdir(results_dir):

            sub_dirs = listdir(results_dir)

            for sub_result in sub_dirs:

                sr_dir_ = opj(results_dir, sub_result)

                fls = listdir(sr_dir_)
                imgs = [im for im in fls if 'nii.gz' in im]
                models = [f_ for f_ in fls if 'glm.pkl' in f_]

                # skip folder like "pdfs"
                if len(imgs) == 0:
                    continue

                self.univariate.results[sub_result] = {}

                # load fmri_model objects
                for mdl_ in models:
                    mdl_task = mdl_.split('_')[1][5:]
                    mdl_runs = mdl_.split('_')[2]
                    mdl_fwhm = mdl_.split('_')[3]
                    #
                    if mdl_task not in self.univariate.model:
                        self.univariate.model[mdl_task] = {}
                    #
                    if mdl_runs not in self.univariate.model[mdl_task]:
                        self.univariate.model[mdl_task][mdl_runs] = {}
                    #
                    if mdl_fwhm not in self.univariate.model[mdl_task][mdl_runs]:
                        self.univariate.model[mdl_task][mdl_runs][mdl_fwhm] = {}
                    #
                    try:
                        with open(opj(sr_dir_, mdl_), 'rb') as ff:
                            self.univariate.model[mdl_task][mdl_runs][mdl_fwhm]['glm'] = cPickle.load(ff)
                    except:
                        print(opj(sr_dir_, mdl_), 'Could not be loaded.')

                for im_ in imgs:
                    im_runs = im_.split('_')[2]
                    im_con = im_.split('_')[3]
                    im_fwhm = im_.split('_')[4]
                    #
                    if im_runs not in self.univariate.results[sub_result]:
                        self.univariate.results[sub_result][im_runs] = {}
                    #
                    if im_con not in self.univariate.results[sub_result][im_runs]:
                        self.univariate.results[sub_result][im_runs][im_con] = {}
                    #
                    if im_fwhm not in self.univariate.results[sub_result][im_runs][im_con]:
                        self.univariate.results[sub_result][im_runs][im_con][im_fwhm] = []

                    self.univariate.results[sub_result][im_runs][im_con][im_fwhm].append(opj(results_dir, sub_result, im_))




    def load_rois(self):

        roi_types = listdir(info.data_paths.rois_dir)

        for roi_type in roi_types:

            if roi_type == 'individual':

                if os.path.isdir(opj(info.data_paths.rois_dir, roi_type, self.ID)):

                    # anatomical or functional
                    roi_type2 = listdir(
                        opj(info.data_paths.rois_dir, roi_type, self.ID))

                    # iterate 'anatomical', 'functional'
                    for rt2 in roi_type2:

                        self.info.ROIs[rt2] = Bunch()
                        rois_ = listdir(
                            opj(info.data_paths.rois_dir, roi_type, self.ID, rt2))

                        for img in [el for el in rois_ if 'nii' in el]:
                            roi_name = img.split('_')[-1].split('-')[0]

                            if not roi_name in self.info.ROIs[rt2].keys():
                                self.info.ROIs[rt2][roi_name] = []

                            self.info.ROIs[rt2][roi_name].append(
                                opj(info.data_paths.rois_dir, roi_type, self.ID, rt2, img))

                        for kk_ in self.info.ROIs[rt2].keys():
                            self.info.ROIs[rt2][kk_].sort(key=lambda x: x.split('-')[-1])

            #
            # neurosynth and other ROIs
            else:
                self.info.ROIs[roi_type] = Bunch()

                rois_ = listdir(opj(info.data_paths.rois_dir, roi_type))

                for img in [el for el in rois_ if 'nii' in el]:
                    # roi_name = img.split('_')[0]+'_'+img.split('_')[1]
                    roi_name = img.split('_')[0]

                    if not roi_name in self.info.ROIs[roi_type].keys():
                        self.info.ROIs[roi_type][roi_name] = []

                    self.info.ROIs[roi_type][roi_name].append(
                        opj(info.data_paths.rois_dir, roi_type, img))

                # sort ROIs such that left in always first element in array
                for kk_ in self.info.ROIs[roi_type].keys():
                    self.info.ROIs[roi_type][kk_].sort(key=lambda x: x.split('-')[-1])


    def clean_images(
            self,
            standardize=True,
            fwhm=False):
        #
        # confound selection
        confound_vars = ['trans_x', 'trans_x_derivative1',
                         'trans_y', 'trans_y_derivative1',
                         'trans_z', 'trans_z_derivative1',
                         'rot_x', 'rot_x_derivative1',
                         'rot_y', 'rot_y_derivative1',
                         'rot_z', 'rot_z_derivative1',
                         'global_signal', 'global_signal_derivative1',
                         'csf', 'csf_derivative1',
                         'a_comp_cor_00',
                         'a_comp_cor_01']

        #
        # iterate functional runs
        # for kk in self.info.func.runs.keys():
            # for kk in [el for el in list(self.info.keys()) if el not in ["run_identifier", "anat", 'ROIs']]:
        for task_ in self.info.func.keys():
            try:
                for kk in self.info.func[task_].keys():

                    #
                    # read confounds .csv
                    confounds_df = pd.read_csv(
                        self.info.func[task_][kk].confounds, delimiter="\t")

                    #
                    # make matrix from df
                    confounds_matrix = confounds_df[confound_vars].values
                    # replace leadning NaNs
                    for col in confounds_matrix.transpose():
                        if np.isnan(col[0]):
                            col[0] = np.nanmean(col)

                    #
                    # clean image
                    clean_img = image.clean_img(
                        self.info.func[task_][kk].img,
                        detrend=True,
                        standardize=standardize,
                        confounds=confounds_matrix,
                        # high_pass=0.05,
                        # low_pass=0.15,
                        t_r=1.5,
                        mask_img=self.info.func[task_][kk].mask)

                    # construct filename extension
                    f_name_extension = ''
                    if fwhm:
                        clean_img = image.smooth_img(
                            clean_img,
                            fwhm=fwhm)
                        f_name_extension = f_name_extension+'_s'+str(fwhm)
                    if standardize:
                        f_name_extension = f_name_extension+'_standardized'

                    #
                    # set TR in nifti header
                    clean_img.header['pixdim'][4] = 1500

                    #
                    # store cleaned image
                    clean_img.to_filename(
                        self.info.func[task_][kk].img[:-12]+"Cleaned"+f_name_extension+"_bold.nii.gz")
                    print("Image stored: ", self.info.func[task_][kk].img[:-12] +
                          "Cleaned"+f_name_extension+"_bold.nii.gz", "stored")
            except:
                pass

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def run_first_level(
        self,
        task_run_dict=dict(),  # if empty all tasks and runs are analyzed
        subject_type='',
        fname_extension_dict={},
        split_task_variants=True,
        fwhm=(None, 5),
        confound_parms={
                'strategy':['motion', 'wm_csf', 'global_signal'], # 'compcor', 'high_pass', 'wm_csf',
                'motion':'derivatives',
                'wm_csf':'derivatives',
                'global_signal':'derivatives',
                'scrub':0,
                'fd_threshold':.2,
                'std_dvars_threshold':3,
                'compcor':'anat_combined',
                'n_compcor':20,
                'demean':False # revommended to be True when using with NiftiMasker, False when using with signal.clean
            },
        concatenate_runs=False,
        permute_events=False,
        hrf_model='glover + derivative + dispersion',
        contrasts_=None,
        sub_folder_extension='',
        output_type="z_score",
        plot_design_matrix=False,
        display_contrast_vectors=False
    ):
        self.univariate.model = Bunch()

        tasks = list(task_run_dict.keys())
        print(tasks)

        for task_id_ in tasks:

            run_selections = task_run_dict[task_id_]
            fname_extension_list = fname_extension_dict[task_id_]

            # iterating task_run_dict[task_id_]
            for irun_selection, run_selection in enumerate(run_selections):
            # run_selections = [[TASK_RUN-X, ...], [TASK_RUN-X, ...]
            # run_selection = [TASK_RUN-X, ...]


                fname_extension = fname_extension_list[irun_selection]
                self.univariate.model[task_id_] = Bunch()
                run_list = [run_selection] # run_list = [[TASK_RUN-X, ...]]
                print(run_list)

                if split_task_variants:
                    task_runs = [rr for rr in self.info.run_identifier
                                 if task_id_ in rr]
                    # get list of runs belonging to task_id_
                    task_variants = np.unique(task_runs)

                    # if there are variants of task_id_ create addtional run lists
                    if np.size(task_variants) > 1:

                        for tv in task_variants:
                            fname_extension_list.append(tv.split('_')[1])
                            run_ids = np.where(tv == np.array(task_runs))[0]

                            run_list.append(
                                [tv.split('_')[0]+'_run-'+str(rid+1)
                                 for rid in run_ids])

                # iterating iterating
                for irun, selected_runs in enumerate(run_list):
                # run_list = [[TASK_RUN-X, ...]]
                # runs = [TASK_RUN-X, ...]
                    selected_run_counts = '-'.join([el[-1] for el in selected_runs])

                    self.univariate.model[task_id_]['run-'+selected_run_counts] = {}

                    for fwhm_ in fwhm:
                        self.univariate.model[task_id_]['run-'+selected_run_counts]['fwhm-' +
                                                        str(fwhm_)] = Bunch()

                        print()
                        print('Analyzing: ', task_id_, '@', fname_extension,
                              '@ fwhm =', fwhm_)
                        print('Runs: ', selected_runs, fname_extension)

                        # fit first level model
                        fmri_glm, design_matrices, func_imgs = first_level.first_level_wrapper(
                            self.info.func[task_id_],
                            task_id_,
                            selected_runs,
                            fwhm_,
                            confound_parms,
                            concatenate_runs=concatenate_runs,
                            permute_events=permute_events,
                            hrf_model=hrf_model)
                        # store first level model
                        # self.univariate.model[task_id_]['run-'+selected_run_counts]['fwhm-' +
                        #                                 str(fwhm_)]['glm'] = fmri_glm
                        # self.univariate.model[task_id_][selected_run_counts]['fwhm_' +
                        #                                 str(fwhm_)]['design_matrices'] = design_matrices
                        # self.univariate.model[task_id_][selected_run_counts]['fwhm_' +
                        #                                 str(fwhm_)]['func_imgs'] = func_imgs

                        # get column names of design matrices for contrast vector
                        dm_columns = []
                        [dm_columns.append(list(dm.columns))
                         for dm in design_matrices]

                        #
                        # compute and save contrasts
                        if fname_extension in ['cubes', 'random', 'rotation', 'V', 'H']:
                            contrast_dict = info.get_contrast_dict(
                                task_id_+fname_extension)
                        else:
                            contrast_dict = info.get_contrast_dict(task_id_.split('_')[0]) # split for cases when task_id is for example 'ec_run-1'

                        # subselect contrasts if contrast_ given
                        contrast_dict_ = {}
                        if contrasts_:
                            for con_ in contrasts_:
                                contrast_dict_[con_] = contrast_dict[con_]
                            #
                            contrast_dict = contrast_dict_

                        #
                        # generate contrast vectors
                        # { kk: contrast_dict[task_id][kk] for kk in ['Eyes1Iris123', 'Cubes1Iris123', 'Eyes1Cubes123', 'CubesIris'] }:#{ kk: contrast_dict[task_id][kk] for kk in ['com_ecEyes_eicIris'] }:#contrast_dict[task_id]:
                        for contrast in contrast_dict:

                            # print(contrast_dict[contrast].name)

                            pos_ = contrast_dict[contrast].pos
                            neg_ = contrast_dict[contrast].neg
                            if len(pos_) > 0:
                                pos_val = 1. / len(pos_)
                            else:
                                pos_val = 0
                            #
                            if len(neg_) > 0:
                                neg_val = -1. / len(neg_)
                            else:
                                neg_val = 0


                            if not contrast_dict[contrast].name == 'Vertical cubes - Horizontal cubes' and \
                                    ('_V' in [el[-2:] for el in pos_] or '_H' in [el[-2:] for el in pos_]):
                                pos_val *= 2
                            if not contrast_dict[contrast].name == 'Vertical cubes - Horizontal cubes' and \
                                    ('_V' in [el[-2:] for el in neg_] or '_H' in [el[-2:] for el in neg_]):
                                neg_val *= 2


                            # if self.univariate.model[task_id_][selected_run_counts][fwhm_None.glm.hrf_model == 'fir':
                            #     pos_ = [el+'_delay_0' for el in pos_]
                            #     neg_ = [el+'_delay_0' for el in neg_]

                            con_matrix = [np.zeros(np.size(dmmm)) for dmmm in dm_columns]  # condition X run

                            for jj, col_ in enumerate(dm_columns):
                                for ii, condition_ in enumerate(col_):
                                    if condition_ in pos_:
                                        con_matrix[jj][ii] = pos_val
                                    if condition_ in neg_:
                                        con_matrix[jj][ii] = neg_val

                                if display_contrast_vectors:
                                    display(pd.DataFrame(
                                        con_matrix[jj], col_)[:-12].transpose())

                            # skip missing conditions
                            if not any([any(el) for el in con_matrix]):
                                continue

                            # Plot design matrices
                            if plot_design_matrix:
                                nm = len(design_matrices)
                                f, ax = plt.subplots(
                                    1, nm, figsize=(3.5*nm, 3))
                                if nm == 1:
                                    ax = [ax]
                                for ii, dm in enumerate(design_matrices):
                                    plotting.plot_design_matrix(dm, ax=ax[ii])
                                plt.show()

                            # get contrast image
                            # try:
                            result_images = fmri_glm.compute_contrast(
                                list(con_matrix), output_type=output_type)

                            if not isinstance(result_images, dict):
                                result_images = {output_type: result_images}

                            for map_type in result_images.keys():
                                map_ = result_images[map_type]

                                ### ROUND BETA ESTIMATES TO INTEGERS
                                # data = np.round(
                                #     map_.get_fdata()).astype(np.float32)
                                # new_img = nb.Nifti1Image(
                                #     data, header=map_.header, affine=map_.affine)
                                # new_img.header.set_data_dtype(np.float32)
                                # new_img.set_data_dtype(np.float32)
                                # print("***Warning*** BETA ESTIMATES ARE ROUNDED TO INTEGERS")

                                # store image file


                                # if not isdir(opj(self.dirs.results_dir,
                                #                  subject_type,
                                #                  'activations',
                                #                  'univariate_contrasts',
                                #                  self.ID)):
                                #     mkdir(opj(self.dirs.results_dir,
                                #               'from_nilearn',
                                #               'univariate_contrasts',
                                #               self.ID))

                                if not sub_folder_extension:
                                    out_dir = opj(
                                        self.dirs.results_dir,
                                        subject_type,
                                        'activations',
                                        'univariate_contrasts',
                                        self.ID,
                                        task_id_)
                                else:
                                    out_dir = opj(
                                        self.dirs.results_dir,
                                        subject_type,
                                        'activations',
                                        'univariate_contrasts',
                                        self.ID,
                                        task_id_,
                                        sub_folder_extension)
                                # make results folder and parents if it does not exist
                                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

                                # if not isdir(out_dir):
                                #     mkdir(out_dir)
                                #     print("Created output folder: ", out_dir)
                                #

                                f_name = self.ID+'_task-'+task_id_+'_run-'+selected_run_counts+'_'+contrast+'_fwhm-' + \
                                    str(fwhm_)+'_'+fname_extension + \
                                    '_'+map_type.replace('_', '-')+'.nii.gz'
                                out_file = opj(
                                    out_dir,
                                    f_name)
                                map_.to_filename(out_file)
                                print('File written: ', out_file)

                                # store fmri_model as pkl object
                                # with open(
                                #         opj(
                                #             self.dirs.results_dir,
                                #             'from_nilearn',
                                #             'univariate_contrasts',
                                #             self.ID,
                                #             task_id_ + sub_folder_extension,
                                #             self.ID + '_task-' + task_id_ + '_run-' + selected_run_counts + \
                                #             '_fwhm-' + str(fwhm_) + '_' + fname_extension + '_glm.pkl'), 'wb') as ff:
                                #     cPickle.dump(fmri_glm, ff)


                                # if output_type == "z_score":
                                #     for height_control in threshold_dict[task_id_].keys():
                                #         for alpha in threshold_dict[task_id_][height_control]:
                                #             # threshold map_ and store activation list
                                #             thresholded_map, threshold = threshold_stats_img(
                                #                 map_, alpha=alpha,
                                #                 height_control=height_control,
                                #                 cluster_threshold=cluster_threshold)
                                #
                                #             thresholded_map.to_filename(
                                #                 opj(self.dirs.results_dir,
                                #                     'from_nilearn',
                                #                     'univariate_contrasts',
                                #                     self.ID+sub_folder_extension,
                                #                     f_name+'_p'+str(alpha).split('.')[-1]+'-'+height_control))
                                # except:
                                #     warnings.warn('Contrast ' + contrast + ' is not available')
        self.load_contrast_images()

    def compute_overlap(
                            self,
                            task_con_dict=None,
                            cut_coords=np.arange(-9, 21, 3),
                            plot_results=True,
                            display_result_table=True,
                            store_=False,
                            plot_mask_components=False):

        #
        if store_:

            ##
            pdf_dir_ = opj(info.data_paths.results_dir,
                         'from_nilearn/univariate_contrasts',
                         self.ID,
                         'pdfs')

            ##
            if not isdir(pdf_dir_):
                ###
                mkdir(pdf_dir_)
                print('***Output folder ', pdf_dir_, 'generated.')

            ##
            summary_file_name = 'overlap'

            ##
            for kk in task_con_dict.keys():
                ###
                summary_file_name = summary_file_name + '_' + \
                  task_con_dict[kk]['task'] + '-' + task_con_dict[kk]['con'] + '-' + task_con_dict[kk]['sign']

            ##
            summary_file_name = summary_file_name+'_summary.txt'
            summary_file = open(opj(pdf_dir_, summary_file_name), "w")

        #
        else:
            ##
            pdf_dir_ = ''
            summary_file_name = '_dummy.txt'
            summary_file = open(opj(pdf_dir_, summary_file_name), "w")

        #
        summary_file.write('***Images used for comparison')
        summary_file.write('\n')

        #
        mask_collector = {}

        center_collector = {}

        #
        for kk in task_con_dict.keys():

            center_collector[kk] = {}

            ##
            mask_collector[kk] = {}

            ##
            if not task_con_dict[kk]['task'] == 'Marquardt2017':

                ###
                if not task_con_dict[kk]['pre_defined_masks']:
                        left_arr, \
                        right_arr, \
                        left_mask, \
                        right_mask, \
                        img_name, \
                        left_distance_to_ref, \
                        right_distance_to_ref = self.compute_masks(
                            task=task_con_dict[kk]['task'],
                            runs=task_con_dict[kk]['runs'],
                            con=task_con_dict[kk]['con'],
                            sign=task_con_dict[kk]['sign'],
                            alpha=task_con_dict[kk]['alpha'],
                            h_control=task_con_dict[kk]['h_control'],
                            cluster_threshold=task_con_dict[kk]['cluster_threshold'],
                            fwhm=task_con_dict[kk]['fwhm'],
                            ref_img=task_con_dict[kk]['ref_img'],
                            use_lbls=task_con_dict[kk]['mask_lbls'],
                            plot_components=plot_mask_components)
                elif type(task_con_dict[kk]['pre_defined_masks'][0]) == str:
                    left_mask = image.load_img(task_con_dict[kk]['pre_defined_masks'][0])
                    right_mask = image.load_img(task_con_dict[kk]['pre_defined_masks'][1])
                    left_arr = left_mask.get_fdata()
                    right_arr = right_mask.get_fdata()
                    img_name = task_con_dict[kk]['pre_defined_masks'][2]
                else: # use prespecified masks coming from compute_masks()
                    left_arr = task_con_dict[kk]['pre_defined_masks'][0]
                    right_arr = task_con_dict[kk]['pre_defined_masks'][1]
                    left_mask = task_con_dict[kk]['pre_defined_masks'][2]
                    right_mask = task_con_dict[kk]['pre_defined_masks'][3]
                    img_name = task_con_dict[kk]['pre_defined_masks'][4]

                ###
                mask_collector[kk]['left_arr'] = left_arr
                mask_collector[kk]['right_arr'] = right_arr
                mask_collector[kk]['left_mask'] = left_mask
                mask_collector[kk]['right_mask'] = right_mask
                mask_collector[kk]['img_name'] = img_name

                ###
                # get the locations of the clusters in array space and the center in image space
                mask_collector[kk]['left_arr_idx'] = np.where(left_arr)
                left_arr_idx_center = np.median(
                  mask_collector[kk]['left_arr_idx'], axis=1)
                mask_collector[kk]['left_center_img_space'] = image.coord_transform(
                  left_arr_idx_center[0], left_arr_idx_center[1], left_arr_idx_center[2], mask_collector[kk]['left_mask'].affine)
                ###
                mask_collector[kk]['right_arr_idx'] = np.where(right_arr)
                right_arr_idx_center = np.median(
                  mask_collector[kk]['right_arr_idx'], axis=1)
                mask_collector[kk]['right_center_img_space'] = image.coord_transform(
                  right_arr_idx_center[0], right_arr_idx_center[1], right_arr_idx_center[2], mask_collector[kk]['right_mask'].affine)
                #
                center_collector[kk]['left'] = np.round(mask_collector[kk]['left_center_img_space'], 2)
                center_collector[kk]['right'] = np.round(mask_collector[kk]['right_center_img_space'], 2)

            ## if Marquardt2017
            else:

                ###
                left_mask = image.load_img(
                  '/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/gm_analysis/fmrius/ROIs/other/Marquardt-2017_GFP_ROI-mask_left.nii.gz')
                right_mask = image.load_img(
                  '/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/gm_analysis/fmrius/ROIs/other/Marquardt-2017_GFP_ROI-mask_right.nii.gz')
                left_arr = left_mask.get_fdata()
                right_arr = right_mask.get_fdata()
                img_name = 'Marquardt-2017_GFP'

                ###
                mask_collector[kk]['left_arr'] = left_arr
                mask_collector[kk]['right_arr'] = right_arr
                mask_collector[kk]['left_mask'] = left_mask
                mask_collector[kk]['right_mask'] = right_mask
                mask_collector[kk]['img_name'] = img_name

                ###
                # get the locations of the clusters in array space and the center in image space
                mask_collector[kk]['left_arr_idx'] = np.where(left_arr)
                left_arr_idx_center = np.median(
                  mask_collector[kk]['left_arr_idx'], axis=1)
                mask_collector[kk]['left_center_img_space'] = image.coord_transform(
                  left_arr_idx_center[0], left_arr_idx_center[1], left_arr_idx_center[2], mask_collector[kk]['left_mask'].affine)
                ##
                mask_collector[kk]['right_arr_idx'] = np.where(right_arr)
                right_arr_idx_center = np.median(
                  mask_collector[kk]['right_arr_idx'], axis=1)
                mask_collector[kk]['right_center_img_space'] = image.coord_transform(
                  right_arr_idx_center[0], right_arr_idx_center[1], right_arr_idx_center[2], mask_collector[kk]['right_mask'].affine)

                ##
                task_con_dict[kk]['alpha'] = 0.001
                task_con_dict[kk]['h_control'] = 'None'
                task_con_dict[kk]['cluster_threshold'] = 5
                task_con_dict[kk]['fwhm'] = 5

            ##
            summary_file.write('\t'+img_name+' @ '+str(task_con_dict[kk]['alpha'])+' ' +
                             str(task_con_dict[kk]['h_control'])+' '+str(task_con_dict[kk]['cluster_threshold']))
            summary_file.write('\n')

        #
        summary_file.write('\n\n')

        #
        # compute the intersection: left
        tcX_coords = []
        for kk in mask_collector.keys():
            tcX_coords.append(
              [(xx, yy, zz) for xx, yy, zz in zip(mask_collector[kk]['left_arr_idx'][0], mask_collector[kk]['left_arr_idx'][1], mask_collector[kk]['left_arr_idx'][2])])

        intersections_left = []
        #
        # iteralte coordinates of first area
        for el in tcX_coords[0]:
            #
            # iterate allother coordinate lists and check is el is in all other coordinate lists
            overlap = []
            for coords in tcX_coords[1:]:
                if el in coords:
                  overlap.append(True)
                else:
                  overlap.append(False)
            #
            if all(overlap):
                intersections_left.append(el)

        #
        # compute the intersection: right
        tcX_coords = []
        for kk in mask_collector.keys():
            tcX_coords.append(
              [(xx, yy, zz) for xx, yy, zz in zip(mask_collector[kk]['right_arr_idx'][0], mask_collector[kk]['right_arr_idx'][1], mask_collector[kk]['right_arr_idx'][2])])

        intersections_right = []
        #
        # iteralte coordinates of first area
        for el in tcX_coords[0]:
            #
            # iterate allother coordinate lists and check is el is in all other coordinate lists
            overlap = []
            for coords in tcX_coords[1:]:
                if el in coords:
                    overlap.append(True)
                else:
                    overlap.append(False)
            #
            if all(overlap):
                intersections_right.append(el)

        #
        # write output: LEFT
        size_collector_left = []
        name_collector_left = []
        summary_file.write('*** Left hemisphere')
        zero_roi_found_ = False
        for kk in mask_collector.keys():
            size_collector_left.append(
              np.shape(mask_collector[kk]['left_arr_idx'])[1])
            name_collector_left.append(kk)
            summary_file.write('\n')
            summary_file.write(
              '\t'+task_con_dict[kk]['task']+' - '+task_con_dict[kk]['con'] + ' (' + task_con_dict[kk]['sign'] + ')')
            summary_file.write('\n')
            roi_size_ = np.shape(mask_collector[kk]['left_arr_idx'])[1]
            if roi_size_ == 0:
                zero_roi_found_ = True
            summary_file.write(
              '\t\t size: '+str(roi_size_))
            summary_file.write('\n')
            summary_file.write('\t\t center of mass: ' + str(np.round(mask_collector[kk]['left_center_img_space'], 2)))
            summary_file.write('\n')
        #
        # write overlap output
        if not zero_roi_found_:
            summary_file.write('\n')
            fracs = []
            for kk in mask_collector.keys():
                if np.shape(mask_collector[kk]['left_arr_idx'])[1] > 0:
                    fracs.append(str(np.round(len(intersections_left)/np.shape(mask_collector[kk]['left_arr_idx'])[1]*100, 2))+'%')

            overlap_collection_left = []
            if fracs and not zero_roi_found_:
                zero_roi_found_ = True
                overlap_collection_left.append(np.max([float(el.split('%')[0]) for el in fracs]))
                ss = ''
                for el in fracs:
                  ss = ss + el + ' / '
                ss = ss[:-3]
                summary_file.write(
                  '\tOverlap: '+str(len(intersections_left))+' (' + ss + ')')
            else:
                summary_file.write('\tNo activity in one of the contrasts.')
        else:
            overlap_collection_left = [np.NaN]
            summary_file.write(
              '\tOverlap: Empty ROI.')
        #
        summary_file.write('\n\n')

        #
        # write output: Right
        size_collector_right = []
        name_collector_right = []
        summary_file.write('*** Right hemisphere')
        zero_roi_found_ = False
        for kk in mask_collector.keys():
            size_collector_right.append(
              np.shape(mask_collector[kk]['right_arr_idx'])[1])
            name_collector_right.append(kk)
            summary_file.write('\n')
            summary_file.write(
              '\t'+task_con_dict[kk]['task']+' - '+task_con_dict[kk]['con'] + ' (' + task_con_dict[kk]['sign'] + ')')
            summary_file.write('\n')
            roi_size_ = np.shape(mask_collector[kk]['right_arr_idx'])[1]
            if roi_size_ == 0:
                zero_roi_found_ = True
            summary_file.write(
              '\t\t size: '+str(roi_size_))
            summary_file.write('\n')
            summary_file.write('\t\t center of mass: ' +
                             str(np.round(mask_collector[kk]['right_center_img_space'], 2)))
            summary_file.write('\n\n')
        #
        # write overlap output
        if not zero_roi_found_:
            summary_file.write('\n')
            fracs = []
            for kk in mask_collector.keys():
                if np.shape(mask_collector[kk]['right_arr_idx'])[1] > 0:
                    fracs.append(str(np.round(len(intersections_right)/np.shape(mask_collector[kk]['right_arr_idx'])[1]*100, 2))+'%')

            overlap_collection_right = []
            if fracs and not zero_roi_found_:
                overlap_collection_right.append(np.max([float(el.split('%')[0]) for el in fracs]))
                ss = ''
                for el in fracs:
                  ss = ss + el + ' / '
                ss = ss[:-3]
                summary_file.write(
                  '\tOverlap: '+str(len(intersections_right))+' (' + ss + ')')
            else:
                summary_file.write('\tNo activity in one of the contrasts.')
        else:
            overlap_collection_right = [np.NaN]
            summary_file.write(
              '\tOverlap: Empty ROI.')

        #
        summary_file.write('\n\n')
        summary_file.close()

        #
        # make intersection images
        keys_ = list(mask_collector.keys())
        cut_arr_left = np.logical_and(
            mask_collector[keys_[0]]['left_arr'], mask_collector[keys_[1]]['left_arr'])
        cut_arr_right = np.logical_and(
            mask_collector[keys_[0]]['right_arr'], mask_collector[keys_[1]]['right_arr'])
        for kk in keys_[2:]:
            cut_arr_left = np.logical_and(
              cut_arr_left, mask_collector[kk]['left_arr'])
            cut_arr_right = np.logical_and(
              cut_arr_right, mask_collector[kk]['right_arr'])

        cut_img_left = image.new_img_like(
          mask_collector[keys_[0]]['left_mask'], cut_arr_left*1)
        cut_img_right = image.new_img_like(
          mask_collector[keys_[0]]['right_mask'], cut_arr_right*1)
        cut_img = image.math_img('a+b', a=cut_img_left, b=cut_img_right)

        #
        # sum of masks
        sum_img_left = image.math_img(
            'np.multiply(a, 1) + np.multiply(b, 1)',
            a=mask_collector[keys_[0]]['left_mask'],
            b=mask_collector[keys_[1]]['left_mask'])
        sum_img_right = image.math_img(
            'np.multiply(a, 1) + np.multiply(b, 1)',
            a=mask_collector[keys_[0]]['right_mask'],
            b=mask_collector[keys_[1]]['right_mask'])
        for kk in keys_[2:]:
            sum_img_left = image.math_img(
                'np.multiply(a, 1) + np.multiply(b, 1)',
                a=sum_img_left,
                b=mask_collector[kk]['left_mask'])
            sum_img_right = image.math_img(
                'np.multiply(a, 1) + np.multiply(b, 1)',
                a=sum_img_right,
                b=mask_collector[kk]['right_mask'])

        sum_img = image.math_img('a+b', a=sum_img_left, b=sum_img_right)

        if plot_results:

            #
            # glass plot
            # f1 = plotting.plot_glass_brain(sum_img, black_bg=True, colorbar=True)
            f1, axs1 = plt.subplots(1, 2, figsize=(15, 5))
            plotting.plot_glass_brain(
                image.math_img(
                    'a + b',
                    a=mask_collector[keys_[0]]['left_mask'],
                    b=mask_collector[keys_[0]]['right_mask']),
                axes=axs1[0],
                black_bg=True,
                title=task_con_dict[keys_[0]]['con'])
            plotting.plot_glass_brain(image.math_img(
                    'a + b',
                    a=mask_collector[keys_[1]]['left_mask'],
                    b=mask_collector[keys_[1]]['right_mask']),
                axes=axs1[1],
                black_bg=True,
                title=task_con_dict[keys_[1]]['con'])

            #
            # contour plot of mask images
            legend_patches = []
            f2 = plotting.plot_anat(
                self.info.anat.img,
                display_mode='z',
                cut_coords=cut_coords, black_bg=True,
                threshold=10e-10, title='Overlap between ')

            for kk in np.array(name_collector_left)[np.argsort(np.multiply(size_collector_left, -1))]:

                if task_con_dict[kk]['task'] in ['MTlocRandom', 'MTlocRotation', 'MTlocCubes', 'Marquardt2017', 'VisualMotion']:
                    color_selector = task_con_dict[kk]['task']
                else:
                    if task_con_dict[kk]['sign'] == 'positive':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]
                    if task_con_dict[kk]['sign'] == 'negative':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[1]
                    if task_con_dict[kk]['sign'] == 'both':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]

                f2.add_contours(
                  mask_collector[kk]['left_mask'],
                  colors=info.colors[color_selector],
                  filled=True)
                legend_patches.append(mpatches.Patch(color=tuple(int(info.colors[color_selector].strip(
                  '#')[i:i+2], 16)/255 for i in (0, 2, 4)), label=color_selector))

            for kk in np.array(name_collector_right)[np.argsort(np.multiply(size_collector_left, -1))]:

                if task_con_dict[kk]['task'] in ['MTlocRandom', 'MTlocRotation', 'MTlocCubes', 'Marquardt2017', 'VisualMotion']:
                    color_selector = task_con_dict[kk]['task']
                else:
                    if task_con_dict[kk]['sign'] == 'positive':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]
                    if task_con_dict[kk]['sign'] == 'negative':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[1]
                    if task_con_dict[kk]['sign'] == 'both':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]

                # if not 'tc3' == kk:
                f2.add_contours(
                  mask_collector[kk]['right_mask'],
                  colors=info.colors[color_selector],
                  filled=True
                )

            f2.add_contours(
                cut_img,
                filled=True,
                colors='w',
                alpha=1, levels=[1])
            legend_patches.append(mpatches.Patch(color='w', label='Overlap'))
            plt.legend(handles=legend_patches)

            #
            # contour plot of mask images
            legend_patches = []
            f3 = plotting.plot_anat(
                self.info.anat.img,
                display_mode='z',
                cut_coords=cut_coords, black_bg=True,
                threshold=10e-10, title='Overlap between ')

            for kk in np.array(name_collector_left)[np.argsort(np.multiply(size_collector_left, -1))]:

                if task_con_dict[kk]['task'] in ['MTlocRandom', 'MTlocRotation', 'MTlocCubes', 'Marquardt2017', 'VisualMotion']:
                    color_selector = task_con_dict[kk]['task']
                else:
                    if task_con_dict[kk]['sign'] == 'positive':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]
                    if task_con_dict[kk]['sign'] == 'negative':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[1]
                    if task_con_dict[kk]['sign'] == 'both':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]

                f3.add_contours(
                    mask_collector[kk]['left_mask'],
                    colors=info.colors[color_selector],
                    filled=False,
                    linewidths=0.5)
                legend_patches.append(
                    mpatches.Patch(
                        color=tuple(int(
                                info.colors[color_selector].strip('#')[i:i+2], 16)/255 for i in (0, 2, 4)),
                                label=color_selector))

            for kk in np.array(name_collector_right)[np.argsort(np.multiply(size_collector_left, -1))]:

                if task_con_dict[kk]['task'] in ['MTlocRandom', 'MTlocRotation', 'MTlocCubes', 'Marquardt2017', 'VisualMotion']:
                    color_selector = task_con_dict[kk]['task']
                else:
                    if task_con_dict[kk]['sign'] == 'positive':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]
                    if task_con_dict[kk]['sign'] == 'negative':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[1]
                    if task_con_dict[kk]['sign'] == 'both':
                        color_selector = re.findall('[A-Z][^A-Z]*', task_con_dict[kk]['con'])[0]

                # if not 'tc3' == kk:
                f3.add_contours(
                  mask_collector[kk]['right_mask'],
                  colors=info.colors[color_selector],
                  filled=False,
                  linewidths=1
                )
            plt.legend(handles=legend_patches)

            # storing is only possible if plot_results is true
            if store_:
                #
                f_name_glass = 'overlap'
                for kk in task_con_dict.keys():
                    f_name_glass = f_name_glass + '_' + \
                        task_con_dict[kk]['task'] + '-' + task_con_dict[kk]['con'] + '-' + task_con_dict[kk]['sign']

                f_name_glass = f_name_glass + '_glass.pdf'
                f1.savefig(opj(pdf_dir_, f_name_glass))

                f_name_trans = 'overlap'
                for kk in task_con_dict.keys():
                    f_name_trans = f_name_trans + '_' + \
                        task_con_dict[kk]['task'] + '-' + task_con_dict[kk]['con'] + '-' + task_con_dict[kk]['sign']

                f_name_trans = f_name_trans + '_transversal_filled.pdf'
                f2.savefig(opj(pdf_dir_, f_name_trans))

                f_name_contours = 'overlap'
                for kk in task_con_dict.keys():
                    f_name_contours = f_name_contours + '_' + \
                        task_con_dict[kk]['task'] + '-' + task_con_dict[kk]['con'] + '-' + task_con_dict[kk]['sign']

                f_name_contours = f_name_contours + '_transversal_contours.pdf'
                f3.savefig(opj(pdf_dir_, f_name_contours))



        if display_result_table:
            summary_file = open(opj(pdf_dir_, summary_file_name), "r")
            print(summary_file.read())

        return overlap_collection_left, overlap_collection_right, center_collector, mask_collector
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------


    def extract_roi_signals(
            self,
            tasks_cond_dict=[],
            subject_type='',
            which_images='img',
            smoothing_fwhm=5,
            confound_load_strategy='',
            clean_image=True,
            confound_parms=[],
            standardize=False,
            load_data=[],
            f_name_extension='',
            roi_selection=[]):

        if not load_data:

            ######################################
            # Extract signals from ROIs
            ######################################
            roi_signals_dict = {}
            roi_type = []
            rois = []

            for rt_ in self.info.ROIs.keys():
                current_rois_ = list(self.info.ROIs[rt_].keys())
                for rr_ in current_rois_:
                    if rr_.split('_')[-1].split('-')[0] in roi_selection:
                        print('*** ' + rr_.split('_')[-1].split('-')[0] + ' ROI added to analysis')
                        rois.append(rr_.split('_')[-1].split('-')[0])
                        roi_type.append(rt_)

            def _extract_parallel(self, tasks_cond_dict, rt_, roi_name_, roi_signals_dict):

                # iterate hemispheres
                for roi_ in self.info.ROIs[rt_][roi_name_]:

                    if 'left' in roi_:
                        hemisphere_ = ' left'
                    elif 'right' in roi_:
                        hemisphere_ = ' right'
                    else:
                        hemisphere_ = ' medial'

                    print('Extrcting signal from: ', roi_name_+hemisphere_)
                    roi_signals_dict[roi_name_+hemisphere_] = {}

                    if rt_ == 'TPJ':

                        #
                        # get coordinates from filename
                        coo = []
                        negative = False
                        for el in roi_.split('_')[-3].split('-'):
                            if el == '':
                                negative = True
                                continue
                            if negative:
                                coo.append(int(el)*-1)
                                negative = False
                            else:
                                coo.append(int(el))

                        #
                        # extracts the mean signal within the seed region
                        nifti_masker = NiftiSpheresMasker(
                            seeds=[coo],
                            radius=5,
                            # smoothing_fwhm=smoothing_fwhm,
                            # standardize=standardize,
                            t_r=1.5
                        )

                    else:

                        nifti_masker = NiftiMasker(
                            mask_img=roi_,
                            # standardize=standardize,  # this should be set to zscore apparently, even though images are already standardized
                            # detrend=True,
                            # smoothing_fwhm=smoothing_fwhm,
                            # memory="nilearn_cache",
                            t_r=1.5,
                            # memory_level=1
                        )

                    #
                    # iterate specified tasks
                    for task_id_ in tasks_cond_dict.keys():

                        roi_signals_dict[roi_name_+hemisphere_][task_id_] = []
                        # func_imgs = []
                        # events_dfs = []

                        for run_ in self.info.func[task_id_].keys():
                            # if task_id_ in run_:
                            # func_imgs.append(
                            #     self.info.func[task_id_][run_][which_images])
                            # events_dfs.append(pd.read_csv(
                            #     self.info.func[task_id_][run_].events, sep='\t'))

                        # for img_ in func_imgs:

                            img_filename = self.info.func[task_id_][run_][which_images]
                            img_ = image.load_img(img_filename)

                            if confound_load_strategy == 'fmriprep_interface':
                                # get confounds
                                confounds_df, sample_mask = fmriprep.load_confounds(
                                    img_filename,
                                    strategy=confound_parms['strategy'],
                                    motion=confound_parms['motion'],
                                    scrub=confound_parms['scrub'],
                                    fd_threshold=confound_parms['fd_threshold'],
                                    std_dvars_threshold=confound_parms['std_dvars_threshold'],
                                    wm_csf=confound_parms['wm_csf'],
                                    global_signal=confound_parms['global_signal'],
                                    compcor=confound_parms['compcor'],
                                    n_compcor=confound_parms['n_compcor'],
                                    demean=confound_parms['demean']
                                )
                                print()
                                print("*** CONFOUNDS ***")
                                print()
                                print(img_filename)
                                print(confounds_df.columns)
                                print()
                                print("*** SAMPLE MASK ***")
                                print()
                                print(sample_mask)

                                #
                                # make matrix from df
                                confounds_matrix = confounds_df.values
                                # # replace leadning NaNs
                                # ### fmriprep.load_confounds does this automatically ###
                                # for col in confounds_matrix.transpose():
                                #     if np.isnan(col[0]):
                                #         col[0] = np.nanmean(col)

                            if confound_load_strategy == 'manual':

                                confound_vars = ['csf', 'csf_derivative1',
                                                 'rot_x', 'rot_x_derivative1',
                                                 'rot_y', 'rot_y_derivative1',
                                                 'rot_z', 'rot_z_derivative1',
                                                 'trans_x', 'trans_x_derivative1',
                                                 'trans_y', 'trans_y_derivative1',
                                                 'trans_z', 'trans_z_derivative1',
                                                 'white_matter', 'white_matter_derivative1',
                                                 'global_signal', 'global_signal_derivative1']

                                #
                                # read confounds .csv
                                confounds_df = pd.read_csv(
                                    self.info.func[task_id_][run_].confounds, delimiter="\t")
                                print()
                                print(img_filename)
                                print(confound_vars)
                                print()

                                #
                                # make matrix from df
                                confounds_matrix = confounds_df[confound_vars].values
                                # replace leadning NaNs
                                for col in confounds_matrix.transpose():
                                    if np.isnan(col[0]):
                                        col[0] = col[1]#np.nanmean(col)

                            if clean_image == 'use_clean_img':

                                #
                                # clean image
                                img_ = image.clean_img(
                                    img_,
                                    detrend=True,
                                    standardize=True,
                                    confounds=confounds_matrix,
                                    # high_pass=0.05,
                                    # low_pass=0.15,
                                    t_r=1.5,
                                    mask_img=self.info.func[task_id_][run_].mask)

                                #
                                # set TR in nifti header
                                img_.header['pixdim'][4] = 1500

                            if smoothing_fwhm and clean_image == 'use_clean_img':
                                img_ = image.smooth_img(
                                    img_,
                                    fwhm=smoothing_fwhm)

                            if confound_load_strategy == 'fmriprep_interface' and 'scrub' in confound_parms['strategy']:
                                print('*** SCRUBBING APPLIED ***')
                                sig_ = nifti_masker.fit_transform(
                                    img_,
                                    sample_mask=sample_mask     # scrubbing
                                )
                            if clean_image == 'use_NiftiMasker':

                                nifti_masker = NiftiMasker(
                                    mask_img=roi_,
                                    standardize=standardize,  # this should be set to zscore apparently, even though images are already standardized
                                    # detrend=True,
                                    smoothing_fwhm=smoothing_fwhm,
                                    # memory="nilearn_cache",
                                    t_r=1.5,
                                    # memory_level=1
                                )

                                sig_ = nifti_masker.fit_transform(
                                    img_,
                                    confounds=confounds_matrix
                                )
                            else:
                                sig_ = nifti_masker.fit_transform(
                                    img_)

                            if rt_ == 'TPJ':
                                roi_signals_dict[roi_name_ +
                                                 hemisphere_][task_id_].append(sig_)
                            else:
                                roi_signals_dict[roi_name_+hemisphere_][task_id_].append(
                                    np.nanmedian(sig_, axis=1))

                return roi_signals_dict

            roi_signals_dict = Parallel(
                n_jobs=-1,
                backend='threading',
                verbose=1)(
                delayed(_extract_parallel)(
                    self, tasks_cond_dict, rt_, roi_name_, roi_signals_dict)
                for rt_, roi_name_ in zip(roi_type, rois)
            )[0]

            out_path_ = opj(info.data_paths.results_dir, subject_type, 'hrf_estimates', self.ID)
            # creat results folder if not exist
            pathlib.Path(out_path_).mkdir(parents=True, exist_ok=True)

            # if not isdir(opj(info.data_paths.results_dir, 'from_nideconv', self.ID)):
            #     os.mkdir(opj(info.data_paths.results_dir, 'from_nideconv', self.ID))
            with open(
                opj(out_path_, 'roi_signals_dict_'+f_name_extension+'.pkl'), 'wb') as ff:
                    cPickle.dump(roi_signals_dict, ff)

        #
        # if path to stored signals dict is given, load it
        elif load_data:
            with open(load_data, 'rb') as ff:
                roi_signals_dict = cPickle.load(ff)

            print('File loaded: ', load_data)

        # self.hrf_estimation.roi_signals = roi_signals_dict

        ######################################
        # Create dataframe with signals
        ######################################
        dat_df_dict = {}

        for task_id_ in tasks_cond_dict.keys():

            dat_df_dict[task_id_] = {}

            #
            for i_roi_, roi_name_ in enumerate(roi_signals_dict.keys()):

                # runs_names = [
                #     rr for rr in self.info.func.runs if task_id_ in rr]
                runs_names = list(self.info.func[task_id_].keys())

                #
                # init variables for data dict
                roi_signal = np.array([])
                td = np.array([])
                sd = np.array([])
                rd = np.array([])
                events_dfs = []
                #
                # init variables for events dict
                re = np.array([])
                se = np.array([])
                con_ons = np.array([])
                con_ids = np.array([])

                #
                for runi_, run_ in enumerate(runs_names):

                    #
                    # data
                    roi_signal = np.append(
                        roi_signal, roi_signals_dict[roi_name_][task_id_][runi_])
                    td = np.append(td, np.arange(
                        0, len(roi_signals_dict[roi_name_][task_id_][runi_])*1.5, 1.5))
                    sd = np.append(
                        sd, [1]*len(roi_signals_dict[roi_name_][task_id_][runi_]))
                    rd = np.append(
                        rd, [runi_+1]*len(roi_signals_dict[roi_name_][task_id_][runi_]))

                    #
                    # events
                    # events_df_ = pd.read_csv(
                    #     self.info.func.runs[run_].events, sep='\t')
                    events_df_ = pd.read_csv(
                        self.info.func[task_id_][run_].events, sep='\t')
                    for con_id_ in tasks_cond_dict[task_id_]:
                        co_ = events_df_[events_df_[
                            'trial_type'].str.startswith(con_id_)]['onset']
                        con_ons = np.append(con_ons, co_)
                        con_ids = np.append(con_ids, [con_id_]*len(co_))
                        se = np.append(se, [1]*len(co_))
                        re = np.append(re, [runi_+1]*len(co_))

                    if not sorted(np.unique(con_ids)) == sorted(tasks_cond_dict[task_id_]):
                        print(np.unique(con_ids), sorted(
                            tasks_cond_dict[task_id_]))
                        warnings.warn('Expected conditions not found')

                #
                # data df
                tuples = [(self.ID, int(rd[ii]), td[ii])
                          for ii, ss in enumerate(sd)]
                index = pd.MultiIndex.from_tuples(
                    tuples, names=["self", "run", 't'])
                series = pd.Series(roi_signal, index=index)
                dat_df = pd.DataFrame()
                dat_df[roi_name_] = series
                if i_roi_ == 0:
                    dat_df_dict[task_id_]['data'] = dat_df
                else:
                    dat_df_dict[task_id_]['data'][roi_name_] = dat_df[roi_name_]

                #
                # events df
                tuples = [(self.ID, int(re[ii]), con_ids[ii])
                          for ii, ss in enumerate(se)]
                index = pd.MultiIndex.from_tuples(
                    tuples, names=["self", "run", 'event_type'])
                events_series = pd.Series(con_ons, index=index)
                e_df_ = pd.DataFrame()
                e_df_['onset'] = events_series
                dat_df_dict[task_id_]['events'] = e_df_

            self.hrf_estimation.roi_signals = dat_df_dict

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def estimate_hrf(
            self,
            task_id=[],
            roi_name=[],
            hemi_rois=[],
            look_at=[],
            basis_set='fourier',
            n_regressors=9,
            interval=[0, 19.5],
            fit_type='ols'):

        if not self.hrf_estimation.roi_signals:
            print('***Extract signal first***')

        else:
            # data_df_ = self.hrf_estimation.roi_signals[task_id]['data'][rois]
            # events_df = self.hrf_estimation.roi_signals[task_id]['events'],
            #
            # display(data_df_)
            # display(self.hrf_estimation.roi_signals[task_id]['events'])

            g_model = GroupResponseFitter(
                self.hrf_estimation.roi_signals[task_id]['data'][hemi_rois],
                self.hrf_estimation.roi_signals[task_id]['events'],
                input_sample_rate=1/1.5,
                concatenate_runs=False,
                oversample_design_matrix=20)

            for ev_ in look_at:
                g_model.add_event(
                    ev_,
                    basis_set=basis_set,
                    n_regressors=n_regressors,
                    interval=interval)

            g_model.fit(type=fit_type)

            if task_id not in self.hrf_estimation.model.keys():
                self.hrf_estimation.model[task_id] = {}
            self.hrf_estimation.model[task_id][roi_name] = g_model

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def compute_masks(
            self,
            img_=None,
            task='MTloc',
            runs='run-1',
            con='MotionStatic',
            con_specs='',
            sign='positive',
            alpha=1e-10,
            h_control="bonferroni",
            cluster_threshold=10,
            fwhm=None,
            ref_img='MTloc',
            use_lbls=False,
            plot_components=False,
            spatial_distance_threshold=False,
            store=False,
            roi_name=''):

        # compute the masks for a specified image
        if img_:
            img_f = img_
            img_name = img_f[0].split('/')[-1]
        else:
            img_f = [el for el in self.univariate.results[task][runs][con]['fwhm-'+str(fwhm)] if ('z-score' in el) and (con_specs in el)]
            img_name = img_f[0].split('/')[-1]

        print(img_f)
        img_f = image.load_img(img_f)


        # if reference image is not specified use Kira's GFP as reference
        # (these will be the subjects individual session masks, when run specific masks are computed)
        if ref_img == 'Marquardt2017':
            left_ref = np.array([14.85, 20.2375, 27.6])  #
            right_ref = np.array([50.36507937, 22.82539683, 24.85714286])  # 
        elif ref_img == 'MTloc':
            left_ref = np.array([17.33057851, 20.76584022, 27.16253444]) ### center coordinates of 'visual motion' in neurosynth
            right_ref = np.array([47.4083045 , 21.7716263 , 27.41176471])
        elif type(ref_img[0]) == np.ndarray: ### if MNI coordinates are given
            left_ref = np.array(image.coord_transform(
                ref_img[0][0],
                ref_img[0][1],
                ref_img[0][2],
                np.linalg.inv(img_f.affine)))
            right_ref = np.array(image.coord_transform(
                ref_img[1][0],
                ref_img[1][1],
                ref_img[1][2],
                np.linalg.inv(img_f.affine)))
        else: ### if path to ROI image is given
            if type(ref_img[0]) == str:
                lr_ = image.load_img(ref_img[0])
                rr_ = image.load_img(ref_img[1])
            elif type(ref_img[0]) == nibabel.nifti1.Nifti1Image:
                lr_ = ref_img[0]
                rr_ = ref_img[1]
            #
            left_ref = np.mean(np.where(lr_.get_fdata()), axis=1)
            right_ref = np.mean(np.where(rr_.get_fdata()), axis=1)



        print('*** Coordinates used as reference: ',
              np.array(image.coord_transform(
                  left_ref[0],
                  left_ref[1],
                  left_ref[2],
                  img_f.affine)),
              np.array(image.coord_transform(
                  right_ref[0],
                  right_ref[1],
                  right_ref[2],
                  img_f.affine)))

        # thresholded_map, threshold = threshold_stats_img(
        #     img_f, alpha=alpha,
        #     height_control=h_control,
        #     cluster_threshold=cluster_threshold)
        #
        # if sign == 'positive':
        #     thresholded_map_arr = image.get_data(
        #         image.math_img('a>0', a=thresholded_map)).astype(bool)
        # elif sign == 'negative':
        #     thresholded_map_arr = image.get_data(
        #         image.math_img('a<0', a=thresholded_map)).astype(bool)
        # elif sign == 'both':
        #     thresholded_map_arr = image.get_data(
        #         image.math_img('np.abs(a)>0', a=thresholded_map)).astype(bool)
        #
        # labels, nlabels = ndimage.label(thresholded_map_arr)
        # cluster_coordinates = [np.mean(
        #     np.where((labels == lbl).astype(int)), axis=1) for lbl in np.unique(labels)]
        #
        #
        #
        # if plot_components:
        #     for lbl in np.unique(labels)[1:]:
        #         left_distance_to_ref = np.sum(np.abs(cluster_coordinates[lbl] - left_ref))
        #         right_distance_to_ref = np.sum(np.abs(cluster_coordinates[lbl] - right_ref))
        #         if remove_farther_away_than:
        #             if min(left_distance_to_ref, right_distance_to_ref) < remove_farther_away_than:
        #                 plotting.plot_glass_brain(image.new_img_like(
        #                     thresholded_map, (labels == lbl).astype(int)), title=lbl)
        #                 plt.show()
        #         else:
        #             plotting.plot_glass_brain(image.new_img_like(
        #                 thresholded_map, (labels == lbl).astype(int)), title=lbl)
        #             plt.show()
        #
        #
        # if not use_lbls:
        #     # find closest cluster to ref
        #     left_id = np.argmin(
        #         np.sum(np.abs(cluster_coordinates - left_ref), axis=1))
        #     right_id = np.argmin(
        #         np.sum(np.abs(cluster_coordinates - right_ref), axis=1))
        #
        #     # if distance of cluster to reference is implausible return None
        #     left_distance_to_ref = np.sum(np.abs(cluster_coordinates[left_id] - left_ref))
        #     right_distance_to_ref = np.sum(np.abs(cluster_coordinates[right_id] - right_ref))
        #     if left_id == 0 or left_distance_to_ref > 7:
        #         left_id = None
        #     if right_id == 0 or right_distance_to_ref > 7:
        #         right_id = None
        #     # if left_id == 0 or np.sum(np.abs(cluster_coordinates[left_id] - left_ref)) > 7:
        #     #     left_id = None
        #     # if right_id == 0 or np.sum(np.abs(cluster_coordinates[right_id] - right_ref)) > 7:
        #     #     right_id = None
        # else:
        #     left_id = use_lbls[0]
        #     right_id = use_lbls[1]
        #     #
        #     #
        #     left_distance_to_ref = np.sum(np.abs(cluster_coordinates[left_id] - left_ref))
        #     right_distance_to_ref = np.sum(np.abs(cluster_coordinates[right_id] - right_ref))
        #
        # left_mask = image.new_img_like(
        #     thresholded_map, (labels == left_id).astype(int))
        # right_mask = image.new_img_like(
        #     thresholded_map, (labels == right_id).astype(int))

        thresholded_map_left, threshold_left = threshold_stats_img(
            img_f, alpha=alpha[0],
            height_control=h_control[0],
            cluster_threshold=cluster_threshold[0])
        thresholded_map_right, threshold_right = threshold_stats_img(
            img_f, alpha=alpha[1],
            height_control=h_control[1],
            cluster_threshold=cluster_threshold[1])

        if sign == 'positive':
            thresholded_map_arr_left = image.get_data(
                image.math_img('a>0', a=thresholded_map_left)).astype(bool)
            thresholded_map_arr_right = image.get_data(
                image.math_img('a>0', a=thresholded_map_right)).astype(bool)
        elif sign == 'negative':
            thresholded_map_arr_left = image.get_data(
                image.math_img('a<0', a=thresholded_map_left)).astype(bool)
            thresholded_map_arr_right = image.get_data(
                image.math_img('a<0', a=thresholded_map_right)).astype(bool)
        elif sign == 'both':
            thresholded_map_arr_left = image.get_data(
                image.math_img('np.abs(a)>0', a=thresholded_map_left)).astype(bool)
            thresholded_map_arr_right = image.get_data(
                image.math_img('np.abs(a)>0', a=thresholded_map_right)).astype(bool)

        labels_left, nlabels_left = ndimage.label(thresholded_map_arr_left)
        labels_right, nlabels_right = ndimage.label(thresholded_map_arr_right)
        cluster_coordinates_left = [
            np.mean(np.where((labels_left == lbl_left).astype(int)), axis=1)[0:3]
            for lbl_left in np.unique(labels_left)]
        cluster_coordinates_right = [
            np.mean(np.where((labels_right == lbl_right).astype(int)), axis=1)[0:3]
            for lbl_right in np.unique(labels_right)]

##############################################
        if plot_components:
            ### plot cluster of left threshold
            print('*** LABELS OF LEFT THRESHOLD')
            for lbl_left in np.unique(labels_left)[1:]:
                # left_distance_to_ref = np.sum(np.abs(cluster_coordinates_left[lbl_left] - left_ref))
                left_distance_to_ref = np.linalg.norm(cluster_coordinates_left[lbl_left] - left_ref)
                if spatial_distance_threshold:
                    if left_distance_to_ref < spatial_distance_threshold:
                        plotting.plot_glass_brain(image.new_img_like(
                            thresholded_map_left,
                            (labels_left == lbl_left).astype(int)),
                            title=str(lbl_left)+' Distance: '+str(round(left_distance_to_ref, 1)))
                        plt.show()
                else:
                    plotting.plot_glass_brain(image.new_img_like(
                        thresholded_map_left,
                        (labels_left == lbl_left).astype(int)),
                        title=str(lbl_left)+' Distance: '+str(round(left_distance_to_ref,1)))
                    plt.show()
            ### plot cluster of right threshold
            print('*** LABELS OF RIGHT THRESHOLD')
            for lbl_right in np.unique(labels_right)[1:]:
                # right_distance_to_ref = np.sum(np.abs(cluster_coordinates_right[lbl_right] - right_ref))
                right_distance_to_ref = np.linalg.norm(cluster_coordinates_right[lbl_right] - right_ref)
                if spatial_distance_threshold:
                    if right_distance_to_ref < spatial_distance_threshold:
                        plotting.plot_glass_brain(image.new_img_like(
                            thresholded_map_right,
                            (labels_right == lbl_right).astype(int)),
                            title=str(lbl_right)+' Distance: '+str(round(right_distance_to_ref, 1)))
                        plt.show()
                else:
                    plotting.plot_glass_brain(image.new_img_like(
                        thresholded_map_right,
                        (labels_right == lbl_right).astype(int)),
                        title=str(lbl_right)+' Distance: '+str(round(right_distance_to_ref, 1)))
                    plt.show()


        if not use_lbls:

            # find closest cluster to ref
            left_id = np.argmin(
                # np.sum(np.abs(cluster_coordinates_left - left_ref), axis=1))
                np.linalg.norm(cluster_coordinates_left - left_ref, axis=1))
            right_id = np.argmin(
                # np.sum(np.abs(cluster_coordinates_right - right_ref), axis=1))
                np.linalg.norm(cluster_coordinates_right - right_ref, axis=1))


            # if distance of cluster to reference is implausible return None
            # left_distance_to_ref = np.sum(np.abs(cluster_coordinates_left[left_id] - left_ref))
            left_distance_to_ref = np.linalg.norm(cluster_coordinates_left[left_id] - left_ref)
            # right_distance_to_ref = np.sum(np.abs(cluster_coordinates_right[right_id] - right_ref))
            right_distance_to_ref = np.linalg.norm(cluster_coordinates_right[right_id] - right_ref)
            if left_id == 0 or left_distance_to_ref > spatial_distance_threshold:
                left_id = None
            if right_id == 0 or right_distance_to_ref > spatial_distance_threshold:
                right_id = None
            # if left_id == 0 or np.sum(np.abs(cluster_coordinates[left_id] - left_ref)) > 7:
            #     left_id = None
            # if right_id == 0 or np.sum(np.abs(cluster_coordinates[right_id] - right_ref)) > 7:
            #     right_id = None
        else:
            left_id = use_lbls[0]
            right_id = use_lbls[1]
            #
            #
            # left_distance_to_ref = np.sum(np.abs(cluster_coordinates_left[left_id] - left_ref))
            left_distance_to_ref = np.linalg.norm(cluster_coordinates_left[left_id] - left_ref)
            # right_distance_to_ref = np.sum(np.abs(cluster_coordinates_right[right_id] - right_ref))
            right_distance_to_ref = np.linalg.norm(cluster_coordinates_right[right_id] - right_ref)

        left_mask = image.new_img_like(
            thresholded_map_left, (labels_left == left_id).astype(int))
        right_mask = image.new_img_like(
            thresholded_map_right, (labels_right == right_id).astype(int))



        if store:
            if con in ['Eyes1Iris123', 'GazeIris']:
                roi_name = 'GFP'
            elif con in ['Cubes1Iris123', 'CubesIris']:
                roi_name = 'CFP'
            elif con in ['MotionStatic']:
                roi_name = 'fMotionArea'
            elif con == 'Gaze1Gaze3':
                roi_name = 'ghGFP'
            elif con == 'Cubes1Cubes3':
                roi_name = 'chGFP'

            if not isdir(opj(info.data_paths.rois_dir, 'individual', self.ID)):
                os.mkdir(opj(info.data_paths.rois_dir, 'individual', self.ID))
                os.mkdir(opj(info.data_paths.rois_dir, 'individual', self.ID, 'functional'))
            if not isdir(opj(info.data_paths.rois_dir, 'individual', self.ID, 'functional')):
                os.mkdir(opj(info.data_paths.rois_dir, 'individual', self.ID, 'functional'))

            left_mask.to_filename(
                opj(info.data_paths.rois_dir, 'individual', self.ID, 'functional',
                    self.ID+"_"+"task-"+task+"_"+con+"-"+sign+"_fwhm-"+str(fwhm)+"_p"+str(alpha[0])+"-"+str(h_control[0])+"_ROI-mask_"+roi_name+"-left.nii"))
            right_mask.to_filename(
                opj(info.data_paths.rois_dir, 'individual', self.ID, 'functional',
                    self.ID+"_"+"task-"+task+"_"+con+"-"+sign+"_fwhm-"+str(fwhm)+"_p"+str(alpha[1])+"-"+str(h_control[1])+"_ROI-mask_"+roi_name+"-right.nii"))

            # store as pdf
            r_img = image.math_img(
                'a+b',
                a=left_mask,
                b=right_mask)

            fig = plotting.plot_glass_brain(r_img)
            fig.savefig(
                opj(info.data_paths.rois_dir, 'individual', self.ID, 'functional',
                    self.ID+"_"+"task-"+task+"_"+con+"-"+sign+"_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_"+roi_name+".pdf"))
            fig.close()

        # get the ROI as array
        left_arr = (labels_left == left_id)
        right_arr = (labels_right == right_id)

        # get the center coordinates of ROI in array space
        left_center_coordinates_arr = np.mean(np.where(left_arr), axis=1)[0:3]
        right_center_coordinates_arr = np.mean(np.where(right_arr), axis=1)[0:3]

        # transform array space coordinates in to image space
        left_center_coordinates_mni = np.array(image.coord_transform(
              left_center_coordinates_arr[0],
              left_center_coordinates_arr[1],
              left_center_coordinates_arr[2],
              img_f.affine))
        right_center_coordinates = np.array(image.coord_transform(
            right_center_coordinates_arr[0],
            right_center_coordinates_arr[1],
            right_center_coordinates_arr[2],
            img_f.affine))

        return left_arr, \
               right_arr, \
               left_mask, \
               right_mask, \
               img_name, \
               left_distance_to_ref, \
               right_distance_to_ref, \
               left_center_coordinates_mni, \
               right_center_coordinates





    def collect_masks(
            self,
            con_run_sets,
            fwhm,
            threshold_values,
            use_lbls=False,
            plot_components=False):

        mask_collector = {}

        for task_id in con_run_sets:
            mask_collector[task_id] = {}

            for con_id in con_run_sets[task_id]:
                print('\n'+con_id)
                mask_collector[task_id][con_id] = {}

                # for target-contrasts use the MotioStatic contrast as reference
                if con_id in ['Eyes1Eyes3', 'Cubes1Cubes3']:
                    ref_img = 'MTloc'
                else:
                    ref_img = 'Marquardt2017'

                # get the reference coordinates from the specified reference image
                # in correspondence to Kira's GFP, i.e. the subject-lvel masks used
                # as reference for run-level masking

                if not (con_run_sets[task_id][con_id]['reference']['left_ref_mask'] and con_run_sets[task_id][con_id]['reference']['right_ref_mask']):

                    print('*** Using', ref_img, 'as reference.')

                    _, _, left_ref_mask, right_ref_mask, _, _, _ = self.compute_masks(
                        img_=con_run_sets[task_id][con_id]['reference']['img'],
                        task=task_id,
                        runs='None',
                        con=con_id,
                        sign='both',
                        alpha=con_run_sets[task_id][con_id]['reference']['alpha'],
                        h_control=con_run_sets[task_id][con_id]['reference']['h_control'],
                        cluster_threshold=3,
                        fwhm=5,
                        ref_img=ref_img,
                        use_lbls=False,
                        plot_components=False,
                        store=False)

                else:

                    left_ref_mask = con_run_sets[task_id][con_id]['reference']['left_ref_mask']
                    right_ref_mask = con_run_sets[task_id][con_id]['reference']['right_ref_mask']
                    print('*** Using', left_ref_mask, right_ref_mask, 'as reference.')

                plotting.plot_glass_brain(
                    image.math_img('a+b', a=left_ref_mask, b=right_ref_mask), title='Reference ROIs')

                for run_i, runs_ in enumerate(con_run_sets[task_id][con_id]['runs']):
                    print(runs_ + ', ',  end = '')

                    mask_collector[task_id][con_id][runs_] = {'left_arr':[], 'right_arr':[]}

                    found_ = [False, False]
                    which_thresh_ = 0
                    masks_of_diff_thresh = {}

                    if not con_run_sets[task_id][con_id]['runs_thrshs']['left'][run_i]:

                        left_best = []
                        right_best = []

                        for ii_, thresh_ in enumerate(threshold_values):

                            left_arr, right_arr, left_mask, right_mask, img_name, left_dist, right_dist = self.compute_masks(
                                    img_=None,
                                    task=task_id,
                                    runs=runs_,
                                    con=con_id,
                                    con_specs=con_run_sets[task_id][con_id]['con_specs'],
                                    sign='both',
                                    alpha=[thresh_[0], thresh_[0]],
                                    h_control=[thresh_[1], thresh_[1]],
                                    cluster_threshold=[3, 3],
                                    fwhm=fwhm,
                                    ref_img=[left_ref_mask, right_ref_mask],
                                    use_lbls=use_lbls,
                                    plot_components=plot_components,
                                    store=False)

                            masks_of_diff_thresh[str(ii_)] = {
                                'left_arr': left_arr,
                                'right_arr': right_arr,
                                'left_mask': left_mask,
                                'right_mask': right_mask,
                                'img_name': img_name,
                                'threshold': thresh_,
                                'left_dist': left_dist,
                                'right_dist': right_dist,
                                'left_var': np.var(np.where(left_arr), axis=1),
                                'right_var': np.var(np.where(right_arr), axis=1),
                                'left_size': np.shape(np.where(left_arr))[1],
                                'right_size': np.shape(np.where(right_arr))[1]
                            }

                            # find most likely cluster minimizing:
                            # (Within cluster variance)/(size of cluster^2) * distance^2
                            left_best.append(np.log(
                                np.mean(np.var(np.where(left_arr), axis=1)) /
                                np.shape(np.where(left_arr))[1]**2 *
                                left_dist**2))
                            right_best.append(np.log(
                                np.mean(np.var(np.where(right_arr), axis=1)) /
                                np.shape(np.where(right_arr))[1]**2 *
                                right_dist**2))

                            # print('***', thresh_)
                            # print(np.mean(np.var(np.where(left_arr), axis=1)),
                            #       np.shape(np.where(left_arr))[1]**2,
                            #       left_dist**2, '=', left_best[-1])
                            # print(np.mean(np.var(np.where(right_arr), axis=1)),
                            #       np.shape(np.where(right_arr))[1]**2 ,
                            #       right_dist**2, '=', right_best[-1])
                            # if runs_ == 'run-6':
                            #     plotting.plot_glass_brain(right_mask)

                        # chose the connected region that minimizes distance to ref and with-region-variance, cf. above
                        left_best_idx = np.argsort(left_best)[0]
                        right_best_idx = np.argsort(right_best)[0]

                    else: # use dpecified threshold
                        alphas_ = [
                                con_run_sets[task_id][con_id]['runs_thrshs']['left'][run_i][0],
                                con_run_sets[task_id][con_id]['runs_thrshs']['right'][run_i][0]]
                        h_controls_ = [
                                con_run_sets[task_id][con_id]['runs_thrshs']['left'][run_i][1],
                                con_run_sets[task_id][con_id]['runs_thrshs']['right'][run_i][1]]
                        left_arr, right_arr, left_mask, right_mask, img_name, left_dist, right_dist = self.compute_masks(
                            img_=None,
                            task=task_id,
                            runs=runs_,
                            con=con_id,
                            con_specs=con_run_sets[task_id][con_id]['con_specs'],
                            sign='both',
                            alpha=alphas_,
                            h_control=h_controls_,
                            cluster_threshold=[3, 3],
                            fwhm=fwhm,
                            ref_img=[left_ref_mask, right_ref_mask],
                            use_lbls=use_lbls,
                            plot_components=plot_components,
                            store=False)

                        masks_of_diff_thresh = {}
                        ii_ = 0
                        left_best_idx = 0
                        right_best_idx = 0
                        masks_of_diff_thresh[str(ii_)] = {
                            'left_arr': left_arr,
                            'right_arr': right_arr,
                            'left_mask': left_mask,
                            'right_mask': right_mask,
                            'img_name': img_name,
                            'threshold': [alphas_, h_controls_],
                            'left_dist': left_dist,
                            'right_dist': right_dist,
                            'left_var': np.var(np.where(left_arr), axis=1),
                            'right_var': np.var(np.where(right_arr), axis=1),
                            'left_size': np.shape(np.where(left_arr))[1],
                            'right_size': np.shape(np.where(right_arr))[1]
                        }

                    # apply some hard distance threshold
                    if not left_dist > 40:
                        mask_collector[task_id][con_id][runs_]['left_arr'] = \
                            masks_of_diff_thresh[str(left_best_idx)]['left_arr']
                        mask_collector[task_id][con_id][runs_]['left_mask'] = \
                            masks_of_diff_thresh[str(left_best_idx)]['left_mask']
                        mask_collector[task_id][con_id][runs_]['left_threshold'] = \
                            masks_of_diff_thresh[str(left_best_idx)]['threshold']
                    else:
                        dummy_img_ = image.load_img(self.info.ROIs.functional.fMotionArea[0])
                        mask_collector[task_id][con_id][runs_]['left_arr'] = np.zeros(
                            np.shape(dummy_img_.get_fdata()))
                        mask_collector[task_id][con_id][runs_]['left_mask'] = image.new_img_like(
                            dummy_img_, np.zeros(np.shape(dummy_img_.get_fdata())))
                        mask_collector[task_id][con_id][runs_]['left_threshold'] = None

                    if not right_dist > 40:
                        mask_collector[task_id][con_id][runs_]['right_arr'] = \
                            masks_of_diff_thresh[str(right_best_idx)]['right_arr']
                        mask_collector[task_id][con_id][runs_]['right_mask'] = \
                            masks_of_diff_thresh[str(right_best_idx)]['right_mask']
                        mask_collector[task_id][con_id][runs_]['right_threshold'] = \
                            masks_of_diff_thresh[str(right_best_idx)]['threshold']
                    else:
                        dummy_img_ = image.load_img(self.info.ROIs.functional.fMotionArea[0])
                        mask_collector[task_id][con_id][runs_]['right_arr'] = np.zeros(
                            np.shape(dummy_img_.get_fdata()))
                        mask_collector[task_id][con_id][runs_]['right_mask'] = image.new_img_like(
                            dummy_img_, np.zeros(np.shape(dummy_img_.get_fdata())))
                        mask_collector[task_id][con_id][runs_]['right_threshold'] = None

                    # # check if mask was found for given thresholds
                    # while (not all(found_)) and (which_thresh_ < len(threshold_values)):
                    #
                    #     #
                    #     left_arr, right_arr, left_mask, right_mask, img_name = self.compute_masks(
                    #         img_=None,
                    #         task=task_id,
                    #         runs=runs_,
                    #         con=con_id,
                    #         sign='both',
                    #         alpha=threshold_values[which_thresh_][0],
                    #         h_control=threshold_values[which_thresh_][1],
                    #         cluster_threshold=3,
                    #         fwhm=fwhm,
                    #         ref_img=[left_ref_mask, right_ref_mask],
                    #         use_lbls=use_lbls,
                    #         plot_components=plot_components,
                    #         store=False)
                    #
                    #     # set found_ if not found yet and size of found spot is larger than 10 voxels
                    #     if (not found_[0]) and np.shape(np.where(left_arr))[1] >= 10 and (not any(
                    #             np.var(np.where(left_arr), axis=1) > 4)):
                    #         # found_[0] = 1  # any(left_arr.flatten())
                    #         mask_collector[task_id][con_id][runs_]['left_arr'] = left_arr
                    #         mask_collector[task_id][con_id][runs_]['left_mask'] = left_mask
                    #         mask_collector[task_id][con_id][runs_]['left_threshold'] = threshold_values[which_thresh_]
                    #         found_[0] = True
                    #     if (not found_[1]) and np.shape(np.where(right_arr))[1] >= 10 and (not any(
                    #             np.var(np.where(right_arr), axis=1) > 4)):
                    #         # found_[1] = 1  # any(right_arr.flatten())
                    #         mask_collector[task_id][con_id][runs_]['right_arr'] = right_arr
                    #         mask_collector[task_id][con_id][runs_]['right_mask'] = right_mask
                    #         mask_collector[task_id][con_id][runs_]['right_threshold'] = threshold_values[which_thresh_]
                    #         found_[1] = True
                    #
                    #     #
                    #     which_thresh_ += 1


                # add empty image if hemisphere is empty
                # for ii_, hemi_ in enumerate(found_):
                #     if not hemi_:
                #         dummy_img_ = image.load_img(self.info.ROIs.functional.fMotionArea[0])
                #         mask_collector[task_id][con_id][runs_][['left', 'right'][ii_] + '_arr'] = np.zeros(
                #             np.shape(dummy_img_.get_fdata()))
                #         mask_collector[task_id][con_id][runs_][['left', 'right'][ii_] + '_mask'] = image.new_img_like(
                #             dummy_img_, np.zeros(np.shape(dummy_img_.get_fdata())))
                #         mask_collector[task_id][con_id][runs_][['left', 'right'][ii_]+'_threshold'] = None

                    # if np.sum(found_) == 1:# not all(found_) and any(found_):
                    #     which_ = np.where(found_)[0][0]
                    #     if which_ == 1:
                    #         which_hemisphere_not = 'left'
                    #         which_hemisphere = 'right'
                    #     else:
                    #         which_hemisphere_not = 'right'
                    #         which_hemisphere = 'left'
                    #
                    #     mask_collector[task_id][con_id][runs_][which_hemisphere_not+'_arr'] = np.zeros(np.shape(mask_collector[task_id][con_id][runs_][which_hemisphere+'_arr']))
                    #     mask_collector[task_id][con_id][runs_][which_hemisphere_not+'_mask'] = image.new_img_like(
                    #         mask_collector[task_id][con_id][runs_][which_hemisphere+'_mask'],
                    #         np.zeros(np.shape(mask_collector[task_id][con_id][runs_][which_hemisphere+'_arr'])))
                    #     mask_collector[task_id][con_id][runs_]['threshold_'+which_hemisphere_not] = False


                    mask_collector[task_id][con_id][runs_]['img_name'] = img_name

        return mask_collector

    def compute_MT_masks(
            self,
            alpha=1e-10,
            h_control="bonferroni",
            fwhm=None):



        left_ref = np.array([16.51, 20.21, 29.05])
        right_ref = np.array([47.95, 22.41, 27.06])

        MTloc_binaries = {}
        for task in self.univariate.results.keys():
            if "MTloc" in task:
                for img in self.univariate.results[task].MotionStatic:
                    if ('z-score' in img) and (str(fwhm) in img) and not ("mask" in img):
                        print(img)
                        thresholded_map, threshold = threshold_stats_img(
                            img, alpha=alpha,
                            height_control=h_control,
                            cluster_threshold=10)
                        MTloc_binaries[task] = image.get_data(
                            image.math_img('a>0', a=thresholded_map)).astype(bool)

                        labels, nlabels = ndimage.label(MTloc_binaries[task])
                        found_left = 0
                        found_right = 0

                        for lbl in range(nlabels):
                            # left_MT_mask = image.new_img_like(img, (labels == lbl).astype(int))
                            # plotting.plot_glass_brain(left_MT_mask)
                            loc = np.mean(
                                np.where((labels == lbl+1).astype(int)), axis=1)

                            if np.abs(np.sum(loc-left_ref)) < 1.5:
                                leftMT_id = lbl+1
                                found_left += 1

                            if np.abs(np.sum(loc-right_ref)) < 1.5:
                                rightMT_id = lbl+1
                                found_right += 1

                        if found_left > 1 or found_right > 1:
                            print("***WARNING*** For task ", task,
                                  " more than 1 ROIs were identified. Check manual!")

                        if found_left == 0 or found_right == 0:
                            print("***WARNING*** For task ", task,
                                  " no ROI was foudn. Check manual!")

                        left_MT_mask = image.new_img_like(
                            img,
                            (labels == leftMT_id).astype(int))

                        right_MT_mask = image.new_img_like(
                            img,
                            (labels == rightMT_id).astype(int))

                        left_MT_mask.to_filename(
                            opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                                self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-left.nii"))

                        right_MT_mask.to_filename(
                            opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                                self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-right.nii"))

                        # left_MT_mask.to_filename(
                        #     img[:-4]+'_'+h_control+'_'+str(alpha)+'_ROImask_left.nii')
                        #
                        # right_MT_mask.to_filename(
                        #     img[:-4]+'_'+h_control+'_'+str(alpha)+'_ROImask_right.nii')

                        print("Mask images stored: ",
                              [opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                                   self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-left.nii"),
                               opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                                   self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-right.nii")])

                        # print("Mask images stored: ",
                        #       [img[:-4]+'_'+h_control+'_'+str(alpha)+'_mask_left.nii',
                        #        img[:-4]+'_'+h_control+'_'+str(alpha)+'_mask_right.nii'])
                        print('')

            # conjunction of different motion localizer tasks
            if len(MTloc_binaries) > 1:
                # MTloc_ is always the combined localizer (if multiple localizer types are used)
                individual_task_binaries = [MTloc_binaries[kk]
                                            for kk in MTloc_binaries.keys() if "MTloc_" not in kk]
                conj_bool = np.logical_or.reduce(individual_task_binaries)
                MTloc_binaries['conj'] = conj_bool

            labels, nlabels = ndimage.label(MTloc_binaries['conj'])
            found_left = 0
            found_right = 0

            for lbl in range(nlabels):
                loc = np.mean(np.where((labels == lbl+1).astype(int)), axis=1)

                if np.abs(np.sum(loc-left_ref)) < 1.5:
                    leftMT_id = lbl+1
                    found_left += 1

                if np.abs(np.sum(loc-right_ref)) < 1.5:
                    rightMT_id = lbl+1
                    found_right += 1

            if found_left > 1 or found_right > 1:
                print("For task ", 'conj',
                      " more than 1 ROIs were identified. Check manual!")

            if found_left == 0 or found_right == 0:
                print("For task ", 'conj',
                      " no ROI was found. Check manual!")

            left_MT_mask = image.new_img_like(
                img,
                (labels == leftMT_id).astype(int))

            right_MT_mask = image.new_img_like(
                img,
                (labels == rightMT_id).astype(int))

            left_MT_mask.to_filename(
                opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                    self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-left-conj.nii"))

            right_MT_mask.to_filename(
                opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                    self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-right-conj.nii"))

            print("Mask images stored: ",
                  [opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                       self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-left-conj.nii"),
                   opj("/", *img.split('/')[:-3], 'ROIs', self.ID,
                       self.ID+"_"+"task-MTloc_MotionStatic_fwhm-"+str(fwhm)+"_p"+str(alpha)+"-"+str(h_control)+"_ROI-mask_fMT-right-conj.nii")])

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def get_decoding_features(
            self,
            runs,
            method='period',
            average_imgs=True,
            img_selection=[2],
            fwhm='img_cleaned_standardized'):

        #
        # make data_dict containing trial data for each trial in ascending order
        #

        #
        # init
        data_dict = {}
        consecutive_trl_cntr = 1

        #
        # iterate runs
        for ir, rr in enumerate(runs):

            # print(self.info.func.runs[rr]['img_cleaned'])
            # if 's5' in self.info.func.runs[rr][fwhm]:
            #     fwhm_fn = '5'
            # else:
            #     fwhm_fn = None

            events_df_ = pd.read_csv(
                self.info.func.runs[rr]['events'], sep='\t')
            func_img_ = image.load_img(self.info.func.runs[rr][fwhm])

            img_times = np.multiply(np.arange(0, func_img_.shape[-1], 1), 1.5)

            #
            # iterate rows of run's events_df
            for row in events_df_.iterrows():
                #
                data_dict[str(consecutive_trl_cntr)] = {}
                trial_period = (row[1]['onset'], row[1]
                                ['onset']+row[1]['duration'])
                data_dict[str(consecutive_trl_cntr)
                          ]['trial_period'] = trial_period
                data_dict[str(consecutive_trl_cntr)
                          ]['trial_type'] = row[1]['trial_type']
                data_dict[str(consecutive_trl_cntr)]['session'] = ir+1
                data_dict[str(consecutive_trl_cntr)]['first_img_id'] = np.where(
                    img_times > trial_period[0])[0][0]
                data_dict[str(consecutive_trl_cntr)]['func_img'] = func_img_
                #
                consecutive_trl_cntr += 1

            #
            # extract the relevant scans of each trial (feature images), make condition and session lists
            #

            #
            # init
            conditions = []
            feature_imgs = []
            sessions = []

            #
            #
            for trl in data_dict.keys():

                img_times = np.multiply(
                    np.arange(0, data_dict[trl]['func_img'].shape[-1], 1), 1.5)

                #
                #
                if method == 'period':  # use_period:

                    #
                    # select images based on trial time, i.e. give a certain period relative to `onset`, images falling into this period a used as features
                    relative_img_times = img_times - \
                        data_dict[trl]['trial_period'][0]
                    min_ = relative_img_times > img_selection[0]
                    max_ = relative_img_times < img_selection[1]
                    features_ = min_ == max_
                    feature_IDs = np.where(features_)[0]

                    if feature_IDs.size > 0:

                        #
                        #
                        if average_imgs:

                            feature_imgs.append(
                                image.mean_img(
                                    image.index_img(data_dict[trl]['func_img'], feature_IDs)))
                            conditions.append(
                                data_dict[trl]['trial_type'].split('_')[0])
                            sessions.append(data_dict[trl]['session'])

                        #
                        # else use individual images, i.e. duplicate trials
                        else:

                            #
                            #
                            for el_ in feature_IDs:

                                feature_imgs.append(
                                    image.index_img(data_dict[trl]['func_img'], el_))

                                conditions.append(
                                    data_dict[trl]['trial_type'].split('_')[0])
                                sessions.append(data_dict[trl]['session'])

                #
                #
                elif method == 'ID':  # use_img_id:

                    #
                    # this implementation allows to select a fixed image ID relative to the first image of the trial
                    # add `img_id_for_decoding` to the first image of the given trial
                    feature_IDs = data_dict[trl]['first_img_id'] + \
                        img_selection-1
                    #
                    # remove scan IDs larger than maximum scans
                    feature_IDs = feature_IDs[feature_IDs <
                                              data_dict[trl]['func_img'].shape[-1]]

                    if feature_IDs.size > 0:
                        #
                        #
                        if average_imgs:

                            feature_imgs.append(
                                image.mean_img(
                                    image.index_img(data_dict[trl]['func_img'], feature_IDs)))
                            conditions.append(
                                data_dict[trl]['trial_type'].split('_')[0])
                            sessions.append(data_dict[trl]['session'])

                        #
                        #
                        else:  # if use individual images

                            #
                            # add as many pseudo-trials as there are images per trial
                            for el_ in feature_IDs:

                                feature_imgs.append(
                                    image.index_img(data_dict[trl]['func_img'], el_))
                                conditions.append(
                                    data_dict[trl]['trial_type'].split('_')[0])
                                sessions.append(data_dict[trl]['session'])

                if feature_IDs.size > 0:
                    #
                    # print info
                    # print(
                    #     trl,
                    #     'Trial onset:', data_dict[trl]['trial_period'][0],
                    #     '| Image IDs:', feature_IDs,
                    #     '| Scan times:', img_times[feature_IDs],
                    #     '| Scan times rel. to onset: ', img_times[feature_IDs]-data_dict[trl]['trial_period'][0])

                    self.multivariate['data'] = Bunch()
                    self.multivariate['data']['info'] = {
                        'Feature selection': method,
                        'Feature image ID': img_selection,
                        'Feature image t': img_times[feature_IDs]-data_dict[trl]['trial_period'][0],
                        'Averaged': average_imgs,
                        'fwhm': fwhm}
                    self.multivariate['data']['feature_imgs'] = feature_imgs
                    self.multivariate['data']['conditions'] = conditions
                    self.multivariate['data']['sessions'] = sessions

                    # print(self.multivariate['data']['info'])

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def searchlight_wrapper(
            self,
            mask_img=[],
            process_mask=None,
            cv=LeaveOneGroupOut()):

        print('Performing searchlight on',
              self.multivariate['data']['info'])
        print('CV method', cv)
        print()

        if not mask_img:
            mask_img = self.info.func.mean_mask

            #
            # init searchlight object
            # cv = CV # LeaveOneGroupOut(), KFold(n_splits=4)
            searchlight = decoding.SearchLight(
                mask_img=mask_img,
                process_mask_img=process_mask,  # process_mask_img, None
                radius=4,
                n_jobs=-1,
                verbose=1,
                cv=cv)

            #
            # fit
            searchlight.fit(
                self.multivariate['data']['feature_imgs'],
                self.multivariate['data']['conditions'],
                groups=self.multivariate['data']['sessions'])

            self.multivariate.searchlight.results.append(
                Bunch(
                    info=self.multivariate['data']['info'],
                    searchlight=searchlight))

    #
    #
    # --------------------------------------------------------------------------

    def decode_from_rois(
            self,
            permutation_test=20,
            rois=['neurosynth', 'functional', 'other', 'TPJ']):

        roi_names = []
        mask_types = []
        for rn in rois:
            print('\t***', rn, 'ROIs added to analysis.')
            rois_of_type = [roi for roi in self.info.ROIs[rn].keys()]
            roi_names.extend(rois_of_type)
            # roi_names = [roi for roi in self.info.ROIs.neurosynth.keys()]
            mask_types.extend([rn]*len(rois_of_type))
            # mask_types = [rn]*len(roi_names)

            # roi_names_fm = [roi for roi in self.info.ROIs.functional.keys()]
            # roi_names.extend(roi_names_fm)
            # mask_types.extend(['functional']*len(roi_names_fm))

            # roi_names_anat = [roi for roi in self.info.ROIs.anatomical.keys()]
            # roi_names.extend(roi_names_anat)
            # mask_types.extend(['anatomical']*len(roi_names_anat))

            # roi_names_ot = [roi for roi in self.info.ROIs.other.keys()]
            # roi_names.extend(roi_names_ot)
            # mask_types.extend(['other']*len(roi_names_ot))

        cv = LeaveOneGroupOut()

        mask_scores = {}
        mask_permuted_scores = {}

        #
        # helper for parallelizing the permutation test
        def perm_test_parallel(condition_list_s, feature_imgs_list, sessions_list, dp_, cv, masker):

            random.shuffle(condition_list_s)

            perm_classifier = Decoder(
                estimator='svc',
                cv=cv,
                mask=masker,
                scoring='roc_auc',
                n_jobs=-1)

            perm_classifier.fit(
                feature_imgs_list,
                condition_list_s,
                groups=sessions_list)

            return np.median(perm_classifier.cv_scores_[dp_.split('X')[0]])

        #
        for mask_type, mask_name in zip(mask_types, roi_names):

            # if mask_name not in exclude_rois:

            mask_scores[mask_name] = {}
            mask_permuted_scores[mask_name] = {}

            for mask_filename in self.info.ROIs[mask_type][mask_name]:

                #
                # if mask is mask image
                # if isinstance(mask_filename, str):
                if 'left' in mask_filename:
                    hemi_ = 'left'
                elif 'right' in mask_filename:
                    hemi_ = 'right'
                else:
                    hemi_ = 'medial'
                #
                masker = NiftiMasker(
                    mask_img=mask_filename,
                    standardize=True)

                #
                # if mask is ROI coordinates
                # elif isinstance(mask_filename, tuple):
                #     if mask_filename[0] < 0:
                #         hemi_ = 'left'
                #     elif mask_filename[0] > 0:
                #         hemi_ = 'right'
                #     #
                #     masker = NiftiSpheresMasker(
                #         seeds=[mask_filename],
                #         radius=5,
                #         # mask_img=self.info.func.mean_mask,
                #         standardize=True
                #     )
                #     print(masker)

            #         print("Working on %s" % mask_name)
                # For decoding, standardizing is often very important
                # if mask_type in ['neurosynth', 'func_masks']:
                #     masker = NiftiMasker(
                #         mask_img=mask_filename,
                #         standardize=True)
                # elif mask_type in ['theoretical', 'anatomical']:
                #     masker = NiftiSpheresMasker(
                #         seeds=[mask_filename],
                #         radius=5,
                #         mask_img=subjects['sub-00'].info.func.mean_mask,
                #         standardize=True
                #     )

                mask_scores[mask_name][hemi_] = {}
                mask_permuted_scores[mask_name][hemi_] = {}

                if 'ec_V' in np.unique(self.info.run_identifier):
                    task_ = 'ec'
                elif 'eic' in np.unique(self.info.run_identifier):
                    task_ = 'eic'

                #
                # iterate decoding pairs (pairs of conditions) specified in info.py
                for dp_ in info.decoding_pairs[task_].keys():

                    try:

                        print("\t\tProcessing %s %s %s" % (mask_name, hemi_, dp_))

                        #
                        # prepare data
                        c1_cat = info.decoding_pairs[task_][dp_][0]
                        c2_cat = info.decoding_pairs[task_][dp_][1]

                        conditions_series = pd.Series(
                            self.multivariate.data.conditions)
                        c1_mask = conditions_series.isin(c1_cat)
                        c2_mask = conditions_series.isin(c2_cat)
                        dat_mask = np.logical_or(c1_mask, c2_mask)

                        conditions_series[c1_mask] = dp_.split('X')[0]
                        conditions_series[c2_mask] = dp_.split('X')[1]

                        conditions_list = list(conditions_series[dat_mask])
                        feature_imgs_list = list(
                            pd.Series(self.multivariate.data.feature_imgs)[dat_mask])
                        sessions_list = list(
                            pd.Series(self.multivariate.data.sessions)[dat_mask])

                        # Specify the classifier to the decoder object.
                        # With the decoder we can input the masker directly.
                        # We are using the svc_l1 here because it is intra subject.
                        decoder = Decoder(
                            estimator='svc',
                            cv=cv,
                            mask=masker,
                            scoring='roc_auc',
                            n_jobs=-1)
                        decoder.fit(
                            feature_imgs_list,
                            conditions_list,
                            groups=sessions_list)
                        mask_scores[mask_name][hemi_][dp_] = decoder.cv_scores_[dp_.split('X')[
                            0]]

                        #
                        # perutation test
                        condition_list_s = conditions_list.copy()

                        mask_permuted_scores[mask_name][hemi_][dp_] = Parallel(
                            n_jobs=1,
                            backend='threading',
                            verbose=1)(
                            delayed(perm_test_parallel)(
                                condition_list_s, feature_imgs_list, sessions_list, dp_, cv, masker) for perm_ in range(permutation_test))

                        # print("Scores: %1.2f +- %1.2f" % (
                        #       np.mean(mask_scores[mask_name][hemi_][dp_]),
                        #       np.std(mask_scores[mask_name][hemi_][dp_])),
                        #      'Chance:  %1.2f +- %1.2f' % (
                        #       np.mean(mask_permuted_scores[mask_name][hemi_][dp_]),
                        #       np.std(mask_permuted_scores[mask_name][hemi_][dp_])))
                        # print()

                    except: # may throw 'ValueError: Ill-posed l1_min_c calculation: l1 will always select zero coefficients for this data'
                        print()
                        print('\t\t***WARNING***: ', mask_name, ' skipped.')
                        print()
                        pass

        self.multivariate.roi_decoding.results.append(
            Bunch(
                info=self.multivariate['data']['info'],
                scores=mask_scores,
                permuted_scores=mask_permuted_scores))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def plot_glass_and_zcuts(
            self,
            figure_dict,
            store_=False):

        for task in figure_dict:

            # #
            # # Load GFP from Marquardt et al. for contour
            # gfp_kira = image.load_img(
            #     '/media/marius/data_ex/ds_kira/derivatives/spm/spm_confounds/second_level/spmT_0001.nii')
            # gfp_kira_thr = threshold_stats_img(
            #     gfp_kira, alpha=.05, height_control='bonferroni', cluster_threshold=10)[0]

            # #
            # # Load MT activation for contour
            # mt_loc_loaded = 0
            # if task != 'MTloc':
            #     try:
            #         idx_ = np.argmax(
            #             [len(el) for el in list(self.univariate.results['MTloc'].keys())])
            #         mTruns_ = list(self.univariate.results['MTloc'].keys())[idx_]
            #         z_map_mt = [im for im in
            #                     self.univariate.results['MTloc'][mTruns_]['MotionStatic']['fwhm-5']
            #                     if 'z-score.nii' in im]
            #         thresholded_map_mt, threshold_mt = threshold_stats_img(
            #             z_map_mt, alpha=0.05, height_control='bonferroni',
            #             cluster_threshold=5)
            #         mt_loc_loaded = 1
            #     except:
            #         pass


            #
            # load contrast dict for figure titles
            contrast_dict = info.get_contrast_dict(task.split('_')[0])

            #
            # load anatomical image for background
            anat_img = self.info.anat.img

            # f, ax = plt.subplots(2*np.size(con_list), 1,
            #                      figsize=(25,10*np.size(con_list)), dpi=100)
            for icon, con in enumerate(figure_dict[task]):
                print()
                print(con)
                print('Specs:', figure_dict[task][con]['specs'])
                # print('------------------')
                # if mt_loc_loaded and task != 'MTloc':
                #     print('MTloc contrast loaded: ', z_map_mt)
                # elif not mt_loc_loaded and task != 'MTloc':
                #     print('No MTloc contrast found...')

                #
                # Load map

                # map_ = [im for im in self.univariate.results[task][figure_dict[task][con]['run_set']][con]
                #         if 'fwhm-'+str(
                #     figure_dict[task][con].fwhm)+'_'+figure_dict[task][con].specs+'_'+figure_dict[task][con].which_map in im]

                map_ = [im for im in self.univariate.results[task][figure_dict[task][con]['run_set']][con][figure_dict[task][con]['fwhm']]
                        if figure_dict[task][con]['specs']+'_'+figure_dict[task][con]['which_map'] in im]

                print('Loaded image: ', map_[0].split('/')[-1])

                #
                # init figures
                if figure_dict[task][con]['glass']:
                    f1 = plt.figure(figsize=(7, 2))
                #
                cut_coords = figure_dict[task][con]['cut_coords']
                cols = 6
                rows = int(np.ceil(len(cut_coords) / cols))
                n = len(cut_coords)
                f2, axs2 = plt.subplots(rows, cols, figsize=(25, rows*5), dpi=300)
                f2.patch.set_facecolor('k')
                f2.suptitle(contrast_dict[con].name, color='w', fontsize=17)
                # f2.tight_layout()

                #
                # threshold the map_
                if figure_dict[task][con]['which_map'] not in ["effect_size", "effect_variance"]:
                    thresholded_map, threshold = threshold_stats_img(
                        map_, alpha=figure_dict[task][con]['alpha'],
                        height_control=figure_dict[task][con]['h_control'],
                        cluster_threshold=figure_dict[task][con]['c_threshold'])
                else:
                    thresholded_map = map_[0]

                #
                # plot glass brain if applicable
                # f, ax = plt.subplots(2, 1, figsize=(25,10), dpi=300)
                if figure_dict[task][con]['glass']:
                    plotting.plot_glass_brain(
                        thresholded_map, plot_abs=False, black_bg=True,
                        colorbar=figure_dict[task][con]['colorbar'], title=contrast_dict[con].name,
                        figure=f1)

                #
                # plot the stat map
                for ax, cut in zip(axs2.flat, cut_coords):
                    display = plotting.plot_stat_map(
                        thresholded_map,
                        vmax=figure_dict[task][con]['vmax'],
                        bg_img=anat_img,
                        threshold=0,
                        display_mode='z',
                        cut_coords=[cut],
                        colorbar=figure_dict[task][con]['colorbar'],
                        black_bg=True,
                        # title=contrast_dict[con].name,
                        axes=ax,
                        figure=f2)

                    if figure_dict[task][con]['plot_kira']:

                        gfp_kira = image.load_img(
                            '/media/marius/data_ex/ds_kira/derivatives/spm/spm_confounds/second_level/spmT_0001.nii')
                        gfp_kira_thr = threshold_stats_img(
                            gfp_kira, alpha=.05, height_control='bonferroni', cluster_threshold=10)[0]

                        display.add_contours(
                            gfp_kira_thr,
                            colors=info.colors['Marquardt2017'],
                            alpha=1,
                            levels=[0],
                            linewidths=1.2)

                    if figure_dict[task][con]['plot_ind_MT'] and task != 'MTloc':#mt_loc_loaded:

                        idx_ = np.argmax(
                            [len(el) for el in list(self.univariate.results['MTloc'].keys())])
                        mTruns_ = list(self.univariate.results['MTloc'].keys())[idx_]
                        z_map_mt = [im for im in
                                    self.univariate.results['MTloc'][mTruns_]['MotionStatic']['fwhm-5']
                                    if 'z-score.nii' in im][0]
                        thresholded_map_mt, threshold_mt = threshold_stats_img(
                            z_map_mt, alpha=0.05, height_control='bonferroni',
                            cluster_threshold=5)

                        display.add_contours(
                            image.math_img('abs(img)', img=thresholded_map_mt),
                            colors=info.colors['MTloc'],
                            alpha=1,
                            levels=[0],
                            linewidths=1.2)

                    if figure_dict[task][con]['plot_ns_MT']:
                        ns_visual_motion_left = image.load_img(
                            '/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/gm_analysis/fmrius/ROIs/neurosynth/visual-motion_MT_thrs-None-100_left.nii.gz')
                        ns_visual_motion_right = image.load_img(
                            '/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/gm_analysis/fmrius/ROIs/neurosynth/visual-motion_MT_thrs-None-100_right.nii.gz')
                        display.add_contours(
                            ns_visual_motion_left,
                            colors='w',
                            alpha=1,
                            levels=[0],
                            linewidths=1.2)
                        display.add_contours(
                            ns_visual_motion_right,
                            colors='w',
                            alpha=1,
                            levels=[0],
                            linewidths=1.2)

                    if figure_dict[task][con]['add_contours']:
                        left_contour_img = image.load_img(
                            figure_dict[task][con]['add_contours'][0])
                        right_contour_img = image.load_img(
                            figure_dict[task][con]['add_contours'][1])

                        if np.size(np.shape(left_contour_img)) > 3:
                            left_contour_img = image.index_img(left_contour_img, 0)
                        if np.size(np.shape(right_contour_img)) > 3:
                            right_contour_img = image.index_img(right_contour_img, 0)

                        display.add_contours(
                            left_contour_img,
                            colors='k',
                            alpha=1,
                            levels=[0],
                            linewidths=1.2)
                        display.add_contours(
                            right_contour_img,
                            colors='k',
                            alpha=1,
                            levels=[0],
                            linewidths=1.2)

                # remove unused axes
                for ax in axs2.flat[n:]:
                    ax.remove()
                # f2.tight_layout()

                # fig = plotting.plot_stat_map(
                #     thresholded_map, vmax=figure_dict[task][con]['vmax'],
                #     bg_img=anat_img, threshold=0, display_mode='z',
                #     cut_coords=figure_dict[task][con]['cut_coords'],
                #     colorbar=figure_dict[task][con]['colorbar'],
                #     black_bg=True, title=contrast_dict[con].name, figure=f2)

                # #
                # # add Kria's GFP contours if applicable
                # if figure_dict[task][con]['plot_kira']:
                #     fig.add_contours(gfp_kira_thr, colors=info.colors['Marquardt2017'],
                #                      alpha=1, levels=[0], linewidths=0.6)

                #
                # add MT contours if applicable
                # if mt_loc_loaded:
                #     fig.add_contours(
                #         image.math_img('abs(img)', img=thresholded_map_mt),
                #         colors=info.colors['MTloc'],
                #         alpha=1, levels=[0], linewidths=0.6)


                if store_:
                    pdf_dir_ = opj(info.data_paths.results_dir,
                                   'from_nilearn/univariate_contrasts',
                                   self.ID,
                                   'pdfs',
                                   figure_dict[task][con]['fwhm'],
                                   con,
                                   figure_dict[task][con]['run_set'])

                    # if not isdir(pdf_dir_):
                    os.makedirs(pdf_dir_, exist_ok=True)
                        # print('***Output folder ', pdf_dir_, 'generated.')

                    if figure_dict[task][con]['glass']:
                        f_name_ = task+'_'+figure_dict[task][con]['run_set']+'_'+con+'_'+figure_dict[task][con]['specs']+'_'+figure_dict[task][con]['fwhm']+'_'+figure_dict[task][con]['h_control']+'-p'+str(
                            figure_dict[task][con]['alpha'])+'-c'+str(figure_dict[task][con]['c_threshold'])+'_glass.pdf'
                        # print(f_name_)
                        # f_name_ = f_name_.replace(' ', '').replace(':', '-')
                        # print(f_name_)
                        f1.savefig(opj(pdf_dir_, f_name_))

                    f_name_ = task+'_'+figure_dict[task][con]['run_set']+'_'+con+'_'+figure_dict[task][con]['specs']+'_'+figure_dict[task][con]['fwhm']+'_'+figure_dict[task][con]['h_control']+'-p'+str(
                        figure_dict[task][con]['alpha'])+'-c'+str(figure_dict[task][con]['c_threshold'])+'_trans-cuts.pdf'
                    # f_name_ = f_name_.replace(' ', '').replace(':', '-')
                    f2.savefig(opj(pdf_dir_, f_name_))

                #
                plt.show(block=False)

    #
    #
    # --------------------------------------------------------------------------

    def interactive_plot(
        self,
        figure_dict
    ):

        activated_regions = {}
        for task in figure_dict:

            contrast_dict = info.get_contrast_dict(task.split('_')[0])
            anat_img = self.info.anat.img

            # f, ax = plt.subplots(2*np.size(con_list), 1,
            #                      figsize=(25,10*np.size(con_list)), dpi=100)
            for icon, con in enumerate(figure_dict[task]):
                activated_regions[con] = {}
                print()
                print(con)
                print('------------------')

                map_ = [im for im in self.univariate.results[task][figure_dict[task][con]['run_set']][con][
                    figure_dict[task][con]['fwhm']]
                        if figure_dict[task][con]['specs'] + '_' + figure_dict[task][con]['which_map'] in im]
                # map_ = [im for im in self.univariate.results[task][figure_dict[task][con]['run_set']][con]
                #         if 'fwhm-'+str(
                #     figure_dict[task][con]['fwhm'])+'_'+figure_dict[task][con]['specs']+'_'+figure_dict[task][con]['which_map']+'.nii' in im]
                # map_ = [im for im in self.univariate.results[task][figure_dict[task][con]['run_set']][con]
                #          if 'fwhm-'+str(figure_dict[task][con].fwhm)+'.nii' in im]
                print('Loaded image: ', map_[0].split('/')[-1])

                # f = plt.figure(figsize=(5,2))

                thresholded_map, threshold = threshold_stats_img(
                    map_, alpha=figure_dict[task][con]['alpha'],
                    height_control=figure_dict[task][con]['h_control'],
                    cluster_threshold=figure_dict[task][con]['c_threshold'])
                ###
                # f, ax = plt.subplots(2, 1, figsize=(25,10), dpi=300)

                if figure_dict[task][con].glass:
                    plotting.plot_glass_brain(
                                thresholded_map,
                                plot_abs=False,
                                black_bg=True,
                                colorbar=True,
                                title=contrast_dict[con].name)
                    plt.show()

                display(plotting.view_img(
                    thresholded_map,
                    bg_img=anat_img, threshold=0, display_mode='tiled',
                    cut_coords=figure_dict[task][con]['cut_coords'],
                    title=contrast_dict[con].name))

                clust_frame, peaks_frame = atlasreader.create_output(
                    thresholded_map, cluster_extent=0, atlas=['juelich', 'harvard_oxford'])
                activated_regions[con]['clusters'] = clust_frame
                activated_regions[con]['peaks'] = peaks_frame

            return activated_regions

            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # def make_lisa_events(
            #             self):
            #
            #     #
            #     # iterare functional runs
            #     for kk in [el for el in list(self.info.keys())
            #                if el not in ["run_identifier", "anat"]]:
            #
            #         #
            #         # load events
            #         events_df = pd.read_csv(self.info[kk].events, sep="\t")
            #
            #         #
            #         # make tril_type ID list
            #         trial_type_ids = {}
            #         id_ = 1
            #         for tt in np.sort(events_df.trial_type.unique()):
            #             trial_type_ids[tt] = id_
            #             id_ += 1
            #         events_id_list = [trial_type_ids[el]
            #                           for el in list(events_df.trial_type)]
            #         #
            #         # store trial_type mapping for reconstruction
            #         pd.DataFrame(trial_type_ids, index=[0]).to_csv(
            #             self.info[kk].events[:-4]+"_mapping.txt",
            #             header=True, index=None, sep='\t')
            #
            #         #
            #         # make new events dataframe
            #         events_dict_new = {}
            #         events_dict_new["event"] = events_id_list
            #         events_dict_new["onset"] = events_df.onset
            #         events_dict_new["duration"] = events_df.duration.astype(float)
            #         if "modulation" in events_df.columns:
            #             events_dict_new["amplitude"] = events_df.modulation
            #         else:
            #             events_dict_new["amplitude"] = len(events_df)*[1]
            #         events_df_new = pd.DataFrame(events_dict_new)
            #
            #         #
            #         # store as .txt
            #         events_df_new.to_csv(self.info[kk].events[:-3]+"txt",
            #                              header=None, index=None, sep=' ')
            #         print("File stored: ", self.info[kk].events[:-3]+"txt")

            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
