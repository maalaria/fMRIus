import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.patches import Rectangle
import importlib
import json
import pandas as pd
from os.path import join as opj

from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs, resample_img, mean_img, load_img
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn import masking
from nilearn.interfaces import fmriprep


# sys.path.append('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/python')
# from .src.info import info
# import helpers as myHelpers


def first_level_wrapper(
    subject_data,
    task_id_,
    selected_runs,
    fwhm,
    confound_parms,
    concatenate_runs=False,
    permute_events=False,
    hrf_model='glover + derivative + dispersion'
    ):

    # select runs based on specified task_id
    # if not runs:
    #     runs = [rr for rr in subject_data.keys() if task_id in rr]


    design_matrices = []
    func_imgs = []
    mask_imgs = []

    if not permute_events:
        print('Generating design matrices...')

    # sample_masks = []
    for rr in selected_runs:

        rr = task_id_+'_'+rr

        # get TR
        with open(subject_data[rr]['meta']) as json_file:
            data = json.load(json_file)
        tr = data['RepetitionTime']
        json_file.close()

        # load images and get frame number
        func_imgs.append(load_img(subject_data[rr]['img']))
        mask_imgs.append(load_img(subject_data[rr]['mask']))
        n_scans = func_imgs[-1].shape[-1]  # get scan number
        frame_times = np.arange(n_scans) * tr  # here are the correspoding frame times

        # get events
        events_df = pd.read_csv(subject_data[rr]['events'], sep='\t')
        if permute_events:
            print('Permuting events ...')
            events_df.trial_type = np.random.permutation(events_df.trial_type)

        # confounds

        confounds_df, sample_mask = fmriprep.load_confounds(
            subject_data[rr]['img'],
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
        # confounds_df, sample_mask = fmriprep.load_confounds(
        #     subject_data[rr]['img'],
        #     strategy=['motion', 'high_pass', 'wm_csf', 'compcor', 'scrub', 'non_steady_state'],
        #     motion='full',
        #     scrub=0,
        #     fd_threshold=.5,
        #     std_dvars_threshold=3,
        #     wm_csf='basic',
        #     global_signal='basic',
        #     compcor='anat_combined',
        #     n_compcor='all',
        #     demean=True
        # )

        # sample_masks.append(sample_mask)
        # print(sample_mask)

        # confounds_df = pd.read_csv(subject_data[rr]['confounds'], sep='\t')
        # confounds_df = confounds_df[list(confound_selection)]

        # make design matrix of run
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events_df,
            drift_model=None,#'polynomial',
            # drift_order=3,
            add_regs=confounds_df,
            hrf_model=hrf_model)

        # collect design matrix of run
        design_matrices.append(design_matrix)

    # resample to first image... necessary after preprocssing?
    affine, shape = func_imgs[0].affine, func_imgs[0].shape
    if not permute_events:
        print('Resampling images to first image...')
    for fi in func_imgs:
        fi = resample_img(fi, affine, shape[:3])

    # compute mask intersection
    grp_mask = masking.intersect_masks(mask_imgs, threshold=1)

    # if concatenate_runs, one big design matrix in generated and func_images are concatenated
    # required if cubes_v are compared to cubes_h
    if concatenate_runs:

        if not permute_events:
            print('Fitting first level model (runs concatenated)...')

        concatenated_design_matrix = pd.concat(design_matrices)
        # replace NaN by 0
        concatenated_design_matrix = concatenated_design_matrix.fillna(0)
        # reorder columns to have it nicer
        concatenated_design_matrix = concatenated_design_matrix[
            ['Cubes_1_v', 'Cubes_1_v_derivative', 'Cubes_2_v', 'Cubes_2_v_derivative',
             'Cubes_3_v', 'Cubes_3_v_derivative', 'Cubes_1_h', 'Cubes_1_h_derivative',
             'Cubes_2_h', 'Cubes_2_h_derivative', 'Cubes_3_h', 'Cubes_3_h_derivative',
             'Eyes_1', 'Eyes_1_derivative', 'Eyes_2', 'Eyes_2_derivative', 'Eyes_3', 'Eyes_3_derivative',
             'global_signal', 'csf', 'trans_x', 'trans_y', 'trans_z', 'rot_x',
             'rot_y', 'rot_z', 'drift_1', 'drift_2', 'drift_3', 'constant']]
        concated_func_img = nilearn.image.concat_imgs(func_imgs)
        fmri_glm = FirstLevelModel(
            n_jobs=-1,
            smoothing_fwhm=fwhm,
            # hrf_model=hrf_model, # already in design matrix
            #high_pass=0.01, ### not necessaryly needed of the design matix includes a drift model (both remove slow drifts)
            mask_img=grp_mask)
        fmri_glm = fmri_glm.fit(
            concated_func_img,
            design_matrices=concatenated_design_matrix,
            # sample_masks=sample_masks
        )
        design_matrices = [concatenated_design_matrix] ### just for plotting the design matrix

    else:

        if not permute_events:
            print('Fitting first level model (runs not concatenated)...')

        # fit model
        fmri_glm = FirstLevelModel(
            n_jobs=-1,
            smoothing_fwhm=fwhm,
            # hrf_model=hrf_model, # already in design matrix
            #high_pass=0.01, ### not necessaryly needed of the design matix includes a drift model (both remove slow drifts)
            mask_img=grp_mask)
        fmri_glm = fmri_glm.fit(
            func_imgs,
            design_matrices=design_matrices,
            # sample_masks=sample_masks
        )

    if not permute_events:
        print('FINISHED')

    return fmri_glm, design_matrices, func_imgs
