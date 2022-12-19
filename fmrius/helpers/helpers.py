from nipype.interfaces.dcm2nii import Dcm2niix
from pathlib import Path
import os
import pandas as pd
from nipype.interfaces.base import Bunch
import numpy as np
import random
from os.path import join as opj
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import importlib

from nilearn.glm import threshold_stats_img
from nilearn import image
from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap

# sys.path.append('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/python')
# import info
# importlib.reload(info)





def move2dicom(paths_dict, subject_list):
    for subj in subject_list:# paths_dict['scans']:
        for st in paths_dict['scans'][subj]:
            if st == 'anatomical':

                try:
                    for cp in paths_dict['scans'][subj][st]:

                        try:
                            ### make DICOM Folder
                            os.mkdir( str(Path( cp, 'DICOM' )) )

                            ### get all image filenames
                            folder_content = os.listdir(cp)
                            images = [fn for fn in folder_content if fn != 'DICOM']

                            ### move images to DICOM filder
                            [ os.rename( Path(cp, img), Path(cp, 'DICOM', img) ) for img in images ]

                            print(subj, st, 'moved')

                        except:
                            print([subj, st], "folder exists")
                except:
                    print([subj, st], "empty")

            elif st == 'functional':
                for rn in paths_dict['scans'][subj][st]:

                    try:
                        for cp in paths_dict['scans'][subj][st][rn]:
                            ### make DICOM Folder
                            try:
                                os.mkdir( str(Path( cp, 'DICOM' )) )

                                ### get all image filenames
                                folder_content = os.listdir(cp)
                                images = [fn for fn in folder_content if fn != 'DICOM']

                                ### move images to DICOM filder
                                [ os.rename( Path(cp, img), Path(cp, 'DICOM', img) ) for img in images ]

                                print(subj, st, rn, 'moved')

                            except:
                                print([subj, st], "folder exists")

                    except:
                            print([subj, st], "empty")

#####################################################
#####################################################




def dicom2nifit(paths_dict, subject_list, what_todo):
    #
    ##### Dcm2niix Options #####
    #
    #   -1..-9 : gz compression level (1=fastest..9=smallest, default 6)
    #   -b : BIDS sidecar (y/n/o(o=only: no NIfTI), default y)
    #    -ba : anonymize BIDS (y/n, default y)
    #   -c : comment stored as NIfTI aux_file (up to 24 characters)
    #   -d : diffusion volumes sorted by b-value (y/n, default n)
    #   -f : filename (%a=antenna  (coil) number, %c=comments, %d=description, %e echo number, %f=folder name, %i ID of patient, %j seriesInstanceUID, %k studyInstanceUID, %m=manufacturer, %n=name of patient, %p=protocol, %s=series number, %t=time, %u=acquisition number, %v=vendor, %x=study ID; %z sequence name; default '%f_%p_%t_%s')
    #   -h : show help
    #   -i : ignore derived, localizer and 2D images (y/n, default n)
    #   -m : merge 2D slices from same series regardless of study time, echo, coil, orientation, etc. (y/n, default n)
    #   -o : output directory (omit to save to input folder)
    #   -p : Philips precise float (not display) scaling (y/n, default y)
    #   -s : single file mode, do not convert other images in folder (y/n, default n)
    #   -t : text notes includes private patient details (y/n, default n)
    #   -v : verbose (n/y or 0/1/2 [no, yes, logorrheic], default 0)
    #   -x : crop (y/n, default n)
    #   -z : gz compress images (y/i/n/3, default n) [y=pigz, i=internal, n=no, 3=no,3D]

    for subj in subject_list:# paths_dict['scans']:
        for st in paths_dict['scans'][subj]:

            if st == 'anatomical':
                try:
                    sd = str(Path(paths_dict['scans'][subj][st][0], 'DICOM'))
                    od = paths_dict['scans'][subj][st][0]

                    ### CONVERT
                    if what_todo == 'convert':
                        converter = Dcm2niix()
                        converter.inputs.source_dir = sd
                        converter.inputs.compression = 3
                        converter.inputs.output_dir = od
                        converter.cmdline
                        'dcm2niix -b y -z y -3 -x n -t n -m n'
                        converter.run() # doctest: +SKIP

                    ### rm all but original DICOMs
                    if what_todo == 'delete_nifti':
                        [os.remove(Path(od, fn)) for fn in os.listdir(od) if len(fn) > 6]

                except:
                    print([subj, st], 'empty')

            if st == 'functional':
                for rn in paths_dict['scans'][subj][st]:
                    try:
                        sd = str(Path(paths_dict['scans'][subj][st][rn][0], 'DICOM'))
                        od = paths_dict['scans'][subj][st][rn][0]

                        ### CONVERT
                        if what_todo == 'convert':
                            converter = Dcm2niix()
                            converter.inputs.source_dir = sd
                            converter.inputs.compression = 3
                            converter.inputs.output_dir = od
                            converter.cmdline
                            'dcm2niix -b y -z y -3 -x n -t n -m n'
                            converter.run() # doctest: +SKIP

                        ### rm all but original DICOMs
                        if what_todo == 'delete_nifti':
                            [os.remove(Path(od, fn)) for fn in os.listdir(od) if len(fn) > 6]

                    except:
                        print([subj, st], 'empty')

            if st == 'fieldmap':
                try:
                    for fp in paths_dict['scans'][subj][st]:

                        ### CONVERT
                        if what_todo == 'convert':
                            converter = Dcm2niix()
                            converter.inputs.source_dir = fp
                            converter.inputs.compression = 3
                            converter.inputs.output_dir = fp
                            converter.cmdline
                            'dcm2niix -b y -z y -3 -x n -t n -m n'
                            converter.run() # doctest: +SKIP

                        ### rm all but original DICOMs
                        if what_todo == 'delete_nifti':
                            fns = os.listdir(fp)

                            for item in fns:
                                if item.endswith(".nii.gz") or item.endswith(".json"):
                                    os.remove(os.path.join(fp, item))

                except:
                    print([subj, st], 'empty')



###################################################
###################################################

# def subjectinfo(sub_id, run_id, confound_selection):#paths_dict, confound_selection, arrow_gaze_path):
#
# #     [paths_dict, base_dir, orig_dir, bids_dir, derivatives_dir, scratch_dir, preproc_dir, subjects, bids_subjects, bids_sessions, bids_data_types, bids_runs, bids_task] = getPathDict()
#     import pandas as pd
#     from nipype.interfaces.base import Bunch
#     from os.path import join as opj
#     import numpy as np
#     import info
#
#     events_path = opj(info.root_dir, sub_id, 'func', sub_id+'_task-'+run_id+'_events.tsv')  #paths_dict['bids'][subject_id]['func'][run_id]
#     trialinfo = pd.read_table(events_path)
#     conditions = []
#     onsets = []
#     durations = []
#
#     for group in trialinfo.groupby('trial_type'):
#         conditions.append(group[0])
#         onsets.append(list(group[1].onset))
#         durations.append(group[1].duration.tolist())
#
#     subject_info = []
#     subject_info = [
#         Bunch(
#             conditions=conditions,
#             onsets=onsets,
#             durations=durations
#         )
#     ]
#
#     ### confounds
#     if confound_selection:
#         confounds = pd.read_csv(opj(info.fmriprep_dir, sub_id, "func", sub_id+"_task-"+run_id+"_desc-confounds_timeseries.tsv"),
#                sep="\t", na_values="n/a")
#         # select confounds to use
#         confounds = confounds[confound_selection]
#         # make list and replace leading NaNs by regressor means
#         regressors = []
#         for ii,reg in enumerate(confounds.transpose().values):
#             if np.isnan(reg[0]):
#                 reg[0] = np.nanmean( reg )
#             regressors.append( reg.tolist() )
#
#         subject_info[0].regressor_names = list(confounds.columns)
#         subject_info[0].regressors = regressors
#
#     return subject_info



def nearest_local_maximum(df, coords, sign='both'):

    if sign == 'positive':
        df = df[df['peak_value'] > 0]
    if sign == 'negative':
        df = df[df['peak_value'] < 0]

    distances = []
    for index, row in df.iterrows():
        ref_coords = np.array((row.peak_x, row.peak_y, row.peak_z))
        distances.append(np.linalg.norm(ref_coords - coords))

    return df.index[np.argmin(distances)], \
           np.array( (df.iloc[np.argmin(distances)].peak_x,
                      df.iloc[np.argmin(distances)].peak_y,
                      df.iloc[np.argmin(distances)].peak_z) ),\
           df.iloc[np.argmin(distances)].peak_value



def sphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0



###################
### PLOTTING
###################

# def plotting_wrapper(
#     what,
#     task,
#     con_list,
#     fname_extension,
#     sub_folder,
#     sub_id,
#     alpha,
#     height_control,
#     cluster_threshold,
#     fwhm,
#     subject_data,
#     coords=None):
#
#     ###
#
#     contrast_dict = info.get_contrast_dict()
#     anat_img = subject_data[sub_id]['anat']['nii']
#     ###
#     gfp_kira = image.load_img('/media/marius/data_ex/ds_kira/derivatives/spm/spm_confounds/second_level/spmT_0001.nii')
#     gfp_kira_thr = threshold_stats_img(
#         gfp_kira, alpha=.05, height_control='bonferroni', cluster_threshold=10)[0]
#
#
#     ### MT
#     if not task == 'MTloc' and not what == 'interactive':
#         try:
#             z_map_mt = opj( info.eyes_cubes_dirs.results_dir, 'from_nilearn/univariate_contrasts', sub_folder,
#                 sub_id+'_task-MTloc_MotionStatic'+'_fwhm-'+str(fwhm)+fname_extension+'.nii')
#             thresholded_map_mt, threshold_mt = threshold_stats_img(
#                 z_map_mt, alpha=10**-10, height_control='bonferroni', cluster_threshold=10)
#             print('MTloc contrast loaded: ', z_map_mt)
#         except:
#             print('No MTloc contrast found...')
#
#
#     if what == 'glass_z-cuts':
#
#         f, ax = plt.subplots( 2*np.size(con_list), 1, figsize=(25,10*np.size(con_list)), dpi=100 )
#         for icon,con in enumerate(con_list):
#             print()
#             print(con)
#             print('------------------')
#             z_map = opj( info.eyes_cubes_dirs.results_dir, 'from_nilearn/univariate_contrasts', sub_folder,
#                 sub_id+'_task-'+task+'_'+con+'_fwhm-'+str(fwhm)+fname_extension+'.nii')
#             print('Loaded image: ', z_map)
#             thresholded_map, threshold = threshold_stats_img(
#                 z_map, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
#             ###
#             # f, ax = plt.subplots( 2, 1, figsize=(25,10), dpi=300 )
#             plotting.plot_glass_brain(
#                 thresholded_map, plot_abs=False, black_bg=True, colorbar=True, title=contrast_dict[task][con].name, axes=ax[icon])
#             fig = plotting.plot_stat_map(
#                 thresholded_map,
#                 bg_img=anat_img, threshold=0, display_mode='z', cut_coords=np.arange(-14, 21, 3),
#                 black_bg=True, title=contrast_dict[task][con].name, axes=ax[np.size(con_list)+icon])
#             # fig.add_contours(gfp_kira_thr, colors=info.colors[2], alpha=0.5, levels=[0])
#             try:
#                 fig.add_contours(image.math_img( 'abs(img)', img=thresholded_map_mt ), colors=info.colors[3], alpha=.5, levels=[0])
#             except:
#                 pass
#
#             display(fig)
#
#             ### overview
#             # fig = plotting.plot_stat_map(
#             #     thresholded_map,
#             #     bg_img=anat_img, threshold=0, display_mode='z', black_bg=True, title=contrast_dict[task][con].name+' overview', axes=ax[2])
#             # # fig.add_contours(image.math_img( 'abs(img)', img=thresholded_map_mt ), colors=info.colors[3], alpha=.5, levels=[0])
#             # fig.add_contours(gfp_kira_thr, colors=info.colors[2], alpha=.4, levels=[0])
#
#             ## legend
#             # p_MT = Rectangle((0, 0), 1, 1, fc=info.colors[3])
#             # plt.legend([p_MT], ["MT"], loc='best', bbox_to_anchor=(-10, .6, 0.5, 0.5))
#
#
#     if what == 'interactive':
#
#         for con in con_list:
#             print()
#             print(con)
#             print('------------------')
#             z_map = opj( info.eyes_cubes_dirs.results_dir, 'from_nilearn/univariate_contrasts', sub_folder,
#                 sub_id+'_task-'+task+'_'+con+'_fwhm-'+str(fwhm)+fname_extension+'.nii')
#             print('Loaded image: ', z_map)
#             thresholded_map, threshold = threshold_stats_img(
#                 z_map, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
#             ### interactove plotting
#             # fig = plt.figure(figsize=(20,10))
#             display(plotting.view_img(thresholded_map, bg_img=anat_img,
#                 threshold=0, title=contrast_dict[task][con].name, display_mode='tiled', cut_coords=coords))
#
#
#     if what == 'contour_comparison':
#         f, ax = plt.subplots( 1, 1, figsize=(30,5), dpi=300 )
#
#         ### init for legend
#         colors = info.colors
#         cc = []
#         recs = []
#         labels = []
#
#         fig = plotting.plot_anat( anat_img, display_mode='z', cut_coords=np.arange(-6, 21, 3), black_bg=True,
#             title='Comparison of locations', axes=ax )
#
#         if 'MTloc' in con_list:
#             cc = info.colors['MTloc']
#             fig.add_contours(thresholded_map_mt, filled=True, colors=cc, alpha=0.4, levels=[0])
#             recs.append(  Rectangle((0, 0), 1, 1, fc=cc) )
#             labels.append("MT localizer")
#
#         if 'Kira' in con_list:
#             cc = info.colors['Kira']
#             fig.add_contours(gfp_kira_thr, colors=cc, alpha=0.4, levels=[0])
#             recs.append(  Rectangle((0, 0), 1, 1, fc=cc) )
#             labels.append("Kira")
#
#         for icon,con in enumerate([con for con in con_list if not con in ['MTloc', 'Kira']]):
#             print()
#             print(con)
#             print('------------------')
#             z_map = opj( info.eyes_cubes_dirs.results_dir, 'from_nilearn/univariate_contrasts', sub_folder,
#                 sub_id+'_task-'+task+'_'+con+'_fwhm-'+str(fwhm)+fname_extension+'.nii')
#             print('Loaded image: ', z_map)
#             thresholded_map, threshold = threshold_stats_img(
#                 z_map, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
#             ###
#             if con=='EyesCubes':
#                 cc = info.colors['EyesCubes'][0]
#                 fig.add_contours(thresholded_map, colors=cc, alpha=.6, levels=[0])
#                 recs.append(  Rectangle((0, 0), 1, 1, fc=cc) )
#                 labels.append("Eyes activated")
#                 cc =  info.colors['EyesCubes'][1]
#                 fig.add_contours(image.math_img( 'np.multiply(img, -1)', img=thresholded_map ), colors=cc, alpha=.6, levels=[0])
#                 recs.append(  Rectangle((0, 0), 1, 1, fc=cc) )
#                 labels.append("Cubes activated")
#             else:
#                 cc = info.colors[con]
#                 fig.add_contours(image.math_img( 'np.abs(img)', img=thresholded_map ), colors=cc, alpha=.6, levels=[0])
#                 recs.append(  Rectangle((0, 0), 1, 1, fc=cc) )
#                 labels.append(con)
#
#         plt.legend(recs, labels, loc='best', bbox_to_anchor=(-0.2, 0.4, 0.4, 0.5))
#         display(fig)
