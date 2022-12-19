import numpy as np
from scipy.io import loadmat
import scipy.stats as st
from itertools import compress
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import pandas as pd





def i_dt(xy, fix_duration_threshold_sec, fix_dispersion_threshold, sf):

    ## fix_dispersion_threshold: maximum displacement of samples belonging to one fixation


    fix_idx = []

    if np.shape(xy)[0] < np.shape(xy)[1]:
        xy = np.transpose(xy)
    xy = np.array([xy[ii] for ii in range(np.shape(xy)[0]) if not np.any(np.isnan(xy[ii]))])
    xy_idx = np.arange(len(xy))
    fix_window_length = fix_duration_threshold_sec * sf

    ### I-DT loop:
    while xy_idx.any():

        if len(xy_idx) > fix_window_length:
            current_window_idxs = list(map(int, np.arange(fix_window_length)))
        else:
            current_window_idxs = list(map(int, np.arange(len(xy_idx))))

        current_window_vals = np.array([xy[ii].tolist() for ii in current_window_idxs])
        wd = (np.max(current_window_vals[:,0])-np.min(current_window_vals[:,0])) + (np.max(current_window_vals[:,1])-np.min(current_window_vals[:,1]))

        # if dispersion in current window is smaller than threshold
        if wd <= fix_dispersion_threshold:

            # while this is the case and window does not exceed amount of data points add next point to window
            while wd <= fix_dispersion_threshold and len(current_window_idxs) < len(xy[:,0]):

                ### append to indices of current window
                current_window_idxs.append( np.max(current_window_idxs)+1 )
                ### make new window
                current_window_vals = np.array([xy[ii].tolist() for ii in current_window_idxs])
                wd = (np.max(current_window_vals[:,0])-np.min(current_window_vals[:,0])) + (np.max(current_window_vals[:,1])-np.min(current_window_vals[:,1]))

            ### add indices of fixation to array and delete window from data
            fix_idx.append( np.array([xy_idx[ii] for ii in current_window_idxs]) )

            xy = np.delete(xy, current_window_idxs, 0)
            xy_idx = np.delete(xy_idx, current_window_idxs, 0)

        else: # if dispersion in current window is greater than threshold delete first data point

            xy = np.delete(xy, 0, 0)
            xy_idx = np.delete(xy_idx, 0, 0)


    return fix_idx







def i_vt(xy, sac_vel_threshold, blink_vel_threshold, sf):



    # force shape to [samples , dimensions]
    if np.shape(xy)[0] < np.shape(xy)[1]:
            xy = np.transpose(xy)
    # remove NaN
    xy = np.array([xy[ii] for ii in range(np.shape(xy)[0]) if not np.any(np.isnan(xy[ii]))])
    ### get list of indices
    xy_idx = list(np.arange(len(xy)))


    saccades = []
    c_sac_ = []
    fixations = []
    c_fix_ = []
    blinks = []
    c_blink_ = []

    vels = []

    while len(xy_idx)>1:

    #     current_idx = xy_idx.pop(0)
        current_start = xy[xy_idx[0]]
        current_end = xy[xy_idx[1]]
        current_vel = np.linalg.norm( current_start-current_end ) * sf # dist / s

        vels.append(current_vel)

        ### if it is a saccade
        if current_vel >= sac_vel_threshold and current_vel < blink_vel_threshold:
            ### add to saccade list and empty fixation list
            if c_fix_:
                fixations.append(np.unique(c_fix_))
            c_fix_ = []
            if c_blink_:
                blinks.append(np.unique(c_blink_))
            c_blink_ = []
            # add first and second element to current saccade list, remove the first one
    #         if not c_sac_:
            c_sac_.append( xy_idx.pop(0) )
            c_sac_.append( xy_idx[0] )


        ### if it is a fixation
        elif current_vel < sac_vel_threshold:
            ### add to fixation list and empty saccade list
            if c_sac_:
                saccades.append(np.unique(c_sac_)) # use unique here because
            c_sac_ = []
            if c_blink_:
                blinks.append(np.unique(c_blink_))
            c_blink_ = []
            # add first and second element to current fixation list, remove the first one
    #         if not c_fix_:
            c_fix_.append( xy_idx.pop(0) )
            c_fix_.append( xy_idx[0] )


        elif current_vel > blink_vel_threshold:
            ### add to blink list and empty saccade list
            if c_sac_:
                saccades.append(np.unique(c_sac_))
            c_sac_ = []
            if c_fix_:
                fixations.append(np.unique(c_fix_))
            c_fix_ = []
            # add first and second element to current fixation list, remove the first one
    #         if not c_fix_:
            c_blink_.append( xy_idx.pop(0) )
            c_blink_.append( xy_idx[0] )

    ### add the last event to the respective list
    if c_sac_:
        saccades.append(np.unique(c_sac_))
        c_sac_ = []
    if c_blink_:
        blinks.append(np.unique(c_blink_))
        c_blink_ = []
    if c_fix_:
        fixations.append(np.unique(c_fix_))
        c_fix_ = []

    return saccades, fixations, blinks








def calibration( calib_mat ):

    calib_dict = {'x':calib_mat['EyeX'].transpose(), 'y':calib_mat['EyeY'].transpose()}


    ######################################
    ### get eye events
    ######################################
    sf = 1000
    sac_vel_threshold = 0.6
    blink_vel_threshold = 60

    saccade_list = []
    fixation_list = []
    blink_list = []
    fixations_xvals = []
    fixations_yvals = []
    fix_vecs = []
    for itrl,x_vals in enumerate(calib_dict['x']):
        xy = np.transpose([x_vals, calib_dict['y'][itrl]])
        saccades, fixations, blinks = i_vt( xy, sac_vel_threshold, blink_vel_threshold, sf )
        saccade_list.append(saccades)
        fixation_list.append(fixations)
        blink_list.append(blinks)

        fixations_xvals.append([])
        fixations_yvals.append([])
        fix_vecs.append([])
        for fix in fixations:
            fixations_xvals[itrl].append(np.mean(xy[fix][:,0]))
            fixations_yvals[itrl].append(np.mean(xy[fix][:,1]))
            fix_vecs[itrl].append( np.sqrt(np.mean(xy[fix][:,0]**2 + xy[fix][:,1]**2) ) )

    iq1 , iq3 = np.percentile([el for subl in fix_vecs for el in subl] , [25,75])
    iqr = iq3 - iq1
    lower_outliers = [np.where(ll < iq1-1.5*iqr) for ll in fix_vecs]
    upper_outliers = [np.where(ll > iq3+1.5*iqr) for ll in fix_vecs]
    outliers = [np.append(lo[0], upper_outliers[ii][0]) for ii,lo in enumerate(lower_outliers)]

    # remove extreme outliers fixations
    for itrl, trls_fix in enumerate(fixations_xvals):
        [trls_fix.pop(rem) for rem in sorted(outliers[itrl], reverse=True)]
        [fixations_yvals[itrl].pop(rem) for rem in sorted(outliers[itrl], reverse=True)]


    ############################################
    ### compute transformation
    ############################################
    xdat = [el for sub in fixations_xvals for el in sub]
    ydat = [el for sub in fixations_yvals for el in sub]
    values = np.vstack([xdat, ydat])
    kernel = st.gaussian_kde(values, 0.08)


    ### remove outliers based on estiamted density
    outlier = []
    for xfix,yfix in zip(xdat, ydat):
        lik = kernel([xfix, yfix])
        if lik < 1:
            outlier.append(False)
        else:
            outlier.append(True)

    ### remove outlier fixations
    xdat_clean = np.transpose([el for el in compress( list(zip(xdat, ydat)), outlier )])[0]
    ydat_clean = np.transpose([el for el in compress( list(zip(xdat, ydat)), outlier )])[1]
    # plt.scatter(x_clean, y_clean)

    ### GET THE PEAKS
    xx = np.linspace( 0.5, 4, num=100 )
    yy = np.linspace( 1.5, 3.5, num=100 )
    xxx, yyy = np.meshgrid(xx, yy)

    vals = np.zeros([100,100])
    for iyd,yd in enumerate(yyy):
        cy = list(zip(xxx[0], yd))
        for ixy,xy in enumerate(cy):
            vals[iyd][ixy] = kernel(xy)

    image_max = ndi.maximum_filter(vals, size=3, mode='constant')
    coordinates = peak_local_max(vals, min_distance=5, num_peaks=9)

    peak_coordinates = []
    for coo in coordinates:
        peak_coordinates.append( [xx[coo[1]], yy[coo[0]]] )

    ### sort peaks first along x axis then along y axis
    sorted_idx = np.argsort(np.array(peak_coordinates)[:,0])
    peak_coordinates_sorted = np.array(peak_coordinates)[sorted_idx]
    y_sort_1 = np.argsort(peak_coordinates_sorted[:3][:,1])
    y_sort_2 = np.argsort(peak_coordinates_sorted[3:6][:,1]) + 3
    y_sort_3 = np.argsort(peak_coordinates_sorted[6:][:,1]) + 6
    peak_coordinates_sorted = peak_coordinates_sorted[np.hstack([y_sort_1, y_sort_2, y_sort_3])]


    ### reference locations to which the fixations will be mapped
    ref = []
    for xy in zip( calib_mat['TargetX'][0], calib_mat['TargetY'][0] ):
        ref.append(xy)
    ref = np.unique(ref, axis=0)

    ##################################################3
    ### learn transformation
    # Pad the data with ones, so that our transformation can do translations too
    n = peak_coordinates_sorted.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(peak_coordinates_sorted)
    Y = pad(ref)
    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)
    transform = lambda x: unpad(np.dot(pad(x), A))

    return np.array([[xel, yel] for xel, yel in zip(xdat_clean, ydat_clean)]), kernel, peak_coordinates_sorted, ref, transform, A





def get_RT(
    fp_dict,
    # eye_dict,
    # trl_list,
    # saccade_list,
    sac_vel_threshold=0.6,
    blink_vel_threshold=45,
    sf=1000,
    min_RT=0.2,
    min_target_saccade_duration=0.07,
    plot_=False):



    trl_lists = {}

    for sub_id in fp_dict:

        trl_lists[sub_id] = {}

        print(sub_id)

        for run_id in fp_dict[sub_id]:

            trl_lists[sub_id][run_id] = []
            print(run_id)

            mat = loadmat( fp_dict[sub_id][run_id][0] )
            trl_list = pd.read_csv( fp_dict[sub_id][run_id][1], sep='\t',
                                   names=['gaze_image', 'Instruction', 'gaze_targetID', 'cubes_targetID', 'stim_onset'] )
            eye_dict = {'x':mat['EyeX'].transpose(), 'y':mat['EyeY'].transpose()}

            saccade_list = []
            fixation_list = []
            blink_list = []
            fixations_xvals = []
            fixations_yvals = []
            fix_vecs = []

            for itrl,x_vals in enumerate(eye_dict['x']):
                xy = np.transpose([x_vals, eye_dict['y'][itrl]])
                saccades, fixations, blinks = i_vt( xy, sac_vel_threshold, blink_vel_threshold, sf )

                saccade_list.append(saccades)


            # trl_list = myeye.get_RT( eye_dict, trl_list, saccade_list, plot_=False )

            if plot_:
                plt.figure()

            cue_onset = trl_list['stim_onset']

            RT = []

            for itrl in range(len(saccade_list)):
                found = False

                for sac in saccade_list[itrl]:
                    saccade_onset = [sac/1000][0][0]
                    cRT_ = round(saccade_onset - cue_onset[itrl], 3)

                    ### if saccade has minimum duration, is after cue onset + minimum RT and if it is the first saccade satisfying these conditions
                    if (np.size(sac) > min_target_saccade_duration*1000) and (saccade_onset > cue_onset[itrl]+min_RT) and (not found):
                        found = True
                        RT.append(cRT_)
                        if plot_:
                            plt.plot( eye_dict['x'][itrl][sac], eye_dict['y'][itrl][sac])

                ### collect NaN of no saccade to target was found
                if found == False:
                    RT.append( np.NaN )

            ### make numpy array
            RT = np.array(RT)
            ### add RT to dataframe
            trl_list['RT'] = RT



            trl_lists[sub_id][run_id] = trl_list




    return trl_lists








# ################
# # behavioral analysis: infos about cued target
# def get_target_info(identified_targets_dict, paths_dict, subj, run):
#
#     cm_to_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4} ### mapping from colormapping identifiers to target IDs
#
#     trialDict_correct = {}
#     trialDict_cm = {}
#     trialDict_df = {}
#     trl_cntr_correct = 0
#     trl_cntr_other = 0
#
#     ### loop over trial lists files of current run
#     for blck in paths_dict['trial_lists']['MRI'][subj][run]:
#
#         ### open trial list file
#         f = open(blck)
#         triallist = [line.rstrip('_m.png\n') for line in f]
#         current_task = triallist[0]#[:triallist[0].find('_')]
#         cm_target_ = [trl[-2] for trl in triallist[1:]]
#         cm_target = [cm_to_num[trgt] for trgt in cm_target_]
#         df_target = [int(trl[-1])-1 for trl in triallist[1:]] # -1 to get target IDs from 0:4
#
#         for ii,trl in enumerate(triallist):
#
#             if ii == 0:
#                 trialDict_correct[trl_cntr_correct] = trl ### add the condition
#                 trl_cntr_correct += 1
#                 trialDict_cm[trl_cntr_other] = trl
#                 trialDict_df[trl_cntr_other] = trl
#                 trl_cntr_other += 1
#
#             if current_task[:12] == 'colormapping' and ii > 0:
#                 trialDict_correct[trl_cntr_correct] = {}
#                 trialDict_correct[trl_cntr_correct]['ID'] = cm_target[ii-1]
# #                 print(trialDict_correct[trl_cntr_correct]['ID'])
#                 trialDict_correct[trl_cntr_correct]['c'] = current_task
#                 trl_cntr_correct += 1
#
#             if current_task[:18] == 'directionfollowing' and ii > 0:
#                 trialDict_correct[trl_cntr_correct] = {}
#                 trialDict_correct[trl_cntr_correct]['ID'] = df_target[ii-1]
# #                 print(trialDict_correct[trl_cntr_correct]['ID'])
#                 trialDict_correct[trl_cntr_correct]['c'] = current_task
#                 trl_cntr_correct += 1
#
#             if ii > 0:
#                 trialDict_cm[trl_cntr_other] = cm_target[ii-1]
#                 trialDict_df[trl_cntr_other] = df_target[ii-1]
#                 trl_cntr_other += 1
#
#
#
#     ### store the correct target ID in the dictionary
#     for trl in identified_targets_dict[subj][run]:
#         identified_targets_dict[subj][run][trl]['condition'] = trialDict_correct[trl]['c']
#         identified_targets_dict[subj][run][trl]['correct_target'] = trialDict_correct[trl]['ID']
#         identified_targets_dict[subj][run][trl]['cm_target'] = trialDict_cm[trl]
#         identified_targets_dict[subj][run][trl]['df_target'] = trialDict_df[trl]
#
#     return identified_targets_dict
#
#
#
# def colors(n):
#     ret = []
#     r = int(random.random() * 256)
#     g = int(random.random() * 256)
#     b = int(random.random() * 256)
#     step = 256 / n
#     for i in range(n):
#         r += step
#         g += step
#         b += step
#         r = int(r) % 256
#         g = int(g) % 256
#         b = int(b) % 256
#         ret.append((r/255,g/255,b/255))
#     return ret
