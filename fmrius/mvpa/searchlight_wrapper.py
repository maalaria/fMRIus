import sys
import numpy as np
import pandas as pd
from os.path import join as opj
import _pickle as cPickle

from nilearn import decoding
from sklearn.model_selection import LeaveOneGroupOut, KFold

from nilearn import plotting
from nilearn.glm import threshold_stats_img
from nilearn import image
from nilearn.image import index_img, new_img_like, load_img, get_data

sys.path.append( sys.path.append('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/helpers') )
sys.path.append( sys.path.append('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/modeling/nilearn') )
sys.path.append( sys.path.append('/home/marius/ownCloud/development/python/atlasreader/atlasreader') )

import subject
import first_level
import info
import helpers as myHelpers





#
# init new dict if not existing
# init new dict if not existing
if 'subjects' not in locals():
    subjects = {}

sub_id_list = ['sub-00']
session = ''

for sub_id in sub_id_list:
    subjects[sub_id] = subject.Subject(sub_id, session, info.eyes_cubes_dirs, fwhm_of_cleaned='')


print('*** subjects ', subjects.keys(), 'loaded')



sub_id = 'sub-00'
runs = ['ec_run-1', 'ec_run-2', 'ec_run-3', 'ec_run-4', 'ec_run-5', 'ec_run-6']

events_list = []
func_img_list = []
for rr in runs:
    events_list.append(pd.read_csv(subjects[sub_id].info[rr]['events'], sep='\t'))
    func_img_list.append(image.load_img(subjects[sub_id].info[rr]['func_cleaned']))

func_img_concat = image.concat_imgs(func_img_list)

print('*** functional images prepared')




behavioral_concat = []
chunks_concat = []

for ii, run in enumerate(events_list):

    img_times = np.multiply(np.arange(0, func_img_list[ii].shape[-1], 1), 1.5)
    behavioral = np.array([None]*len(img_times))

    starts = list(events_list[ii]['onset'])[:-1]
    ends = list(events_list[ii]['onset'])[1:]

    trial_periods_list = [(sel, ends[ii]) for ii,sel in enumerate(starts)]
    trial_periods_list.append( (ends[-1], img_times[-1]+0.01) )


    for jj, t_period in enumerate(trial_periods_list):

        cond = np.array(events_list[ii].iloc[jj]['trial_type'])
        img_ids = np.sort(list(set(np.where(img_times > t_period[0])[0]) & set(np.where(img_times < t_period[1])[0])))

        #
        # select dimensionality

        # using all images of a trial
#         np.put(behavioral, img_ids, cond)

        # using only the x-th image of each trial (i.e. one value per trial only, consider PCA over time for data reduction?)
        try:
            np.put(behavioral, img_ids[3], cond) # 3rd image <=> 4.5s
        except:
            pass

    ### make sure everything is string
    behavioral = behavioral.astype(str)


    ### REPLACE ACCORDING TO INTENDED DECODING
    ### Eyes vs Cubes
    behavioral = np.char.replace(behavioral, "Eyes_1", 1)
    behavioral = np.char.replace(behavioral, "Eyes_2", 1)
    behavioral = np.char.replace(behavioral, "Eyes_3", 1)
    behavioral = np.char.replace(behavioral, "Cubes_1_V", 2)
    behavioral = np.char.replace(behavioral, "Cubes_2_V", 2)
    behavioral = np.char.replace(behavioral, "Cubes_3_V", 2)
    behavioral = np.char.replace(behavioral, "Cubes_1_H", 2)
    behavioral = np.char.replace(behavioral, "Cubes_2_H", 2)
    behavioral = np.char.replace(behavioral, "Cubes_3_H", 2)


    # behavioral = np.char.replace(behavioral, "Cubes_1_v", "Cubes_1")
    # behavioral = np.char.replace(behavioral, "Cubes_2_v", "Cubes_2")
    # behavioral = np.char.replace(behavioral, "Cubes_3_v", "Cubes_3")
    # behavioral = np.char.replace(behavioral, "Cubes_1_h", "Cubes_1")
    # behavioral = np.char.replace(behavioral, "Cubes_2_h", "Cubes_2")
    # behavioral = np.char.replace(behavioral, "Cubes_3_h", "Cubes_3")


    behavioral_concat.extend(behavioral)
    chunks_concat.extend([ii]*len(behavioral)) # array with session labels

behavioral = pd.DataFrame( {'label':behavioral_concat, 'chunk':chunks_concat } )
conditions = behavioral['label']

condition_mask = behavioral['label'].isin([1, 2])

conditions = conditions[condition_mask]
func_img_indexed = index_img(func_img_concat, condition_mask)
session_label = behavioral['chunk'][condition_mask]


print('*** dataframes prepared')





#
# prepare masks

mask_img = '/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_space-MNI152NLin2009cAsym_desc-brain_mean-mask.nii.gz'

process_mask = get_data(mask_img).astype(int)
picked_slice = 25
process_mask[..., (picked_slice + 17):] = 0
process_mask[..., :picked_slice] = 0
process_mask[:, 35:] = 0 # select transversal section
process_mask_img = new_img_like(mask_img, process_mask)





#
# init s
cv = KFold(n_splits=4) # LeaveOneGroupOut(), KFold(n_splits=4)

searchlight = decoding.SearchLight(
    mask_img=mask_img,
    process_mask_img=process_mask_img, # process_mask_img, None
    estimator='svr',
    radius=4,
    n_jobs=1,
    verbose=1,
    cv=cv)

print('*** searchlight initialized, fitting model...')





#
# fit
searchlight.fit(func_img_indexed, np.array(conditions))#, groups=np.array(session_label))







#
#
path_ = opj(info.eyes_cubes_dirs.results_dir,
                    'from_nilearn',
                    'searchlight',
                    subjects['sub-00'].ID,
                    'whole-brain_radius-4_4Fold'+'.pkl')
with open(path_, 'wb') as ff:
                cPickle.dump(searchlight,
                             ff)


print('*** results stored: ', path_)
