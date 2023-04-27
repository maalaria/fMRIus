#!/bin/bash

#
# sub-00
#
# MTloc
#

in_path=


vlisa_onesample -in /home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_nilearn/univariate_contrasts/sub-00/sub-00_task-MTloc_run-*_MotionStatic_fwhm-None_z_score.nii \
                -out /home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/2ndLevel/sub-00_task-MTloc_MotionStatic.v \
                -mask /media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_space-MNI152NLin2009cAsym_desc-brain_mean-mask.nii \
                -alpha 1