#!/bin/bash

#
# sub-00
#
# MTloc
#
in_file1='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-1_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'
in_file2='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-2_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'
in_file3='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-3_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'
in_file4='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-4_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'
in_file5='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-5_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'
in_file6='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-6_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'
in_file7='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-7_space-MNI152NLin2009cAsym_desc-preprocCleaned_bold.nii.gz'


design1='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-1_events.txt'
design2='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-2_events.txt'
design3='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-3_events.txt'
design4='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-4_events.txt'
design5='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-5_events.txt'
design6='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-6_events.txt'
design7='/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-7_events.txt'

# MotionStatic run-1
design_out='/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/sub-00_task-MTloc_MotionStatic_design.txt'
out_file_v='/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/sub-00_task-MTloc_MotionStatic.v'
out_file_nii='/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/sub-00_task-MTloc_MotionStatic.nii'
vlisa_prewhitening  -in $in_file1 $in_file2 $in_file3 $in_file4 $in_file5 $in_file6 $in_file7 \
                    -design $design1 $design2 $design3 $design4 $design5 $design6 $design7 \
                    -hemo gamma_1 \
                    -col1 true \
                    -contrast 0 1 0 -1 0 \
                    -perm 1000 \
                    -alpha 1.0 \
                    -out $out_file_v \
                    -plotdesign $design_out
vnifti -in $out_file_v -out $out_file_nii





# StaticMotion
# design_out='/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/sub-00_task-MTloc_StaticMotion_design.txt'
# out_file_v='/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/sub-00_task-MTloc_StaticMotion.v'
# out_file_nii='/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_lisa/sub-00/sub-00_task-MTloc_StaticMotion.nii'
# vlisa_prewhitening  -in $in_file1 $in_file2 $in_file3 $in_file4 $in_file5 $in_file6 \
#                     -design $design1 $design2 $design3 $design4 $design5 $design6 \
#                     -col1 true \
#                     -contrast 0 -1 1 \
#                     -perm 1000 \
#                     -alpha 1.0 \
#                     -out $out_file_v \
#                     -plotdesign $design_out
# vnifti -in $out_file_v -out $out_file_nii