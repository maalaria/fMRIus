#!/bin/bash

surf_root=/home/marius/freesurfer/subjects/fsaverage/surf
root_dir=/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/results/from_nilearn/univariate_contrasts


subs=( sub-00 )

# in_files=(  sub-00_task-MTloc_MotionStatic_fwhm-None_p05-fdr
#             sub-00_task-MTloc_MotionStatic_fwhm-5_p05-fdr
#             sub-00_task-ec_EyesCubes_fwhm-None_p05-fdr
#             sub-00_task-ec_EyesCubes_fwhm-5_p05-fdr)



for sub_id in "${subs[@]}"; do

  echo
  echo "###########"
  echo $sub_id

  for in_file in $root_dir/$sub_id/*.nii; do
    echo $in_file
    
    for surf_type in pial; do

      for hemi in lh rh; do

        wb_command -volume-to-surface-mapping $"${in_file%.*}".nii $surf_root/$hemi.$surf_type.surf.gii $"${in_file%.*}".$hemi.$surf_type.func.gii -trilinear

      done

    done

  done

done


# mris_convert /home/marius/freesurfer/subjects/sub-00/surf/lh.pial_semi_inflated /home/marius/freesurfer/subjects/fsaverage/surf/lh.pial_semi_inflated.surf.gii
# mris_convert /home/marius/freesurfer/subjects/sub-00/surf/rh.pial_semi_inflated /home/marius/freesurfer/subjects/fsaverage/surf/rh.pial_semi_inflated.surf.gii