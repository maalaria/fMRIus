%% Use MarsBaR to make spherical ROIs


%% Set general options

outDir = '/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/gm_analysis/src/ROIs/TPJ';
sphereRadius = 5; % mm


% coordinates are nvoxels rows by 3 columns for X,Y,Z
coords = csvread('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/gm_analysis/src/ROIs/TPJ/dat/tpj_coords_combined_selected-minD-4_MNI.csv');

% extract roi_space from image
roi_space = maroi('classdata', 'spacebase')
mask_info = niftiinfo('/media/marius/data_ex/gaze-motion/derivatives/fmriprep/sub-00/func/sub-00_space-MNI152NLin2009cAsym_desc-brain_mean-mask.nii')
roi_space.dim = mask_info.ImageSize
roi_space.mat = mask_info.Transform.T'

% (alternatively, or better, you could put these in a text file and read
% them in using the dlmread function)



%% Error checking: directory exists, MarsBaR is in path
if ~isdir(outDir)
    mkdir(outDir);
end

if ~exist('marsbar')
    error('MarsBaR is not installed or not in your matlab path.');
end


%% Make rois
fprintf('\n');

for i=1:size(coords,1)
    thisCoord = coords(i,:);
    
    fprintf('Working on ROI %d/%d...', i, size(coords,1));
    
    roiLabel = sprintf('%i-%i-%i', thisCoord(1), thisCoord(2), thisCoord(3));
    
    sphereROI = maroi_sphere(struct('centre', thisCoord, 'radius', sphereRadius));
    
    outName = fullfile(outDir, sprintf('GengVossel2013-TPJ_%s_%dmm-sphere_right', roiLabel, sphereRadius));
    
    % save MarsBaR ROI (.mat) file
    %    saveroi(sphereROI, [outName '.mat']);
    
    % save the Nifti (.nii) file
    mars_rois2img(sphereROI, [outName '.nii'], roi_space)
    
    fprintf('done.\n');
    
end


fprintf('\nAll done. %d ROIs written to %s.')