import mne
import os.path as op
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # unused but required for 3d scatter plot
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.spatialimages import SpatialImage
from numpy.linalg import norm
from tqdm import tqdm
from atlas_utils import mri_to_mni, inds_to_coords

###############################################################################
# settings
###############################################################################

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'fsaverage'
subj_grid_spacing = 5

methods = ['aal', 'allen', 'basc', 'craddock', 'destrieux',
           'harvard_oxford', 'msdl', 'pauli', 'smith',
           'talairach', 'yeo']

method = methods[0]

print('Using %s atlas' % method)

###############################################################################
# load atlas
###############################################################################

if method == 'aal':

    ###############################################################################
    # AAL atlas
    # Everything works out of the box, no adjustment to labels necessary
    ###############################################################################

    from nilearn.datasets import fetch_atlas_aal

    atlas = fetch_atlas_aal(data_dir=data_path)
    maps = 'maps'
    atlas_img = nib.load(atlas[maps])

    value_to_label_dict = dict()
    for idx, label in enumerate(atlas['labels']):
        value_to_label_dict[int(atlas['indices'][idx])] = label

    value_to_label_dict['0'] = 'Background'

elif method == 'allen':

    ###############################################################################
    # Allen 2011 atlas
    # Atlas constructed using 75 ICA components but only 28 of those are named
    # All unidentified components will be labelled by their component number.
    #
    # Since value 0 in data array is a valid label (background labels have value
    # 0 everywhere) we have to shift all label indices up by 1 and set background
    # voxels to 0.
    ###############################################################################

    from nilearn.datasets import fetch_atlas_allen_2011

    atlas = fetch_atlas_allen_2011(data_dir=data_path)
    maps = 'maps'
    img = nib.load(atlas[maps])
    img_data = img.get_fdata()

    value_to_label_dict = dict()
    for label in atlas['rsn_indices']:
        for val in label[1]:
            value_to_label_dict[val + 1] = label[0]

    for val in range(img_data.shape[3]):
        if (val + 1) not in value_to_label_dict.keys():
            value_to_label_dict[val + 1] = 'Component ' + str(val)

    value_to_label_dict[0] = 'Background'

    # allen 2011 is an atlas for 75 ICA components
    # use for each voxel the index of the component with maximum value as label

    # find voxels where all values are 0 (background)
    atlas_data_max = img_data.max(axis=3)
    indices_unassigned = np.where(atlas_data_max == 0)

    # get component with maximum value
    atlas_data = img_data.argmax(axis=3)

    # shift components up by 1 and then set background voxels back to 0
    atlas_data = atlas_data + 1
    atlas_data[indices_unassigned] = 0

    atlas_img = SpatialImage(atlas_data, img.affine)

elif method == 'basc':

    ###############################################################################
    # BASC multiscale 2015 atlas
    # Background is properly labelled as 0 already.
    # Problem:
    # Does not have named labels, just network numbers
    ###############################################################################

    from nilearn.datasets import fetch_atlas_basc_multiscale_2015
    atlas = fetch_atlas_basc_multiscale_2015(data_dir=data_path)
    maps = 'scale064'
    atlas_img = nib.load(atlas[maps])

    value_to_label_dict = dict()
    for val in range(1, int(maps[-3:]) + 1):
        value_to_label_dict[val] = 'Network ' + str(val)

    value_to_label_dict[0] = 'Background'

elif method == 'craddock':

    ###############################################################################
    # Craddock 2012 atlas
    # Problem:
    # How to interpret atlas data of shape (47, 56, 46, 43)?
    # Seems to be probabilities again but not really sure
    # Data points relatively sparse
    ###############################################################################

    from nilearn.datasets import fetch_atlas_craddock_2012
    atlas = fetch_atlas_craddock_2012(data_dir=data_path)
    maps = 'scorr_mean'
    img = nib.load(atlas[maps])
    img_data = img.get_fdata()

    value_to_label_dict = dict()
    for val in range(img_data.shape[3]):
        value_to_label_dict[val + 1] = 'Network ' + str(val)

    value_to_label_dict['0'] = 'Background'

    # find voxels where all probabilities are 0 (background)
    atlas_data_max = img_data.max(axis=3)
    indices_unassigned = np.where(atlas_data_max == 0)

    # get component with maximum probability
    atlas_data = img_data.argmax(axis=3)

    # shift components up by 1 and then set background voxels back to 0
    atlas_data = atlas_data + 1
    atlas_data[indices_unassigned] = 0

    atlas_img = SpatialImage(atlas_data, img.affine)

elif method == 'destrieux':

    ###############################################################################
    # Destrieux 2009 atlas
    # Everything works out of the box, no adjustment to labels necessary
    # Does not have a cerebellum but otherwise fine
    ###############################################################################

    from nilearn.datasets import fetch_atlas_destrieux_2009
    atlas = fetch_atlas_destrieux_2009(data_dir=data_path)
    maps = 'maps'
    atlas_img = nib.load(atlas[maps])

    value_to_label_dict = dict()
    for label in atlas['labels']:
        value_to_label_dict[int(label[0])] = label[1]

elif method == 'harvard_oxford':

    ###############################################################################
    # Harvard Oxford atlas
    # Everything works out of the box, no adjustment to labels necessary
    # Does not have a cerebellum but otherwise fine
    ###############################################################################

    from nilearn.datasets import fetch_atlas_harvard_oxford
    atlas = fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr0-1mm', data_dir=data_path)
    maps = 'maps'
    atlas_img = nib.load(atlas[maps])

    value_to_label_dict = dict()
    for val, label in enumerate(atlas['labels']):
        value_to_label_dict[val] = label

elif method == 'msdl':

    ###############################################################################
    # MSDL atlas
    # Atlas data has shape (40, 48, 35, 39) with the last column being the
    # probabilities for each label. Use label with highest probablility for each
    # voxel.
    # Since value 0 in data array is a valid label (background labels have probability
    # 0 everywhere) we have to shift all label indices up by 1 and set background
    # voxels to 0.
    # Problem:
    # Does not work too well since the atlas data seems very sparse -> long distances
    ###############################################################################

    from nilearn.datasets import fetch_atlas_msdl
    atlas = fetch_atlas_msdl(data_dir=data_path)
    maps = 'maps'
    img = nib.load(atlas[maps])
    img_data = img.get_fdata()

    value_to_label_dict = dict()
    for val, label in enumerate(atlas['labels']):
        value_to_label_dict[val + 1] = label

    value_to_label_dict[0] = 'Background'

    # MSDL data has shape (40, 48, 35, 39) with the last column being the probabilities for each label
    # use for each voxel the label with highest probablity

    # find voxels where all probabilities are 0 (background)
    atlas_data_max = img_data.max(axis=3)
    indices_unassigned = np.where(atlas_data_max == 0)

    # get component with maximum probability
    atlas_data = img_data.argmax(axis=3)

    # shift components up by 1 and then set background voxels back to 0
    atlas_data = atlas_data + 1
    atlas_data[indices_unassigned] = 0

    atlas_img = SpatialImage(atlas_data, img.affine)

elif method == 'pauli':

    ###############################################################################
    # Pauli 2017 atlas
    # Atlas data has (193, 229, 193, 16) with the last column being the
    # probabilities for each label. Use label with highest probablility for each
    # voxel.
    # Since value 0 in data array is a valid label (background labels have probability
    # 0 everywhere) we have to shift all label indices up by 1 and set background
    # voxels to 0.
    # Problem:
    # There does not seem to be any overlap at all with fsaverage in MNI space
    ###############################################################################

    from nilearn.datasets import fetch_atlas_pauli_2017
    atlas = fetch_atlas_pauli_2017(data_dir=data_path)
    maps = 'maps'
    img = nib.load(atlas[maps])
    img_data = img.get_fdata()

    value_to_label_dict = dict()
    for val, label in enumerate(atlas['labels']):
        value_to_label_dict[val + 1] = label

    value_to_label_dict[0] = 'Background'

    #  data has shape (193, 229, 193, 16) with the last column being the probabilities for each label
    # use for each voxel the label with highest probablity

    # find voxels where all probabilities are 0 (background)
    atlas_data_max = img_data.max(axis=3)
    indices_unassigned = np.where(atlas_data_max == 0)

    # get component with maximum probability
    atlas_data = img_data.argmax(axis=3)

    # shift components up by 1 and then set background voxels back to 0
    atlas_data = atlas_data + 1
    atlas_data[indices_unassigned] = 0

    atlas_img = SpatialImage(atlas_data, img.affine)

elif method == 'smith':

    ###############################################################################
    # Smith 2009 atlas
    # Atlas constructed  ICA decomposition. Labels are just the component numbers.
    #
    # Since value 0 in data array is a valid label (background labels have values
    # 0 everywhere) we have to shift all label indices up by 1 and set background
    # voxels to 0.
    # Does not have a cerebellum but otherwise fine
    ###############################################################################

    from nilearn.datasets import fetch_atlas_smith_2009
    atlas = fetch_atlas_smith_2009(data_dir=data_path)
    maps = 'rsn20'
    img = nib.load(atlas[maps])
    img_data = img.get_fdata()

    value_to_label_dict = dict()
    for val in range(img_data.shape[3]):
        value_to_label_dict[val + 1] = 'Component ' + str(val)

    value_to_label_dict[0] = 'Background'

    # find voxels where all values are 0 (background)
    atlas_data_max = img_data.max(axis=3)
    indices_unassigned = np.where(atlas_data_max == 0)

    # get component with maximum value
    atlas_data = img_data.argmax(axis=3)

    # shift components up by 1 and then set background voxels back to 0
    atlas_data = atlas_data + 1
    atlas_data[indices_unassigned] = 0

    atlas_img = SpatialImage(atlas_data, img.affine)

elif method == 'talairach':

    ###############################################################################
    # Talairach atlas
    # Everything works out of the box, no adjustment to labels necessary
    # Does not have a cerebellum but otherwise fine
    ###############################################################################

    from nilearn.datasets import fetch_atlas_talairach
    atlas = fetch_atlas_talairach(level_name='ba', data_dir=data_path)
    maps = 'maps'
    atlas_img = atlas[maps]

    value_to_label_dict = dict()
    for idx, label in enumerate(atlas['labels']):
        value_to_label_dict[idx] = label

elif method == 'yeo':

    ###############################################################################
    # Yeo 2011 atlas
    # Everything works out of the box, no adjustment to labels necessary
    # Does not have a cerebellum but otherwise fine
    ###############################################################################

    from nilearn.datasets import fetch_atlas_yeo_2011
    atlas = fetch_atlas_yeo_2011(data_dir=data_path)
    maps = 'thick_17'
    img = nib.load(atlas[maps])
    img_data = img.get_fdata()
    atlas_data = np.squeeze(img_data)

    value_to_label_dict = dict()
    for val in range(len(np.unique(atlas_data))):
        value_to_label_dict[val] = str(val)

    atlas_img = SpatialImage(atlas_data, img.affine)

else:
    raise ValueError("Method %s is unknown." % method)

###############################################################################
# get labels for mni_coordinates
###############################################################################

atlas_data = atlas_img.get_fdata()
affine = atlas_img.affine

mni_inuse = np.where(atlas_data > 0)
# get for mni_inuse the label indices
mni_label_inds = atlas_data[mni_inuse]


###############################################################################
# transform subject source space to MNI space
###############################################################################

src_fpath = op.join(subjects_dir, subject, subject + '_vol-%.2f-src.fif' % subj_grid_spacing)
src = mne.read_source_spaces(src_fpath)

rr = src[0]['rr']

rr_inuse = rr[np.where(src[0]['inuse'] > 0)]

rr_inuse_mni = mri_to_mni(rr_inuse, subject=subject,
                          subjects_dir=subjects_dir)

###############################################################################
# transform indices for aal_img data to MNI coordinates
###############################################################################

xmax, ymax, zmax = atlas_data.shape

coords_shape = list(atlas_data.shape)
coords_shape.append(3)
atlas_coords = np.empty(coords_shape)

for idx_x in range(xmax):
    for idx_y in range(ymax):
        for idx_z in range(zmax):

            atlas_coords[idx_x, idx_y, idx_z] = inds_to_coords(idx_x, idx_y, idx_z, affine)

# plot only coords inuse in scatterplot
atlas_coords_inuse = atlas_coords[mni_inuse]

###############################################################################
# check if inds_to_coords and mni_pos live in the same space
# -> plot scatterplot coords in  mni_pos and coords from inds_to_coords
###############################################################################

print("Check if subject brain and MNI brain overlap")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xa = atlas_coords_inuse.T[0][::121]
ya = atlas_coords_inuse.T[1][::121]
za = atlas_coords_inuse.T[2][::121]

xm = rr_inuse_mni.T[0][::6]
ym = rr_inuse_mni.T[1][::6]
zm = rr_inuse_mni.T[2][::6]

ax.scatter(xa, ya, za, c='r', marker='o', alpha=0.3)
ax.scatter(xm, ym, zm, c='b', marker='x', alpha=0.3)

plt.show()

###############################################################################
# assign labels to the voxels of the subject brain
###############################################################################

# get for each mni_pos the coords_inuse that are closest
# get the index of coords_inuse_closest
# use value_to_label_dict to get label name
# create list where each voxel has a label

# flatten all but the last dimension in coords
atlas_data_fl = atlas_data.flatten()
atlas_coords_fl = atlas_coords.reshape(-1, atlas_coords.shape[-1])

atlas_data_inuse = atlas_data[mni_inuse]
# next nearest neighbor
min_dist_nnn = np.sqrt(2) * subj_grid_spacing

subj_label_list = []
for pos in tqdm(rr_inuse_mni):

    dist = norm(atlas_coords_inuse - pos, axis=1)

    idx = dist.argmin()

    # use sqrt(2) * 5 for next nearest neighbors since the subject grid spacing is 5 mm
    if dist[idx] < min_dist_nnn:
        subj_label_list.append(value_to_label_dict[int(atlas_data_inuse[idx])])

    else:
        subj_label_list.append('unassigned')

subj_labels = np.array(subj_label_list)

print('Number of "unassigned" labels:', np.where(subj_labels == 'unassigned')[0].shape)

###############################################################################
# Plot the "unassigned" volume label for the three versions
###############################################################################

target_label = 'unassigned'
# target_label = aal['labels'][0]

print('Plotting "unassigned" labels')

# voxels in target label
tl_inds = np.where(subj_labels == target_label)[0]
# rest of the voxels
rl_inds = np.where(subj_labels != target_label)[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xt = rr_inuse_mni[tl_inds].T[0]
yt = rr_inuse_mni[tl_inds].T[1]
zt = rr_inuse_mni[tl_inds].T[2]

xr = rr_inuse_mni[rl_inds].T[0][::9]
yr = rr_inuse_mni[rl_inds].T[1][::9]
zr = rr_inuse_mni[rl_inds].T[2][::9]

ax.scatter(xt, yt, zt, c='r', marker='x', alpha=0.3)
ax.scatter(xr, yr, zr, c='b', marker='o', alpha=0.3)

plt.show()
