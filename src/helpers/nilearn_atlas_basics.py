import mne
import os.path as op
import numpy as np
import nibabel as nib
from nibabel.spatialimages import SpatialImage
from atlas_utils import get_atlas_volume_labels

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'fsaverage'

vsrc_fpath = op.join(subjects_dir, subject, subject + '_vol-5.00-src.fif')
vsrc = mne.read_source_spaces(vsrc_fpath)

methods = ['aal', 'allen', 'basc', 'craddock', 'destrieux',
           'harvard_oxford', 'msdl', 'pauli', 'smith',
           'talairach', 'yeo']

method = methods[10]

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


labels = get_atlas_volume_labels(subject, subjects_dir, vsrc, atlas_img=atlas_img,
                                 value_to_label_dict=value_to_label_dict, min_dist=5)

