import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


def mri_to_mni(coo_mri_ras, subject, subjects_dir, ignore_warning=True):
    """
    Based on mne.source_space.head_to_mni

    Transform coordinates in MRI RAS space to MNI space.

    Parameters
    ----------
    coo_mri_ras : np.array of shape (n, 3)
    subject : str
        Subject ID.
    subjects_dir : str
        Path to the subject directory

    Returns
    -------
    coo_mni : np.array of shape (n, 3)
        Transformed data.
    """

    from mne.utils import get_subjects_dir
    from mne.source_space import _read_talxfm
    from mne.transforms import apply_trans

    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    # convert to MNI coordinates
    xfm = _read_talxfm(subject, subjects_dir)

    coo_mni = apply_trans(xfm['trans'], coo_mri_ras * 1000)

    # these are floats close to integers -> convert to integers
    coo_mni_r = np.rint(coo_mni)

    if len(np.where(np.abs(coo_mni - coo_mni_r) > 1e-3)[0]) > 0 and not ignore_warning:
        raise Warning("Transformed coordinates are not close to integers.")
    else:
        coo_mni = np.array(coo_mni_r, dtype=int)

    return coo_mni


def inds_to_coords(i, j, k, affine):
    """
    Return X, Y, Z coordinates for i, j, k

    Parameters
    ----------
    i, j, k : int
        indices for aal_img.get_fdata()
    affine : np.array of shape (4, 4)
        The affine transformation from indices to coordinates
        given by nibabel.nifti1.Nifti1Image.affine

    Returns
    -------
    coords : np.array of shape (3, )
        The indices i, j, k transformed to coordinates in mm.

    References
    ----------
    https://nipy.org/nibabel/coordinate_systems.html#applying-the-affine

    """

    M = affine[:3, :3]
    abc = affine[:3, 3]

    return M.dot([i, j, k]) + abc


def get_atlas_volume_labels(subject, subjects_dir, vsrc, atlas_img, value_to_label_dict, min_dist=5):
    """
    Get the atlas labels for the voxels in the volume source space.

    Parameters
    ----------
    subject : str
        Subject ID
    subjects_dir : str
        Path to the subjects directory:
    vsrc : mne.SourceSpaces
        Volume source space for the subject in MRI space NOT
        head space.
    atlas_img : nibabel.spatialimages.SpatialImage
        Contains the atlas data and the affine transformation.
    value_to_label_dict : dict
        Dictionary to translate the value at each voxel to a label.
    min_dist : float
        Minimum distance in mm between a source space voxel in MNI
        space and its nearest neighbor voxel from atlas space for
        them to be considered at the same position.

    Returns
    -------
    subj_labels : np.array of shape (n_inuse,)
        Returns a label for each voxel inuse in vsrc.
    """

    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    ###############################################################################
    # get voxels inuse
    ###############################################################################

    mni_inuse = np.where(atlas_data > 0)

    ###############################################################################
    # transform subject source space to MNI space
    ###############################################################################

    rr = vsrc[0]['rr']
    rr_inuse = rr[np.where(vsrc[0]['inuse'] > 0)]

    rr_inuse_mni = mri_to_mni(rr_inuse, subject=subject,
                              subjects_dir=subjects_dir)

    ###############################################################################
    # transform indices for atlas_img data to MNI coordinates
    ###############################################################################

    coords_shape = list(atlas_data.shape)
    coords_shape.append(3)
    atlas_coords = np.empty(coords_shape)

    for idx_x in range(atlas_data.shape[0]):
        for idx_y in range(atlas_data.shape[1]):
            for idx_z in range(atlas_data.shape[2]):

                atlas_coords[idx_x, idx_y, idx_z] = inds_to_coords(idx_x, idx_y,
                                                                   idx_z, affine)

    # plot only coords inuse in scatterplot
    atlas_coords_inuse = atlas_coords[mni_inuse]

    ###############################################################################
    # assign labels to the voxels of the subject brain
    ###############################################################################

    # get for each mni_pos the coords_inuse that are closest
    # get the index of coords_inuse_closest
    # use ind_to_label_dict to get label name
    # create list where each voxel has a label

    # take only those voxels which are part of the MNI brain
    atlas_data_inuse = atlas_data[mni_inuse]
    subj_label_list = []

    for pos in tqdm(rr_inuse_mni):

        dist = norm(atlas_coords_inuse - pos, axis=1)

        idx = dist.argmin()

        if dist[idx] < min_dist:
            subj_label_list.append(value_to_label_dict[int(atlas_data_inuse[idx])])

        else:
            subj_label_list.append('unassigned')

    subj_labels = np.array(subj_label_list)

    return subj_labels
