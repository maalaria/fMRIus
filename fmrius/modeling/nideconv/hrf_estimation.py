import sys

sys.path.append('/home/marius/ownCloud/development/python/nideconv')

import nideconv
from nideconv import simulate
from nideconv.utils import double_gamma_with_d
from nideconv.nifti import NiftiResponseFitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nilearn import image
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker

sys.path.append( sys.path.append('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/helpers') )
sys.path.append( sys.path.append('/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/analysis/modeling/nilearn') )
sys.path.append( sys.path.append('/home/marius/ownCloud/development/python/atlasreader/atlasreader') )

import subject
import info


def estimate_hrf_wrapper(subject, ROIs, ):
