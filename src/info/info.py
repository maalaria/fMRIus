import os
from os.path import join as opj
import pandas as pd
from sklearn.utils import Bunch


colors = Bunch(
    Iris='#40acb0',
    Iris123='#fdae61',
    IrisXEyes_Cubes='#fdae61',
    Iris_1='#f28418',
    Iris_2='#f0a459',
    Iris_3='#f5c493',
    Iris1='#f28418',
    Iris2='#f0a459',
    Iris3='#f5c493',
    # Eyes='#d7191c',
    # Eyes1='#d7191c',
    # Eyes2='#d46365',
    # Eyes3='#eda8aa',
    # Eyes_1='#d7191c',
    # Eyes_2='#d46365',
    # Eyes_3='#eda8aa',
    Gaze='#33633d',
    Gaze1='#d7191c',
    Gaze2='#d46365',
    Gazes3='#eda8aa',
    Gaze_1='#d7191c',
    Gaze_2='#d46365',
    Gaze_3='#eda8aa',
    Cubes='#ad3fad',
    Cubes123='#2c7bb6',
    CubesXEyes_Iris='#2c7bb6',
    Cubes1='#1872b5',
    Cubes2='#4c88b5',
    Cubes3='#89b6d9',
    Cubes_1='#1872b5',
    Cubes_2='#4c88b5',
    Cubes_3='#89b6d9',
    Motion='#7fbf7b',
    Static='#67a9cf',
    MTloc='#3AEAB3',  # '#ffffbf',CFF64E
    MotionStatic='#3AEAB3',
    MTlocRandom='#3AEAB3',
    MTlocRotation='#3ac9ea',
    MTlocCubes='#3aea5b',
    Any='k',
    Marquardt2017='#F049F1',  # '#ff69b4',
    EyesCubes='#d7191c',
    Eyes1Cubes1='#d7191c',
    Eyes1Cubes123='#d7191c',
    EyesXCubes='#921154',
    Eyes1Eyes3='#d7191c',
    Cubes1Cubes3='#2c7bb6',
    Iris1Iris3='#fdae61',
    Eyes1Iris123='#d7191c',
    EyesXIris='#e952eb',
    Eyes1Iris1='#e952eb',
    Cubes1Iris123='#2cb6ac',
    Cubes1Iris1='#2cb6ac',
    CubesIris='#2cb6ac',
    CubesXIris='#2cb6ac',
    VisualMotion='#ffffff'
)

data_paths = Bunch(
    root_dir='/media/marius/data_ex/gaze-motion',
    bids_dir='/media/marius/data_ex/gaze-motion/dsgm',
    qc_dir='/media/marius/data_ex/gaze-motion/mriqc',
    derivatives_dir='/media/marius/data_ex/gaze-motion/derivatives',
    fmriprep_dir='/media/marius/data_ex/gaze-motion/derivatives/fmriprep',
    lisa_dir="/media/marius/data_ex/gaze-motion/derivatives/lisa",
    scratch_dir='/media/marius/data_ex/gaze-motion/derivatives/scratch',
    results_dir='/mnt/fileserv1_neurologie/Research_groups/mgoerner/projects/scientific/gaze-motion/results',
    rois_dir='/mnt/fileserv1_neurologie/Research_groups/mgoerner/projects/scientific/gaze-motion/analysis/fMRIus/src/ROIs',
)


### specify existing runs in /fmriprep/[sub]/func/ in the order of the experiment

func_run_dict = {
    'sub-00': {
        'ses-gc': {
            'MTloc': ['MTloc_run-1', 'MTloc_run-2', 'MTloc_run-3', 'MTloc_run-4', 'MTloc_run-5', 'MTloc_run-6',
                      'MTloc_run-7'],
            'gc': ['gc_run-1', 'gc_run-2', 'gc_run-3', 'gc_run-4', 'gc_run-5', 'gc_run-6'],
            'experimental_order': ['MTloc_run-1', 'MTloc_run-2', 'MTloc_run-3', 'MTloc_run-4', 'MTloc_run-5',
                                   'MTloc_run-6', 'MTloc_run-7', 'gc_run-1', 'gc_run-2', 'gc_run-3',
                                   'gc_run-4', 'gc_run-5', 'gc_run-6']
        },
        'ses-com': {
            'MTloc': ['MTloc_run-1', 'MTloc_run-2', 'MTloc_run-3', 'MTloc_run-4', 'MTloc_run-5', 'MTloc_run-6'],
            'gic': ['gic_run-1', 'gic_run-2', 'gic_run-3'],
            'co': ['co_run-1', 'co_run-2', 'co_run-3'],
            'experimental_order': ['MTloc_run-1', 'co_run-1', 'MTloc_run-2', 'co_run-2', 'co_run-3', 'MTloc_run-3',
                                   'gic_run-1', 'gic_run-2', 'MTloc_run-4', 'MTloc_run-5', 'gic_run-3', 'MTloc_run-6']
        }
    },
    # 'sub-000': {
    #     'MTloc': ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6'],
    #     'co': ['run-1', 'run-2', 'run-3'],
    #     'eic': ['run-1', 'run-2', 'run-3'],
    #     'experimental_order': ['MTloc_run-1', 'co_run-1', 'MTloc_run-2', 'co_run-2', 'co_run-3', 'MTloc_run-3', 'eic_run-1',
    #     'eic_run-2', 'MTloc_run-4', 'MTloc_run-5', 'eic_run-3', 'MTloc_run-6']
    # },
    'sub-0000': {
        'ses-sep': {
            'faceLoc': ['faceLoc_run-1', 'faceLoc_run-2', 'faceLoc_run-3', 'faceLoc_run-4', 'faceLoc_run-5'],
            'go': ['go_run-1', 'go_run-2', 'go_run-3', 'go_run-4',],
            'co': ['co_run-1', 'co_run-2', 'co_run-3', 'co_run-4'],
            'experimental_order': ['go_run-1', 'faceLoc_run-1', 'go_run-2', 'faceLoc_run-2',
                                   'co_run-1', 'faceLoc_run-3', 'go_run-3', 'faceLoc_run-4',
                                   'co_run-2', 'faceLoc_run-5', 'co_run-3', 'go_run-4', 'co_run-4']
        },
        'ses-com': {
            'MTloc': ['MTloc_run-1', 'MTloc_run-2', 'MTloc_run-3', 'MTloc_run-4', 'MTloc_run-5', 'MTloc_run-6'],
            'gic': ['gic_run-1', 'gic_run-2', 'gic_run-3', 'gic_run-4', 'gic_run-5', 'gic_run-6'],
            'experimental_order': ['gic_run-1', 'MTloc_run-1', 'gic_run-2', 'MTloc_run-2', 'gic_run-3',
                                    'MTloc_run-3', 'gic_run-4', 'MTloc_run-4', 'gic_run-5', 'MTloc_run-5',
                                    'gic_run-6', 'MTloc_run-6']
        }
    }
}

# func_run_dict = {
#     'sub-00': [
#         'MTloc_run-1', 'MTloc_run-2', 'MTloc_run-3', 'MTloc_run-4', 'MTloc_run-5', 'MTloc_run-6', 'MTloc_run-7',
#         'ec_run-1', 'ec_run-2', 'ec_run-3', 'ec_run-4', 'ec_run-5', 'ec_run-6'],
#     'sub-000': [
#         'MTloc_run-1', 'co_run-1', 'MTloc_run-2', 'co_run-2', 'co_run-3', 'MTloc_run-3', 'eic_run-1',
#         'eic_run-2', 'MTloc_run-4', 'MTloc_run-5', 'eic_run-3', 'MTloc_run-6'],
    # 'sub-00000': [
    #     'MTloc_run-01', 'MTloc_run-02', 'MTloc_run-03', 'MTloc_run-04', 'MTloc_run-05', 'MTloc_run-06',
    #     'MTloc_run-07', 'MTloc_run-08', 'MTloc_run-09', 'MTloc_run-10', 'MTloc_run-11', 'MTloc_run-12', 'MTloc_run-13',
    #     'com_run-1', 'com_run-2', 'com_run-3', 'com_run-4', 'com_run-5', 'com_run-6',
    #     'com_run-7', 'com_run-8', 'com_run-9']
#}


areas_sorted_for_plotting = ['fMotionArea', 'Marquardt-2017_GFP', 'visual-motion_MTplus', 'temporoparietal-junction_TPJ', 'theory-mind_TPJ', 'Kraemer-2020_IFJ', 'eye-fields_FEF', 'eye-fields_PEF',
                             'face-recognition_FFA', 'face-recognition_OFA', 'face-ffa_STS-FA', 'face-recognition_antFA', 'theory-mind_precuneus', 'theory-mind_mid-temp-cortex', 'theory-mind_ant-mid-temp-gyr',
                             'theory-mind_infFrontalGyr', 'theory-mind_mOFC', 'theory-mind_medAntPFC', 'Amygdala', 'SC']


############################
### MAKE SUBJECT_DATA BUNCH
def get_subject_data(
    sub_id,
    session,
    dirs,
    func_run_dict=func_run_dict  # from info.func_run_dict
):

    #
    # init Bunch object
    subject_data = Bunch()

    #
    # exp = subjects_experiment
    func_runs = func_run_dict

    #
    # get session sequence of cube arrangement
    # subject_data['run_identifierrun_identifier'] = list(pd.read_csv(
    #     opj(dirs.bids_dir, sub_id, 'nrec/trial_lists', 'run_identifier.tsv'), sep='\t', header=None)[0])



    subject_data = Bunch()

    #
    # anat
    subject_data['anat'] = Bunch()
    if os.path.isdir(opj(dirs.fmriprep_dir, sub_id, 'anat')):
        anat_dir = os.listdir(opj(dirs.fmriprep_dir, sub_id, 'anat'))
        for ff in anat_dir:
            if 'space-MNI152NLin2009cAsym_desc-brain_mask.nii' in ff:
                subject_data['anat']['mask'] = opj(
                    dirs.fmriprep_dir, sub_id, 'anat', ff)
            if 'space-MNI152NLin2009cAsym_desc-preproc_T1w.nii' in ff:
                subject_data['anat']['img'] = opj(
                    dirs.fmriprep_dir, sub_id, 'anat', ff)
            if 'L_pial.surf.gii' in ff:
                subject_data['anat']['left_surf_pial'] = opj(
                    dirs.fmriprep_dir, sub_id, 'anat', ff)
            if 'R_pial.surf.gii' in ff:
                subject_data['anat']['right_surf_pial'] = opj(
                    dirs.fmriprep_dir, sub_id, 'anat', ff)
            if 'L_inflated.surf.gii' in ff:
                subject_data['anat']['left_surf_inflated'] = opj(
                    dirs.fmriprep_dir, sub_id, 'anat', ff)
            if 'R_inflated.surf.gii' in ff:
                subject_data['anat']['right_surf_inflated'] = opj(
                    dirs.fmriprep_dir, sub_id, 'anat', ff)
    else:
        for ses_ in func_run_dict[sub_id]:
            anat_dir = os.listdir(opj(dirs.fmriprep_dir, sub_id, ses_, 'anat'))
            for ff in anat_dir:
                if 'space-MNI152NLin2009cAsym_desc-brain_mask.nii' in ff:
                    subject_data['anat']['mask'] = opj(
                        dirs.fmriprep_dir, sub_id, ses_, 'anat', ff)
                if 'space-MNI152NLin2009cAsym_desc-preproc_T1w.nii' in ff:
                    subject_data['anat']['img'] = opj(
                        dirs.fmriprep_dir, sub_id, ses_, 'anat', ff)
                if 'L_pial.surf.gii' in ff:
                    subject_data['anat']['left_surf_pial'] = opj(
                        dirs.fmriprep_dir, sub_id, ses_, 'anat', ff)
                if 'R_pial.surf.gii' in ff:
                    subject_data['anat']['right_surf_pial'] = opj(
                        dirs.fmriprep_dir, sub_id, ses_, 'anat', ff)
                if 'L_inflated.surf.gii' in ff:
                    subject_data['anat']['left_surf_inflated'] = opj(
                        dirs.fmriprep_dir, sub_id, ses_, 'anat', ff)
                if 'R_inflated.surf.gii' in ff:
                    subject_data['anat']['right_surf_inflated'] = opj(
                        dirs.fmriprep_dir, sub_id, ses_, 'anat', ff)

    #
    # func
    subject_data['func'] = Bunch()
    func_dir = os.listdir(opj(dirs.fmriprep_dir, sub_id, session, 'func'))

    # check if there is a functional mean mask
    mm_ = [el for el in func_dir if 'mean-mask.nii.gz' in el]
    if mm_:
        subject_data['func']['mean_mask'] = opj(
            dirs.fmriprep_dir, sub_id, session, 'func', mm_[0])
    else:
        subject_data['func']['mean_mask'] = None

    #
    # iterate tasks and functioanl runs specified in info.func_run_dict
    # subject_data['func']['runs'] = Bunch()
    if session == '':
        func_runs_ = list(func_runs[sub_id])
    else:
        func_runs_ = list(func_runs[sub_id][session])
    #
    for task_id in [el for el in func_runs_ if not el == 'experimental_order']:
        subject_data['func'][task_id] = Bunch()
        # subject_data['func'][task_id]['runs'] = Bunch()
        if not session == '':
            task_runs_ = func_runs[sub_id][session][task_id]
        else:
            task_runs_ = func_runs[sub_id][task_id]
        for fr in task_runs_:
            run_files = [frf for frf in func_dir if fr in frf and 'task-'+task_id in frf]
            subject_data['func'][task_id][fr] = Bunch()

            #
            # iterate files in ../fmriprep/func/ (filered by name of functional run)
            for ff in run_files:
                if 'brain_mask.nii' in ff:
                    subject_data['func'][task_id][fr]['mask'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if 'desc-preproc_bold.json' in ff:
                    subject_data['func'][task_id][fr]['meta'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if 'preproc_bold.nii' in ff:
                    subject_data['func'][task_id][fr]['img'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if 'preprocCleaned_bold.nii' in ff:
                    subject_data['func'][task_id][fr]['img_cleaned'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if ('preprocCleaned_s5' in ff) and ('_standardized' not in ff):
                    subject_data['func'][task_id][fr]['img_cleaned_smoothed'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if ('preprocCleaned_s5' in ff) and ('_standardized' in ff):
                    subject_data['func'][task_id][fr]['img_cleaned_smoothed_standardized'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if ('preprocCleaned_s5' not in ff) and ('_standardized' in ff):
                    subject_data['func'][task_id][fr]['img_cleaned_standardized'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if ('confounds_timeseries.tsv' in ff) or ('confounds_regressors.tsv' in ff):
                    subject_data['func'][task_id][fr]['confounds'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                if 'events.tsv' in ff:
                    subject_data['func'][task_id][fr]['events'] = opj(
                        dirs.fmriprep_dir, sub_id, session, 'func', ff)
                #
                # this following was used for testing the LISA algorithm
                # if 'events.txt' in ff:
                #     subject_data['func'][fr]['events_lisa'] = opj(dirs.fmriprep_dir, sub_id, session, 'func', ff )
                # if 'events_mapping.txt' in ff:
                #     subject_data['func'][fr]['events_lisa_mapping'] = opj(dirs.fmriprep_dir, sub_id, session, 'func', ff )

    return subject_data


def get_contrast_dict(task_id):

    contrast_dict = Bunch(

        ### task gc *** WITH HV ***
        # gc=Bunch(
        #
        #     ## effects
        #     Gaze=Bunch(
        #         name='Gaze',
        #         pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
        #         neg=[]),
        #     Cubes=Bunch(
        #         name='Cubes',
        #         pos=['Cubes_1_V', 'Cubes_2_V', 'Cubes_3_V',
        #              'Cubes_1_H', 'Cubes_2_H', 'Cubes_3_H'],
        #         neg=[]),
        #
        #     GazeCubes=Bunch(
        #         name="Gaze - Cubes",
        #         pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
        #         neg=['Cubes_1_V', 'Cubes_2_V', 'Cubes_3_V', 'Cubes_1_H', 'Cubes_2_H', 'Cubes_3_H']),
        #     Gaze1Gaze3=Bunch(
        #         name="Gaze: left target - right target",
        #         pos=['Gaze_1'],
        #         neg=['Gaze_3']),
        #     Cubes1Cubes3=Bunch(
        #         name="Cubes: left target - right target",
        #         pos=['Cubes_1_V', 'Cubes_1_H'],
        #         neg=['Cubes_3_V', 'Cubes_3_H']),
        #     CubesVCubesH=Bunch(
        #         name="Vertical cubes - Horizontal cubes",
        #         pos=['Cubes_1_V', 'Cubes_2_V', 'Cubes_3_V'],
        #         neg=['Cubes_1_H', 'Cubes_2_H', 'Cubes_3_H']),
        #     Gaze1Cubes3=Bunch(
        #         name="Gaze left target - Cubes right target",
        #         pos=['Gaze_1'],
        #         neg=['Cubes_3_V', 'Cubes_3_H'])
        # ),

        ### task gc *** WITHOUT HV ***
        gc=Bunch(

            ## effects
            Gaze=Bunch(
                name='Gaze',
                pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
                neg=[]),
            Cubes=Bunch(
                name='Cubes',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=[]),

            GazeCubes=Bunch(
                name="Gaze - Cubes",
                pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
                neg=['Cubes_1', 'Cubes_2', 'Cubes_3']),
            Gaze1Gaze3=Bunch(
                name="Gaze: left target - right target",
                pos=['Gaze_1'],
                neg=['Gaze_3']),
            Cubes1Cubes3=Bunch(
                name="Cubes: left target - right target",
                pos=['Cubes_1'],
                neg=['Cubes_3']),
            Gaze1Cubes3=Bunch(
                name="Gaze left target - Cubes right target",
                pos=['Gaze_1'],
                neg=['Cubes_3'])
        ),

        ### task MTloc
        MTloc=Bunch(
            MotionStatic=Bunch(
                name="Motion - Static",
                pos=['Motion'],
                neg=['Static']),
            # Motion=Bunch(
            #     name="Motion",
            #     pos=['Motion'],
            #     neg=[]),
            # Static=Bunch(
            #     name="Static",
            #     pos=['Static'],
            #     neg=[])
        ),

        ### task faceLoc
        faceLoc=Bunch(
            FaceOther=Bunch(
                name="Faces - Other",
                pos=['Face'],
                neg=['Other']),
            # Face=Bunch(
            #     name="Faces",
            #     pos=['Face'],
            #     neg=[]),
            # Other=Bunch(
            #     name="Other",
            #     pos=['Other'],
            #     neg=[]),
        ),

        ###############################
        ### sub-000
        ### task eic/gic
        eic=Bunch(

            ## effects
            Gaze=Bunch(
                name='Gaze',
                pos=['Gaze_1'],
                neg=[]),
            Cubes=Bunch(
                name='Cubes',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=[]),
            Iris=Bunch(
                name='Iris',
                pos=['Iris_1', 'Iris_2', 'Iris_3'],
                neg=[]),

            ## general contrasts
            Gaze1Iris123=Bunch(
                name='Eye-direction (left target) - Iris-color all targets',
                pos=['Gaze_1'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            Cubes1Iris123=Bunch(
                name='Cubes (left target) - Iris-color all targets',
                pos=['Cubes_1'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            Gaze1Cubes123=Bunch(
                name='Eye-direction (left target) - Cubes (all targets)',
                pos=['Gaze_1'],
                neg=['Cubes_1', 'Cubes_2', 'Cubes_3']),
            CubesIris=Bunch(
                name='Cubes - Iris-color',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),

            ### target specific contrasts
            Gaze1Iris1=Bunch(
                name='Left target: Eye-direction - Iris-color',
                pos=['Gaze_1'],
                neg=['Iris_1']),
            Gaze1Cubes1=Bunch(
                name='Left target: Eye-direction - Cubes',
                pos=['Gaze_1'],
                neg=['Cubes_1']),
            Cubes1Iris1=Bunch(
                name='Left target: Cubes - Iris',
                pos=['Cubes_1'],
                neg=['Iris_1']),
            Cubes2Iris2=Bunch(
                name='Middle target: Cubes - Iris',
                pos=['Cubes_2'],
                neg=['Iris_2']),
            Cubes3Iris3=Bunch(
                name='Right target: Cubes - Iris',
                pos=['Cubes_3'],
                neg=['Iris_3']),

            ### hemipshere contrasts
            Iris1Iris3=Bunch(
                name='Iris-color: left target - right target',
                pos=['Iris_1'],
                neg=['Iris_3']),
            Cubes1Cubes3=Bunch(
                name='Cubes: left target - right target',
                pos=['Cubes_1'],
                neg=['Cubes_3'])

        ),

        gic=Bunch(

            ## effects
            Gaze=Bunch(
                name='Gaze',
                pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
                neg=[]),
            Cubes=Bunch(
                name='Cubes',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=[]),
            Iris=Bunch(
                name='Iris',
                pos=['Iris_1', 'Iris_2', 'Iris_3'],
                neg=[]),
            Gaze1=Bunch(
                name='Gaze left target',
                pos=['Gaze_1'],
                neg=[]),
            Cubes1=Bunch(
                name='Cubes left target',
                pos=['Cubes_1'],
                neg=[]),
            Iris1=Bunch(
                name='Iris left target',
                pos=['Iris_1'],
                neg=[]),

            ## general contrasts
            GazeIris=Bunch(
                name='Gaze - Iris-color',
                pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            CubesIris=Bunch(
                name='Cubes - Iris-color',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            GazeCubes=Bunch(
                name='Gaze - Cubes',
                pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
                neg=['Cubes_1', 'Cubes_2', 'Cubes_3']),
            CubesGaze=Bunch(
                name='Cubes - Gaze',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=['Gaze_1', 'Gaze_2', 'Gaze_3']),


            ### hemipshere contrasts
            Gaze1Gaze3=Bunch(
                name='Gaze: left target - right target',
                pos=['Gaze_1'],
                neg=['Gaze_3']),
            Iris1Iris3=Bunch(
                name='Iris-color: left target - right target',
                pos=['Iris_1'],
                neg=['Iris_3']),
            Cubes1Cubes3=Bunch(
                name='Cubes: left target - right target',
                pos=['Cubes_1'],
                neg=['Cubes_3']),
            Gaze1Iris123=Bunch(
                name='Eye-direction (left target) - Iris-color all targets',
                pos=['Gaze_1'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            Gaze3Iris123=Bunch(
                name='Eye-direction (right target) - Iris-color all targets',
                pos=['Gaze_3'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            Cubes1Iris123=Bunch(
                name='Cubes-direction (left target) - Iris-color all targets',
                pos=['Cubes_1'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            Cubes3Iris123=Bunch(
                name='Cubes-direction (right target) - Iris-color all targets',
                pos=['Cubes_3'],
                neg=['Iris_1', 'Iris_2', 'Iris_3']),
            Gaze1Cubes3=Bunch(
                name='Gaze-direction (left target) - Cubes-direction (right target)',
                pos=['Gaze_1'],
                neg=['Cubes_3']),
            Gaze1Iris3=Bunch(
                name='Gaze-direction (left target) - Iris-color (right target)',
                pos=['Gaze_1'],
                neg=['Iris_3']),
            Cubes1Iris3=Bunch(
                name='Cubes-direction (left target) - Iris-color (right target)',
                pos=['Cubes_1'],
                neg=['Iris_3']),
            Gaze3Cubes1=Bunch(
                name='Gaze-direction (right target) - Cubes-direction (left target)',
                pos=['Gaze_3'],
                neg=['Cubes_1']),
            Gaze1Cubes123=Bunch(
                name='Gaze-direction (left target) - Cubes-direction (all targets)',
                pos=['Gaze_1'],
                neg=['Cubes_1', 'Cubes_2', 'Cubes_3']),

        ),


        ### task co
        co=Bunch(
            Cubes=Bunch(
                name='Cubes',
                pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
                neg=[]),
            Cubes1=Bunch(
                name='Cubes (left target)',
                pos=['Cubes_1'],
                neg=[]),
            Cubes2=Bunch(
                name='Cubes (middle target)',
                pos=['Cubes_2'],
                neg=[]),
            Cubes3=Bunch(
                name='Cubes (right target)',
                pos=['Cubes_3'],
                neg=[]),
            Cubes1Cubes3=Bunch(
                name='Cubes only: left target - right target',
                pos=['Cubes_1'],
                neg=['Cubes_3']),
            Ignore=Bunch(
                name="Ignore",
                pos=['Ignore'],
                neg=[]),
            IgnoreCubes=Bunch(
                name="Ignore - Cubes",
                pos=['Ignore'],
                neg=['Cubes_1', 'Cubes_2', 'Cubes_3'])
        ),

        ### task co
        go=Bunch(
            Gaze=Bunch(
                name='Gaze',
                pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
                neg=[]),
            Gaze1=Bunch(
                name='Gaze (left target)',
                pos=['Gaze_1'],
                neg=[]),
            Gaze2=Bunch(
                name='Gaze (middle target)',
                pos=['Gaze_2'],
                neg=[]),
            Gaze3=Bunch(
                name='Gaze (right target)',
                pos=['Gaze_3'],
                neg=[]),
            Gaze1Gaze3=Bunch(
                name='Gaze only: left target - right target',
                pos=['Gaze_1'],
                neg=['Gaze_3']),
            Ignore=Bunch(
                name="Ignore",
                pos=['Ignore'],
                neg=[]),
            IgnoreGaze=Bunch(
                name="Ignore - Gaze",
                pos=['Ignore'],
                neg=['Gaze_1', 'Gaze_2', 'Gaze_3'])
        ),

        ###############################
        ### sub-00000
        ### task eic
        # com=Bunch(
        #
        #     ecGaze_eicIris=Bunch(
        #         name='Eye-direction (ec) - Iris-color (eic)',
        #         pos=['ec_Gaze_1', 'ec_Gaze_2', 'ec_Gaze_3', 'eic_Gaze_1'],
        #         neg=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),
        #     ecCubesV_eicIris=Bunch(
        #         name='Cubes (ecV) - Iris-color (eic)',
        #         pos=['ec_Cubes_1_V', 'ec_Cubes_2_V', 'ec_Cubes_3_V'],
        #         neg=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),
        #     eicIris_ecGaze=Bunch(
        #         name='Iris-color (eic) - Eye-direction (ec)',
        #         neg=['ec_Gaze_1', 'ec_Gaze_2', 'ec_Gaze_3'],
        #         pos=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),
        #     eicIris_ecCubesV=Bunch(
        #         name='Iris-color (eic) - Cubes (ecV)',
        #         neg=['ec_Cubes_1_V', 'ec_Cubes_2_V', 'ec_Cubes_3_V'],
        #         pos=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),

            # com_eicCubes_eicIris=Bunch(
            #     name='Cubes (eic) - Iris-color (eic)',
            #     pos=['eic_Cubes_1', 'eic_Cubes_2', 'eic_Cubes_3'],
            #     neg=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),
            # com_eicGaze1_eicIris123=Bunch(
            #     name='Eye_direction 1 (eic) - Iris-color 123 (eic)',
            #     pos=['eic_Gaze_1'],
            #     neg=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),
            # com_eicCubes1_eicIris123=Bunch(
            #     name='Cubes 1 (eic) - Iris-color 123 (eic)',
            #     pos=['eic_Cubes_1'],
            #     neg=['eic_Iris_1', 'eic_Iris_2', 'eic_Iris_3']),
            # com_eicGaze1_eicCubes123=Bunch(
            #     name='Eye-direction 1 (eic) - Cubes 123 (eic)',
            #     pos=['eic_Gaze_1'],
            #     neg=['eic_Cubes_1', 'eic_Cubes_2', 'eic_Cubes_3']),
            # com_eicGaze1_eicCubes1=Bunch(
            #     name='Eye-direction 1 (eic) - Cubes 1 (eic)',
            #     pos=['eic_Gaze_1'],
            #     neg=['eic_Cubes_1']),

            # com_ecGaze_ecCubes=Bunch(
            #     name='Eye_direction (ec) - Cubes (ec)',
            #     pos=['ec_Gaze_1', 'ec_Gaze_2', 'ec_Gaze_3'],
            #     neg=['ec_Cubes_1_V', 'ec_Cubes_2_V', 'ec_Cubes_3_V', 'ec_Cubes_1_H', 'ec_Cubes_2_H', 'ec_Cubes_3_H']),
            # com_ecGaze1_ecCubes1V=Bunch(
            #     name='Eye-direction 1 (eic) - Cubes 1 (eic)',
            #     pos=['ec_Gaze_1'],
            #     neg=['ec_Cubes_1_V']),
            # com_ecGaze1_ecCubes1H=Bunch(
            #     name='Eye-direction 1 (ec) - Cubes 1 (ec)',
            #     pos=['ec_Gaze_1'],
            #     neg=['ec_Cubes_1_H']),

            # com_ecVCubes1_eicCubes3=Bunch(
            #     name='Cubes ecV-eic: left-right',
            #     pos=['ec_Cubes_1_V'],
            #     neg=['eic_Cubes_3']),
            # com_ecCubes1V_ecCubes3V=Bunch(
            #     name='Cubes ecV-ecV: left-right',
            #     pos=['ec_Cubes_1_V'],
            #     neg=['ec_Cubes_3_V']),
            # com_ecCubes1H_ecCubes3H=Bunch(
            #     name='Cubes ecH-ecH: left-right',
            #     pos=['ec_Cubes_1_H'],
            #     neg=['ec_Cubes_3_H']),
            # com_eicCubes1_eicCubes3=Bunch(
            #     name='Cubes eic-eic: left-right',
            #     pos=['eic_Cubes_1'],
            #     neg=['eic_Cubes_3'])
        # )

    )

    contrast_dict['gcH'] = Bunch(
        GazeCubesH=Bunch(
            name="Gaze - Horizontal cubes",
            pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
            neg=['Cubes_1_H', 'Cubes_2_H', 'Cubes_3_H']),
        Cubes1HCubes3H=Bunch(
            name="Horizontal cubes: left target - right target",
            pos=['Cubes_1_H'],
            neg=['Cubes_3_H'])
    )
    contrast_dict['gcV'] = Bunch(
        GazeCubesV=Bunch(
            name="Gaze - Vertical cubes",
            pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
            neg=['Cubes_1_V', 'Cubes_2_V', 'Cubes_3_V']),
        Cubes1VCubes3V=Bunch(
            name="Vertical cubes: left target - right target",
            pos=['Cubes_1_V'],
            neg=['Cubes_3_'])
    )
    contrast_dict['MTlocCubes'] = Bunch(
        MotionStatic=Bunch(
            name="Motion - Static (Cubes)",
            pos=['Motion'],
            neg=['Static'])
    )
    contrast_dict['MTlocRotation'] = Bunch(
        MotionStatic=Bunch(
            name="Motion - Static (Rotating dots)",
            pos=['Motion'],
            neg=['Static'])
    )
    contrast_dict['MTlocRandom'] = Bunch(
        MotionStatic=Bunch(
            name="Motion - Static (RDM)",
            pos=['Motion'],
            neg=['Static'])
    )

    return contrast_dict[task_id]


decoding_pairs = Bunch(
    gic=Bunch(
        GazeXCubes=[['Gaze'], ['Cubes']],
        CubesXIris=[['Cubes'], ['Iris']],
        GazeXIris=[['Gaze'], ['Iris']],
        IrisXGaze_Cubes=[['Iris'], ['Gaze', 'Cubes']],
        CubesXGaze_Iris=[['Cubes'], ['Gaze', 'Iris']]
    ),
    gc=Bunch(
        GazeXCubes=[['Gaze'], ['Cubes']]
    )
)



data_files_for_behavior = {
    'sub-00': {
        'gc_run-1':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-17-22-318.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-1.txt'],
        'gc_run-2':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-31-40-722.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-2.txt'],
        'gc_run-3':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-46-05-995.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-3.txt'],
        'gc_run-4':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-05-20-639.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-4.txt'],
        'gc_run-5':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-19-11-291.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-5.txt'],
        'gc_run-6':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-32-42-220.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-6.txt']
    },

    'sub-000': {
        'co_run-1':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/gfp_experiment2021-05-17-12-04-02-023.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/trial_lists/co_run-1.txt'],
        'co_run-2':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/gfp_experiment2021-05-17-12-16-12-251.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/trial_lists/co_run-2.txt'],
        'co_run-3':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/gfp_experiment2021-05-17-12-23-09-643.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/trial_lists/co_run-3.txt'],
        'eic_run-1':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/gfp_experiment2021-05-17-12-35-41-627.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/trial_lists/eic_run-1.txt'],
        'eic_run-2':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/gfp_experiment2021-05-17-12-53-55-556.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/trial_lists/eic_run-2.txt'],
        'eic_run-3':
            ['/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/gfp_experiment2021-05-17-13-15-35-508.mat',
             '/media/marius/data_ex/gaze-motion/dsgm/sub-000/nrec/trial_lists/eic_run-3.txt'],
    }
}


#
# data_paths = {'sub-00':
#               {
#                   'anat':{
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/anat/sub-00_T1w.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/anat/sub-00_T1w.json'
#                   },
#                   'fmap':[
#                       '/media/marius/data_ex/dsgm/sub-00/fmap/sub-00_magnitude1.nii',
#                       '/media/marius/data_ex/dsgm/sub-00/fmap/sub-00_magnitude2.nii',
#                       '/media/marius/data_ex/dsgm/sub-00/fmap/sub-00_phasediff.nii'
#                   ],
#                   'MTloc_run-1':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-1_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-1_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-1_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-13-53-01-020.mat'
#                   },
#                   'MTloc_run-2':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-2_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-2_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-2_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-13-56-54-891.mat'
#                   },
#                   'MTloc_run-3':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-3_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-3_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-3_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-13-59-53-714.mat'
#                   },
#                   'MTloc_run-4':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-4_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-4_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-4_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-4_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-14-03-37-228.mat'
#                   },
#                   'MTloc_run-5':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-5_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-5_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-5_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-5_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-14-06-33-220.mat'
#                   },
#                   'MTloc_run-6':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-6_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-6_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-6_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-6_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-14-09-28-705.mat'
#                   },
#                   'MTloc_run-7':{
#                       'nii_preproc':'/media/marius/data_ex/dsgm_derivatives/fmriprep/sub-00/func/sub-00_task-MTloc_run-7_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-7_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-7_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-MTloc_run-7_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/experiment2021-04-20-14-12-34-552.mat'
#                   },
#                   ####################################
#                   ####################################
#                   'gmV_run-1':{
#                       'trial_list':'/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/stimuli/trial_lists/sub-00/session-1.txt',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-1_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-1_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-1_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-17-22-318.mat'
#                   },
#                   'gmH_run-1':{
#                       'trial_list':'/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/stimuli/trial_lists/sub-00/session-2.txt',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-1_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-1_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-1_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-31-40-722.mat'
#                   },
#                   'gmV_run-2':{
#                       'trial_list':'/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/stimuli/trial_lists/sub-00/session-3.txt',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-2_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-2_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-2_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-46-05-995.mat'
#                   },
#                   'gmV_run-3':{
#                       'trial_list':'/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/stimuli/trial_lists/sub-00/session-4.txt',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-3_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-3_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmV_run-3_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-05-20-639.mat'
#                   },
#                   'gmH_run-2':{
#                       'trial_list':'/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/stimuli/trial_lists/sub-00/session-5.txt',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-2_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-2_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-2_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-19-11-291.mat'
#                   },
#                   'gmH_run-3':{
#                       'trial_list':'/home/marius/ownCloud/PhD/projects/scientific/gaze-motion/stimuli/trial_lists/sub-00/session-6.txt',
#                       'nii':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-3_bold.nii',
#                       'json':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-3_bold.json',
#                       'events':'/media/marius/data_ex/dsgm/sub-00/func/sub-00_task-gmH_run-3_bold.tsv',
#                       'nrec':'/media/marius/data_ex/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-32-42-220.mat'
#                   }
#               }
#              }
