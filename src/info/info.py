import os
from os.path import join as opj
import pandas as pd
from sklearn.utils import Bunch


colors = Bunch(
    ###
)

### SET FIRST ###
data_paths = Bunch(
    root_dir='',
    bids_dir='',
    qc_dir='',
    derivatives_dir='',
    fmriprep_dir='',
    scratch_dir='',
    results_dir='',
    rois_dir='',
)


### specify existing runs in /fmriprep/[sub]/func/ in the order of the experiment
func_run_dict = {
    'SUBJECT': {
        'SESSION': {
            'TASK': ['RUN']
        }
    }
}


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
        TASK=Bunch(
            name='CONTRAST',
            pos=[],
            neg=[]
        )

        ### EXAMPLE ***
        # gc=Bunch(
        #     ## effects
        #     Gaze=Bunch(
        #         name='Gaze',
        #         pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
        #         neg=[]),
        #     Cubes=Bunch(
        #         name='Cubes',
        #         pos=['Cubes_1', 'Cubes_2', 'Cubes_3'],
        #         neg=[]),
        #
        #     GazeCubes=Bunch(
        #         name="Gaze - Cubes",
        #         pos=['Gaze_1', 'Gaze_2', 'Gaze_3'],
        #         neg=['Cubes_1', 'Cubes_2', 'Cubes_3']),
        #     Gaze1Gaze3=Bunch(
        #         name="Gaze: left target - right target",
        #         pos=['Gaze_1'],
        #         neg=['Gaze_3']),
        #     Cubes1Cubes3=Bunch(
        #         name="Cubes: left target - right target",
        #         pos=['Cubes_1'],
        #         neg=['Cubes_3']),
        #     Gaze1Cubes3=Bunch(
        #         name="Gaze left target - Cubes right target",
        #         pos=['Gaze_1'],
        #         neg=['Cubes_3'])
        # ),

    return contrast_dict[task_id]


decoding_pairs = Bunch(
    TASK=Bunch(
        CON1_XCON2=[['CON1'], ['CON2']],
    )
    # gc=Bunch(
    #     GazeXCubes=[['Gaze'], ['Cubes']]
    # )
)



data_files_for_behavior = {

    ### EXAMPLE ###
    # 'sub-00': {
    #     'gc_run-1':
    #         ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-17-22-318.mat',
    #          '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-1.txt'],
    #     'gc_run-2':
    #         ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-31-40-722.mat',
    #          '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-2.txt'],
    #     'gc_run-3':
    #         ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-14-46-05-995.mat',
    #          '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-3.txt'],
    #     'gc_run-4':
    #         ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-05-20-639.mat',
    #          '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-4.txt'],
    #     'gc_run-5':
    #         ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-19-11-291.mat',
    #          '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-5.txt'],
    #     'gc_run-6':
    #         ['/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/gfp_experiment2021-04-20-15-32-42-220.mat',
    #          '/media/marius/data_ex/gaze-motion/dsgm/sub-00/nrec/trial_lists/session-6.txt']
    # }
}
