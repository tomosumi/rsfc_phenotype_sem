"""
Module for preprocessing fmri data
"""

import argparse
import os
import os.path as op
from pdb import set_trace
from subprocess import call
from time import time

import numpy as np
from math import comb
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

DATA_ROOT_DIR = "/home/cezanne/t-haitani/hcp_data/data"
HCP_ROOT_DIR = "/home/cezanne/t-haitani/hcp_data"
PARCELLATION_DIR = op.join(HCP_ROOT_DIR, "derivatives", "Python", "parcellation")
ATLAS_DIR_DICT = {
    "Schaefer": op.join(PARCELLATION_DIR, "Schaefer"),
    "Gordon": op.join(PARCELLATION_DIR, "Gordon"),
    "Glasser": op.join(PARCELLATION_DIR, "Glasser"),
}
N_RSFC_DICT = {'Schaefer': 432, 'Gordon': 365}

SESSIONS = ["rfMRI_REST1_RL", "rfMRI_REST1_LR", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]

NODE_SUMMARY_PATH_DICT = {
    "Schaefer": "/home/cezanne/t-haitani/hcp_data/derivatives/parcellation/schaefer_s2_summary_xyz.csv",
    "Gordon": "/home/cezanne/t-haitani/hcp_data/derivatives/parcellation/gordon_333_s2_summary.csv",
    "Glasser": None,
}


NETWORK_ORDER_DICT_SCHAEFER = {
    "Vis": 0,
    "SomMot": 1,
    "DorsAttn": 2,
    "SalVentAttn": 3,
    "Limbic": 4,
    "LimbicTian": 5,
    "Cont": 6,
    "Default": 7,
}

NETWORK_ORDER_DICT_GORDON = {
    "Auditory": 0,
    "Visual": 1,
    "SMhand": 2,
    "SMmouth": 3,
    "DorsalAttn": 4,
    "VentralAttn": 5,
    "Salience": 6,
    "CinguloOperc": 7,
    "CinguloParietal": 8,
    "Subcortex": 9,
    "FrontoParietal": 10,
    "Default": 11,
    "RetrosplenialTemporal": 12,
    "NA": 13
}

NETWORK_ORDER_NESTED_DICT = {
    "Schaefer": NETWORK_ORDER_DICT_SCHAEFER,
    "Gordon": NETWORK_ORDER_DICT_GORDON,
}

NETWORK_ORDER_DICT_PUB_SCHAEFER = {
    "Visual": 0,
    "Somatomotor": 1,
    "Dorsal Attention": 2,
    "Ventral Attention": 3,
    "Limbic": 4,
    "Subcortex": 5,
    "Frontoparietal": 6,
    "Default": 7,
}

NETWORK_ORDER_DICT_PUB_GORDON = {
    "Auditory": 0,
    "Visual": 1,
    "Supplementary motor (hand)": 2,
    "Supplementary motor (mouth)": 3,
    "Dorsal Attention": 4,
    "Ventral Attention": 5,
    "Salience": 6,
    "Cingulo-opercular": 7,
    "Cingulo-parietal": 8,
    "Subcortex": 9,
    "Frontoparietal": 10,
    "Default": 11,
    "Retrosplenial-temporal": 12,
    "No assignment": 13
}

NETWORK_AGG_PUB_DICT_SCHAEFER = {
    key: value
    for key, value in zip(NETWORK_ORDER_DICT_SCHAEFER.keys(), NETWORK_ORDER_DICT_PUB_SCHAEFER.keys())
}
NETWORK_AGG_PUB_DICT_GORDON = {
    key: value
    for key, value in zip(
        NETWORK_ORDER_DICT_GORDON.keys(), NETWORK_ORDER_DICT_PUB_GORDON.keys()
    )
}
NETWORK_AGG_PUB_NESTED_DICT = {
    "Schaefer": NETWORK_AGG_PUB_DICT_SCHAEFER,
    "Gordon": NETWORK_AGG_PUB_DICT_GORDON,
}

TIAN_ATLAS_DIR = (
    "/home/cezanne/t-haitani/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/"
)
ATLAS_DIR = "/home/cezanne/t-haitani/hcp_data/atlas"

SCHAEFER_400_SUBCORTEX_S2 = op.join(
    TIAN_ATLAS_DIR,
    "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S2.dlabel.nii",
)
GORDON_333_SUBCORTEX_S2 = op.join(
    TIAN_ATLAS_DIR, "Gordon333.32k_fs_LR_Tian_Subcortex_S2.dlabel.nii"
)
GLASSER_360_SUBCORTEX_S2 = op.join(
    TIAN_ATLAS_DIR,
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S2.dlabel.nii",
)

PARCEL_NAME_DICT = {
    "Schaefer": "Schaefer_400_subcortex_s2",
    "Gordon": "Gordon_333_subcortex_s2",
    "Glasser": "Glasser_360_subcortex_s2",
}

PARCEL_FILE_DICT = {
    "Schaefer": SCHAEFER_400_SUBCORTEX_S2,
    "Gordon": GORDON_333_SUBCORTEX_S2,
    "Glasser": GLASSER_360_SUBCORTEX_S2,
}

HCP_PREPROCESS = "Atlas_MSMAll_hp2000_clean"

FAMILY_PATH = '/home/cezanne/t-haitani/hcp_data/RESTRICTED_tomosumi_9_9_2024_3_21_50.csv'
SUBJECT_ANALYSIS_PATH = '/home/cezanne/t-haitani/hcp_data/derivatives/subjects_set_for_analysis_861.csv'
SUBJECT_CB = '/home/cezanne/t-haitani/hcp_data/derivatives/subjects_cb_filtered.txt'

SUBJECT_LIST = list()
for folder in next(os.walk(op.join(DATA_ROOT_DIR)))[1]:
    SUBJECT_LIST.append(folder)

nib.imageglobals.logger.setLevel(40)


def generate_subject_list() -> list[int]:
    """
    generate a list of subjects
    """
    # create subject list
    subject_list = []
    for folder in next(os.walk(DATA_ROOT_DIR))[1]:
        subject_list.append(folder)
    return subject_list


def generate_gsr_strings(gsr_type):
    """
    generate gsr string from gsr conditions
    """
    if gsr_type == "nogs":
        gsr_str = ""
    elif gsr_type == "gs":
        gsr_str = "_gs"
    return gsr_str


def get_ordered_array_of_node_summary(parcellation="Schaefer"):
    """
    Get dataframe of ordered node summary
    """
    node_summary_path = NODE_SUMMARY_PATH_DICT.get(parcellation)
    network_order_dict = NETWORK_ORDER_NESTED_DICT.get(parcellation)
    network_agg_pub_dict = NETWORK_AGG_PUB_NESTED_DICT.get(parcellation)
    
    node_summary = pd.read_csv(node_summary_path).iloc[:, 1:].reset_index()
    node_summary["net"] = node_summary["net"].replace("Limbic_tian", "LimbicTian")
    node_summary["net"] = pd.Categorical(
        node_summary["net"], categories=network_order_dict.keys(), ordered=True
    )
    node_summary["net"] = node_summary["net"].cat.rename_categories(
        network_agg_pub_dict
    )

    node_summary_ordered = (
        node_summary.sort_values(["net", "hem"])
        .rename(columns={"index": "old_index"})
        .reset_index(drop=True)
        .reset_index()
    )
    ordered_array = node_summary_ordered["old_index"]

    return ordered_array


def generate_fcs(filename_suffix, n_nodes, parcellation="Schaefer"):
    """
    Generate and save summary array of RSFC
    """
    fc_array = np.zeros(
        shape=(
            int(n_nodes * (n_nodes - 1) / 2),
            int(len(next(os.walk(DATA_ROOT_DIR))[1])),
            4,
        )
    )
    fc_array[:] = np.nan

    node_summary_path = NODE_SUMMARY_PATH_DICT.get(parcellation)
    ordered_array = get_ordered_array_of_node_summary(parcellation)

    for i, subject in enumerate(tqdm(SUBJECT_LIST)):
        for j, run in enumerate(SESSIONS):
            for pt_file in os.listdir(op.join(DATA_ROOT_DIR, subject, run)):
                if pt_file.endswith(filename_suffix):
                    input_file = (
                        nib.load(op.join(DATA_ROOT_DIR, subject, run, pt_file))
                        .get_fdata()
                        .T
                    )
                    # Reorder parcellation
                    inp_rs = np.corrcoef(input_file[ordered_array])
                    inp_rs = inp_rs[np.triu_indices(inp_rs.shape[0], 1)]
                    fc_array[:, i, j] = inp_rs

    np.save(
        op.join(
            f"/home/cezanne/t-haitani/hcp_data/derivatives/Python/parcellation/{parcellation}",
            "ordered_fc_" + filename_suffix.replace(".ptseries.nii", ".npy"),
        ),
        fc_array,
    )


def concat_fmri_files_all(gsr_list=["nogs", "gs"], parcellation="Schaefer"):
    """
    Concatenate fmri files and calculate FC in all sessions
    """
    subject_list = generate_subject_list()
    ordered_array = get_ordered_array_of_node_summary(
        parcellation
    )
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    n_rsfc = N_RSFC_DICT.get(parcellation)

    # Subject level loop
    for subject in tqdm(subject_list):
        subject_dir = op.join(DATA_ROOT_DIR, subject)

        # GS level loop
        for gs in gsr_list:
            file_exis_list = []
            gs_str = "_gs" if gs == "gs" else ""
            subject_data = np.array([]).reshape(0, n_rsfc)
            save_file = op.join(
                subject_dir,
                f"all_spreg_0.25{gs_str}_{parcel_name}_full_data_exist_ordered.tsv",
            )

            for folder_name in [
                "rfMRI_REST1_RL",
                "rfMRI_REST1_LR",
                "rfMRI_REST2_LR",
                "rfMRI_REST2_RL",
            ]:
                target_folder = op.join(subject_dir, folder_name)
                # Search target file
                filename = f"{folder_name}_{HCP_PREPROCESS}_spreg_0.25{gs_str}_demeaned_{parcel_name}.ptseries.nii"
                target_file = op.join(target_folder, filename)
                if op.isfile(target_file):
                    bold_data = nib.load(target_file).get_fdata()
                    subject_data = np.concatenate([subject_data, bold_data], axis=0)
                    file_exis_list.append(True)
                else:
                    file_exis_list.append(False)

            # Calculate FC
            if all(file_exis_list):
                corr_fc = np.corrcoef(subject_data.T[ordered_array])
                # Save FC file
                np.savetxt(save_file, corr_fc)


def concat_fmri_files_by_days(
        parcellation="Schaefer", 
        over_write=False,
        read_file=True,
        ordered=True
        ):
    """
    Concatenate fmri files and calculate FC in the same day
    """
    subject_list = generate_subject_list()
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    ordered_array = get_ordered_array_of_node_summary(parcellation)
    n_nodes = len(ordered_array)
    
    # Subject level loop
    for gs in ["gs", "nogs"]:
        gs_str = "_gs" if gs == "gs" else ""
        fc_array = np.zeros(
            shape=(
                int(n_nodes * (n_nodes - 1) / 2),
                int(len(subject_list)),
                2,
            )
        )
        for i, subject in enumerate(subject_list):
            subject_dir = op.join(DATA_ROOT_DIR, subject)
            print(f"Processing {subject}.")
            # Day level loop
            for j, day in enumerate(["1", "2"]):
                # make directories by day
                day_dir = op.join(subject_dir, f"day{day}")
                os.makedirs(day_dir, exist_ok=True)

                save_file = op.join(
                    day_dir, f"day{day}_spreg_0.25{gs_str}_{parcel_name}.tsv"
                )
                # Search target file
                if not op.isfile(save_file) or over_write or read_file:
                    filename_lr = f"rfMRI_REST{day}_LR_{HCP_PREPROCESS}_spreg_0.25{gs_str}_demeaned_{parcel_name}.ptseries.nii"
                    filename_rl = f"rfMRI_REST{day}_RL_{HCP_PREPROCESS}_spreg_0.25{gs_str}_demeaned_{parcel_name}.ptseries.nii"
                    lr_dir = op.join(
                        DATA_ROOT_DIR, subject, f"rfMRI_REST{day}_LR", filename_lr
                    )
                    rl_dir = op.join(
                        DATA_ROOT_DIR, subject, f"rfMRI_REST{day}_RL", filename_rl
                    )
                    # Read and save data if file exists
                    if op.isfile(lr_dir) and op.isfile(rl_dir):
                        day_lr_data = nib.load(lr_dir).get_fdata()
                        day_rl_data = nib.load(rl_dir).get_fdata()
                        #print(gs, day_lr_data.shape, day_rl_data.shape)
                        # Concat day data with opposite phase encoding directions
                        concat_data = np.concatenate(
                            [day_lr_data.T, day_rl_data.T], axis=1
                        )
                        # Calculate FC
                        concat_data = concat_data[ordered_array] if ordered else concat_data
                        corr_fc = np.corrcoef(concat_data)
                        corr_fc = corr_fc[np.triu_indices(corr_fc.shape[0], 1)]
                        fc_array[:, i, j] = corr_fc
                        # Save FC file
                        if not op.isfile(save_file):
                            np.savetxt(save_file, corr_fc)
                        if over_write:
                            np.savetxt(save_file, corr_fc)
        ordered_prefix = 'ordered_' if ordered else ''
        np.save(
            op.join(
                f"/home/cezanne/t-haitani/hcp_data/derivatives/Python/parcellation/{parcellation}",
                f"{gs}_{ordered_prefix}fc_spreg_0.25_day_combined_{parcel_name}.npy",
            ),
            fc_array,
        )


def combine_fc_all_sub_day(parcellation="Schaefer"):
    """
    Combine FCs in all subjects and save a numpy file with and without gsr conditions by days
    This function may be used after conducting concat_fmri_files_by_days()
    """
    sub_list = generate_subject_list()
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    parcellation_folder = ATLAS_DIR_DICT.get(parcellation)
    n_rsfc = N_RSFC_DICT.get(parcellation)
    # gsr type level loop
    for gsr in ["nogs", "gs"]:
        gsr_str = generate_gsr_strings(gsr)
        # 2 represents a number of days
        output_array = np.empty(shape=(comb(n_rsfc, 2), len(sub_list), 2))
        output_array[:] = np.nan
        for i, subject in enumerate(tqdm(sub_list)):
            # print(f'Processing {subject}.')
            for j, day in enumerate(["day1", "day2"]):
                tsv_file = op.join(
                    DATA_ROOT_DIR,
                    subject,
                    day,
                    f"{day}_spreg_0.25{gsr_str}_{parcel_name}.tsv",
                )
                if op.isfile(tsv_file):
                    fc_day_array = np.loadtxt(tsv_file)
                    fc_inp = fc_day_array[np.triu_indices(fc_day_array.shape[0], 1)]
                    output_array[:, i, j] = fc_inp
        np.save(
            op.join(
                parcellation_folder,
                f"fc_spreg_0.25{gsr_str}_demeaned_{parcel_name}_day_combine.npy",
            ),
            output_array,
        )


def combine_fc_all_sub_full(gsr_list=["nogs", "gs"], parcellation="Schaefer"):
    """
    Combine FCs in all subjects and save a numpy file
    with and without gsr conditions with all data
    """
    sub_list = generate_subject_list()
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    parcellation_folder = ATLAS_DIR_DICT.get(parcellation)
    n_rsfc = N_RSFC_DICT.get(parcellation)
    # gsr type level loop
    for gsr in gsr_list:
        gsr_str = generate_gsr_strings(gsr)
        output_array = np.empty(shape=(comb(n_rsfc, 2), len(sub_list)))
        output_array[:] = np.nan
        for i, subject in enumerate(tqdm(sub_list)):
            # print(f'Processing {subject}.')
            tsv_file = op.join(
                DATA_ROOT_DIR,
                subject,
                f"all_spreg_0.25{gsr_str}_{parcel_name}_full_data_exist_ordered.tsv",
            )
            if op.isfile(tsv_file):
                fc_day_array = np.loadtxt(tsv_file)
                fc_inp = fc_day_array[np.triu_indices(fc_day_array.shape[0], 1)]
                output_array[:, i] = fc_inp
        np.save(
            op.join(
                parcellation_folder,
                f"fc_spreg_0.25{gsr_str}_demeaned_{parcel_name}_all_combine_full_data_exist_ordered.npy",
            ),
            output_array,
        )


def compare_all_vs_mean_fc(parcellation="Schaefer"):
    """
    Compare FCs between full and mean of runs
    """
    parcellation_folder = ATLAS_DIR_DICT.get(parcellation)
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    cor_array = np.empty(shape=(93096, 2))
    for j, gsr in enumerate(["nogs", "gs"]):
        print(f"Processing {gsr}")
        gsr_str = "_gs_" if gsr == "gs" else "_"
        all_array = np.arctanh(
            np.load(
                op.join(
                    parcellation_folder,
                    f"fc_spreg_0.25{gsr_str}demeaned_{parcel_name}_all_combine.npy",
                )
            )
        )
        runs_array = np.arctanh(
            np.load(
                op.join(
                    parcellation_folder,
                    f"fc__spreg_0.25{gsr_str}demeaned_{parcel_name}.npy",
                )
            )
        )
        runs_array_mean = np.mean(runs_array, axis=2)
        for i in range(runs_array_mean.shape[1]):
            cor = np.corrcoef(all_array[:, i], runs_array_mean[:, i])[0, 1]
            cor_array[i, j] = cor

    return cor_array


def train_test_split_family(random_seed=0, k_fold:int=2, select_cb=True):
    """
    Split ID based on family
    """
    subjects_path = SUBJECT_ANALYSIS_PATH if not select_cb else SUBJECT_CB
    subjects = np.loadtxt(subjects_path, dtype=str)
    family_df = pd.read_csv(FAMILY_PATH)
    family_df['Subject'] = family_df['Subject'].astype(str)
    family_df.query('Subject in @subjects', inplace=True)

    unique_family = family_df['Family_ID'].unique()
    train_family_ids, test_family_ids = train_test_split(unique_family, test_size=1/k_fold, random_state=random_seed)
    
    train_ids = family_df.query(
            'Family_ID in @train_family_ids'
            ).groupby('Family_ID')['Subject'].sample(n=1, random_state=random_seed).sort_values().astype(object)
    valid_ids = family_df.query(
            'Family_ID in @test_family_ids'
            ).groupby('Family_ID')['Subject'].sample(n=1, random_state=random_seed).sort_values().astype(object)
    return set(map(str, train_ids)), set(map(str, valid_ids))


def spike_gs_regression_mod(
    subject_list=SUBJECT_LIST,
    sessions=SESSIONS,
    parcellation="Schaefer",
    gsr=False,
    spike_regression=True,
    spike_thres=0.25,
    over_write=True,
    remove_volumes=4,
    subject_ids=None,
    fig_suffix_add="",
    save_fig=False,
):
    """
    Conduct spike regression (and GSR)
    """
    if subject_ids == ["None"]:
        subject_list_for_loop = subject_list
    else:
        subject_list_for_loop = [i for i in subject_list if i in subject_ids]
    parcellation_name = PARCEL_NAME_DICT.get(parcellation)
    parcellation_file = PARCEL_FILE_DICT.get(parcellation)
    print(subject_list_for_loop)

    def replace_parcel_filename(input_file, parcellation_name):
        return input_file.replace(".dtseries.nii", f"_{parcellation_name}.ptseries.nii")

    def conduct_parcellation(
        input_file, parcellation_name, parcellation_file, over_write
    ):
        parcel_file = replace_parcel_filename(input_file, parcellation_name)
        cmd = f"/home/ncd/t-haitani/Downloads/workbench/bin_linux64/wb_command \
        -cifti-parcellate {input_file} {parcellation_file} COLUMN {parcel_file}"
        if not over_write:
            if not op.isfile(parcel_file):
                print(f"Generating {parcel_file}")
                call(cmd, shell=True)
        else:
            print(f"Generating {parcel_file}")
            call(cmd, shell=True)

    for subject in subject_list_for_loop:
        for run in sessions:
            mri_file = op.join(
                DATA_ROOT_DIR,
                subject,
                run,
                f"{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii",
            )
            spike_regressor_file = op.join(
                DATA_ROOT_DIR,
                subject,
                run,
                f"spike_regressor_with_rms_{spike_thres}.txt",
            )
            if over_write:
                if (op.isfile(mri_file)) and (op.isfile(spike_regressor_file)):
                    print(f"Processing {run} in {subject}")
                    filename_original = (
                        f"{run}_Atlas_MSMAll_hp2000_clean_demeaned.dtseries.nii"
                    )
                    gsr_str = "gs_" if gsr else ""
                    filename_spreg = f"{run}_Atlas_MSMAll_hp2000_clean_spreg_{spike_thres}_{gsr_str}demeaned.dtseries.nii"

                    cifti_file_original = op.join(
                        DATA_ROOT_DIR, subject, run, filename_original
                    )
                    cifti_file_spreg = op.join(
                        DATA_ROOT_DIR, subject, run, filename_spreg
                    )

                    filelist = [
                        cifti_file_original,
                        cifti_file_spreg,
                        #                        cifti_file_original.replace("dtseries", "ptseries"),
                        #                        cifti_file_spreg.replace("dtseries", "ptseries"),
                    ]
                    # regression on fmri data
                    # check whether spike regression (and GSR) should be conducted
                    if not all([op.isfile(f) for f in filelist]):
                        mri_data = nib.load(mri_file)
                        mri_data_d = mri_data.get_fdata().T[:, remove_volumes:]

                        # scaling
                        mri_data_d = mri_data_d / np.median(mri_data_d) * 1000

                        # demeaned for visualization
                        mri_data_mean_s = mri_data_d.mean(axis=1, keepdims=True)
                        mri_data_demeaned = mri_data_d - mri_data_mean_s

                        # get header data
                        time_axis, brain_model_axis = [
                            mri_data.header.get_axis(i) for i in range(mri_data.ndim)
                        ]
                        time_series = nib.cifti2.SeriesAxis(
                            start=0, step=0.8, size=mri_data_d.shape[1]
                        )
                        # save demeaned dtseries data
                        img = nib.Cifti2Image(
                            mri_data_demeaned.T,
                            header=(time_series, brain_model_axis),
                            nifti_header=mri_data.nifti_header,
                        )
                        img.to_filename(cifti_file_original)

                        # conduct spike regression and/or GSR
                        gs = np.mean(mri_data_demeaned, axis=0, keepdims=True).T
                        spike_regressor = np.loadtxt(spike_regressor_file)[
                            remove_volumes:
                        ]

                        num_spike = int(spike_regressor.sum())
                        X = np.zeros(shape=(len(spike_regressor), num_spike))
                        for k, l in enumerate(np.argwhere(spike_regressor == 1)):
                            X[l, k] = 1

                        if gsr:
                            if gs.shape[0] == X.shape[0]:
                                X = np.concatenate([gs, X], axis=1)

                        if mri_data_demeaned.shape[1] == X.shape[0]:
                            residual = mri_data_demeaned.T - X.dot(
                                np.linalg.pinv(X)
                            ).dot(mri_data_demeaned.T)

                            # save spike regressed mri data as cifti
                            img = nib.Cifti2Image(
                                residual,
                                header=(time_series, brain_model_axis),
                                nifti_header=mri_data.nifti_header,
                            )
                            img.to_filename(cifti_file_spreg)
                    # Check whether parcellation should be conducted
                    if not op.isfile(
                        replace_parcel_filename(cifti_file_original, parcellation_name)
                    ):
                        conduct_parcellation(
                            cifti_file_original,
                            parcellation_name=parcellation_name,
                            parcellation_file=parcellation_file,
                            over_write=over_write,
                        )
                    if not op.isfile(
                        replace_parcel_filename(cifti_file_spreg, parcellation_name)
                    ):
                        conduct_parcellation(
                            cifti_file_spreg,
                            parcellation_name=parcellation_name,
                            parcellation_file=parcellation_file,
                            over_write=over_write,
                        )
                    # save figures based on above output
                    if save_fig:
                        if gsr:
                            filename_suffix = "_gsr"
                        else:
                            filename_suffix = ""
                        figfile = op.join(
                            HCP_ROOT_DIR,
                            "derivatives",
                            "Python",
                            "parcellation",
                            parcellation,
                            "qc",
                            f"rms_025_spreg_{parcellation_name}",
                            f"{subject}_{run}_{parcellation_name}{filename_suffix}{fig_suffix_add}.png",
                        )
                        if not op.isfile(figfile):
                            relative_rms = np.loadtxt(
                                op.join(
                                    DATA_ROOT_DIR,
                                    subject,
                                    run,
                                    "Movement_RelativeRMS.txt",
                                )
                            )[4:]
                            before_filename = f"{run}_Atlas_MSMAll_hp2000_clean_demeaned_{parcellation_name}.ptseries.nii"
                            before_spreg = (
                                nib.load(
                                    op.join(
                                        DATA_ROOT_DIR, subject, run, before_filename
                                    )
                                )
                                .get_fdata()
                                .T
                            )

                            after_filename = f"{run}_Atlas_MSMAll_hp2000_clean_spreg_{spike_thres}_{gsr_str}demeaned_{parcellation_name}.ptseries.nii"
                            after_spreg = (
                                nib.load(
                                    op.join(DATA_ROOT_DIR, subject, run, after_filename)
                                )
                                .get_fdata()
                                .T
                            )

                            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

                            y_labels = [
                                round(i, 1)
                                for i in np.arange(0, before_spreg.shape[0], 100)
                            ]

                            ax1.plot(relative_rms)
                            ax1.axhline(
                                0.25, linestyle="--", color="black", linewidth=0.5
                            )
                            ax1.set_ylabel("relative RMS")

                            sns.heatmap(
                                data=before_spreg, ax=ax2, cmap="Greys", cbar=False
                            )
                            ax2.set_title("Before spike regression")
                            ax2.set_yticks(y_labels)
                            ax2.set_yticklabels(y_labels)

                            sns.heatmap(
                                data=after_spreg, ax=ax3, cmap="Greys", cbar=False
                            )
                            if gsr:
                                ax3_title_suffix = " and GSR"
                            else:
                                ax3_title_suffix = ""
                            ax3.set_title(f"After spike regression{ax3_title_suffix}")
                            ax3.set_yticks(y_labels)
                            ax3.set_yticklabels(y_labels)

                            sns.heatmap(
                                data=before_spreg - after_spreg,
                                ax=ax4,
                                cmap="Greys",
                                cbar=False,
                            )
                            ax4.set_title("Before - After")
                            x_labels = [round(i, 1) for i in np.arange(0, 1200, 120)]
                            ax4.set_xticks(x_labels)
                            ax4.set_xticklabels(x_labels)
                            ax4.set_yticks(y_labels)
                            ax4.set_yticklabels(y_labels)

                            fig.tight_layout()

                            fig.subplots_adjust(top=0.9)

                            if gsr:
                                title_suffix = " with GSR"
                            else:
                                title_suffix = ""
                            fig.suptitle(
                                f"{run} in {subject} using {parcellation_name}{title_suffix}"
                            )
                            print(f"Saving figure {run} in {subject}")
                            fig.savefig(figfile)

                            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gsr", action="store_true")
    parser.add_argument("--parcellation", type=str)
    parser.add_argument("--subject_ids", nargs="*", type=str)
    parser.add_argument("--save_fig", action="store_true")
    parser.add_argument("--fig_suffix_add", default="", type=str)

    inputs = parser.parse_args()
    gsr = inputs.gsr
    parcellation = inputs.parcellation
    subject_ids = inputs.subject_ids
    # For adopting slurm script (preprocess.sh)
    print(len(subject_ids[0]))
    if len(subject_ids[0]) > 6:
        subject_ids = [
            i.replace("(", "").replace(")", "") for i in subject_ids[0].split(" ")
        ]
    save_fig = inputs.save_fig
    fig_suffix_add = inputs.fig_suffix_add

    print(subject_ids)

    spike_gs_regression_mod(
        subject_list=SUBJECT_LIST,
        sessions=SESSIONS,
        parcellation=parcellation,
        gsr=gsr,
        spike_regression=True,
        spike_thres=0.25,
        over_write=True,
        remove_volumes=4,
        subject_ids=subject_ids,
        fig_suffix_add=fig_suffix_add,
    )
