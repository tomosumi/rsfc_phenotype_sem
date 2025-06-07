"""
Module for selecting models, condudcting analysis including power analysis, and visualising results
"""
from collections import defaultdict
from dataclasses import dataclass
import datetime
from functools import reduce
from itertools import combinations
from multiprocessing import cpu_count
import os
import os.path as op
from operator import add, le, ge, itemgetter
from pdb import set_trace
import re
from shutil import copy, move
from subprocess import call
from time import time
from typing import (
    Any,
    Literal,
    TypedDict,
    Annotated,
    Optional,
    Mapping,
    Union,
    Generator,
    Iterator,
    DefaultDict,
)
from warnings import warn

import hcp_utils as hcp
from joblib import Parallel, delayed
from math import ceil
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.style as mplstyle
from matplotlib_venn import venn2, venn3, venn3_circles
from mord import OrdinalRidge, LogisticAT
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import netplotbrain
import nibabel as nib
from nilearn import datasets
from nilearn.plotting import find_parcellation_cut_coords, plot_surf_stat_map, plot_img_on_surf
from nilearn.surface import vol_to_surf, load_surf_mesh
from nptyping import NDArray, Shape, Float, Int, Bool
import numexpr
import numpy as np
from numpy.linalg import inv, cholesky, LinAlgError
import numpy.ma as ma
import numpy.typing as npt
import pandas as pd
from patchworklib import load_ggplot, Brick
from plotnine import (
    aes,
    after_stat,
    coord_cartesian,
    coord_flip,
    element_blank,
    element_rect,
    element_text,
    facet_grid,
    facet_wrap,
    geom_abline,
    geom_bar,
    geom_blank,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_errorbar,
    geom_histogram,
    geom_hline,
    geom_jitter,
    geom_line,
    geom_point,
    geom_rect,
    geom_smooth,
    geom_text,
    geom_violin,
    geom_vline,
    ggplot,
    ggsave,
    ggtitle,
    guides,
    guide_legend,
    labs,
    position_dodge,
    scale_color_discrete,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_discrete,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_continuous,
    stat_smooth,
    stat_summary,
    theme,
    theme_bw,
    theme_classic,
    theme_void,
)
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy.optimize import brentq, root
from scipy.stats import t, norm, spearmanr, pearsonr, chi2, linregress
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from venn import venn

from apply_model import *
from preprocess import train_test_split_family

HCP_ROOT_DIR = "/home/cezanne/t-haitani/hcp_data"
SUBJECT_ANALYSIS_PATH = '/home/cezanne/t-haitani/hcp_data/derivatives/subjects_set_for_analysis_861.csv'

DEMO_PATH = op.join(HCP_ROOT_DIR, 'unrestricted_tomosumi_10_11_2022_1_24_36.csv')
COVARIATES_PATH = op.join(HCP_ROOT_DIR, 'derivatives', 'covariates.csv')

PARCELLATION_DIR = op.join(HCP_ROOT_DIR, "derivatives", "Python", "parcellation")
ATLAS_DIR_DICT = {
    "Schaefer": op.join(PARCELLATION_DIR, "Schaefer"),
    "Gordon": op.join(PARCELLATION_DIR, "Gordon"),
    "Glasser": op.join(PARCELLATION_DIR, "Glasser"),
}
N_RSFC_DICT = {'Schaefer': 93096, 'Gordon': 66430}

FA_ORDER_DIR_REL = '/home/cezanne/t-haitani/hcp_data/derivatives/Python/fa_result/reliability'
FA_ORDER_DIR = '/home/cezanne/t-haitani/hcp_data/derivatives/Python/fa_result/'

NODE_SUMMARY_PATH_DICT = {
    "Schaefer": "/home/cezanne/t-haitani/hcp_data/derivatives/parcellation/schaefer_s2_summary_xyz.csv",
    "Gordon": "/home/cezanne/t-haitani/hcp_data/derivatives/parcellation/gordon_333_s2_summary.csv",
    "Glasser": None,
}

NODE_SUMMARY_PATH = "/home/cezanne/t-haitani/hcp_data/derivatives/parcellation/schaefer_s2_summary_xyz.csv"
NODE_REORDERED_PATH = (
    "/home/cezanne/t-haitani/hcp_data/derivatives/parcellation/ordered_edges.pkl"
)

NEO_FFI_DIR = (
    "/home/cezanne/t-haitani/hcp_data/derivatives/Python/parcellation/Schaefer/NEO_FFI"
)
NEO_FFI_SCALES = [
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
]

NIH_COGNITION_DIR = op.join(SCHAEFER_DIR, "NIH_cognition")
NIH_COGNITION_SCALES = ["Total", "Fluid", "Crystal"]
NIH_COGNITION_SCALES_MAPPING = {'CogFluidComp_Unadj': 'Fluid', 'CogCrystalComp_Unadj': 'Crystal', 'CogTotalComp_Unadj': 'Total'}
FLUID_COGNITION = ["PicSeq", "CardSort", "Flanker", "ProcSpeed", "ListSort"]
CRYSTAL_COGNITION = ["ReadEng", "PicVocab"]
ALL_COGNITION = FLUID_COGNITION + CRYSTAL_COGNITION

ASR_BROAD_SCALES = ["Total", "Internalizing", "Externalizing", "ThoughtAttentionOthers"]
ASR_BROAD_SCALES_MOD = ["All", "Internalizing", "Externalizing", "Others"]
ASR_INT_SCALES_PUB = ["Anxious/Depressed", "Withdrawn", "Somatic complaints"]
ASR_EXT_SCALES_PUB = ["Aggressive behavior", "Rule Breaking Behavior", "Intrusive"]
ASR_OTHER_SCALES_PUB = ["Thought problems", "Attention problems", "Other problems"]
ASR_ALL_SCALES_PUB = ASR_INT_SCALES_PUB + ASR_EXT_SCALES_PUB + ASR_OTHER_SCALES_PUB

SUBSCALE_N_DICT = {
        'NIH_Cognition': {'Total': 7, 'Fluid': 5, 'Crystal': 2},
        'ASR': {'All': 9, 'Internalizing': 3, 'Externalizing': 3, 'Others': 3},
        'NEO_FFI': {'Neuroticism': 12, 'Extraversion': 12, 'Openness': 12, 'Agreeableness': 12, 'Conscientiousness': 12}
        }

ASR_INT_SCALES = ["ASR_Anxd_Raw", "ASR_Witd_Raw", "ASR_Soma_Raw"]
ASR_EXT_SCALES = ["ASR_Aggr_Raw", "ASR_Rule_Raw", "ASR_Intr_Raw"]
ASR_OTHER_SCALES = ["ASR_Thot_Raw", "ASR_Attn_Raw", "ASR_Oth_Raw"]
ASR_ALL_SCALES = ASR_INT_SCALES + ASR_EXT_SCALES + ASR_OTHER_SCALES
ASR_DICT = {'All': ASR_ALL_SCALES, 'Internalizing': ASR_INT_SCALES, 'Externalizing': ASR_EXT_SCALES, 'Others': ASR_OTHER_SCALES}

NEO_FFI_DICT = defaultdict(dict)
NEO_FFI_SCALES = [
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
]

for i, scale_name in enumerate(NEO_FFI_SCALES):
    remainder = 0 if i == 4 else i + 1
    NEO_FFI_DICT[scale_name] = [f'NEORAW_{j:02}' for j in range(1, 61) if j % 5 == remainder]


SCALES_DICT = {
        'both': {
            'NIH_Cognition': {
                'Total': [i + '_Unadj' for i in ALL_COGNITION],
                'Fluid': [i + '_Unadj' for i in FLUID_COGNITION],
                'Crystal': [i + '_Unadj' for i in CRYSTAL_COGNITION]
                },
            'ASR': ASR_DICT,
            'NEO_FFI': NEO_FFI_DICT
        },
        'fc': {
            'NIH_Cognition': {
                'Total': ['CogTotalComp_Unadj'],
                'Fluid': ['CogFluidComp_Unadj'],
                'Crystal': ['CogCrystalComp_Unadj']
                },
            'ASR': {'All': ['All'], 'Internalizing': ['Internalizing'], 'Externalizing': ['Externalizing'], 'Others': ['Others']},
            'NEO_FFI': {
                "Neuroticism": ['Neuroticism'],
                "Extraversion": ['Extraversion'],
                "Openness": ['Openness'],
                "Agreeableness": ['Agreeableness'],
                "Conscientiousness": ['Conscientiousness']
                }
            }
        }

FIT_INDICES = [
    "DoF",
    "DoF_baseline",
    "chi2",
    "chi2_pvalue",
    "chi2_baseline",
    "CFI",
    "GFI",
    "AGFI",
    "NFI",
    "TLI",
    "RMSEA",
    "AIC",
    "BIC",
    "LogLik",
    "SRMR",
]

FIT_INDICES_OP_DICT = {
    "DoF": None,
    "DoF_baseline": None,
    "chi2": None,
    "chi2_pvalue": le,
    "chi2_baseline": None,
    "CFI": ge,
    "GFI": ge,
    "AGFI": ge,
    "NFI": ge,
    "TLI": ge,
    "RMSEA": le,
    "AIC": le,
    "BIC": le,
    "LogLik": le,
    "SRMR": le,
}

MODEL_TRAIT = ["model_fc", "model_trait", "model_both"]
MODEL_TRAIT_STR = [i.replace("model_", "") for i in MODEL_TRAIT]
MODEL_POSITION_DICT = {"fc": 0, "trait": 1, "both": 2}

pd.options.mode.chained_assignment = None  # default='warn'

FitIndicesList = Literal[tuple(FIT_INDICES)]
GSR_DICT = {"nogs": "Without GSR", "gs": "With GSR"}
DROP_DICT = {"not_dropped": "Original model", "dropped": "Modified model"}
DROP_DICT2 = {False: "Original", True: "Modified"}
MODEL_DICT = {
    "model_fc": "Model FC",
    "model_trait": "Model trait",
    "model_both": "Model FC and trait",
}
MODEL_DICT_ADD_MEAN_PCA = {
    "mean": "Mean",
    "pca": "PCA",
    "model_fc": "Model FC",
    "model_trait": "Model trait",
    "model_both": "Model FC and trait",
}
FIT_INDICES_PUB = ["SRMR", "RMSEA", "CFI"]

TRAIT_TYPES = ["cognition", "mental", "personality"]

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
    'No': 13,
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

NETWORK_ORDER_FOR_PUB_NESTED_LIST = {
        'Schaefer': list(NETWORK_ORDER_DICT_PUB_SCHAEFER.keys()),
        'Gordon': list(NETWORK_ORDER_DICT_PUB_GORDON.keys())
        }

TIAN_ATLAS_DIR = (
    "/home/cezanne/t-haitani/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/"
)

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

N_NODES_DICT = {'Schaefer': 432, 'Gordon': 365}

class ParamPositionDict(TypedDict):
    cor_position: int
    fc_load_positions: list[int]
    trait_load_positions: list[int]
    fc_error_positions: list[int]
    trait_error_positions: list[int]


TraitType = Literal["cognition", "personality", "mental"]
TraitScaleName = Literal["NIH_Cognition", "NEO_FFI", "ASR"]


def calc_icc(
        fc_filename_dict,
        sub_list='spreg_0.25_N_861_rms_percentage_0.1.csv',
        parcellation='Schaefer',
        return_df=False
        ):
    """
    Calculate ICC from numpy array
    fc_filename should be concatenated by days
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    sub_bool = np.loadtxt(op.join(HCP_ROOT_DIR, 'derivatives', sub_list)).astype(int)
    
    edges_df = get_edge_summary(
            parcellation=parcellation, 
            network_hem_order=True
            )
    nodes_df = get_nodes_df(parcellation=parcellation)
    
    if not return_df:
        mat_df = get_empty_df_for_hmap(parcellation)
        fig, ax = plt.subplots()

    for gsr_type in ['nogs', 'gs']:
        print(f'Processing {gsr_type} condition')
        fc_filename = fc_filename_dict.get(gsr_type)
        fc_array = np.load(op.join(atlas_dir, fc_filename))
        fc_array = fc_array[:, np.where(sub_bool)[0], :]
        fc_array = np.arctanh(fc_array)
        icc_array = np.empty(shape=len(fc_array))

        for i in range(fc_array.shape[0]):
            fc_data = fc_array[i, ...]
            T = fc_data.sum()
            T_i = fc_data.sum(axis=1)
            T_j = fc_data.sum(axis=0)
            ## calculate mean squares from sum of squares and degrees of freedom
            # MS_T
            SS_T = (fc_data**2).sum() - (T**2) / fc_data.size
            dof_T = fc_data.size - 1
            # MS_A
            SS_A = np.sum((T_i**2) / fc_data.shape[1]) - (T**2) / fc_data.size
            dof_A = fc_data.shape[0] - 1
            MS_A = SS_A / dof_A
            # MS_B
            SS_B = np.sum((T_j**2) / fc_data.shape[0]) - (T**2) / fc_data.size
            dof_B = fc_data.shape[1] - 1
            MS_B = SS_B / dof_B
            # MS_AB
            SS_AB = np.sum(fc_data**2) - (T**2) / fc_data.size - SS_A - SS_B
            dof_AB = dof_A * dof_B
            MS_AB = SS_AB / dof_AB
            # MS_E
            SS_E = SS_T - SS_A - SS_B - SS_AB
            dof_E = fc_data.size - fc_data.shape[0] * fc_data.shape[1]
            MS_E = 0 if dof_E == 0 else SS_E / dof_E
            # Calculate ICC
            icc_21 = (MS_A - MS_AB) / (MS_A + (fc_data.shape[1]-1)*MS_AB + (fc_data.shape[1]/fc_data.shape[0]) * (MS_B - MS_AB))
            icc_array[i] = icc_21
        if not return_df:
            edges_df['icc'] = icc_array
            wide_df = get_wide_df_hmap(edges_df, value_col_name='icc')
            mat_df = fill_mat_df(mat_df, wide_df, gsr_type)
        else:
            edges_df[f'{gsr_type}_icc'] = icc_array
    
    if return_df:
        return edges_df

    print('Drawing heatmap')
    draw_hmaps_fcs(
        mat_df,
        nodes_df,
        cmap='Oranges',
       # save_dir=folder,
    #    save_filename=None,
        ax=ax,
        parcellation=parcellation
       # cbar_ax=cbar_ax,
       # vmin=hmap_vmin,
       # vmax=hmap_vmax,
       # iteration=iteration,
    )


def set_font(font="Arial"):
    """
    Set font to draw figure for publication
    """
    # matplotlib.rc('font', family=font)
    mpl.rc("font", family="sans-serif")
    mpl.rc("font", serif=font)
    mpl.rc("text", usetex=False)


def replace_and_reorder_column(
    df, var_name=None, var_dict=None, var_name_dict: dict[str:dict] = None
):
    """replace and reorder values in df for visualization"""

    def replace_var_name_dict(var_name, var_dict):
        df[var_name] = df[var_name].replace(var_dict)
        df[var_name] = pd.Categorical(df[var_name], categories=var_dict.values())
        return df

    if var_name_dict is not None:
        for var_name, var_dict in var_name_dict.items():
            df = replace_var_name_dict(var_name, var_dict)
    else:
        df = replace_var_name_dict(var_name, var_dict)
    return df


def rename_scale_names_of_parcellation(
    parent_dir: str,
    original_list: list[str],
    new_list: list[str],
    trait_scale_name: str,
    target_subdirs: list[str],
):
    """function for renaming scale names of files on output file based on parcellation"""
    for i, new_scale_name in enumerate(new_list):
        target_dir = op.join(parent_dir, trait_scale_name, new_scale_name)
        for target_subdir in target_subdirs:
            target_dir_rename = op.join(target_dir, target_subdir)
            target_files = os.listdir(target_dir_rename)
            for target_file in target_files:
                os.rename(
                    op.join(target_dir_rename, target_file),
                    op.join(target_dir_rename, target_file).replace(
                        "_" + original_list[i] + "_", "_" + new_scale_name + "_"
                    ),
                )


def rename_scale_names_of_fa_parameters(
    parent_dir: str,
    original_list: list[str],
    new_list: list[str],
    trait_scale_name: str,
):
    """function for renaming scale names of files on parameters of factor models"""
    target_dir = op.join(parent_dir, trait_scale_name)
    target_files = (
        file_
        for file_ in os.listdir(target_dir)
        if op.isfile(op.join(target_dir, file_))
    )
    for target_file in target_files:
        for i, new_scale_name in enumerate(new_list):
            target_file_replaced = re.sub(
                f"^{original_list[i]}_", new_scale_name + "_", target_file
            )
            target_file_with_dir = op.join(target_dir, target_file)
            if op.isfile(target_file_with_dir):
                os.rename(
                    target_file_with_dir, op.join(target_dir, target_file_replaced)
                )


def get_strings_from_filename(
    filename: str, var_list=None, include_nofa_model=True
) -> Union[Iterator[str], dict[str]]:
    """get objects from filename of memmap file"""
    string_dict = {}
    string_dict["trait_type"] = reduce(
        add, re.findall("Trait_[a-zA-Z]+_Scale", filename)
    ).split("_")[1]
    scale_name = reduce(add, re.findall("Scale_[a-zA-Z]+", filename)).split("_")[1]
    string_dict["scale_name"] = scale_name

    for none_key in ["trait_type", "scale_name"]:
        if string_dict[none_key] == "None":
            string_dict[none_key] = None

    string_dict["trait_scale_name"] = get_scale_name_from_trait(
        string_dict["trait_type"]
    )

    string_dict["sample_n"] = reduce(add, re.findall("sampleN_[0-9]+", filename)).split(
        "_"
    )[1]

    if filename.startswith(("pearson", "spearman")):
        string_dict["data_type"] = "correlation"
    elif filename.startswith("fit_indices"):
        string_dict["data_type"] = "fit_indices"
    elif filename.startswith("factor_scores"):
        string_dict["data_type"] = "factor_scores"
    elif filename.startswith("params"):
        string_dict["data_type"] = "parameters"
    elif filename.startswith('Model_implied_vcov'):
        string_dict['data_type'] = 'model_vcov'

    if filename.startswith("pearson"):
        string_dict["cor_type"] = "pearson"
    elif filename.startswith("spearman"):
        string_dict["cor_type"] = "spearman"

    string_dict["num_iter"] = int(
        reduce(add, re.findall("edgeN_[0-9]+", filename)).split("_")[1]
    )

    string_dict["gsr_type"] = reduce(add, re.findall("gs|nogs", filename))
    # MSST may be modified
    if '_both_' in filename or '_fc_' in filename or 'onlyFC' in filename:
        control_end_str = 'MSST' if 'MSST' in filename else 'OutStd'
    elif '_mean_' in filename:
        control_end_str = 'session'
    if 'controlling_' in filename:
        string_dict["control"] = reduce(
            add, re.findall(f"controlling_[a-z]+.*{control_end_str}", filename)
        ).split("_")[1:-1]
    elif 'controllingBefore' in filename:
        try:
            string_dict["control_before"] = reduce(
                add, re.findall(f"controllingBefore_[a-z]+.*{control_end_str}", filename)
            ).split("_")[1:-1]
        except:
            string_dict['control_before'] = None

    if not 'controlling_' in filename:
        string_dict["control"] = None
    else:
        if reduce(add, string_dict["control"]) == "none":
            string_dict["control"] = None
    
    string_dict["cov_cor"] = True if "CovCor" in filename else False
    string_dict["phase_encoding"] = True if "PE_" in filename else False
    string_dict["order_in_day"] = True if "OrderInDay" in filename else False
    string_dict["use_lavaan"] = True if "lavaan" in filename else False
    string_dict["day_cor"] = True if "DayCor" in filename else False
    string_dict["add_marker"] = True if "Marker" in filename else False
    string_dict['multistate_single_trait'] = True if 'MultiStateSingleTrait' in filename or 'MSST' in filename else False
    string_dict['bi_factor'] = True if 'Bifactor' in filename else False
    string_dict['add_CU'] = True if '_CU_' in filename else False
    string_dict['mean_structure'] = True if 'MeanStr' in filename else False
    if "_day_" in filename:
        string_dict["fc_unit"] = "day"
    elif "_session_" in filename:
        string_dict["fc_unit"] = "session"
    else:
        string_dict["fc_unit"] = None

    model_names = reduce(add, re.findall("Model_.*_Trait", filename))
    if len(model_names) > 0:
        model_names = model_names.replace("_Trait", "").split("_")
        model_names = [i for i in model_names if not i in ['Model', 'implied', 'vcov']]
        if include_nofa_model:
            string_dict["model_type"] = [
                "model_" + i if i not in ["mean", "pca"] else i for i in model_names
            ]
        else:
            string_dict["model_type"] = [
                "model_" + i for i in model_names if i not in ["mean", "pca"]
            ]

    # get dropped variables
    if "drop" in filename:
        drop_str = reduce(add, re.findall("drop_(.*)_\d{4}", filename))
        if string_dict["trait_type"] != "cognition":
            drop_vars_list = drop_str.split("_")
        else:
            drop_vars_list = drop_str.replace("_Unadj", "").split("_")
            drop_vars_list = [i + "_Unadj" for i in drop_vars_list]
    else:
        drop_vars_list = None
    string_dict["drop_vars_list"] = drop_vars_list

    string_dict["fixed_load"] = True if "FixedLoad" in filename else False

    string_dict["date_time"] = reduce(
        add,
        re.findall("_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", filename),
    )

    if var_list is not None:
        string_dict_gen = tuple(string_dict[var] for var in var_list)
        return string_dict_gen
    return string_dict


def copy_memmap_output_data(
    filename_input: str,
    dtype_memmap="float32", 
    n_array=None,
    parcellation='Schaefer',
    search_trash=False,
    family_cv=False
) -> Union[
    NDArray[Shape["Num_iter, Num_models"], Float],
    NDArray[Shape["Num_iter, 15, Num_modelfa"], Float],
    NDArray[Shape["Num_iter, N_sample, 2"], Float],
    NDArray[Shape["Num_iter, 2"], Float],
]:
    """
    function for reading output data (memmap file with .dat extension))
    extension (.dat) is not necesary for 'filename'
    filename should be specified outside the function
    """
    if not ("Trait_None" in filename_input):
        trait_type, scale_name = get_strings_from_filename(
            filename_input,
            ["trait_type", "scale_name"],
        )
        folder = op.join(get_scale_name_from_trait(trait_type), scale_name)
    else:
        folder = "reliability"

    (data_type, model_type, num_iter, sample_n, use_lavaan) = get_strings_from_filename(
        filename_input,
        ["data_type", "model_type", "num_iter", "sample_n", "use_lavaan"],
    )
    # modify variable because of long filename
    use_lavaan = True
    if n_array is not None:
        num_iter = int(num_iter / n_array)

    if data_type == "correlation":
        memmap_shape = (num_iter, len(model_type))
        if (
            len(reduce(add, re.findall("sampleN_(.*)_Fold", filename_input)).split("_"))
            == 2
        ):
            memmap_shape = (num_iter, 2)

    elif data_type == "fit_indices":
        model_fas = get_factor_models(model_type)
        model_num = len(model_fas)
        memmap_shape = (num_iter, 15, model_num)

    elif data_type == "factor_scores":
        num_dim2 = 2 if not "Trait_None" in filename_input else 1
        if ('MultiStateSingleTrait' in filename_input) or ('MSST' in filename_input) or ('Bifactor' in filename_input):
            num_dim2 += 2
        memmap_shape = (num_iter, int(sample_n), num_dim2)

    elif data_type == "parameters":
        param_num, var_num = get_param_num_from_filename(filename_input)
        ncol = 3 if not use_lavaan else 5
        memmap_shape = (num_iter, param_num, ncol)

    elif data_type in ['residuals', 'model_vcov']:
        _, var_num = get_param_num_from_filename(filename_input)
        memmap_shape = (num_iter, var_num, var_num)
#    else:
#        raise NameError('filename should include "correlation" or "fit_indices".')
    parent_dir = op.join(ATLAS_DIR_DICT.get(parcellation), folder, data_type)
    if search_trash:
        parent_dir = op.join(parent_dir, 'trash')
    if not 'combine' in filename_input:
        data = np.memmap(
            filename=op.join(parent_dir, filename_input),
            mode="r",
            dtype=dtype_memmap,
            shape=memmap_shape,
        )
        data_copied = data.copy()
        return np.array(data_copied)
    else:
        target_dir = op.join(parent_dir, 'combined')
        if family_cv:
            target_dir = op.join(target_dir, 'split_half_cv')
        return np.load(op.join(target_dir, filename_input))


def get_index_of_r_within_range(
    data, min_r: float, max_r: float
) -> NDArray[Shape["*, *"], Bool]:
    """
    function for getting index for r values which is out of range (<= -1 or >=1)
    """
    boolean_r_within_range = (min_r < data) & (data < max_r)
    return boolean_r_within_range


def print_r_out_of_range_n(
    boolean_r_within_range, model_fa_list, num_iter, min_r, max_r
):
    """function for printing number of values which is out of range"""
    # caluclate number of correlations which is out of range per column
    r_within_range_ns = np.sum(boolean_r_within_range, axis=0)
    for i, model in enumerate(model_fa_list):
        r_out_of_range_n = num_iter - r_within_range_ns[i]
        print(
            f"{model.capitalize()}: Number of correlation values out of range (smaller than {min_r} or greater than {max_r}) is {r_out_of_range_n}/{num_iter}."
        )


ModelFA = Literal["model_fc", "model_trait", "model_both", "model_onlyFC"]


def get_param_num_from_filename(filename):
    """
    Get number of parameters from filename
    """
    (
        model_type,
        control,
        num_iter,
        gsr_type,
        cov_cor,
        day_cor,
        phase_encoding,
        use_lavaan,
        trait_type,
        scale_name,
        fc_unit,
        drop_vars_list,
        fixed_load,
        order_in_day,
        multistate_single_trait,
        bi_factor,
        add_CU,
        add_marker,
        mean_structure
    ) = get_strings_from_filename(
        filename,
        [
            "model_type",
            "control",
            "num_iter",
            "gsr_type",
            "cov_cor",
            "day_cor",
            "phase_encoding",
            "use_lavaan",
            "trait_type",
            "scale_name",
            "fc_unit",
            "drop_vars_list",
            "fixed_load",
            "order_in_day",
            "multistate_single_trait",
            "bi_factor",
            'add_CU',
            'add_marker',
            'mean_structure'
        ],
        include_nofa_model=False,
    )

    if type(model_type) is list and len(model_type) == 1:
        model_type = model_type[0]

    param_num, var_num = calculate_param_num(
        model_type,
        control,
        cov_cor,
        phase_encoding,
        day_cor,
        use_lavaan,
        fc_unit,
        trait_type,
        scale_name,
        drop_vars_list,
        return_nvars=True,
        order_in_day=order_in_day,
        multistate_single_trait=multistate_single_trait,
        bi_factor=bi_factor,
        add_CU=add_CU,
        add_method_marker=add_marker,
        mean_structure=mean_structure
    )
    return param_num, var_num


def get_parameters_from_filename(filename):
    """
    Get parameters from filename
    """
    param_num, var_num = get_param_num_from_filename(filename)
    if trait_type is None:
        folder = "reliability"
    else:
        trait_scale_name = select_folder_from_trait(trait_type)
        folder = op.join(trait_scale_name, scale_name)


def generate_params_dict(
    filename: str,
    dtype_memmap="float32",
    residuals=False,
    n_arrays=None,
    parcellation='Schaefer',
    family_cv=False,
#    read_combined_data=False,
    **kwargs,
) -> dict[ModelFA : NDArray[Shape["Num_iter, Num_param, 3"], Float]]:
    """
    function for copying mammap file storing parameter estimates
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    (
        model_type,
        control,
        num_iter,
        gsr_type,
        cov_cor,
        day_cor,
        phase_encoding,
        use_lavaan,
        trait_type,
        scale_name,
        fc_unit,
        drop_vars_list,
        fixed_load,
        order_in_day,
        add_marker
    ) = get_strings_from_filename(
        filename,
        [
            "model_type",
            "control",
            "num_iter",
            "gsr_type",
            "cov_cor",
            "day_cor",
            "phase_encoding",
            "use_lavaan",
            "trait_type",
            "scale_name",
            "fc_unit",
            "drop_vars_list",
            "fixed_load",
            "order_in_day",
            "add_marker"
        ],
        include_nofa_model=False,
    )
    if trait_type is None:
        folder = "reliability"
    else:
        trait_scale_name = select_folder_from_trait(trait_type)
        folder = op.join(trait_scale_name, scale_name)
    # create empty dictionary storing parameter outputs
    if n_arrays:
        num_iter = int(num_iter / n_arrays)
    params_dict = {}
    ncol = 3 if not use_lavaan else 5
    filename_after_model_type = reduce(add, re.findall("Trait_.*", filename))
    # reading data per model
    for model in model_type:
        # set number of parameters
        param_num, var_num = calculate_param_num(
            model,
            control,
            cov_cor,
            phase_encoding,
            day_cor,
            use_lavaan,
            fc_unit,
            remove_vars_list=drop_vars_list,
            trait_type=trait_type,
            scale_name=scale_name,
            return_nvars=True,
            order_in_day=order_in_day,
            add_method_marker=add_marker
        )
        # read a memmap file
        model_string = model.replace("model", "Model")
        # if trait_type is not None else model
        if not residuals:
            prefix, memmap_shape, parent_folder = (
                "params_",
                (num_iter, param_num, ncol),
                "parameters",
            )
        else:
            prefix, memmap_shape, parent_folder = (
                "std_residuals_",
                (num_iter, var_num, var_num),
                "residuals",
            )

        param_dir = op.join(atlas_dir, folder, parent_folder)
        combined = 'combine' in filename
        if combined:
            param_dir = op.join(param_dir, 'combined')
            if family_cv:
                param_dir =op.join(param_dir, 'split_half_cv')
        if model in ["model_fc", "model_trait"]:
            fixed_load_str = "_FixedLoad" if fixed_load else ""
            # this part should be modified when more than two files are stored in a directory
            filename_list = [
                i
                for i in os.listdir(param_dir)
                if model.replace("model", "Model") in i
                and fixed_load_str in i
                and f"_{gsr_type}_" in i
            ]
            if kwargs.get("drop") is not None:
                if kwargs.get("drop") == "drop":
                    filename_list = [
                        i for i in filename_list if "drop" in i
                    ]
                else:
                    filename_list = [
                        i for i in filename_list if "drop" not in i
                    ]

            if len(filename_list) == 1:
                filename_read = filename_list[0]
            else:
                raise Exception("Target filename could not be identified.")
        elif model in ["model_both", 'model_onlyFC']:
            filename_read = f"{prefix}{model_string}_{filename_after_model_type}"

        if not 'combined' in filename_read:
            param_data = np.memmap(
                filename=op.join(
                    param_dir,
                    filename_read,
                ),
                dtype=dtype_memmap,
                mode="r",
                shape=memmap_shape,
            )
            param_data_copied = param_data.copy()
        else:
            param_data_copied = np.load(op.join(param_dir, filename_read))
        # insert parameter
        params_dict[model] = np.array(param_data_copied)
    return params_dict


def add_drop_vars_to_specified_files(
    trait_type, drop_vars_dict, start_end_datetime_dict, n_edge: int, sample_n: int
):
    """
    add strings representing dropped variables to filename with specific time
    """
    trait_scale_name = get_scale_name_from_trait(trait_type)
    save_dirs = [
        "parameters",
        "factor_scores",
        "residuals",
        "fit_indices",
        "correlation",
    ]
    datetime_start = datetime.datetime(*start_end_datetime_dict["start"])
    datetime_end = datetime.datetime(*start_end_datetime_dict["end"])

    trait_scale_dir = op.join(SCHAEFER_DIR, trait_scale_name)

    for key, item in drop_vars_dict.items():
        subscale_dir = op.join(trait_scale_dir, key)
        for save_dir in save_dirs:
            subscale_save_dir = op.join(subscale_dir, save_dir)
            for filename in os.listdir(subscale_save_dir):
                if str(n_edge) in filename and str(sample_n) in filename:
                    print(filename)
                    datetime_file = subset_datetime_from_filename(filename)
                    if datetime_start < datetime_file < datetime_end:
                        drop_suffix = generate_drop_suffix(trait_type, item)
                        datetime_str = datetime_file.strftime("%Y-%m-%d %H:%M:%S")
                        if not drop_suffix in filename:
                            filename_new = (
                                filename.replace(
                                    "_" + datetime_str + ".dat", drop_suffix
                                )
                                + "_"
                                + datetime_str
                                + ".dat"
                            )
                            os.rename(
                                op.join(subscale_save_dir, filename),
                                op.join(subscale_save_dir, filename_new),
                            )


def get_param_position_dict(
    filename: str,
) -> ParamPositionDict:
    """function for getting dictionary of parameter positions"""
    (
        model_type,
        control,
        num_iter,
        gsr_type,
        cov_cor,
        day_cor,
        phase_encoding,
        use_lavaan,
        trait_type,
        scale_name,
        fc_unit,
        drop_vars_list,
        fixed_load,
        order_in_day,
        add_marker,
    ) = get_strings_from_filename(
        filename,
        [
            "model_type",
            "control",
            "num_iter",
            "gsr_type",
            "cov_cor",
            "day_cor",
            "phase_encoding",
            "use_lavaan",
            "trait_type",
            "scale_name",
            "fc_unit",
            "drop_vars_list",
            "fixed_load",
            "order_in_day",
            "add_marker",
        ],
        include_nofa_model=False,
    )

    param_order_file = generate_param_order_filename(
        control,
        model_type[0],
        cov_cor,
        phase_encoding,
        day_cor,
        use_lavaan,
        fc_unit,
        trait_type,
        scale_name,
        drop_vars_list,
        fix_loadings_to_one=fixed_load,
        order_in_day=order_in_day,
        add_method_marker=add_marker,
    )
    param_position_dict: ParamPositionDict = get_param_positions(
        param_order_file, use_lavaan, control, model_type[0], fc_unit
    )

    return param_position_dict


def get_parameters(
    params_input,
    #    : Union[
    #        dict[ModelFA : NDArray[Shape["Num_iter, Num_param, 3"], Float]],
    #        NDArray[Shape["Num_iter, Num_param, 3"], Float]
    #        ],
    model: str,
    param_position_dict: ParamPositionDict,
    get_load=False,
    get_cor=True,
    target="std_estimates",
    add_marker=False,
) -> tuple[
    NDArray[Shape["Num_iter, Num_fc_error_param"], Float],
    NDArray[Shape["Num_iter, Num_trait_error_param"], Float],
    Optional[NDArray[Shape["Num_iter, 1"], Float]],
]:
    """
    function for getting values of error variances of indicators and
    inter-factor correlation for evaluating local model misspecification
    """
    if target == "unstd_se":
        array_index = 2
    elif target == "std_estimates":
        array_index = 0
    elif target == "unstd_estimates":
        array_index = 1

    param_str = "load" if get_load else "error"
    if type(params_input) is dict:
        params_array = params_input[model][:, :, array_index]
    elif type(params_input) is np.ndarray:
        params_array = params_input[:, :, array_index]

    trait_exist = model in MODEL_TRAIT
    if trait_exist:
        fc_param = params_array[
            :, [int(i) for i in param_position_dict[f"fc_{param_str}_positions"]]
        ]
        trait_param = params_array[
            :, [int(i) for i in param_position_dict[f"trait_{param_str}_positions"]]
        ]
        method_param = params_array[
            :, [int(i) for i in param_position_dict[f"fc_cov_positions"]]
        ]

    if "day" in model:
        fc_param_day1 = params_array[
            :, [int(i) for i in param_position_dict[f"fc_{param_str}_positions_day1"]]
        ]
        fc_param_day2 = params_array[
            :, [int(i) for i in param_position_dict[f"fc_{param_str}_positions_day2"]]
        ]
        fc_param = (fc_param_day1, fc_param_day2)
        trait_param = None
    if "onlyFC" in model:
        fc_param = params_array[
            :, [int(i) for i in param_position_dict[f"fc_{param_str}_positions"]]
        ]
        trait_param = None
        dict_key = "fc_cov_positions" if not add_marker else "fc_marker_positions"
        method_param = params_array[:, [int(i) for i in param_position_dict[dict_key]]]

    if get_cor and not "onlyFC" in model:
        cor_param = params_array[:, param_position_dict["cor_position"]]
    else:
        cor_param = None
    return fc_param, trait_param, cor_param, method_param


class ErrorVarsMinMaxDict(TypedDict):
    minimum_fc_error_var: float
    maximum_fc_error_var: float
    minimum_trait_error_var: float
    maximum_trait_error_var: float


class CorMinMaxDict(TypedDict):
    minimum: float
    maximum: float


def get_specified_models(
    params_dict: dict[ModelFA : NDArray[Shape["Num_iter, Num_param, Ncol"], Float]],
    model: ModelFA,
    param_position_dict: ParamPositionDict,
    error_vars_dict: ErrorVarsMinMaxDict,
    cor_min_max_dict: CorMinMaxDict = None,
) -> NDArray[Shape["Num_iter, 4"], Bool]:
    """
    function for getting booleans representing whether models are specified in terms of fc and trait error variances
    """
    # get error variances of fc and trait indicators
    fc_error_vars, trait_error_vars, cor, method_param = get_parameters(
        params_dict, model, param_position_dict
    )
    boolean_locally_misspecified_array = examine_misspecified_model_from_parameters(
        fc_error_vars,
        cor,
        error_vars_dict,
        trait_error_vars,
        cor_min_max_dict,
        print_error=False,
    )

    # reverse booleans
    boolean_locally_specified_array = ~boolean_locally_misspecified_array
    return boolean_locally_specified_array.T


FitLocations = Literal["fc", "trait", "cor", "all"]


def get_edges_of_locally_misspecified_models(
    boolean_locally_misspecified_array: NDArray[
        Shape["Num_iter, Num_modelfa, Ncol"], Bool
    ],
    model_num: int,
    model,
) -> dict[FitLocations, set[int]]:
    """function for getting edges in specified models"""
    # create dictionary of specified edges
    boolean_locally_misspecified_array = ~boolean_locally_misspecified_array
    if "fc" in model or "trait" in model or "both" in model:
        dict_keys = ["fc", "trait", "cor", "all"]
    elif "onlyFC" in model:
        dict_keys = ["all"]
    elif "day" in model:
        dict_keys = ["fc", "cor", "all"]
    locally_misspecified_edge_dict = {
        key: set(
            reduce(add, np.where(boolean_locally_misspecified_array[:, model_num, i]))
        )
        for i, key in enumerate(dict_keys)
    }
    return locally_misspecified_edge_dict


def get_misfit_edges_locally_and_globally(
    boolean_local,
    boolean_global,
    model_fa_list,
):
    """function for getting sets of edges passing local and global fit criteria"""
    # get boolean representing local fit
    misfit_edges_dict = defaultdict(dict)
    array_edges_local, array_models_local = np.where(~boolean_local[:, :, 3])
    array_edges_global, array_models_global = np.where(~boolean_global)

    for i, model in enumerate(model_fa_list):
        local_index = array_models_local == i
        global_index = array_models_global == i
        misfit_edges_dict[model]["local"] = set(array_edges_local[local_index])
        misfit_edges_dict[model]["global"] = set(array_edges_global[global_index])
    return misfit_edges_dict


def get_model_strings(filename) -> tuple[list[ModelFA], str, list[str]]:
    """function for getting strings representing model names"""
    string_dict = get_strings_from_filename(filename)
    model_type = string_dict["model_type"]
    model_fa_list = [i for i in model_type if "model" in i]
    models_fa_str = "Model_" + "_".join(
        [i.replace("model_", "") for i in model_fa_list]
    )
    model_nofa_list = [i for i in model_type if "model" not in i]
    return model_fa_list, models_fa_str, model_nofa_list


def select_models_from_local_fits(
    filename_cor: str,
    error_vars_dict,
    model_type_list,
    cor_min_max_dict=None,
    dtype_memmap="float32",
) -> NDArray[Shape["Num_iter, Num_modelfa, Ncol"], Bool]:
    """
    function for selecting models which passed local and global fit criteria
    """
    string_dict = get_strings_from_filename(filename_cor)
    # if 'Trait_None' in filename_cor:
    #     trait_type, scale_name = string_dict["trait_type"], string_dict['scale_name']
    #     ncol = 4
    # else:
    #     trait_type, scale_name = None, None
    #     ncol = 1 if cor_min_max_dict is None else 3

    filename_after_cor_type = reduce(add, re.findall("Model_.*", filename_cor))
    (
        num_iter,
        control,
        cov_cor,
        phase_encoding,
        day_cor,
        use_lavaan,
        trait_type,
        scale_name,
        fc_unit,
        drop_vars_list,
    ) = (
        string_dict["num_iter"],
        string_dict["control"],
        string_dict["cov_cor"],
        string_dict["phase_encoding"],
        string_dict["day_cor"],
        string_dict["use_lavaan"],
        string_dict["trait_type"],
        string_dict["scale_name"],
        string_dict["fc_unit"],
        string_dict["drop_vars_list"],
    )
    if trait_type is None:
        ncol = 1 if cor_min_max_dict is None else 3
    else:
        ncol = 4

    # model_fa_list = get_model_strings(filename_cor)[0]

    ## identify and explore locally misspecified models
    # order of axis 2 is fc_var, trait_var, cor, overall

    boolean_locally_specified_array = np.empty(
        shape=(num_iter, len(model_type_list), ncol), dtype="bool"
    )
    params_dict = generate_params_dict(
        filename_after_cor_type,
    )
    for i, model in enumerate(model_type_list):
        # generate boolean index representing model specification
        param_position_dict = get_param_position_dict(
            control,
            model,
            cov_cor,
            phase_encoding,
            day_cor,
            use_lavaan,
            fc_unit,
            trait_type,
            scale_name,
        )
        boolean_locally_specified_array_per_model = get_specified_models(
            params_dict, model, param_position_dict, error_vars_dict, cor_min_max_dict
        )
        boolean_locally_specified_array[
            :, i, :
        ] = boolean_locally_specified_array_per_model.T

    return boolean_locally_specified_array


def get_n_of_local_misfit(boolean_locally_specified_array):
    """function for getting number of locally misfit models"""
    locally_misspecified_ns = boolean_locally_specified_array.shape[0] - np.sum(
        boolean_locally_specified_array, axis=0
    )
    return locally_misspecified_ns


def get_strings_of_thresholds_of_global_fit_indices(
    fit_indices_thresholds_dict: dict,
) -> str:
    """
    function for getting strings representing thresholds of global fit indices
    """
    str_global_fit_thresholds = ""
    for item in fit_indices_thresholds_dict.items():
        operation = FIT_INDICES_OP_DICT[item[0]].__name__
        if operation == "le":
            output_str = item[0] + " < " + str(item[1])
        elif operation == "ge":
            output_str = item[0] + " > " + str(item[1])
        else:
            raise NameError('operation should be "le" or "ge".')
        str_global_fit_thresholds = str_global_fit_thresholds + ", " + output_str
    # remove first new line
    return str_global_fit_thresholds[1:]


def generate_caption_of_figure_on_parameter_thresholds(
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
) -> str:
    """
    function for generating caption of figure from thresholds of parameters
    """
    fc_errors_caption = f'{error_vars_dict["minimum_fc_error_var"]} < Error variance of indicators of FC < {error_vars_dict["maximum_fc_error_var"]}'
    trait_error_caption = f'{error_vars_dict["minimum_trait_error_var"]} < Error variance of indicators of trait < {error_vars_dict["maximum_trait_error_var"]}'
    cor_caption = f'{cor_min_max_dict["minimum"]} < Inter-factor correlation < {cor_min_max_dict["maximum"]}'
    global_fit_thresholds_caption = get_strings_of_thresholds_of_global_fit_indices(
        fit_indices_thresholds_dict
    )
    caption = (
        fc_errors_caption
        + "\n"
        + trait_error_caption
        + "\n"
        + cor_caption
        + "\n"
        + global_fit_thresholds_caption
    )
    return caption


def generate_filename_suffix_on_local_fit(
    error_vars_dict, cor_min_max_dict=None
) -> str:
    """
    function for generating string of filename representing thresholds of local fits
    """
    fc_error_thresholds = f'FC_ERROR_VAR_from_{error_vars_dict["minimum_fc_error_var"]}_to_{error_vars_dict["maximum_fc_error_var"]}'
    if (
        "minimum_trait_error_var" in error_vars_dict.keys()
        and "maximum_trait_error_var" in error_vars_dict.keys()
    ):
        trait_error_thresholds = f'_TRAIT_ERROR_VAR_from_{error_vars_dict["minimum_trait_error_var"]}_to_{error_vars_dict["maximum_trait_error_var"]}'
    else:
        trait_error_thresholds = ""
    if cor_min_max_dict is not None:
        cor_thresholds = (
            f'_COR_from_{cor_min_max_dict["minimum"]}_to_{cor_min_max_dict["maximum"]}'
        )
    else:
        cor_thresholds = ""
    local_thresholds = fc_error_thresholds + trait_error_thresholds + cor_thresholds
    return local_thresholds


def generate_filename_suffix_on_global_fit(fit_indices_thresholds_dict) -> str:
    """
    function for generating string of filename representing thresholds of global fits
    """
    global_fit_thresholds = ""
    for item in fit_indices_thresholds_dict.items():
        fit_threshold = "_".join([str(i) for i in item])
        global_fit_thresholds += "_" + fit_threshold
    return global_fit_thresholds


def get_n_of_misfit_models(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    model_type_list: list[ModelFA],
    dtype_memmap="float32",
) -> np.ndarray:
    """get count of misspecified or misfit models"""
    string_dict = get_strings_from_filename(filename_cor, include_nofa_model=False)
    model_fas, trait_type, scale_name, gsr_type = (
        string_dict["model_type"],
        string_dict["trait_type"],
        string_dict["scale_name"],
        string_dict["gsr_type"],
    )
    boolean_locally_specified_array = select_models_from_local_fits(
        filename_cor, error_vars_dict, cor_min_max_dict, model_type_list
    )
    (
        boolean_globally_specified_array,
        boolean_globally_specified_array_each,
    ) = select_models_from_global_fits(
        filename_cor, fit_indices_thresholds_dict, model_type_list
    )
    bool_array_inclusion = select_edges_locally_and_globally(
        boolean_locally_specified_array,
        boolean_globally_specified_array,
    )
    locally_misspecified_n = get_n_of_local_misfit(boolean_locally_specified_array)[
        :, 3
    ]
    globally_misfit_n, globally_misfit_n_each = get_n_of_global_misfit(
        boolean_globally_specified_array, boolean_globally_specified_array_each
    )
    overall_misfit_n = bool_array_inclusion.shape[0] - np.sum(
        bool_array_inclusion, axis=0
    )

    return locally_misspecified_n, globally_misfit_n, overall_misfit_n


def get_set_of_misfit_models(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    dtype_memmap="float32",
):
    """function for getting set of edges which led to misfit models"""


def get_n_of_misfit_models_from_traits(
    trait_type_list,
    n_edge: int,
    sample_n: int,
    est_method: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    model_type_list: list[ModelFA],
    dtype_memmap="float32",
) -> pd.DataFrame:
    """
    function for getting counts of misspecified models per scale of trait type
    """
    filename_cor_list = get_latest_files_with_args(
        trait_type_list=trait_type_list,
        n_edge=n_edge,
        sample_n=sample_n,
        est_method=est_method,
        data_type="correlation",
    )
    n_scales_trait_list = get_n_of_scales_from_trait(trait_type_list)
    # 3 represents Local, Global, and Overall
    num_model_iter = len(model_type_list) * 3
    misfit_count_array = np.empty(
        shape=2 * num_model_iter * sum(n_scales_trait_list), dtype=int
    )
    models_list, gsr_type_list, scale_name_list, trait_type_list = [], [], [], []
    for i, filename_cor in enumerate(filename_cor_list):
        string_dict = get_strings_from_filename(filename_cor, include_nofa_model=False)
        model_fas, gsr_type, scale_name, trait_type = (
            string_dict["model_type"],
            string_dict["gsr_type"],
            string_dict["scale_name"],
            string_dict["trait_type"],
        )
        (
            locally_misspecified_n,
            globally_misfit_n,
            overall_misfit_n,
        ) = get_n_of_misfit_models(
            filename_cor=filename_cor,
            error_vars_dict=error_vars_dict,
            cor_min_max_dict=cor_min_max_dict,
            fit_indices_thresholds_dict=fit_indices_thresholds_dict,
            model_type_list=model_type_list,
            dtype_memmap=dtype_memmap,
        )
        n_misfits = np.concatenate(
            [locally_misspecified_n, globally_misfit_n, overall_misfit_n]
        )
        misfit_count_array[i * num_model_iter : (i + 1) * num_model_iter] = n_misfits
        models_list.append(model_type_list * 3)
        gsr_type_list.append(gsr_type)
        scale_name_list.append(scale_name)
        trait_type_list.append(trait_type)
    count_df = pd.DataFrame(
        {
            "gsr_type": sum([[i] * num_model_iter for i in gsr_type_list], []),
            "scale_name": sum([[i] * num_model_iter for i in scale_name_list], []),
            "trait_type": sum([[i] * num_model_iter for i in trait_type_list], []),
            "misfit_type": sum(
                [
                    ["Local"] * len(model_type_list)
                    + ["Global"] * len(model_type_list)
                    + ["Overall"] * len(model_type_list)
                ]
                * 2
                * sum(n_scales_trait_list),
                [],
            ),
            "model": sum(models_list, []),
            "misfit_n": misfit_count_array,
        }
    )
    count_df["misfit_type"] = count_df["misfit_type"].astype("category")
    return count_df


def visualize_misfit_n(
    count_df: pd.DataFrame,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    n_edge: int,
    plt_close=False,
) -> None:
    """
    function for visualising count of misfit as seaborn FacetGrid
    """
    for trait_type in count_df["trait_type"].unique():
        for misfit_type in count_df["misfit_type"].unique():
            if count_df["model"].nunique() == 1:
                g = sns.FacetGrid(
                    data=count_df.query(
                        "trait_type == @trait_type & misfit_type == @misfit_type"
                    ),
                    col="gsr_type",
                    margin_titles=True,
                )
                g.map(sns.barplot, "scale_name", "misfit_n")
                for ax in g.axes.flatten():
                    ax.set_xticks(ax.get_xticks())
                    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
                    ax.set_xlabel("")
                    ax.set_ylabel("Number of excluded edges")
                    ax.set_ylim(bottom=0, top=n_edge)
            else:
                g = sns.FacetGrid(
                    data=count_df.query(
                        "trait_type == @trait_type & misfit_type == @misfit_type"
                    ),
                    col="scale_name",
                    row="gsr_type",
                    margin_titles=True,
                )
                g.map(sns.barplot, "model", "misfit_n")
            g.fig.suptitle(f"Number of {misfit_type.lower()} misfit in {trait_type}")
            caption = generate_caption_of_figure_on_parameter_thresholds(
                error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
            )
            g.fig.text(0, -0.125, caption)
            g.fig.tight_layout()
            trait_scale_name = get_scale_name_from_trait(trait_type)
            fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
            if (misfit_type == "Local") or (misfit_type == "Overall"):
                local_fit_suffix = generate_filename_suffix_on_local_fit(
                    error_vars_dict, cor_min_max_dict
                )
            if (misfit_type == "Global") or (misfit_type == "Overall"):
                global_fit_suffix = generate_filename_suffix_on_global_fit(
                    fit_indices_thresholds_dict
                )
            if misfit_type == "Overall":
                fig_filename = f"Count_misfit_{trait_type}_{misfit_type}_{local_fit_suffix}_{global_fit_suffix}.png"
            elif misfit_type == "Local":
                fig_filename = (
                    f"Count_misfit_{trait_type}_{misfit_type}_{local_fit_suffix}.png"
                )
            elif misfit_type == "Global":
                fig_filename = (
                    f"Count_misfit_{trait_type}_{misfit_type}_{global_fit_suffix}.png"
                )

            g.fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")
            if plt_close:
                plt.close()


def visualize_misfit_n_from_thresholds(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    est_method: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    model_type_list: list[ModelFA],
    dtype_memmap="float32",
    plt_close=False,
) -> None:
    """
    function for visualising number of misfits from recently applied model
    """
    count_df = get_n_of_misfit_models_from_traits(
        trait_type_list,
        n_edge,
        sample_n,
        est_method,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        model_type_list,
        dtype_memmap="float32",
    )
    visualize_misfit_n(
        count_df,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        n_edge,
        plt_close=plt_close,
    )


def draw_n_of_misspecified_models(
    filename_cor: str,
    locally_misspecified_n: int,
    globally_misfit_n: int,
    overall_misfit_n: int,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    dtype_memmap="float32",
) -> None:
    """
    function for comparing number of misfits per variables and gsr type
    """
    string_dict = get_strings_from_filename(filename_cor, include_nofa_model=False)
    model_fas, trait_type, scale_name, gsr_type = (
        string_dict["model_type"],
        string_dict["trait_type"],
        string_dict["scale_name"],
        string_dict["gsr_type"],
    )

    misfit_df = pd.DataFrame(
        {
            "Local_n": locally_misspecified_n,
            "Global_n": globally_misfit_n,
            "Overall_n": overall_misfit_n,
            "Model": model_fas,
        }
    )
    long_misfit_df = misfit_df.melt(
        id_vars="Model",
        value_vars=[fit + "_n" for fit in ["Local", "Global", "Overall"]],
        var_name="Fit",
        value_name="n",
    )

    gsr_suffix = generate_gsr_suffix(gsr_type)
    trait_scale_name = get_scale_name_from_trait(trait_type)
    fig_title = f"Count of misfit or misspecified models in {scale_name} of {trait_scale_name} {gsr_suffix}"

    g = sns.FacetGrid(data=long_misfit_df, col="Fit")
    g.map(sns.barplot, "Model", "n")
    g.set_ylabels("Count")
    g.set_xlabels("")
    g.fig.suptitle(fig_title)
    caption = generate_caption_of_figure_on_parameter_thresholds(
        error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    g.fig.text(0, -0.125, caption)
    g.set(ylim=(0, bool_array_inclusion.shape[0]))
    g.fig.tight_layout()

    fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, scale_name, "figures")
    local_fit_suffix = generate_filename_suffix_on_local_fit(
        error_vars_dict, cor_min_max_dict
    )
    global_fit_suffix = generate_filename_suffix_on_global_fit(
        fit_indices_thresholds_dict
    )
    fig_filename = f"Count_misfit_{trait_scale_name}_{scale_name}_{gsr_type}_{local_fit_suffix}{global_fit_suffix}.png"
    g.fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")


def select_models_from_global_fits(
    filename_cor: str,
    fit_indices_thresholds_dict: dict[FitIndicesList:float],
    model_type_list: list[ModelFA],
    dtype_memmap="float32",
) -> tuple[
    NDArray[Shape["Num_iter, 1"], Bool],
    NDArray[Shape["Num_iter, Num_fit_indices"], Bool],
]:
    """
    function for selecting models from global fits
    return ndarray with shape of (num_iter, n_fa_models) and (num_iter, n_of_fit_indices, n_fa_models)
    """

    models_fa_str = get_model_strings(filename_cor)[1]
    filename_after_model_type = reduce(add, re.findall("Trait_.*", filename_cor))
    filename_fit = f"fit_indices_{models_fa_str}_{filename_after_model_type}"
    (
        boolean_globally_specified_array,
        boolean_globally_specified_array_each,
    ) = examine_global_misfit_of_model(
        filename_fit, fit_indices_thresholds_dict, model_type_list, fit_all_or_any="all"
    )
    return boolean_globally_specified_array, boolean_globally_specified_array_each


def get_n_of_global_misfit(
    boolean_globally_specified_array, boolean_globally_specified_array_each
) -> tuple[np.ndarray, np.ndarray]:
    """
    function for calculating number of global misfits
    retruning shape of (n_of_fa_model, ) and (n_of_fit_indices, n_of_fa_models)
    """
    globally_misfit_n = boolean_globally_specified_array.shape[0] - np.sum(
        boolean_globally_specified_array, axis=0
    )
    globally_misfit_n_each = boolean_globally_specified_array.shape[0] - np.sum(
        boolean_globally_specified_array_each, axis=0
    )
    return globally_misfit_n, globally_misfit_n_each


@dataclass
class ValueRange:
    low: int
    high: int


CorRange = Annotated[float, ValueRange(-1, 1)]


def examine_misspecified_model_from_parameters(
    fc_error_vars: NDArray[Shape["Num_iter, Num_fc_error_param"], Float],
    cor: CorRange,
    error_vars_dict: ErrorVarsMinMaxDict,
    trait_error_vars: NDArray[Shape["Num_iter, Num_trait_error_param"], Float] = None,
    cor_min_max_dict: CorMinMaxDict = None,
    print_error=True,
) -> NDArray[Shape["Num_iter, Ncol"], Bool]:
    """
    function for examining model misspecification from standardized error variance of FC and trait
    """
    num_iter = len(fc_error_vars)
    misspecified_fc_var = np.any(
        (fc_error_vars > error_vars_dict["maximum_fc_error_var"])
        | (fc_error_vars < error_vars_dict["minimum_fc_error_var"]),
        axis=1,
    )
    if trait_error_vars is not None:
        misspecified_trait_var = np.any(
            (trait_error_vars > error_vars_dict["maximum_trait_error_var"])
            | (trait_error_vars < error_vars_dict["minimum_trait_error_var"]),
            axis=1,
        )
    else:
        misspecified_trait_var = np.empty(shape=num_iter)
        misspecified_trait_var[:] = np.nan
    if cor_min_max_dict is not None:
        misspecified_cor = (cor < cor_min_max_dict["minimum"]) | (
            cor > cor_min_max_dict["maximum"]
        )
    else:
        misspecified_cor = np.empty(shape=num_iter)
        misspecified_cor[:] = np.nan

    if print_error:
        if misspecified_fc_var:
            print(
                "Misspecified in FC: error variances of indicators = ",
                np.round(fc_error_vars, 3),
            )

        if misspecified_trait_var:
            print(
                "Misspecified in trait: error variances of indicators = ",
                np.round(trait_error_vars, 3),
            )
    # trait variables and correlation thresholds exist
    if not np.all(np.isnan(misspecified_trait_var)) and not np.all(
        np.isnan(misspecified_cor)
    ):
        misspecified_locally = (
            misspecified_fc_var | misspecified_trait_var | misspecified_cor
        )
        boolean_locally_misspecified_array = np.array(
            [
                misspecified_fc_var,
                misspecified_trait_var,
                misspecified_cor,
                misspecified_locally,
            ]
        )
    # trait variable does not exist but correlation thresholds exist
    elif np.all(np.isnan(misspecified_trait_var)) and not np.all(
        np.isnan(misspecified_cor)
    ):
        misspecified_locally = misspecified_fc_var | misspecified_cor
        boolean_locally_misspecified_array = np.array(
            [
                misspecified_fc_var,
                misspecified_cor,
                misspecified_locally,
            ]
        )
    # tarit variable and correlation thresholds do not exist
    elif np.all(np.isnan(misspecified_trait_var)) and np.all(
        np.isnan(misspecified_cor)
    ):
        misspecified_locally = misspecified_fc_var
        boolean_locally_misspecified_array = misspecified_locally[:, np.newaxis]
    # trait vairable exists but correlation thresholds do not exist
    elif not np.all(np.isnan(misspecified_trait_var)) and np.all(
        np.isnan(misspecified_cor)
    ):
        misspecified_locally = None
    return boolean_locally_misspecified_array


def get_names_and_thresholds_of_fit_indices(
    fit_indices_thresholds_dict: dict[FitIndicesList:float],
) -> tuple[list[FitIndicesList], list[float]]:
    """
    function for getting a list of names and set thresholds of input fit indices
    """
    fit_indices = list(fit_indices_thresholds_dict.keys())
    fit_thresholds = list(fit_indices_thresholds_dict.values())
    return fit_indices, fit_thresholds


AllAny = Literal["all", "any"]


def examine_global_misfit_of_model(
    filename_fit: str,
    fit_indices_thresholds_dict: dict[FitIndicesList:float],
    model_type_list: list[ModelFA],
    fit_all_or_any: AllAny = "all",
) -> tuple[
    NDArray[Shape["Num_iter, Num_model_fa"], Bool],
    NDArray[Shape["Num_iter, Num_fit_indices, Num_model_fa"], Bool],
]:
    """
    function for examining model misfits using fit indices and FIT_INDICES_OP_DICT
    """
    # get order number of input fit indices in FIT_INDICES_LIST
    fit_indices, fit_thresholds = get_names_and_thresholds_of_fit_indices(
        fit_indices_thresholds_dict
    )
    fit_indices_order = [list(FIT_INDICES_OP_DICT.keys()).index(i) for i in fit_indices]
    # get operations
    ops = list(itemgetter(*fit_indices)(FIT_INDICES_OP_DICT))

    model_fits = copy_memmap_output_data(
        filename_fit,
    )
    model_type = get_strings_from_filename(
        filename_fit, ["model_type"], include_nofa_model=False
    )[0]

    model_index = [i for i, model in enumerate(model_type) if model in model_type_list]
    # conduct processing after calculating correlations
    fits_interested = model_fits[:, fit_indices_order][..., model_index]
    if len(fits_interested.shape) == 2:
        fits_interested = fits_interested[:, :, np.newaxis]
    boolean_global_each = np.empty(shape=fits_interested.shape)
    for i, fit_threshold in enumerate(fit_thresholds):
        for j, _ in enumerate(model_type_list):
            # range in number of factor models
            boolean_global_each[:, i, j] = ops[i](
                fits_interested[:, i, j], fit_thresholds[i]
            )
    if fit_all_or_any == "all":
        boolean_global = np.all(boolean_global_each, axis=1)
    else:
        boolean_global = np.any(boolean_global_each, axis=1)

    return boolean_global, boolean_global_each


class ModelFitIndex(TypedDict):
    FitIndicesList: set[int]


def get_edges_of_globally_misfit_models(
    filename_fit,
    fit_indices_thresholds_dict: dict[FitIndicesList:float],
    model_type_list,
    add_global=False,
) -> Mapping[ModelFA, ModelFitIndex]:
    """function for getting edges where global misfit is found"""
    fit_indices, _ = get_names_and_thresholds_of_fit_indices(
        fit_indices_thresholds_dict
    )
    boolean_global, boolean_global_each = examine_global_misfit_of_model(
        filename_fit,
        fit_indices_thresholds_dict,
        model_type_list,
        fit_all_or_any="all",
    )
    model_fa_list = get_strings_from_filename(filename_fit, ["model_type"])
    if add_global:
        boolean_global_each = np.concatenate(
            [boolean_global_each, boolean_global[:, np.newaxis, :]], axis=1
        )
        fit_indices = fit_indices + ["all"]
    array_edges, array_fit_indices, array_models = np.where(
        ~boolean_global_each.astype("bool")
    )

    globally_misfit_edges_dict = defaultdict(dict)
    for i, model in enumerate(reduce(add, model_fa_list)):
        for j, fit_index in enumerate(fit_indices):
            bool_index = (array_models == i) & (array_fit_indices == j)
            array_edges_misfit = array_edges[bool_index]
            globally_misfit_edges_dict[model][fit_index] = set(array_edges_misfit)

    return globally_misfit_edges_dict


def select_edges_locally_and_globally(
    boolean_locally_specified_array: NDArray[Shape["Num_iter, Num_modelfa, 4"], Bool],
    boolean_globally_specified_array: NDArray[Shape["Num_iter"], Bool],
) -> NDArray[Shape["Num_iter, Num_modelfa"], Bool]:
    """
    function for selecting edges from boolean representing local and global fits
    """
    # select edges
    bool_array_inclusion = (
        boolean_locally_specified_array[:, :, 3] & boolean_globally_specified_array
    )
    return bool_array_inclusion


def get_boolean_array_of_selected_edges(bool_array_inclusion, model_nofa_list):
    """
    function for getting reshaped boolean array representing selected edges considering no fa models
    """
    bool_array_inclusion_reshaped = np.c_[
        np.ones(
            shape=(bool_array_inclusion.shape[0], len(model_nofa_list)), dtype="bool"
        ),
        bool_array_inclusion,
    ]
    return bool_array_inclusion_reshaped


def delete_sample_n_duplicated_files():
    """function for deleting files with sampleN is repeated more than two times"""
    for trait_type in ["personality", "cognition", "mental"]:
        # select folder
        folder = get_scale_name_from_trait(trait_type)
        # select scale names
        if trait_type == "personality":
            scale_name_list = NEO_FFI_SCALES
        elif trait_type == "cognition":
            scale_name_list = NIH_COGNITION_SCALES
        elif trait_type == "mental":
            scale_name_list = ASR_BROAD_SCALES

        for scale_name in scale_name_list:
            file_folder = op.join(SCHAEFER_DIR, folder, scale_name, "correlation")
            files = os.listdir(file_folder)
            files = [i for i in files if "Fold" in i]
            delete_file_list = [
                i
                for i in files
                if len(reduce(add, re.findall("sampleN_(.*)_Fold", i)).split("_")) > 2
            ]
            # delete files
            for delete_file in delete_file_list:
                os.remove(op.join(file_folder, delete_file))


def get_latest_single_file_with_args(
    trait_scale_name,
    scale_name,
    n_edge,
    sample_n,
    data_type,
    est_method,
    gsr_type,
    model_type,
    cov_cor: bool = False,
    phase_encoding: bool = False,
    factor_scores=False,
):
    """get a single filename with multiple arguments"""
    file_folder = op.join(SCHAEFER_DIR, trait_scale_name, scale_name, data_type)
    # get sample size in each fold from sample_n, split_ratio, and fold_n
    n_edge_files = [
        i
        for i in os.listdir(file_folder)
        if (str(n_edge) in i)
        and (gsr_type in i)
        and (str(sample_n) in i)
        and (est_method in i)
        and (model_type in i)
    ]

    if cov_cor:
        n_edge_files = [i for i in n_edge_files if "CovCor" in i]
    else:
        n_edge_files = [i for i in n_edge_files if not "CovCor" in i]

    if phase_encoding:
        n_edge_files = [i for i in n_edge_files if "PE" in i]
    else:
        n_edge_files = [i for i in n_edge_files if "PE" not in i]
    n_edge_files = sort_list_by_time(n_edge_files)
    return n_edge_files


def get_latest_pair_files_with_args(
    trait_scale_name,
    scale_name,
    n_edge,
    sample_n,
    split_ratio,
    fold_k,
    data_type,
    est_method,
    gsr_type,
    comb_model_type,
    ext: str,
):
    """
    function for getting a single filename with arguments
    """
    file_folder = op.join(SCHAEFER_DIR, trait_scale_name, scale_name, data_type)
    # get sample size in each fold from sample_n, split_ratio, and fold_n
    fold_n = int(1 / split_ratio)
    if fold_n == fold_k + 1:
        sample_n1 = ceil(sample_n / fold_n)
    else:
        sample_n1 = int(sample_n / fold_n)
    sample_n2 = sample_n - sample_n1
    n_edge_files1 = [
        i
        for i in os.listdir(file_folder)
        if (str(n_edge) in i)
        and (gsr_type in i)
        and (str(sample_n1) in i)
        and not (str(sample_n2) in i)
        and (est_method in i)
        and (f"Fold_{fold_k}" in i)
        and (comb_model_type in i)
        and (ext in i)
    ]
    n_edge_files2 = [
        i
        for i in os.listdir(file_folder)
        if (str(n_edge) in i)
        and (gsr_type in i)
        and (str(sample_n2) in i)
        and not (str(sample_n1) in i)
        and (est_method in i)
        and (f"Fold_{fold_k}" in i)
        and (comb_model_type in i)
        and (ext in i)
    ]
    n_edge_files1 = sort_list_by_time(n_edge_files1)
    n_edge_files2 = sort_list_by_time(n_edge_files2)

    return n_edge_files1[-1], n_edge_files2[-1]


def get_latest_combined_single_file_with_args(
    trait_scale_name,
    scale_name,
    n_edge,
    sample_n,
    split_ratio,
    fold_k,
    data_type,
    est_method,
    gsr_type,
    comb_model_type,
    ext: str,
    get_sample_n=False,
) -> str:
    """
    get a single filename combined in each fold
    """
    file_folder = op.join(SCHAEFER_DIR, trait_scale_name, scale_name, data_type)
    # get sample size in each fold from sample_n, split_ratio, and fold_n
    fold_n = int(1 / split_ratio)
    if fold_n == fold_k + 1:
        sample_n1 = ceil(sample_n / fold_n)
    else:
        sample_n1 = int(sample_n / fold_n)
    sample_n2 = sample_n - sample_n1
    n_edge_files = [
        i
        for i in os.listdir(file_folder)
        if (str(n_edge) in i)
        and (gsr_type in i)
        and (str(sample_n1) in i)
        and (str(sample_n2) in i)
        and (est_method in i)
        and (f"Fold_{fold_k}" in i)
        and (comb_model_type in i)
        and (ext in i)
    ]
    n_edge_files = sort_list_by_time(n_edge_files)
    if get_sample_n:
        return n_edge_files[-1], sample_n1, sample_n2
    return n_edge_files[-1]


def subset_datetime_from_filename(filename):
    """
    subset datetime from filename
    """
    datetime_str = datetime.datetime.strptime(
        reduce(
            add,
            re.findall("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", filename),
        ),
        "%Y-%m-%d %H:%M:%S",
    )
    return datetime_str


def sort_list_by_time(list_obj):
    list_obj.sort(
        key=lambda x: datetime.datetime.strptime(
            reduce(
                add,
                re.findall("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", x),
            ),
            "%Y-%m-%d %H:%M:%S",
        )
    )
    return list_obj


# function should be modified (simplified)
def get_latest_files_with_args(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    est_method: str,
    data_type: str,
    model_type_list: list[str] = None,
    drop_vars_list_dict: Optional[dict[str, list[str]]] = None,
    split_ratio: float = None,
    score_type: str = None,
    combined=False,
    ext: str = None,
    cov_cor=True,
    day_cor=False,
    PE=False,
    fixed_load=True,
):
    """
    function for getting latest filenames processed
    filename_dict is added to store multiple filenames with cross-validation
    trait_type_list is often used with a single element including ['cognition']
    """
    filename_list_dict = defaultdict(dict)
    fixed_load_str = "FixedLoad" if fixed_load else ""
    # select folder
    for trait_type in trait_type_list:
        if trait_type is not None:
            folder = get_scale_name_from_trait(trait_type)
            scale_name_list = get_subscale_list(folder)
        else:
            folder, scale_name_list = "reliability", [None]

        for gsr_type in ["_nogs_", "_gs_"]:
            gsr_type_key_str = gsr_type.replace("_", "")
            for scale_name in scale_name_list:
                print(gsr_type, scale_name)
                file_folder = "/".join(
                    filter(None, (SCHAEFER_DIR, folder, scale_name, data_type))
                )
                if split_ratio is None:
                    # get filenames
                    # when model_type_list includes FAModel
                    if all(i in MODEL_TRAIT_STR for i in model_type_list):
                        target_files_list = [
                            i
                            for i in os.listdir(file_folder)
                            if (str(n_edge) in i)
                            and (gsr_type in i)
                            and (str(sample_n) in i)
                            and (est_method in i)
                            and all(j in i for j in model_type_list)
                            and fixed_load_str in i
                        ]
                    else:
                        target_files_list = [
                            i
                            for i in os.listdir(file_folder)
                            if (str(n_edge) in i)
                            and (gsr_type in i)
                            and (str(sample_n) in i)
                            and all(j in i for j in model_type_list)
                        ]
                    if day_cor:
                        target_files_list = [
                            i for i in target_files_list if "DayCor" in i
                        ]
                    else:
                        target_files_list = [
                            i for i in target_files_list if not "DayCor" in i
                        ]
                    target_files_list_not_dropped = [
                        i for i in target_files_list if not "drop_" in i
                    ]
                    if drop_vars_list_dict is not None:
                        drop_vars_list = drop_vars_list_dict.get(scale_name)
                    else:
                        drop_vars_list = None

                    if drop_vars_list is not None:
                        target_files_list_add = [
                            target_file
                            for target_file in target_files_list
                            if all(
                                drop_var in target_file for drop_var in drop_vars_list
                            )
                        ]
                    if ext is not None and len(target_files_list) > 0:
                        target_files_list = [i for i in target_files_list if ext in i]
                if combined:
                    fold_n = int(1 / split_ratio)
                    for k in range(fold_n):
                        if k + 1 < fold_n:
                            sample_n1 = int(sample_n * split_ratio)
                        else:
                            sample_n1 = ceil(sample_n * split_ratio)
                        sample_n2 = sample_n - sample_n1
                        target_files_list = [
                            i
                            for i in os.listdir(file_folder)
                            if (str(n_edge) in i)
                            and (gsr_type in i)
                            and ((str(sample_n1) in i) or (str(sample_n2) in i))
                            and (est_method in i)
                            and (f"Fold_{k}" in i)
                        ]
                        target_files_list = [
                            i
                            for i in target_files_list
                            if (str(sample_n1) in i)
                            and (str(sample_n2) in i)
                            and (score_type in i)
                        ]
                        if ext is not None:
                            target_files_list = [
                                i for i in target_files_list if ext in i
                            ]
                        target_files_list.sort(
                            key=lambda x: datetime.datetime.strptime(
                                reduce(
                                    add,
                                    re.findall(
                                        "split_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", x
                                    ),
                                ),
                                "%Y-%m-%d %H:%M:%S",
                            )
                        )
                        return target_files_list[-1]
                # reorder files based on datetime
                if len(target_files_list) > 0:
                    target_files_list_not_dropped.sort(
                        key=lambda x: datetime.datetime.strptime(
                            reduce(
                                add,
                                re.findall("_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", x),
                            ),
                            "%Y-%m-%d %H:%M:%S",
                        )
                    )
                filename_list_dict[gsr_type_key_str][scale_name] = defaultdict(dict)
                if scale_name is not None:
                    filename_list_dict[gsr_type_key_str][scale_name][
                        "not_dropped"
                    ] = target_files_list_not_dropped[-1]
                else:
                    filename_list_dict[gsr_type_key_str] = target_files_list_not_dropped

                if drop_vars_list is not None:
                    if len(target_files_list_add) > 0:
                        # not sorted! should be sorted by time
                        target_files_list_add = sort_list_by_time(target_files_list_add)
                        filename_list_dict[gsr_type_key_str][scale_name][
                            "dropped"
                        ] = target_files_list_add[-1]
                    else:
                        filename_list_dict[gsr_type_key_str][scale_name][
                            "dropped"
                        ] = None

    return filename_list_dict


def get_n_of_scales_from_trait(trait_type_list) -> list:
    """
    function for getting number of scales of trait measure
    """
    n_scales_list = []
    for trait_type in trait_type_list:
        if trait_type == "personality":
            n_scales = len(NEO_FFI_SCALES)
        elif trait_type == "cognition":
            n_scales = len(NIH_COGNITION_SCALES)
        elif trait_type == "mental":
            n_scales = len(ASR_BROAD_SCALES)
        else:
            raise NameError('triat_type should be "personality" or "cognition".')
        n_scales_list.append(n_scales)
    return n_scales_list


GSRType = Literal["nogs", "gs"]


def generate_gsr_suffix(gsr_type: GSRType, capitalize=False) -> str:
    """
    function for generating suffix of gsr type
    """
    if gsr_type == "gs":
        gsr_suffix = "with GSR"
    elif gsr_type == "nogs":
        gsr_suffix = "without GSR"
    else:
        raise NameError('Input should be "gs" or "nogs".')
    if capitalize:
        gsr_suffix = gsr_suffix.capitalize().replace("gsr", "GSR")
    return gsr_suffix


class LocallyMisspecifiedLocations(TypedDict):
    FitLocations: set[int]


def get_set_of_locally_globlly_misfit_edges(
    filename_cor,
    error_vars_dict,
    fit_indices_thresholds_dict,
    model_type_list: list[str],
    cor_min_max_dict=None,
) -> dict[ModelFA, set[int]]:
    """
    function for getting misfit edges both locally and globally
    """

    def get_set_of_locally_misspecified_edges() -> Mapping[
        ModelFA, LocallyMisspecifiedLocations
    ]:
        """
        function for getting sets of edges with locally misspecified models
        """
        boolean_locally_specified_array = select_models_from_local_fits(
            filename_cor, error_vars_dict, model_type_list, cor_min_max_dict
        )
        model_fa_list, _, _ = get_model_strings(filename_cor)
        locally_misspecified_edge_dict = defaultdict(dict)
        for model_index, model in enumerate(model_type_list):
            locally_misspecified_edge_dict[
                model
            ] = get_edges_of_locally_misspecified_models(
                boolean_locally_specified_array, model_index, model
            )
        return locally_misspecified_edge_dict

    # if 'Trait' in filename_cor:
    locally_misspecified_dict = get_set_of_locally_misspecified_edges()
    # else:
    #     locally_misspecified_dict = get_set_of_locally_misspecified_edges_rel(
    #         filename_cor, error_vars_dict, model_type_list, cor_min_max_dict,
    #     )
    def get_set_of_globally_misfit_edges() -> Mapping[ModelFA, ModelFitIndex]:
        """
        function for getting sets of edges with globally misfit model
        """
        model_fa_list, _, _ = get_model_strings(filename_cor)
        (
            boolean_globally_specified_array,
            boolean_globally_specified_array_each,
        ) = select_models_from_global_fits(
            filename_cor, fit_indices_thresholds_dict, model_type_list
        )

        models_fa_str = get_model_strings(filename_cor)[1]
        filename_after_model_type = reduce(add, re.findall("Trait_.*", filename_cor))
        filename_fit = f"fit_indices_{models_fa_str}_{filename_after_model_type}"
        globally_misfit_edges_dict = get_edges_of_globally_misfit_models(
            filename_fit,
            fit_indices_thresholds_dict,
            model_type_list,
            add_global=True,
        )

        return globally_misfit_edges_dict

    globally_misfit_dict = get_set_of_globally_misfit_edges()
    # model_fas = locally_misspecified_dict.keys()
    misfit_dict = {}
    for model in model_type_list:
        misfit_dict[model] = (
            locally_misspecified_dict[model]["all"] | globally_misfit_dict[model]["all"]
        )
    return misfit_dict


def add_coordinate_to_nodes(node_summary_path):
    """
    function for adding coordinate information to nodes
    """
    nodes_df = pd.read_csv(node_summary_path)
    folder = "/home/cezanne/t-haitani/hcp_data/atlas/Tian2020MSA_v1.4/Tian2020MSA/3T/Cortex-Subcortex/MNIvolumetric"
    filename = "Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S2_MNI152NLin6Asym_1mm.nii.gz"
    coordinate_xyz = find_parcellation_cut_coords(op.join(folder, filename))
    for index, xyz in enumerate(["x", "y", "z"]):
        nodes_df[xyz] = coordinate_xyz[:, index]
    save_folder = "/".join(node_summary_path.split("/")[:-1])
    save_filename = node_summary_path.split("/")[-1].replace(".csv", "_xyz.csv")
    nodes_df.to_csv(op.join(save_folder, save_filename))


def get_edge_summary(
    parcellation='Schaefer',
    network_hem_order=False,
    over_write=False,
) -> pd.DataFrame:
    """
    function for generating node summary
    """
    node_summary_path = NODE_SUMMARY_PATH_DICT.get(parcellation)
    ordered_suffix = "_ordered" if network_hem_order else ""
    network_agg_pub_dict = NETWORK_AGG_PUB_NESTED_DICT.get(parcellation)
    filename = node_summary_path.split(".csv")[0] + "_edge" + ordered_suffix + ".pkl"
    #nodes_df = get_nodes_df(parcellation)
    if (not op.isfile(filename)) or over_write:
        node_summary = pd.read_csv(node_summary_path)
        dict_edges = defaultdict(list)
        node_summary["net"] = node_summary["net"].replace("Limbic_tian", "LimbicTian")

        if network_hem_order:
            network_order_dict = NETWORK_ORDER_NESTED_DICT.get(parcellation)
            network_agg_pub_dict = NETWORK_AGG_PUB_NESTED_DICT.get(parcellation)
            node_summary["net"] = pd.Categorical(
                node_summary["net"], categories=network_order_dict.keys(), ordered=True
            )
            node_summary["net"] = node_summary["net"].cat.rename_categories(
                network_agg_pub_dict
            )
            node_summary.sort_values(["net", "hem"], inplace=True)
            node_summary.reset_index(inplace=True)
        for edge in combinations(node_summary["node"], 2):
            for node_n in [1, 2]:
                dict_edges["node" + str(node_n)].append(edge[node_n - 1])

                if "_" in edge[node_n - 1]:
                    dict_edges["node" + str(node_n) + "_hem"].append(
                        edge[node_n - 1].split("_")[1]
                    )
                    dict_edges["node" + str(node_n) + "_net"].append(
                        edge[node_n - 1].split("_")[2]
                    )
                    net_sub = edge[node_n - 1].split("_")[3]

                    if not re.findall("[1-9]", net_sub):
                        dict_edges["node" + str(node_n) + "_net_sub"].append(net_sub)
                    else:
                        dict_edges["node" + str(node_n) + "_net_sub"].append(None)

                elif "-" in edge[node_n - 1]:
                    dict_edges["node" + str(node_n) + "_hem"].append(
                        edge[node_n - 1].split("-")[-1].upper()
                    )
                    dict_edges["node" + str(node_n) + "_net"].append("LimbicTian")
                    dict_edges["node" + str(node_n) + "_net_sub"].append(
                        reduce(
                            add,
                            re.findall("[A-Z]+[a-z]?", edge[node_n - 1].split("-")[0]),
                        )
                    )

        edges_df = pd.DataFrame(dict_edges)
        edges_df["node1"] = pd.Categorical(
            edges_df['node1'], categories=node_summary['node'][:-1]
        )
        edges_df["node2"] = pd.Categorical(
            edges_df['node2'], categories=node_summary['node'][1:]
        )
        edges_df.reset_index(inplace=True, names="edge")
        edges_df.sort_values(['node1', 'node2'], inplace=True)
        if over_write:
            filename = filename.split(".pkl")[0] + "_mod.pkl"
        edges_df.to_pickle(filename)
    else:
        edges_df = pd.read_pickle(filename)

    # rename categories
    for column in ["node1_net", "node2_net"]:
        edges_df[column] = pd.Categorical(edges_df[column], categories=NETWORK_ORDER_NESTED_DICT.get(parcellation).keys())
        edges_df[column] = edges_df[column].cat.rename_categories(network_agg_pub_dict)

#    for column in ["node1", "node2"]:
#        edges_df[column] = pd.Categorical(
#            edges_df[column], categories=edges_df[column].unique(), ordered=True
#        )

    return edges_df


#def get_edge_summary(
#    parcellation='Schaefer',
#    network_hem_order=False,
#    over_write=False,
#) -> pd.DataFrame:
#    """
#    function for generating node summary
#    """
#    node_summary_path = NODE_SUMMARY_PATH_DICT.get(parcellation)
#    ordered_suffix = "_ordered" if network_hem_order else ""
#    filename = node_summary_path.split(".csv")[0] + "_edge" + ordered_suffix + ".pkl"
#    network_agg_pub_dict = NETWORK_AGG_PUB_NESTED_DICT.get(parcellation)
#    
#    if (not op.isfile(filename)) or over_write:
#        node_summary = pd.read_csv(node_summary_path)
#        dict_edges = defaultdict(list)
#        node_summary["net"] = node_summary["net"].replace("Limbic_tian", "LimbicTian")
#
#        if network_hem_order:
#            network_order_dict = NETWORK_ORDER_NESTED_DICT.get(parcellation)
#            node_summary.fillna('No', inplace=True)
#            node_summary["net"] = pd.Categorical(
#                node_summary["net"], categories=list(network_order_dict.keys()), ordered=True
#            )
#            node_summary["net"] = node_summary["net"].cat.rename_categories(
#                network_agg_pub_dict
#            )
#            node_summary.sort_values(["net", "hem"], inplace=True)
#            node_summary.reset_index(inplace=True)
#
#        if parcellation == 'Gordon':
#            nodes_dict = node_summary.set_index('node').to_dict('index')
#
#        dict_edges["node1_hem"], dict_edges['node2_hem'] = [], []
#        dict_edges["node1_net"], dict_edges['node2_net'] = [], []
#        dict_edges["node1_net_sub"], dict_edges['node2_net_sub'] = [], []
#        hem_list, net_list, net_sub_list = [], [], [] 
#        for edge in combinations(node_summary["node"], 2):
#            for node_n in [1, 2]:
#                dict_edges["node" + str(node_n)].append(edge[node_n - 1])
#                # Processing cortical nodes
#                if "_" in edge[node_n - 1]:
#                    node = edge[node_n - 1]
#                    if parcellation == 'Schaefer':
#                        # they should be modified to work
#                        hem_list.append(node.split("_")[1])
#                        net_list.append(node.split("_")[2])
#                        net_sub = node.split("_")[3]
#                        if not re.findall("[1-9]", net_sub):
#                            net_sub_list.append(net_sub)
#                        else:
#                            net_sub_list.append(None)
#                    elif parcellation == "Gordon":
#                        dict_edges[f'node{node_n}_hem'].append(nodes_dict.get(node).get('hem'))
#                        dict_edges[f'node{node_n}_net'].append(nodes_dict.get(node).get('net'))
#                        dict_edges[f'node{node_n}_net_sub'].append(None)
#                # Processing subcortical nodes
#                elif "-" in edge[node_n - 1]:
#                    dict_edges[f'node{node_n}_hem'].append(edge[node_n - 1].split("-")[-1].upper())
#                    if parcellation == 'Schaefer':
#                        net_list.append('LimbicTian')
#                        net_sub_list.append(
#                                reduce(add, re.findall("[A-Z]+[a-z]?", edge[node_n - 1].split("-")[0]))
#                                )
#                    elif parcellation == 'Gordon':
#                        dict_edges[f'node{node_n}_net'].append('Subcortex')
#                        dict_edges[f'node{node_n}_net_sub'].append(None)
#        edges_df = pd.DataFrame(dict_edges)
#        edges_df["node1_net"] = pd.Categorical(
#            edges_df.node1_net, categories=network_agg_pub_dict.values()
#        )
#        edges_df["node2_net"] = pd.Categorical(
#            edges_df.node2_net, categories=network_agg_pub_dict.values()
#        )
#        edges_df.reset_index(inplace=True, names="edge")
#        edges_df = edges_df.sort_values(["node1_net", "node2_net"])
#        #edges_df.fillna('NA', inplace=True)
#        print('Saving pickle file.')
#        if over_write:
#            filename = filename.split(".pkl")[0] + "_mod.pkl"
#        edges_df.to_pickle(filename)
#    else:
#        edges_df = pd.read_pickle(filename)
#
#    # rename and order categories
#    for column in ["node1_net", "node2_net"]:
#        edges_df[column] = pd.Categorical(edges_df[column])
#        edges_df[column] = edges_df[column].cat.rename_categories(network_agg_pub_dict)
#
##    for column in ["node1", "node2"]:
##        edges_df[column] = pd.Categorical(
##            edges_df[column], categories=edges_df[column].unique(), ordered=True
##        )
#
#    return edges_df


def map_misfit_edge_to_networks(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    model_type_list,
) -> pd.DataFrame:
    """
    function for mapping misfit edges to networks
    """
    misfit_dict = get_set_of_locally_globlly_misfit_edges(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        model_type_list,
    )
    edges_df = get_edge_summary(node_summary_path)
    model_fas = misfit_dict.keys()
    misfit_edges_models_df = pd.DataFrame()
    for model in model_fas:
        misfit_set = misfit_dict.get(model)
        misfit_edges_df = edges_df.query("edge in @misfit_set").assign(model=model)
        misfit_edges_models_df = pd.concat(
            [misfit_edges_models_df, misfit_edges_df], axis=0
        )
    return misfit_edges_models_df


def generate_set_of_networks(long_edges_df: pd.DataFrame) -> pd.DataFrame:
    """generate set of combinations of networks"""
    long_edges_df["net_comb"] = list(
        zip(long_edges_df.node1_net, long_edges_df.node2_net)
    )
    long_edges_df["net_set"] = long_edges_df["net_comb"].apply(lambda x: frozenset(x))
    long_edges_df["net_set2"] = long_edges_df["net_set"].apply(
        lambda x: sorted([i for i in x], key=lambda y: NETWORK_ORDER_DICT_PUB_SCHAEFER[y])
    )
    long_edges_df["net_set2"] = long_edges_df.apply(
        lambda x: "_".join(x["net_set2"]), axis=1
    )
    long_edges_df.drop("net_set", axis=1)
    return long_edges_df


def get_summary_of_edges_per_network(
    long_edges_df: pd.DataFrame, col_name: str, apply_func: str
) -> pd.DataFrame:
    """
    function for getting mean value of a column by combinations of networks
    """
    long_edges_df = generate_set_of_networks(long_edges_df)

    if apply_func == "mean":
        long_edges_df = long_edges_df.groupby("net_set2")[col_name].mean()
    elif apply_func == "median":
        long_edges_df = long_edges_df.groupby("net_set2")[col_name].median()
    elif apply_func == "std":
        long_edges_df = long_edges_df.groupby("net_set2")[col_name].std()

    wide_edges_df = generate_wide_net_df(long_edges_df, col_name)
    # fill lower traiangle with nan
    il1 = np.tril_indices(len(NETWORK_ORDER_DICT), -1)
    edges_array = np.array(wide_edges_df)
    edges_array[il1] = np.nan
    edges_wide_df = pd.DataFrame(
        edges_array,
        index=NETWORK_ORDER_DICT.keys(),
        columns=NETWORK_ORDER_DICT.keys(),
    )
    return edges_wide_df


def get_counts_of_edges_per_network(long_edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    function for getting counts of combinations of edges
    """
    long_edges_df = generate_set_of_networks(long_edges_df)
    edges_count_df = long_edges_df["net_set2"].value_counts()
    edges_count_wide_df = generate_wide_net_df(edges_count_df, "count")
    return edges_count_wide_df


def generate_wide_net_df(long_edges_df: pd.DataFrame, summary_col: str) -> pd.DataFrame:
    """
    function for generating wide dataframe ordered by networks
    """
    edges_long_df = long_edges_df.reset_index().assign(
        node_list=lambda x: x["net_set2"].str.split("_")
    )
    edges_long_df["node1"] = pd.Categorical(
        edges_long_df["node_list"].apply(lambda x: x[0]),
        categories=NETWORK_ORDER_DICT_PUB.keys(),
    )
    edges_long_df["node2"] = pd.Categorical(
        edges_long_df["node_list"].apply(lambda x: x[-1]),
        categories=NETWORK_ORDER_DICT_PUB.keys(),
    )
    edges_wide_df = edges_long_df.pivot_table(
        index="node1",
        columns="node2",
        values=summary_col,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    # since divide of matrix produces nan, nan need not be created in thin function
    # il1 = np.tril_indices(len(NETWORK_ORDER_DICT), -1)
    edges_array = np.array(edges_wide_df)
    # edges_count_array[il1] = np.nan
    edges_wide_df = pd.DataFrame(
        edges_array,
        index=NETWORK_ORDER_DICT_PUB.keys(),
        columns=NETWORK_ORDER_DICT_PUB.keys(),
    )
    return edges_wide_df


def get_n_of_edges_per_network(node_summary_path: str) -> pd.DataFrame:
    """function for getting dataframe"""
    long_edges_df = get_edge_summary(node_summary_path, network_hem_order=True)
    edges_count_wide_df = get_counts_of_edges_per_network(long_edges_df)
    return edges_count_wide_df


def get_n_of_misfit_edges_per_network(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    model: str,
) -> pd.DataFrame:
    """
    function for getting count of misfit edges
    """
    misfit_edges_models_df = map_misfit_edge_to_networks(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        node_summary_path,
        [model],
    )
    misfit_edges_df = misfit_edges_models_df.query("model == @model")
    misfit_edges_count_wide_df = get_counts_of_edges_per_network(misfit_edges_df)
    return misfit_edges_count_wide_df


def generate_suffixes_and_saving_folder(filename: str) -> tuple[str, str, str]:
    """
    function for generating suffix of figure title
    """
    trait_type, scale_name, gsr_type = get_strings_from_filename(
        filename, ["trait_type", "scale_name", "gsr_type"], include_nofa_model=False
    )
    gsr_suffix = generate_gsr_suffix(gsr_type)
    trait_scale_name = get_scale_name_from_trait(trait_type)
    fig_title_suffix = f"in {scale_name} of {trait_scale_name} {gsr_suffix}"
    fig_name_suffix = f"{trait_scale_name}_{scale_name}_{gsr_type}"
    save_fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, scale_name, "figures")

    return fig_title_suffix, fig_name_suffix, save_fig_folder


def generate_fig_name_suffix_from_thresholds(
    error_vars_dict,
    fit_indices_thresholds_dict,
    cor_min_max_dict,
) -> str:
    """
    function for generating suffix of figure filename
    """
    local_fit_suffix = generate_filename_suffix_on_local_fit(
        error_vars_dict, cor_min_max_dict
    )
    global_fit_suffix = generate_filename_suffix_on_global_fit(
        fit_indices_thresholds_dict
    )
    fig_filename_suffix_thresholds = f"{local_fit_suffix}{global_fit_suffix}"
    return fig_filename_suffix_thresholds


def compare_residuals_between_models(
    trait_scale_name: str,
    scale_name: str,
    model_base: str,
    model_edge_loop: str,
    sample_n: int,
    model_fit_obj: str,
    control: list[str] = None,
    drop_vars: list[str] = None,
):
    """
    function for comparing residuals between models
    """
    model_base_dir = op.join(
        SCHAEFER_DIR, trait_scale_name, scale_name, "residuals", model_base
    )
    model_edge_loop_dir = op.join(
        SCHAEFER_DIR, trait_scale_name, scale_name, "residuals", model_edge_loop
    )

    file_list_base = os.listdir(model_base_dir)
    file_list_base = [
        i
        for i in file_list_base
        if (str(sample_n) in i)
        and (model_fit_obj in i)
        and all([j in i for j in control])
    ]
    if drop_vars is not None:
        file_base = reduce(
            add, [i for i in file_list_base if all([k in i for k in drop_vars])]
        )
    else:
        file_base = reduce(add, [i for i in file_list_base if not "drop" in i])
    mat_base = pd.read_csv(op.join(model_base_dir, file_base), index_col=0)
    cols = mat_base.columns

    file_edge_loop_list = os.listdir(model_edge_loop_dir)
    file_edge_loop_list = [
        i
        for i in file_edge_loop_list
        if (str(sample_n) in i)
        and (model_fit_obj in i)
        and all([j in i for j in control])
    ]
    if drop_vars is not None:
        file_edge_loops = [
            i for i in file_edge_loop_list if all([k in i for k in drop_vars])
        ]
    else:
        file_edge_loops = [i for i in file_edge_loop_list if not "drop" in i]
    for file_edge in file_edge_loops:
        df_mat = pd.read_csv(op.join(model_edge_loop_dir, file_edge), index_col=0)
        print(mat_base - df_mat.loc[cols, cols])


def compare_fit_between_models(
    filename_fit1: str,
    filename_fit2: str,
    vis_model_type: str,
    fit_indices_dict_with_xlim,
    **kwargs,
):
    """
    function for comparing model fits between models
    """
    for filename_fit in [filename_fit1, filename_fit2]:
        draw_hist_of_fit_indices(
            # list includes mimimum of x axis, maximum of x axis, and binwidth of plot
            # fit_indices_dict_with_xlim argument moved to loop_for_draw_hist_of_fit_indices()
            filename_fit,
            vis_model_type,
            fit_index_xlim=None,
            fit_index=None,
            fit_indices_dict_with_xlim=fit_indices_dict_with_xlim,
            dtype_memmap="float32",
            fig_size=(12, 4),
            save_fig=True,
            plt_close=False,
            ax=None,
            **kwargs,
        )


def compare_cor_between_models(
    filename_cor1: str,
    filename_cor2: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    model_type_list,
):
    """
    function for comparing outputs of the models
    """
    model_type1 = get_strings_from_filename(filename_cor1, ["model_type"])
    model_type2 = get_strings_from_filename(filename_cor2, ["model_type"])
    model_type = reduce(add, [i for i in model_type1 if i in model_type2])
    model_indices = [model_type.index(i) for i in model_type_list]

    model_fa_list = [i for i in model_type if "model" in i]
    model_fa_indices = [model_fa_list.index(i) for i in model_type_list]
    # copy data
    dat1 = copy_memmap_output_data(filename_cor1)[:, model_indices]
    dat2 = copy_memmap_output_data(filename_cor2)[:, model_indices]
    # select models
    bool_array1 = get_boolean_fit_locally_globally(
        filename_cor1,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        model_type_list,
    )[:, model_fa_indices]
    bool_array2 = get_boolean_fit_locally_globally(
        filename_cor2,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        model_type_list,
    )[:, model_fa_indices]
    # insert nan in rejected models
    dat1[~bool_array1] = np.nan
    dat2[~bool_array2] = np.nan

    if len(model_type_list) == 1:
        fig, ax = plt.subplots()
        ax.scatter(dat1, dat2, s=1)
        abline(1, 0, ax)


def draw_heatmap_on_proportion_of_misfit_edges_per_network(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    edges_count_wide_df,
    node_summary_path: str,
    model: str,
    fig_size=(12, 4),
    plt_close=False,
    ax=None,
    **kwargs,
) -> pd.DataFrame:
    """
    function for getting proportion of misfit edges per combinations of network
    """
    gsr_type, scale_name = get_strings_from_filename(
        filename_cor, ["gsr_type", "scale_name"]
    )
    gsr_suffix = generate_gsr_suffix(gsr_type)

    misfit_edges_count_wide_df = get_n_of_misfit_edges_per_network(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        node_summary_path,
        model,
    )
    prop_edges_df = misfit_edges_count_wide_df / edges_count_wide_df

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(prop_edges_df, annot=True, fmt=".1%", ax=ax)
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.125, caption)
        fig.suptitle(fig_title)
        fig.tight_layout()

        fig_filename_suffix_thresholds = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_filename = f"heatmap_prop_misfit_edges_{fig_name_suffix}_{fig_filename_suffix_thresholds}.png"

        fig.savefig(op.join(save_fig_folder, fig_filename), bbox_inches="tight")
        if plt_close:
            plt.close()
    else:
        sns.heatmap(
            prop_edges_df,
            annot=True,
            fmt=".1%",
            ax=ax,
            vmax=kwargs["vmax"],
            vmin=kwargs["vmin"],
            annot_kws={"fontsize": kwargs["annot_fsize"]},
        )
        ax.set_title(f"{scale_name} {gsr_suffix}")
    return prop_edges_df


def loop_for_draw_heatmap_on_proportion_of_misfit_edges_per_network(
    n_edge: int,
    sample_n: int,
    trait_type_list: list[str],
    est_method: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    vis_model: str,
    plt_close=False,
    fig_size=(12, 4),
    **kwargs,
) -> None:
    """
    function for looping for drawing heatmap of proportion of misfit edges per network
    """
    edges_count_wide_df = get_n_of_edges_per_network(node_summary_path)
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, est_method, "correlation"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        elif trait_type == "mental":
            ncol = 4
        fig, axes = plt.subplots(
            2,
            ncol,
            figsize=(ncol * 3, 6),
        )
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
            for i, filename_cor in enumerate(filename_list_gsr_type):
                draw_heatmap_on_proportion_of_misfit_edges_per_network(
                    filename_cor,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    edges_count_wide_df,
                    node_summary_path,
                    vis_model,
                    ax=axes[j, i],
                    **kwargs,
                )
        fig.suptitle(
            f"Heatmap of proportion of excluded edges in {trait_scale_name} of {vis_model} with {est_method} (N = {sample_n}, number of edges = {n_edge})"
        )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.125, caption)
        fig.tight_layout()
        fig_filename_suffix_thresholds = generate_fig_name_suffix_from_thresholds(
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
        )
        fig_name = f"heatmap_prop_{trait_scale_name}_{est_method}_sampleN_{sample_n}_edgeN_{n_edge}_{fig_filename_suffix_thresholds}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def visualize_mean_error_variance_based_on_networks(
    filename_cor,
    vis_model_type,
    node_summary_path,
    vis_fc_or_trait="FC",
    target="std_estimates",
    ax=None,
    **kwargs,
):
    """
    function for calculating mean error variance considering networks
    """
    filename_after_cor_type = reduce(add, re.findall("Model_.*", filename_cor))

    params_dict = generate_params_dict(filename_after_cor_type)
    model_type, trait_type, control, scale_name, gsr_type = get_strings_from_filename(
        filename_after_cor_type,
        ["model_type", "trait_type", "control", "scale_name", "gsr_type"],
    )
    folder = get_scale_name_from_trait(trait_type)

    gsr_suffix = generate_gsr_suffix(gsr_type)

    param_position_dict = get_param_position_dict(
        trait_type, control, scale_name, vis_model_type
    )
    # get error variances
    fc_param, trait_param, cor_param = get_parameters(
        params_dict, vis_model_type, param_position_dict, target=target
    )
    if fc_param.shape[1] > 1:
        fc_param_mean = fc_param.mean(axis=1)

    edges_df = get_edge_summary(node_summary_path)
    np_index = edges_df.index
    fc_param_mean, trait_param = fc_param_mean[np_index], trait_param[np_index]
    fc_col_name = f"FC_{vis_model_type}"
    trait_col_name = f"trait_{vis_model_type}"
    edges_df[fc_col_name] = fc_param_mean
    edges_df[trait_col_name] = trait_param
    edges_df["net_comb"] = list(zip(edges_df.node1_net, edges_df.node2_net))
    edges_df["net_set"] = edges_df["net_comb"].apply(lambda x: frozenset(x))
    edges_df["net_set2"] = edges_df["net_set"].apply(
        lambda x: sorted([i for i in x], key=lambda y: NETWORK_ORDER_DICT[y])
    )
    edges_df["net_set2"] = edges_df.apply(lambda x: "_".join(x["net_set2"]), axis=1)
    if type(vis_model_type) is str:
        vis_model_type = [vis_model_type]
    edges_mean_df = (
        edges_df.melt(
            id_vars=["edge", "net_set2"],
            value_vars=[fc_col_name, trait_col_name],
            value_name="error_var",
            var_name="fc_or_trait",
        )
        .groupby(["fc_or_trait", "net_set2"])["error_var"]
        .mean()
        .reset_index()
        .assign(node_list=lambda x: x["net_set2"].str.split("_"))
    )
    edges_mean_df["node1"] = pd.Categorical(
        edges_mean_df["node_list"].apply(lambda x: x[0]),
        categories=NETWORK_ORDER_DICT.keys(),
    )
    edges_mean_df["node2"] = pd.Categorical(
        edges_mean_df["node_list"].apply(lambda x: x[-1]),
        categories=NETWORK_ORDER_DICT.keys(),
    )
    if vis_fc_or_trait == "FC":
        edges_mean_df = edges_mean_df.query("fc_or_trait == @fc_col_name")
    elif vis_fc_or_trait == "trait":
        edges_mean_df = edges_mean_df.query("fc_or_trait == @trait_col_name")
    edges_mean_wide_df = edges_mean_df.pivot_table(
        index="node1",
        columns="node2",
        values="error_var",
    )
    if ax is not None:
        sns.heatmap(
            edges_mean_wide_df,
            annot=True,
            ax=ax,
            vmax=kwargs["vmax"],
            vmin=kwargs["vmin"],
            annot_kws={"fontsize": kwargs["annot_fsize"], "fontfamily": "serif"},
        )
        ax.set_title(f"{scale_name} {gsr_suffix}")
        ax.set_xlabel("")
        ax.set_ylabel("")


def loop_for_visualize_mean_error_variance_based_on_networks(
    trait_type_list,
    n_edge,
    sample_n,
    est_method,
    vis_model_type,
    node_summary_path,
    **kwargs,
):
    """loop function"""
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, est_method, "correlation"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        elif trait_type == "mental":
            ncol = 4
        fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
            for i, filename_cor in enumerate(filename_list_gsr_type):
                visualize_mean_error_variance_based_on_networks(
                    filename_cor,
                    vis_model_type,
                    node_summary_path,
                    ax=axes[j, i],
                    **kwargs,
                )
        fig.suptitle(
            f"Heatmap of error variance in {trait_scale_name} of {vis_model_type} with {est_method} (N = {sample_n}, number of edges = {n_edge})"
        )
        fig.tight_layout()
        fig_name = f"heatmap_{vis_model_type}_{trait_scale_name}_{est_method}_sampleN_{sample_n}_edgeN_{n_edge}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def get_cor_array_passed_thresholds(
    filename_cor: str, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
) -> tuple[list[str], NDArray[Shape["Num_iter, Num_models"], Float]]:
    """
    function for getting array of correlation to draw scatter plot
    """
    model_type = tuple(
        get_strings_from_filename(filename_cor, ["model_type"], include_nofa_model=True)
    )[0]
    bool_array_inclusion = get_boolean_fit_locally_globally(
        filename_cor, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    cor_array = copy_memmap_output_data(filename_cor)
    model_nofa_list = [i for i in model_type if "model" not in i]
    bool_add = np.ones(shape=(cor_array.shape[0], len(model_nofa_list))).astype("bool")
    bool_array_inclusion = np.concatenate([bool_add, bool_array_inclusion], axis=1)
    cor_array[~bool_array_inclusion] = np.nan
    return model_type, cor_array


def abline(slope, intercept, ax):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, "--", color="black", linewidth=1)


def draw_scatter_cor_models(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type_list: list[str] = None,
    fig_size=(6, 3),
    ax=None,
) -> None:
    """
    function for drawing scatterplots of correlation values
    """
    model_type, cor_array = get_cor_array_passed_thresholds(
        filename_cor, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    trait_type, scale_name, gsr_type = get_strings_from_filename(
        filename_cor,
        ["trait_type", "scale_name", "gsr_type"],
        # include_nofa_model=True,
    )
    folder = get_scale_name_from_trait(trait_type)
    model_index = [model_type.index(i) for i in vis_model_type_list]
    cor_array = cor_array[:, model_index]
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        x, y = cor_array[:, 0], cor_array[:, 1]
        ax.scatter(x, y, s=1)
        ax.set_xlabel(f"Correlation (Pearson's r) calculated by {models[0]}")
        ax.set_ylabel(f"Correlation (Pearson's r)\n calculated by {models[1]}")

        (
            fig_title_suffix,
            fig_name_suffix,
            save_fig_folder,
        ) = generate_suffixes_and_saving_folder(filename_cor)
        fig_title = f"Scatterplot of correlations {fig_title_suffix}"
        fig.suptitle(fig_title)
        # add slope of one and regression line
        abline(1, 0, ax)
        idx = np.isfinite(x) & np.isfinite(y)
        slope, intercept = np.polyfit(x[idx], y[idx], deg=1)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, "-", color="red", linewidth=1)

        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.15, caption)
        corr_coef = ma.corrcoef(
            ma.masked_invalid(cor_array[:, 0]),
            ma.masked_invalid(cor_array[:, 1]),
            rowvar=False,
        )[0, 1]
        plt.text(
            0.005,
            0.99,
            f"Pearson's r = {round(corr_coef, 2)}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

        cor_array_na_removed = cor_array[~np.isnan(cor_array).any(axis=1)]
        spearman_res = spearmanr(cor_array_na_removed)
        corr_s = spearman_res.correlation
        plt.text(
            0.005,
            0.90,
            f"Spearman's rho = {round(corr_s, 2)}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        plt.text(
            0.005,
            0.81,
            f"Improvement factor = {round(slope, 2)}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        fig.tight_layout()

        fig_filename_suffix_thresholds = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )

        filename = f"Scatterplot_{models[0]}_{models[1]}_{fig_name_suffix}_{fig_filename_suffix_thresholds}.png"
        fig.savefig(op.join(save_fig_folder, filename), bbox_inches="tight")
    else:
        # draw scatter plots of correlations (z values)
        cor_array = np.arctanh(cor_array)
        x, y = cor_array[:, 0], cor_array[:, 1]
        ax.scatter(x, y, s=1)
        # add slope of one
        abline(1, 0, ax)
        idx = np.isfinite(x) & np.isfinite(y)
        # add regression line
        slope, intercept = np.polyfit(x[idx], y[idx], deg=1)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, "-", color="red", linewidth=1)
        corr_coef = ma.corrcoef(
            ma.masked_invalid(cor_array[:, 0]),
            ma.masked_invalid(cor_array[:, 1]),
            rowvar=False,
        )[0, 1]

        cor_array_na_removed = cor_array[~np.isnan(cor_array).any(axis=1)]
        spearman_res = spearmanr(cor_array_na_removed)
        corr_s = spearman_res.correlation
        ax.text(
            0.1,
            0.9,
            f"r = {round(corr_coef, 2)}\nrho = {round(corr_s, 2)}\nIF = {round(slope, 2)}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        ax.set_xlabel(f"z value by {vis_model_type_list[0]}")
        ax.set_ylabel(f"z value by {vis_model_type_list[1]}")
        gsr_suffix = generate_gsr_suffix(gsr_type)
        ax.set_title(f"{scale_name} {gsr_suffix}")


def loop_for_draw_scatter_cor_models(
    trait_type_list,
    n_edge: int,
    sample_n: int,
    est_method: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type_list: list[str] = None,
    plt_close=False,
):
    """
    function for drawing scatterplots using loop
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, est_method, "correlation"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        elif trait_type == "mental":
            ncol = 4
        fig, axes = plt.subplots(
            2,
            ncol,
            figsize=(ncol * 3, 6),
        )
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
            for i, filename_cor in enumerate(filename_list_gsr_type):
                draw_scatter_cor_models(
                    filename_cor,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    vis_model_type_list=vis_model_type_list,
                    ax=axes[j, i],
                )
        fig.suptitle(
            f"Scatterplots of z values (inter-factor correlation) in {trait_scale_name} of {' and '.join(vis_model_type_list)} with {est_method} (N = {sample_n}, number of edges = {n_edge})"
        )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.125, caption)
        fig.tight_layout()
        fig_filename_suffix_thresholds = generate_fig_name_suffix_from_thresholds(
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
        )
        fig_name = f"scatterplots_{'_'.join(vis_model_type_list)}_{trait_scale_name}_{est_method}_sampleN_{sample_n}_edgeN_{n_edge}_{fig_filename_suffix_thresholds}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def save_files(save_path, filename1, filename2, comb_model_type) -> None:
    """use in combine_memmap_cv() to save files"""
    # get index of comb_type
    _, _, model_nofa_list = get_model_strings(filename1)
    col_index = model_nofa_list.index(comb_model_type)
    # copy fc-trait correlation data of train and test samples
    dat1 = copy_memmap_output_data(filename1)
    dat2 = copy_memmap_output_data(filename2)
    # generate combined array
    cor_fc_trait_array = np.vstack([dat1[:, col_index], dat2[:, col_index]]).T
    # save combined array
    np.save(save_path, cor_fc_trait_array)


def combine_memmap_cv(
    trait_type_list,
    n_edge,
    sample_n,
    est_method,
    split_ratio,
    comb_model_type,
    over_write=False,
) -> None:
    """
    function for combining memmap files of no-factor models generated through cv
    """
    fold_n = int(1 / split_ratio)
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        subscale_list = get_subscale_list(trait_scale_name)
        for scale_name in subscale_list:
            for g, gsr_type in enumerate(["nogs", "gs"]):
                for k in range(fold_n):
                    gsr_suffix = generate_gsr_suffix(gsr_type)
                    print(
                        f"Processing fold {k} of {scale_name} in {trait_scale_name} {gsr_suffix}"
                    )
                    filename1, filename2 = get_latest_pair_files_with_args(
                        trait_scale_name,
                        scale_name,
                        n_edge,
                        sample_n,
                        split_ratio,
                        k,
                        "correlation",
                        est_method,
                        "_" + gsr_type + "_",
                        comb_model_type,
                        ext=".dat",
                    )
                    # generate new filename
                    filename = combine_memmap_filenames(
                        filename1, filename2, comb_model_type
                    )
                    save_path = op.join(
                        SCHAEFER_DIR,
                        trait_scale_name,
                        scale_name,
                        "correlation",
                        filename,
                    )

                    if not over_write:
                        if not op.isfile(save_path):
                            save_files(save_path, filename1, filename2, comb_model_type)
                    else:
                        save_files(save_path, filename1, filename2, comb_model_type)


def subset_fit_filenames(combined_filename, ext: str) -> tuple[str, str]:
    """
    function for subsettig filenames from combined ones
    """
    # subset sampleN in filename1
    filename1_sample_n = reduce(add, re.findall("sampleN_([0-9]+)", combined_filename))
    filename2_sample_n = reduce(add, re.findall("([0-9]+)_Fold", combined_filename))
    filename1_time = reduce(
        add,
        re.findall("split_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", combined_filename),
    )
    filename2_time = reduce(
        add,
        re.findall("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})" + ext, combined_filename),
    )
    if "factor_scores" in combined_filename:
        filename_suffix = reduce(
            add, re.findall("factor_scores_(.*)", combined_filename)
        )

    filename1_suffix = filename_suffix.replace(filename2_sample_n + "_", "")
    filename1_fit = "fit_indices_" + filename1_suffix.replace(
        "_" + filename2_time, ""
    ).replace(ext, ".dat")

    filename2_suffix = filename_suffix.replace(filename1_sample_n + "_", "")
    filename2_fit = "fit_indices_" + filename2_suffix.replace(
        "_" + filename1_time, ""
    ).replace(ext, ".dat")

    return filename1_fit, filename2_fit


def draw_scatter_fc_trait_cor_cv(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    est_method: str,
    split_ratio: float,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_type_list: list[str],
    cor_type="z",
):
    """
    function for drawing scatterplots on correlation of fc-trait correlations between samples
    """
    fold_n = int(1 / split_ratio)
    # sort vis_type_list to Model_* be the first
    vis_type_list = sorted(vis_type_list, key=lambda x: (x != "Model_*", x))
    # trait level loop
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        # set figures
        subscale_list = get_subscale_list(trait_scale_name)
        # scale level loop
        for s, scale_name in enumerate(subscale_list):
            # create output array (2 represents length of gsr_type)
            cor_array = np.empty(shape=(fold_n, 2, len(vis_type_list)))
            cor_array[:] = np.nan

            fig, axes = plt.subplots(
                2 * len(vis_type_list),
                fold_n,
                figsize=(12, 8),
                sharex=True,
                sharey=True,
            )
            # gsr type level loop
            for g, gsr_type in enumerate(["nogs", "gs"]):
                gsr_suffix = generate_gsr_suffix(gsr_type)

                # fold level loop
                for k in range(fold_n):
                    for v, vis_model_type in enumerate(vis_type_list):
                        # get filenames
                        (
                            filename,
                            sample_n1,
                            sample_n2,
                        ) = get_latest_combined_single_file_with_args(
                            trait_scale_name,
                            scale_name,
                            n_edge,
                            sample_n,
                            split_ratio,
                            k,
                            "correlation",
                            est_method,
                            gsr_type,
                            vis_model_type,
                            ext=".npy",
                            get_sample_n=True,
                        )
                        # print(
                        #    f"Processing fold {k} in {scale_name} of {trait_type} {gsr_suffix}"
                        # )
                        cor_array_vis = np.load(
                            op.join(
                                SCHAEFER_DIR,
                                trait_scale_name,
                                scale_name,
                                "correlation",
                                filename,
                            )
                        )
                        if cor_type == "z":
                            cor_array_vis = np.arctanh(cor_array_vis)
                        # get filenames of fit indices
                        if "factor_scores" in filename:
                            filename1_fit, filename2_fit = subset_fit_filenames(
                                filename, ".npy"
                            )
                            # select edges from fits
                            fit_passed_bool1 = get_boolean_fit_locally_globally(
                                filename1_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                model_type_list=[vis_model_type.lower()],
                            )
                            fit_passed_bool2 = get_boolean_fit_locally_globally(
                                filename2_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                model_type_list=[vis_model_type.lower()],
                            )
                            fit_passed_both = fit_passed_bool1 & fit_passed_bool2
                        # subset data
                        cor_array_vis = cor_array_vis[np.squeeze(fit_passed_both), :]
                        # draw scatterplots
                        row_index = g * v + v + g
                        if (g == 1) and (v == 0):
                            row_index = 2
                        ax = axes[row_index, k]
                        ax.scatter(cor_array_vis[:, 1], cor_array_vis[:, 0], s=0.5)
                        ax.set_title(f"{vis_model_type} {gsr_suffix} in fold {k + 1}")
                        ax.set_ylabel(f"sample 1 (n = {sample_n1})")
                        ax.set_xlabel(f"sample 2 (n = {sample_n2})")
                        # add regression line
                        # slope, intercept = np.polyfit(cor_array_vis[0], cor_array_vis[1], deg=1)
                        # x_vals = np.array(ax.get_xlim())
                        # y_vals = intercept + slope * x_vals
                        # ax.plot(x_vals, y_vals, "-", color="black", linewidth=1)
                        # add annotation of correlation
                        pearson_r = pearsonr(cor_array_vis[:, 0], cor_array_vis[:, 1])[
                            0
                        ]
                        spearman_r = spearmanr(
                            cor_array_vis[:, 0], cor_array_vis[:, 1]
                        )[0]
                        ax.text(
                            0.01,
                            0.99,
                            f"Pearson's r = {pearson_r:.3f}\nSpearman's rho = {spearman_r:.3f}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                        )

                fig.suptitle(
                    f"Scatterplots of FC-trait correlations ({cor_type} values) in split samples in {scale_name} of {trait_scale_name}"
                )
                caption = generate_caption_of_figure_on_parameter_thresholds(
                    error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
                )
                fig.text(
                    0,
                    -0.125,
                    f"Edges are removed based on the following thresholds.\n{caption}",
                )
                fig.tight_layout()
                # save figure
                fig_filename_suffix_thresholds = (
                    generate_fig_name_suffix_from_thresholds(
                        error_vars_dict,
                        cor_min_max_dict,
                        fit_indices_thresholds_dict,
                    )
                )
                save_fig_folder = op.join(
                    SCHAEFER_DIR, trait_scale_name, scale_name, "figures"
                )
                os.makedirs(save_fig_folder, exist_ok=True)
                filename = f'scatterplots_fold_{k}_{cor_type}_{"_".join(vis_type_list)}_{fig_filename_suffix_thresholds}.png'
                fig.savefig(op.join(save_fig_folder, filename))


def vis_n_of_dropped_edges_for_kfold_cv(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    est_method: str,
    split_ratio: float,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_type_list: list[str],
):
    """
    visualize number of dropped edges in folds
    """
    fold_n = int(1 / split_ratio)
    # sort vis_type_list to Model_* be the first
    vis_type_list = sorted(vis_type_list, key=lambda x: (x != "Model_*", x))
    # trait level loop
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        # create an output array
        subscale_list = get_subscale_list(trait_scale_name)
        # 2 and 2 represent number of gsr_type and number of split samples, respectively
        n_edge_array = np.empty(shape=(len(subscale_list), fold_n, 2, 2))
        n_edge_array[:] = np.nan
        for s, scale_name in enumerate(subscale_list):
            # gsr type level loop
            for g, gsr_type in enumerate(["nogs", "gs"]):
                gsr_suffix = generate_gsr_suffix(gsr_type)
                # fold level loop
                for k in range(fold_n):
                    for v, vis_model_type in enumerate(vis_type_list):
                        # get filenames
                        (
                            filename,
                            sample_n1,
                            sample_n2,
                        ) = get_latest_combined_single_file_with_args(
                            trait_scale_name,
                            scale_name,
                            n_edge,
                            sample_n,
                            split_ratio,
                            k,
                            "correlation",
                            est_method,
                            gsr_type,
                            vis_model_type,
                            ext=".npy",
                            get_sample_n=True,
                        )
                        # get filenames of fit indices
                        if "factor_scores" in filename:
                            filename1_fit, filename2_fit = subset_fit_filenames(
                                filename, ".npy"
                            )
                            # select edges from fits
                            fit_passed_bool1 = get_boolean_fit_locally_globally(
                                filename1_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                model_type_list=[vis_model_type.lower()],
                            )
                            fit_passed_bool2 = get_boolean_fit_locally_globally(
                                filename2_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                model_type_list=[vis_model_type.lower()],
                            )
                            n_fit_not_passed_bool1 = int(np.sum(~fit_passed_bool1))
                            n_fit_not_passed_bool2 = int(np.sum(~fit_passed_bool2))
                            n_edge_array[s, k, g, :] = (
                                n_fit_not_passed_bool1,
                                n_fit_not_passed_bool2,
                            )
        # 2 and 2 represent number of gsr_type and number of split samples, respectively
        n_edge_array_reshaped = n_edge_array.reshape(
            len(subscale_list) * fold_n * 2, 2, order="F"
        )
        n_edge_pd = pd.DataFrame(n_edge_array_reshaped, columns=["sample1", "sample2"])
        n_edge_pd["subscale"] = subscale_list * fold_n * 2
        n_edge_pd["Fold"] = reduce(
            add, [list(str(i + 1)) * len(subscale_list) for i in range(fold_n)] * 2
        )
        n_edge_pd["gsr_type"] = ["nogs"] * len(subscale_list) * fold_n + ["gs"] * len(
            subscale_list
        ) * fold_n
        long_n_edge_pd = n_edge_pd.melt(
            id_vars=["subscale", "Fold", "gsr_type"],
            value_vars=["sample1", "sample2"],
            var_name="sample",
            value_name="n_edge",
        )
        long_n_edge_pd["sample"] = long_n_edge_pd["sample"].replace(
            ["sample1", "sample2"],
            [f"sample 1 (n = {sample_n1})", f"sample 2 (n = {sample_n2})"],
        )
        # draw figures
        sns.set(font_scale=2)
        g = sns.catplot(
            data=long_n_edge_pd,
            x="Fold",
            y="n_edge",
            col="subscale",
            row="gsr_type",
            hue="sample",
            kind="bar",
            margin_titles=True,
        )
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        # set title
        g.fig.suptitle(
            f"Number or removed edges based on thresholds in {trait_scale_name}"
        )
        # add caption on thresholds
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        g.fig.text(
            0,
            -0.15,
            f"Edges are removed based on the following thresholds.\n{caption}",
        )
        sns.move_legend(g, "upper right", bbox_to_anchor=(1, 0))
        g.fig.tight_layout()
        # legend_labels = [f'sample 1 (n = {sample_n1})', f'sample 2 (n = {sample_n2})']
        # g.add_legend(legend_data={key: value for key, value in zip(legend_labels, g._legend_data.values())})
        # save figures
        fig_filename_suffix_thresholds = generate_fig_name_suffix_from_thresholds(
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
        )
        save_fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig_filename = (
            f"n_edge_removed_{trait_scale_name}_{fig_filename_suffix_thresholds}.png"
        )
        g.fig.savefig(op.join(save_fig_folder, fig_filename), bbox_inches="tight")


def get_boolean_index_of_both_passed_edges(
    filename1_fit,
    filename2_fit,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type,
):
    """get boolean of edges where both split samples passed thresholds"""
    fit_passed_bool1 = get_boolean_fit_locally_globally(
        filename1_fit,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        model_type_list=[vis_model_type.lower()],
    )
    fit_passed_bool2 = get_boolean_fit_locally_globally(
        filename2_fit,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        model_type_list=[vis_model_type.lower()],
    )
    fit_passed_both = fit_passed_bool1 & fit_passed_bool2
    return fit_passed_both


def draw_heatmaps_networks_of_kfolds(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    est_method: str,
    split_ratio: float,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_type_list: list[str],
    node_summary_path: str,
    cor_type="z",
):
    """
    draw heatmaps of percentages of edges with greater power considering networks with kfold cv
    """
    fold_n = int(1 / split_ratio)
    # sort vis_type_list to Model_* be the first
    vis_type_list = sorted(vis_type_list, key=lambda x: (x != "Model_*", x))

    # read and edit dataframe on edges
    edges_df = get_edge_summary(node_summary_path)
    edges_df["net_comb"] = list(zip(edges_df.node1_net, edges_df.node2_net))
    edges_df["net_set"] = edges_df["net_comb"].apply(lambda x: frozenset(x))
    edges_df["net_set2"] = edges_df["net_set"].apply(
        lambda x: sorted([i for i in x], key=lambda y: NETWORK_ORDER_DICT[y])
    )
    edges_df["net_set2"] = edges_df.apply(lambda x: "_".join(x["net_set2"]), axis=1)
    # to reorder array
    np_index = edges_df.index
    # trait level loop
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        # set figures
        subscale_list = get_subscale_list(trait_scale_name)
        # scale level loop
        for s, scale_name in enumerate(subscale_list):
            # gsr type level loop
            for g, gsr_type in enumerate(["nogs", "gs"]):
                gsr_suffix = generate_gsr_suffix(gsr_type)
                # fold level loop
                for k in range(fold_n):
                    for v, vis_model_type in enumerate(vis_type_list):
                        # get filenames
                        (
                            filename,
                            sample_n1,
                            sample_n2,
                        ) = get_latest_combined_single_file_with_args(
                            trait_scale_name,
                            scale_name,
                            n_edge,
                            sample_n,
                            split_ratio,
                            k,
                            "correlation",
                            est_method,
                            gsr_type,
                            vis_model_type,
                            ext=".npy",
                            get_sample_n=True,
                        )
                        # load a target file
                        cor_array_vis = np.load(
                            op.join(
                                SCHAEFER_DIR,
                                trait_scale_name,
                                scale_name,
                                "correlation",
                                filename,
                            )
                        )
                        if cor_type == "z":
                            cor_array_vis = np.arctanh(cor_array_vis)
                        # reorder array
                        cor_array_vis = cor_array_vis[np_index]
                        # get filenames of fit indices
                        if "factor_scores" in filename:
                            filename1_fit, filename2_fit = subset_fit_filenames(
                                filename, ".npy"
                            )
                            # select edges from fits
                            fit_passed_both = get_boolean_index_of_both_passed_edges(
                                filename1_fit,
                                filename2_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                vis_model_type,
                            )


def vis_cor_of_cor_fc_trait_cv(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    est_method: str,
    split_ratio: float,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_type_list: list[str],
    cor_type="z",
    calc_cor_type="pearson",
):
    """
    function for calculating and visualising correlation of fc-trait correlations between samples
    """
    fold_n = int(1 / split_ratio)
    output_array = np.empty(
        shape=(
            n_edge,
            4,
            fold_n,
        )
    )
    # sort vis_type_list to Model_* be the first
    vis_type_list = sorted(vis_type_list, key=lambda x: (x != "Model_*", x))
    # trait level loop
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)

        # set figures
        subscale_list = get_subscale_list(trait_scale_name)
        ax_num = len(subscale_list)
        fig, axes = plt.subplots(2, ax_num, sharey=True, figsize=(12, 6))
        # gsr type level loop
        for g, gsr_type in enumerate(["nogs", "gs"]):
            gsr_suffix = generate_gsr_suffix(gsr_type)

            # scale level loop
            for s, scale_name in enumerate(subscale_list):
                # create output array (2 represents length of gsr_type)
                cor_array = np.empty(shape=(fold_n, 2, len(vis_type_list)))
                cor_array[:] = np.nan

                # fold level loop
                for k in range(fold_n):
                    for v, vis_model_type in enumerate(vis_type_list):
                        # get filenames
                        filename = get_latest_combined_single_file_with_args(
                            trait_scale_name,
                            scale_name,
                            n_edge,
                            sample_n,
                            split_ratio,
                            k,
                            "correlation",
                            est_method,
                            gsr_type,
                            vis_model_type,
                            ext=".npy",
                        )
                        print(
                            f"Processing fold {k} in {scale_name} of {trait_type} {gsr_suffix}"
                        )
                        cor_array_analysis = np.load(
                            op.join(
                                SCHAEFER_DIR,
                                trait_scale_name,
                                scale_name,
                                "correlation",
                                filename,
                            )
                        )
                        cor_array_z_analysis = np.arctanh(cor_array_analysis)
                        # get filenames of fit indices
                        if "factor_scores" in filename:
                            filename1_fit, filename2_fit = subset_fit_filenames(
                                filename, ".npy"
                            )
                            # select edges from fits
                            fit_passed_bool1 = get_boolean_fit_locally_globally(
                                filename1_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                model_type_list=[vis_model_type.lower()],
                            )
                            fit_passed_bool2 = get_boolean_fit_locally_globally(
                                filename2_fit,
                                error_vars_dict,
                                cor_min_max_dict,
                                fit_indices_thresholds_dict,
                                model_type_list=[vis_model_type.lower()],
                            )
                            fit_passed_both = fit_passed_bool1 & fit_passed_bool2
                        # subset data
                        cor_array_analysis = cor_array_analysis[
                            np.squeeze(fit_passed_both), :
                        ]
                        cor_array_z_analysis = cor_array_z_analysis[
                            np.squeeze(fit_passed_both), :
                        ]

                        if cor_type == "z":
                            target_array = cor_array_z_analysis
                        elif cor_type == "r":
                            target_array = cor_array_analysis
                        if calc_cor_type == "pearson":
                            calc_func = pearsonr
                        elif calc_cor_type == "spearman":
                            calc_func = spearmanr
                        print(f"Processing {vis_model_type}")
                        r = calc_func(target_array[:, 0], target_array[:, 1])[0]
                        cor_array[k, g, v] = r
                # create dataframe
                cor_array_reshaped = cor_array.reshape(
                    fold_n * 2, len(vis_type_list), order="F"
                )
                cor_pd = pd.DataFrame(cor_array_reshaped, columns=vis_type_list)
                cor_pd["Fold"] = [i + 1 for i in range(fold_n)] * 2
                cor_pd["gsr_type"] = ["nogs"] * 4 + ["gs"] * 4
                long_cor_pd = cor_pd.melt(
                    id_vars=["Fold", "gsr_type"],
                    value_vars=vis_type_list,
                    var_name="Model",
                    value_name="r",
                )
                # draw bars
                sns.barplot(
                    long_cor_pd.query("gsr_type == @gsr_type"),
                    x="Fold",
                    y="r",
                    hue="Model",
                    ax=axes[g, s],
                )
                # set title with scale name
                axes[g, s].set_title(f"{scale_name} {gsr_suffix}")
        fig.suptitle(
            f"{calc_cor_type.capitalize()} correlation of FC-trait correlations ({cor_type} values) between split samples in {trait_scale_name} (total sample size is {sample_n} and split ratio is 1:{fold_n - 1})"
        )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(
            0,
            -0.125,
            f"Edges are removed based on the following thresholds.\n{caption}",
        )
        fig.tight_layout()


def combine_memmap_filenames(filename1, filename2, comb_type):
    """
    function for combining filenames of no factor models
    """
    # subset sampleN in filename2
    filename2_sample_n = reduce(add, re.findall("sampleN_([0-9]+)", filename2))
    # replace sampleN in filename1 with sampleNs in filename1 and filename2
    filename1_sample_str = reduce(add, re.findall("sampleN_[0-9]+", filename1))
    filename2_sample_n = reduce(add, re.findall("sampleN_([0-9]+)", filename2))
    new_sample_n_str = filename1_sample_str + "_" + filename2_sample_n
    filename = filename1.replace(filename1_sample_str, new_sample_n_str)
    # subset time in filename2
    filename2_time = reduce(
        add, re.findall("split_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", filename2)
    )

    filename = filename.replace(".dat", "") + "_" + filename2_time
    # replace calculation names
    model_str = reduce(add, re.findall("Model_(.*)_Trait", filename))
    filename = filename.replace(model_str, comb_type)
    return filename


def examine_reliability_of_edges(
    trait_type_list, n_edge, sample_n, est_method, model_type
):
    """
    examine inter-session relaibility of FC
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        subscale_list = get_subscale_list(trait_scale_name)
        for scale_name in subscale_list:
            for g, gsr_type in enumerate(["nogs", "gs"]):
                pass


def calc_cor_fscores(
        sample_n,
        parcellation='Schaefer',
        controls=['age', 'gender', 'MeanRMS'],
        mean_fc_filenames_dict={},
        subjects_filename=None,
        test_n_edge=None
        ):
    """
    Calculate correlations between factor score estimates of traits and RSFC
    subjects_filename argument should be modified when conducting cross-validation
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    fscore_dir = op.join(atlas_dir, 'reliability', 'factor_scores', 'combined')
    fc_fscore_filenames = os.listdir(fscore_dir)
    controlling_suffix = '_'.join(controls)
    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    n_edge = len(edges_df) if test_n_edge is None else test_n_edge
    # empty list for creating df
    cor_list, trait_type_list, scale_name_list, gsr_list, cor_type_list = [], [], [], [], []
    # create list for selecting fc data
    mri_subjects_list = [i for i in os.listdir(op.join(HCP_ROOT_DIR, 'data')) if not i.startswith('.')]
    subjects_list = pd.read_csv(op.join(HCP_ROOT_DIR, 'derivatives', subjects_filename), header=None).loc[:, 0].to_list()
    subjects_list = [str(i) for i in subjects_list]
    mri_subjects_bool_list = [i in subjects_list for i in mri_subjects_list]
    for g, gsr_type in enumerate(["nogs", "gs"]):
        fc_array = np.arctanh(np.load(op.join(atlas_dir, mean_fc_filenames_dict.get(gsr_type))))
        fc_array = fc_array[:, mri_subjects_bool_list, :]
        fc_means = np.mean(fc_array, axis=2)
        for trait_type in TRAIT_TYPES:
            print(f'Processing {trait_type}')
            trait_scale_name = get_scale_name_from_trait(trait_type)
            subscale_list = get_subscale_list(trait_scale_name)
            # get factor scores of traits
            fscore_trait = pd.read_csv(op.join(atlas_dir, trait_scale_name, 'tables', 'fscore.csv'))
            # get summary scores of traits
            trait_summary_data = read_trait_data(trait_type=trait_type, subjects_filename=subjects_filename)
            for scale_name in subscale_list:
                print(f'Processing {scale_name}')
                trait_fscores = fscore_trait[scale_name]
                trait_summary = trait_summary_data[scale_name]
                print(f'Processing {gsr_type}')
                # filter fscore filenames
                filenames = [
                        i for i in fc_fscore_filenames 
                        if f'_{gsr_type}_' in i 
                        and (('MSST' in i) or ('MultiStateSingleTrait' in i))
                        and controlling_suffix in i 
                        and f'sampleN_{sample_n}' in i
                        ]
                filename = sort_list_by_time(filenames)[-1]
                # get latest fscore filename
                fc_fscores = np.load(op.join(fscore_dir, filename))[..., 2]
                cor_array = np.empty(shape=n_edge*2)
                for i in range(n_edge):
                    # calculate correlation between factor scores
                    cor_array[i] = np.corrcoef(fc_fscores[i, ...], trait_fscores)[0, 1]
                    # calculate correlation between mean scores
                    cor_array[n_edge+i] = np.corrcoef(fc_means[i, ...], trait_summary)[0, 1]
                # add elements to list to create df
                trait_type_list.append([trait_type] * n_edge * 2)
                scale_name_list.append([scale_name] * n_edge * 2)
                gsr_list.append([gsr_type] * n_edge * 2)
                cor_type_list.append(['score'] * n_edge + ['mean'] * n_edge)
                cor_list.append(cor_array)
    # number of scales (12), gsr (2), mean vs fscores (2)
    iteration = 12 * 2 * 2
    edges_df = edges_df.loc[:n_edge-1, :]
    results_df = pd.DataFrame(
            {
                'edge': edges_df['edge'].to_list() * iteration,
                'node1': edges_df['node1'].to_list() * iteration,
                'node2': edges_df['node2'].to_list() * iteration,
                'net1': edges_df['node1_net'].to_list() * iteration,
                'net2': edges_df['node2_net'].to_list() * iteration,
                'trait_type': reduce(add, trait_type_list),
                'scale_name': reduce(add, scale_name_list),
                'gsr_type': reduce(add, gsr_list),
                'cor_type': reduce(add, cor_type_list),
                'cor': np.array(cor_list).flatten()
                }
            )
#    edges_df['trait_type'] = reduce(add, trait_type_list)
#    edges_df['scale_name'] = reduce(add, scale_name_list)
#    edges_df['gsr_type'] = reduce(add, gsr_list)
#    edges_df['cor'] = np.array(cor_list).flatten()
    return results_df

def calc_cor_of_fscores(
    trait_type_list,
    n_edge,
    sample_n,
    est_method,
    split_ratio,
    model_type,
):
    """
    function for calculating correlation of factor scores
    """
    fold_n = int(1 / split_ratio)
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        subscale_list = get_subscale_list(trait_scale_name)
        for scale_name in subscale_list:
            for g, gsr_type in enumerate(["nogs", "gs"]):
                for k in range(fold_n):
                    gsr_suffix = gsr_suffix = generate_gsr_suffix(gsr_type)
                    print(
                        f"Processing fold {k} in {scale_name} of {trait_type} {gsr_suffix}"
                    )
                    filename1, filename2 = get_latest_pair_files_with_args(
                        trait_scale_name,
                        scale_name,
                        n_edge,
                        sample_n,
                        split_ratio,
                        k,
                        "factor_scores",
                        est_method,
                        "_" + gsr_type + "_",
                        model_type,
                    )
                    fscore_dat1 = copy_memmap_output_data(filename1)
                    fscore_dat2 = copy_memmap_output_data(filename2)

                    # subset sampleN in filename2
                    filename2_sample_n = reduce(
                        add, re.findall("sampleN_([0-9]+)", filename2)
                    )
                    # replace sampleN in filename1 with sampleNs in filename1 and filename2
                    filename1_sample_str = reduce(
                        add, re.findall("sampleN_[0-9]+", filename1)
                    )
                    filename2_sample_n = reduce(
                        add, re.findall("sampleN_([0-9]+)", filename2)
                    )
                    new_sample_n_str = filename1_sample_str + "_" + filename2_sample_n
                    filename = filename1.replace(filename1_sample_str, new_sample_n_str)
                    # subset time in filename2
                    filename2_time = reduce(
                        add,
                        re.findall(
                            "split_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", filename2
                        ),
                    )
                    filename = (
                        "pearson_" + filename.replace(".dat", "") + "_" + filename2_time
                    )

                    # create array storing outputs
                    cor_fc_trait_array = np.empty(shape=(n_edge, 2), dtype="float32")
                    cor_fc_trait_array[:] = np.nan
                    # calculate correlaion of factor scores within train and test samples
                    for i in range(n_edge):
                        cor_fc_trait_array[i, 0] = np.corrcoef(fscore_dat1[i, :, :].T)[
                            0, 1
                        ]
                        cor_fc_trait_array[i, 1] = np.corrcoef(fscore_dat2[i, :, :].T)[
                            0, 1
                        ]
                    # save npy file
                    np.save(
                        op.join(
                            SCHAEFER_DIR,
                            trait_scale_name,
                            scale_name,
                            "correlation",
                            filename,
                        ),
                        cor_fc_trait_array,
                    )


def draw_figures_on_cor_of_fc_trait_cor(
    cor_of_cor_fc_trait_array, fold_n, vis_type, scale_name, trait_scale_name
):
    long_data = (
        pd.DataFrame(
            cor_of_cor_fc_trait_array,
            columns=[i + "_" + j for i in ["r", "z"] for j in ["pearson", "spearman"]],
        )
        .assign(
            fold=[i for i in range(fold_n)] * 2,
            gsr_type=["nogs"] * fold_n + ["gs"] * fold_n,
        )
        .melt(
            id_vars=["fold", "gsr_type"],
            value_vars=["r_pearson", "r_spearman", "z_pearson", "z_spearman"],
        )
        .assign(
            cor_type=lambda x: x["variable"].str.split("_").apply(lambda x: x[0]),
            calc_cor_type=lambda x: x["variable"].str.split("_").apply(lambda x: x[1]),
        )
    )
    # draw figure
    g = sns.FacetGrid(
        data=long_data, row="calc_cor_type", col="cor_type", margin_titles=True
    )
    g.map(
        sns.barplot,
        "fold",
        "value",
        "gsr_type",
        palette="dark:#1f77b4",
        order=long_data["fold"].unique(),
        hue_order=["nogs", "gs"],
    )
    g.axes[0, 0].set_ylabel("r value")
    g.axes[1, 0].set_ylabel("r value")
    # set title
    if "model" in vis_type:
        title_suffix = f"factor scores of {vis_type}"
    else:
        title_suffix = vis_type
    g.fig.suptitle(
        f"Correlation between samples in {scale_name} of {trait_scale_name} using {title_suffix}"
    )
    g.add_legend()
    g.fig.tight_layout()
    # save figure
    g.fig.savefig(
        op.join(
            SCHAEFER_DIR,
            trait_scale_name,
            "figures",
            f"{fold_n}_fold_cv_fc_trait_cor_{vis_type}_{scale_name}.png",
        )
    )


def vcorrcoef(X, y):
    """
    function for conducting vectorized calculation of correlation
    """
    X_mean = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    y_mean = np.mean(y)
    r_numerator = np.sum((X - X_mean) * (y - y_mean), axis=1)
    r_denominator = np.sqrt(
        np.sum((X - X_mean) ** 2, axis=1) * np.sum((y - y_mean) ** 2)
    )
    r = r_numerator / r_denominator
    return r


def draw_venn_diagrams(
    filename_cor: str,
    error_vars_dict: ErrorVarsMinMaxDict,
    fit_indices_thresholds_dict: dict[FitIndicesList:float],
    cor_min_max_dict: CorMinMaxDict,
    model_type_list,
    dtype_memmap="float32",
    fig_size=(12, 4),
    save_fig=True,
    plt_close=False,
):
    """
    function for drawing venn diagrams of excluded edges
    """
    model_fa_list, models_fa_str, model_nofa_list = get_model_strings(filename_cor)
    num_iter, scale_name, trait_type, model_type, gsr_type = get_strings_from_filename(
        filename_cor,
        ["num_iter", "scale_name", "trait_type", "model_type", "gsr_type"],
        include_nofa_model=True,
    )

    gsr_suffix = generate_gsr_suffix(gsr_type)
    boolean_locally_specified_array = select_models_from_local_fits(
        filename_cor, error_vars_dict, cor_min_max_dict, model_type_list
    )

    folder = get_scale_name_from_trait(trait_type)
    fig, axes = plt.subplots(1, len(model_fa_list), figsize=fig_size)
    fig.suptitle(
        f"Venn diagram of locally misspecified edges in {scale_name} of {folder} {gsr_suffix}"
    )

    boolean_locals = [
        "indicators of FC",
        "indicators of trait",
        "inter-factor correlation",
        "any of them",
    ]

    for model_index, model in enumerate(model_fa_list):
        # print number of models with misspeifications
        for j, model_part in enumerate(boolean_locals):
            print(
                f"{scale_name} of {folder} {gsr_suffix}: In {model.title()}, number of misspecification in {model_part} is {num_iter - boolean_locally_specified_array[:, model_index, j].sum()}/{num_iter}"
            )
        # draw venn diagram representing edges where
        # fc or trait indicators and inter-factor correlation are specified
        locally_misspecified_edge_dict = get_edges_of_locally_misspecified_models(
            boolean_locally_specified_array,
            model_index,
        )
        venn3_subsets = [
            locally_misspecified_edge_dict["fc"],
            locally_misspecified_edge_dict["trait"],
            locally_misspecified_edge_dict["cor"],
        ]
        venn3(
            subsets=venn3_subsets,
            set_labels=(
                "FC",
                "Trait",
                "Correlation",
            ),
            ax=axes[model_index],
        )
        venn3_circles(subsets=venn3_subsets, ax=axes[model_index])
        axes[model_index].set_title(model)

    if save_fig:
        local_params_thresholds = generate_filename_suffix_on_local_fit(
            error_vars_dict, cor_min_max_dict
        )
        fig_filename = f"venn_diagram_local_misfit_{models_fa_str}_{local_params_thresholds}_{gsr_type}.png"
        fig_folder = op.join(SCHAEFER_DIR, folder, scale_name, "figures")
        os.makedirs(fig_folder, exist_ok=True)
        fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")
    if plt_close:
        plt.close()
    # identify and explore globally misfit models
    (
        boolean_globally_specified_array,
        boolean_globally_specified_array_each,
    ) = select_models_from_global_fits(
        filename_cor, fit_indices_thresholds_dict, model_type_list
    )
    # generate a dictionary storing sets of edges which passed cutoff criteria of fit indices
    globally_misfit_edges_dict = get_edges_of_globally_misfit_models(
        filename_fit, fit_indices_thresholds_dict, model_type_list
    )
    fit_indices, _ = get_names_and_thresholds_of_fit_indices(
        fit_indices_thresholds_dict
    )

    fig, axes = plt.subplots(1, len(model_fa_list), figsize=fig_size)
    fig.suptitle(
        f"Venn diagram of edges which did not passed cutoff critiria of global fit indices in {scale_name} of {folder} {gsr_suffix}"
    )
    for i, model in enumerate(model_fa_list):
        # draw venn diagrams
        if len(fit_indices) == 2:
            venn2(
                [
                    globally_misfit_edges_dict[model][fit_indices[0]],
                    globally_misfit_edges_dict[model][fit_indices[1]],
                ],
                (fit_indices[0], fit_indices[1]),
                ax=axes[i],
            )
            axes[i].set_title(model)
        # print number of misfits in each model
        for j, fit_index in enumerate(fit_indices):
            n_of_misfits = int(
                num_iter - boolean_globally_specified_array_each[:, j, i].sum()
            )
            print(
                f"{scale_name} of {folder} {gsr_suffix}: In {model.capitalize()}, number of misfits evaluated by {fit_index} is {n_of_misfits}/{num_iter}."
            )
    if save_fig:
        global_fit_thresholds = generate_filename_suffix_on_global_fit(
            fit_indices_thresholds_dict
        )
        fig_filename = f"venn_diagram_global_misfit_{models_fa_str}_{global_fit_thresholds}_{gsr_type}.png"
        fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")
    if plt_close:
        plt.close()
    # draw venn diagrams of selected edges by local or global misfits
    misfit_edges_dict = get_misfit_edges_locally_and_globally(
        boolean_locally_specified_array,
        boolean_globally_specified_array,
        model_fa_list,
    )

    fig, axes = plt.subplots(1, len(model_fa_list), figsize=fig_size)
    fig.suptitle(
        f"Venn diagram of edges which dod not pass local and global fit criteria in {scale_name} of {folder} {gsr_suffix}"
    )
    for i, model in enumerate(model_fa_list):
        venn2(
            [misfit_edges_dict[model]["local"], misfit_edges_dict[model]["global"]],
            ("Local fit", "Global fit"),
            ax=axes[i],
        )
        axes[i].set_title(model)
    if save_fig:
        fig_filename = (
            f"venn_diagram_local_global_misfit_{models_fa_str}_{gsr_type}.png"
        )
        fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")
    if plt_close:
        plt.close()


def draw_correlation_hist(
    filename_cor: str,
    error_vars_dict: ErrorVarsMinMaxDict,
    fit_indices_thresholds_dict: dict[FitIndicesList:float],
    cor_min_max_dict: CorMinMaxDict,
    model_type_list,
    dtype_memmap="float32",
    fig_size=(12, 4),
    save_fig=True,
    plt_close=False,
) -> None:
    """
    function for drawing histgorams of correlation calculated through models
    """
    model_fa_list, models_fa_str, model_nofa_list = get_model_strings(filename_cor)
    num_iter, scale_name, trait_type, model_type, gsr_type = get_strings_from_filename(
        filename_cor,
        ["num_iter", "scale_name", "trait_type", "model_type", "gsr_type"],
        include_nofa_model=True,
    )
    folder = get_scale_name_from_trait(trait_type)

    gsr_suffix = generate_gsr_suffix(gsr_type)
    boolean_locally_specified_array = select_models_from_local_fits(
        filename_cor, error_vars_dict, cor_min_max_dict, model_type_list
    )
    (
        boolean_globally_specified_array,
        boolean_globally_specified_array_each,
    ) = select_models_from_global_fits(
        filename_cor, fit_indices_thresholds_dict, model_type_list
    )
    # read and select correlation data
    cor_data = copy_memmap_output_data(
        filename_cor,
    )
    bool_array_inclusion = select_edges_locally_and_globally(
        boolean_locally_specified_array,
        boolean_globally_specified_array,
    )
    bool_array_inclusion_reshaped = get_boolean_array_of_selected_edges(
        bool_array_inclusion, model_nofa_list
    )
    cor_data[~bool_array_inclusion_reshaped] = np.nan

    # remove rows with missing values and create long format dataframe
    long_data = (
        pd.DataFrame(cor_data, columns=model_type)
        .reset_index(names="edge")
        .melt(id_vars="edge", var_name="model", value_name="r")
    )

    # draw kdeplot
    fig, ax = plt.subplots(figsize=fig_size)
    sns.kdeplot(data=long_data, x="r", hue="model", ax=ax)
    fig.suptitle(
        f"Density plot of correlation in {scale_name} in {folder} {gsr_suffix}"
    )
    local_fit_suffix = generate_filename_suffix_on_local_fit(
        error_vars_dict, cor_min_max_dict
    )
    global_fit_suffix = generate_filename_suffix_on_global_fit(
        fit_indices_thresholds_dict
    )
    if save_fig:
        fig_folder = op.join(SCHAEFER_DIR, folder, scale_name, "figures")
        fig_filename = f"cor_kde_{models_fa_str}_{local_fit_suffix}{global_fit_suffix}_{gsr_type}.png"
        fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")
    if plt_close:
        plt.close()


def loop_for_draw_correlation_hist(
    trait_type_list,
    n_edge: int,
    sample_n: int,
    error_vars_dict,
    fit_indices_thresholds_dict,
    cor_min_max_dict,
    save_fig=True,
    plt_close=True,
):
    """
    function for looping draw_correlation_hist()
    """
    filename_cor_list = get_latest_files_with_args(
        trait_type_list=trait_type_list,
        n_edge=n_edge,
        sample_n=sample_n,
        data_type="correlation",
    )
    for filename_cor in filename_cor_list:
        draw_correlation_hist(
            filename_cor,
            error_vars_dict,
            fit_indices_thresholds_dict,
            cor_min_max_dict,
            dtype_memmap="float32",
            fig_size=(12, 4),
            save_fig=save_fig,
            plt_close=plt_close,
        )


VisParam = Literal["FC", "trait", "cor"]


def check_models_rsfc(
    cov_cor: bool, day_cor: bool, phase_encoding: bool, order_in_day: bool
):
    # specify model
    if cov_cor and not phase_encoding and not day_cor and not order_in_day:
        model = "Model RSFC 1"
    elif cov_cor and day_cor and not phase_encoding and not order_in_day:
        model = "Model RSFC 2"
    elif cov_cor and phase_encoding and not day_cor and not order_in_day:
        model = "Model RSFC 3"
    elif cov_cor and order_in_day and not day_cor and not phase_encoding:
        model = "Model RSFC 4"
    return model


def separate_fit_and_correlation_dat_by_models(trait_type):
    """
    separate .dat files of fit indices and correlations
    """
    # get Model_fc_trait_both files
    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(trait_scale_name)

    if trait_type == "mental":
        drop_list = ["not_dropped"]
    elif trait_type in ["cognition", "personality"]:
        drop_list = ["not_dropped", "dropped"]

    for data_type in ["fit_indices", "correlation"]:
        # get latest *_Model_fc_trait_both_* files
        model_fc_trait_both_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            ["fc", "trait", "both"],
            kwargs.get("drop_vars_list_dict"),
        )
        # get latest *_Model_fc_trait_* files
        model_fc_trait_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            ["fc", "trait"],
            kwargs.get("drop_vars_list_dict"),
        )
        # change file values
        for gsr_type in ["nogs", "gs"]:
            for scale_name in subscales:
                for drop in drop_list:
                    print(data_type, gsr_type, scale_name, drop)
                    filename_fc_trait_both = (
                        model_fc_trait_both_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    filename_fc_trait = (
                        model_fc_trait_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    # get time point of filename_fc_trait
                    if (
                        filename_fc_trait_both is not None
                        and filename_fc_trait is not None
                    ):
                        pass


def add_columns_to_df_with_vars(df, vars_list: list[str]):
    """add columns to df using local variables"""
    for var in vars_list:
        df[var] = globals().get(var)
    return df


def add_outputs_of_model_both_to_files(
    trait_type, old_model_list, additional_model_list, **kwargs
):
    """
    add outputs of model_both to outputs of Model_fc_trait with FixedLoad
    """
    # get Model_fc_trait_both files
    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(trait_scale_name)

    if trait_type == "mental":
        drop_list = ["not_dropped"]
    elif trait_type in ["cognition", "personality"]:
        drop_list = ["not_dropped", "dropped"]

    for data_type in ["fit_indices", "correlation"]:
        # get latest *_Model_fc_trait_both_* files
        model_old_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            old_model_list,
            kwargs.get("drop_vars_list_dict"),
        )
        # get latest *_Model_fc_trait_* files
        model_additional_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            additional_model_list,
            kwargs.get("drop_vars_list_dict"),
        )
        # change file values
        for gsr_type in ["nogs", "gs"]:
            for scale_name in subscales:
                for drop in drop_list:
                    print(data_type, gsr_type, scale_name, drop)
                    filename_old = (
                        model_old_files_list.get(gsr_type).get(scale_name).get(drop)
                    )
                    filename_additional = (
                        model_additional_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    # get time point of filename_fc_trait
                    if filename_old is not None and filename_additional is not None:
                        time_suffix_additional = get_strings_from_filename(
                            filename_additional, ["date_time"]
                        )[0]
                        time_suffix_old = get_strings_from_filename(
                            filename_old, ["date_time"]
                        )[0]

                        filename_new = filename_old.replace(
                            time_suffix_old, time_suffix_additional
                        )
                        file_dir = op.join(
                            SCHAEFER_DIR, trait_scale_name, scale_name, data_type
                        )
                        if filename_old != filename_new:
                            copy(
                                op.join(file_dir, filename_old),
                                op.join(file_dir, filename_new),
                            )


def replace_model_trait_cog_dropped(**kwargs):
    """
    replace fit indices, correlation, and parameter estimates of model_trait in NIH Cognition
    """
    for data_type in ["fit_indices", "correlation"]:
        model_old_files_list = get_latest_files_with_args(
            ["cognition"],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            ["model_fc", "model_trait", "model_both"],
            kwargs.get("drop_vars_list_dict"),
        )
        # get latest *_Model_fc_trait_* files
        model_additional_files_list = get_latest_files_with_args(
            ["cognition"],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            ["model_fc", "model_trait"],
            kwargs.get("drop_vars_list_dict"),
        )
        target_file_old = model_old_files_list.get("fluid")


def modify_cog_crystal_dat(save_dat=False, rename_param_file=False, **kwargs):
    """
    modify output files of Crystal subscale in NIH toolbox
    Thns function may be expanded to change other .dat files
    """
    # Processing outputs of fit indices and correlation
    for data_type in ["fit_indices", "correlation"]:
        # get latest *_Model_fc_trait_both_* files
        model_old_files_list = get_latest_files_with_args(
            ["cognition"],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            ["fc", "trait", "both"],
            kwargs.get("drop_vars_list_dict"),
        )
        # get latest *_Model_fc_trait_* files
        model_additional_files_list = get_latest_files_with_args(
            ["cognition"],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            ["trait", "both"],
            kwargs.get("drop_vars_list_dict"),
        )
        # change file values
        for gsr_type in ["nogs", "gs"]:
            for scale_name in ["Crystal"]:
                for drop in ["not_dropped", "dropped"]:
                    filename_old = (
                        model_old_files_list.get(gsr_type).get(scale_name).get(drop)
                    )
                    filename_additional = (
                        model_additional_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    if filename_old is not None and filename_additional is not None:
                        old_dat = copy_memmap_output_data(filename_old)
                        additional_dat = copy_memmap_output_data(filename_additional)

                        old_models = (
                            re.findall("Model_.*_Trait", filename_old)[0]
                            .replace("Model_", "")
                            .replace("_Trait", "")
                            .split("_")
                        )
                        additional_models = (
                            re.findall("Model_.*_Trait", filename_additional)[0]
                            .replace("Model_", "")
                            .replace("_Trait", "")
                            .split("_")
                        )
                        # get index of models
                        old_models_index = [
                            MODEL_POSITION_DICT.get(value) for value in old_models
                        ]
                        additional_model_index = [
                            MODEL_POSITION_DICT.get(value)
                            for value in additional_models
                        ]
                        # replace values
                        if old_models == ["fc", "trait", "both"]:
                            if data_type == "fit_indices":
                                old_dat[:, :, additional_model_index] = additional_dat
                            if data_type == "correlation":
                                old_dat[:, additional_model_index] = additional_dat
                        replaced_dat = old_dat
                        # replace time of old filename
                        suffix_new = reduce(
                            add, re.findall("Trait_.*", filename_additional)
                        )
                        suffix_replaced = reduce(
                            add, re.findall("Trait_.*", filename_old)
                        )
                        model_old_file_new = filename_old.replace(
                            suffix_replaced, suffix_new
                        )
                        # save replaced .dat file of fit indices and correlation
                        dat_dir = op.join(SCHAEFER_DIR, "NIH_Cognition", "Crystal")
                        if save_dat is True:
                            if old_models == ["fc", "trait", "both"]:
                                save_filename = op.join(
                                    dat_dir, data_type, model_old_file_new
                                )
                                if not op.exists(save_filename):
                                    data_memmap_old = np.memmap(
                                        filename=save_filename,
                                        shape=old_dat.shape,
                                        mode="w+",
                                        dtype="float32",
                                    )
    # rename filenames of outputs of parameter estimates
    not_additional_models = [i for i in old_models if i not in additional_models]
    params_dir = op.join(dat_dir, "parameters")
    target_file_param_list = [
        i
        for i in os.listdir(params_dir)
        if str(kwargs.get("n_edge")) in i
        and str(kwargs.get("sample_n")) in i
        and kwargs.get("est_method") in i
        and "_".join(not_additional_models) in i
        and "FixedLoad" in i
        and gsr_type in i
    ]
    target_file_param = sort_list_by_time(target_file_param_list)[-1]
    target_file_param_new = target_file_param.replace(suffix_replaced, suffix_new)
    new_filepath = op.join(params_dir, target_file_param_new)
    if rename_param_file is True:
        if not op.exists(new_filepath):
            os.rename(op.join(params_dir, target_file_param), new_filepath)


def modify_output_files(
    trait_type, old_model_list, additional_model_list, save_file=False, **kwargs
):
    """
    modify output files by combining .dat files
    """
    # combine model_fc and model_trait in *_Model_fc_trait_* with
    # model_both in *_Model_fc_trait_both_*
    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(trait_scale_name)

    if trait_type == "mental":
        drop_list = ["not_dropped"]
    elif trait_type in ["cognition", "personality"]:
        drop_list = ["not_dropped", "dropped"]

    for data_type in ["fit_indices", "correlation"]:
        # get latest *_Model_fc_trait_both_* files
        model_old_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            old_model_list,
            kwargs.get("drop_vars_list_dict"),
            fixed_load=False,
        )
        # get latest *_Model_fc_trait_* files
        model_additional_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            additional_model_list,
            kwargs.get("drop_vars_list_dict"),
        )
        # change file values
        for gsr_type in ["nogs", "gs"]:
            for scale_name in subscales:
                for drop in drop_list:
                    filename_old = (
                        model_old_files_list.get(gsr_type).get(scale_name).get(drop)
                    )
                    filename_additional = (
                        model_additional_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    if filename_old is not None and filename_additional is not None:
                        if (
                            "FixedLoad" not in filename_old
                            and "FixedLoad" in filename_additional
                        ):

                            # copy and get data
                            old_dat = copy_memmap_output_data(filename_old)
                            additional_dat = copy_memmap_output_data(
                                filename_additional
                            )

                            # combine data
                            if data_type == "fit_indices":
                                # replace outputs of model_fc and model_trait
                                combined_data = np.concatenate(
                                    [additional_dat, old_dat[:, :, [-1]]], axis=2
                                )
                            if data_type == "correlation":
                                combined_data = np.concatenate(
                                    [additional_dat, old_dat[:, [-1]]], axis=1
                                )

                            if save_file:
                                print(
                                    f"Modifying {filename_old} and {filename_additional}"
                                )
                                # add '_FixedLoad_' to filename
                                split_str = f'Est_{kwargs.get("est_method")}'
                                # save files according to filename of Model_fc_trait_both
                                save_filename_old = (
                                    filename_old.split(split_str)[0]
                                    + split_str
                                    + "_FixedLoad"
                                    + filename_old.split(split_str)[1]
                                )
                                save_filename_additional = (
                                    filename_additional.split(split_str)[0]
                                    + split_str
                                    + "_FixedLoad"
                                    + filename_additional.split(split_str)[1]
                                )

                                save_filepath_old = op.join(
                                    SCHAEFER_DIR,
                                    trait_scale_name,
                                    scale_name,
                                    data_type,
                                    save_filename_old,
                                )
                                save_filepath_additional = op.join(
                                    SCHAEFER_DIR,
                                    trait_scale_name,
                                    scale_name,
                                    data_type,
                                    save_filename_additional,
                                )

                                combined_data_memmap_old = np.memmap(
                                    filename=save_filepath_old,
                                    shape=combined_data.shape,
                                    mode="w+",
                                    dtype="float32",
                                )
                                combined_data_memmap_additional = np.memmap(
                                    filename=save_filepath_additional,
                                    shape=combined_data.shape,
                                    mode="w+",
                                    dtype="float32",
                                )

                                combined_data_memmap_old[
                                    :
                                ] = combined_data_memmap_additional[:] = combined_data


def rename_cor_fit_filenames_from_models(
    trait_type, old_model_list, additional_model_list, **kwargs
):
    """
    rename filenames of outputs of correlation and fit indices to match models
    """
    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(trait_scale_name)

    if trait_type == "mental":
        drop_list = ["not_dropped"]
    elif trait_type in ["cognition", "personality"]:
        drop_list = ["not_dropped", "dropped"]

    for data_type in ["fit_indices", "correlation"]:
        # get latest *_Model_fc_trait_both_* files
        print(data_type)
        model_old_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            old_model_list,
            kwargs.get("drop_vars_list_dict"),
            fixed_load=False,
        )
        print("Model fc and trait.")
        model_additional_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            additional_model_list,
            kwargs.get("drop_vars_list_dict"),
        )
        # change file names of parameters in model_both
        for gsr_type in ["nogs", "gs"]:
            for scale_name in subscales:
                for drop in drop_list:
                    filename_old = (
                        model_old_files_list.get(gsr_type).get(scale_name).get(drop)
                    )
                    filename_additional = (
                        model_additional_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    if filename_old is not None and filename_additional is not None:
                        if (
                            "FixedLoad" not in filename_old
                            and "FixedLoad" in filename_additional
                        ):
                            pass


def rename_time_in_cor_fit_filenames_after_modifying(
    trait_type, old_model_list, additional_model_list, **kwargs
):
    """
    rename filenames of parameters in accordance with
    those of fit indices and correlation
    """
    # combine model_fc and model_trait in *_Model_fc_trait_* with
    # model_both in *_Model_fc_trait_both_*
    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(trait_scale_name)

    if trait_type == "mental":
        drop_list = ["not_dropped"]
    elif trait_type in ["cognition", "personality"]:
        drop_list = ["not_dropped", "dropped"]

    for data_type in ["fit_indices", "correlation"]:
        # get latest *_Model_fc_trait_both_* files
        print(data_type)
        print("Model fc, trait, and both.")
        model_old_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            old_model_list,
            kwargs.get("drop_vars_list_dict"),
            fixed_load=kwargs.get("fixed_load"),
        )
        print("Model fc and trait.")
        model_additional_files_list = get_latest_files_with_args(
            [trait_type],
            kwargs.get("n_edge"),
            kwargs.get("sample_n"),
            kwargs.get("est_method"),
            data_type,
            additional_model_list,
            kwargs.get("drop_vars_list_dict"),
        )
        # change file names of parameters in model_both
        for gsr_type in ["nogs", "gs"]:
            for scale_name in subscales:
                for drop in drop_list:
                    model_additional_file = (
                        model_additional_files_list.get(gsr_type)
                        .get(scale_name)
                        .get(drop)
                    )
                    model_old_file = (
                        model_old_files_list.get(gsr_type).get(scale_name).get(drop)
                    )
                    if model_additional_file is not None and model_old_file is not None:
                        # change suffix in model_fc_trait_both_file to match that in model_fc_trait_file
                        suffix_new = reduce(
                            add, re.findall("Trait_.*", model_additional_file)
                        )
                        suffix_replaced = reduce(
                            add, re.findall("Trait_.*", model_old_file)
                        )
                        model_old_file_new = model_old_file.replace(
                            suffix_replaced, suffix_new
                        )
                        file_dir = op.join(
                            SCHAEFER_DIR, trait_scale_name, scale_name, data_type
                        )
                        new_filename = op.join(file_dir, model_old_file_new)
                        if "FixedLoad_FixedLoad_" in new_filename:
                            new_filename = new_filename.replace(
                                "FixedLoad_FixedLoad_", "FixedLoad_"
                            )
                        if not op.exists(new_filename):
                            print(
                                f"Copying and renaming {suffix_replaced} with {suffix_new}."
                            )
                            os.rename(op.join(file_dir, model_old_file), new_filename)


def rename_time_of_model_both_in_param_filenames(trait_type, **kwargs):
    """
    rename filenames of parameters in accordance with
    those of fit indices and correlation
    """
    # combine model_fc and model_trait in *_Model_fc_trait_* with
    # model_both in *_Model_fc_trait_both_*
    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(trait_scale_name)

    if trait_type == "mental":
        drop_list = ["not_dropped"]
    elif trait_type in ["cognition", "personality"]:
        drop_list = ["not_dropped", "dropped"]

    # get latest *_Model_fc_trait_* files
    model_fc_trait_files_list = get_latest_files_with_args(
        [trait_type],
        kwargs.get("n_edge"),
        kwargs.get("sample_n"),
        kwargs.get("est_method"),
        "parameters",
        ["fc"],
        kwargs.get("drop_vars_list_dict"),
    )
    params_model_both_files_list = get_latest_files_with_args(
        [trait_type],
        kwargs.get("n_edge"),
        kwargs.get("sample_n"),
        kwargs.get("est_method"),
        "parameters",
        ["both"],
        kwargs.get("drop_vars_list_dict"),
        fixed_load=False,
    )
    # change file names of parameters in model_both
    for gsr_type in ["nogs", "gs"]:
        for scale_name in subscales:
            for drop in drop_list:
                params_model_both_file = (
                    params_model_both_files_list.get(gsr_type).get(scale_name).get(drop)
                )
                model_fc_trait_file = (
                    model_fc_trait_files_list.get(gsr_type).get(scale_name).get(drop)
                )
                if (
                    params_model_both_file is not None
                    and model_fc_trait_file is not None
                ):
                    if "FixedLoad" in model_fc_trait_file:
                        # change suffix in params_model_both_file to match that in model_fc_trait_both_file
                        suffix_new = reduce(
                            add, re.findall("Trait_.*", model_fc_trait_file)
                        )
                        suffix_replaced = reduce(
                            add, re.findall("Trait_.*", params_model_both_file)
                        )
                        params_model_both_file_new = params_model_both_file.replace(
                            suffix_replaced, suffix_new
                        )
                        file_dir = op.join(
                            SCHAEFER_DIR, trait_scale_name, scale_name, "parameters"
                        )
                        new_filename = op.join(file_dir, params_model_both_file_new)

                        if not op.exists(new_filename):
                            print(
                                f"Copying and renaming {suffix_replaced} with {suffix_new}."
                            )
                            copy(
                                op.join(file_dir, params_model_both_file), new_filename
                            )


def remove_NEORAW_from_fa_result_filenames():
    """
    remove 'NEORAW' from filenames to match change of pipeline
    """
    neo_ffi_fa_result_dir = op.join(
        HCP_ROOT_DIR, "derivatives", "Python", "fa_result", "NEO_FFI"
    )

    for filename in os.listdir(neo_ffi_fa_result_dir):
        if "NEORAW" in filename:
            filename_new = filename.replace("NEORAW_", "")
            original_filename = op.join(neo_ffi_fa_result_dir, filename)
            new_filename = op.join(neo_ffi_fa_result_dir, filename_new)
            os.rename(original_filename, new_filename)


def save_rdata_for_visualize_params_of_indicators_fc_trait(
    trait_type,
    vis_model_type_list: ModelFA,
    dtype_memmap="float32",
    target="std_estimates",
    get_load=False,
    analysis_time_lag=False,
    **kwargs,
) -> None:
    """
    function for visualising error variance of indicators
    vis_param should be FC or trait
    target shoule be std_estimates, unstd_estimates, or unstd_se
    """
    data_dict = defaultdict(dict)

    filename_list_dict = get_latest_files_with_args(
        [trait_type],
        kwargs["num_iter"],
        kwargs["sample_n"],
        kwargs["est_method"],
        kwargs["data_type"],
        kwargs["model_type_list"],
        kwargs.get("drop_vars_list_dict"),
    )
    trait_scale_name = get_scale_name_from_trait(trait_type)
    fc_params_pd_all, trait_params_pd_all = pd.DataFrame(), pd.DataFrame()

    for vis_model_type in vis_model_type_list:

        if vis_model_type in ["model_fc", "model_both"]:
            fc_column_names = ["Q" + str(i) for i in range(1, 5)]
        elif vis_model_type == "model_trait":
            fc_column_names = ["estimate"]

        for g, gsr_type in enumerate(["nogs", "gs"]):
            subscale_level_dict = filename_list_dict.get(gsr_type)
            for scale_name in subscale_level_dict:

                if vis_model_type == "model_fc":
                    trait_column_names = ["estimate"]
                elif vis_model_type in ["model_trait", "model_both"]:
                    trait_column_names = get_subscale_list(
                        trait_scale_name, scale_name=scale_name, get_all_subscales=True
                    )

                for drop in filename_list_dict.get(gsr_type).get(scale_name):
                    print(vis_model_type, scale_name, drop)
                    for f, filename in enumerate(
                        [filename_list_dict.get(gsr_type).get(scale_name).get(drop)]
                    ):
                        if filename is not None:
                            print(filename)
                            filename_after_models = reduce(
                                add, re.findall("Model_.*", filename)
                            )

                            def get_params():
                                params_dict = generate_params_dict(
                                    filename_after_models,
                                    analysis_time_lag=analysis_time_lag,
                                    **{"drop": drop},
                                )
                                (
                                    trait_type,
                                    scale_name,
                                    model_type,
                                    control,
                                    gsr_type,
                                    cov_cor,
                                    fc_unit,
                                    phase_encoding,
                                    day_cor,
                                    use_lavaan,
                                    drop_vars_list,
                                ) = get_strings_from_filename(
                                    filename_after_models,
                                    [
                                        "trait_type",
                                        "scale_name",
                                        "model_type",
                                        "control",
                                        "gsr_type",
                                        "cov_cor",
                                        "fc_unit",
                                        "phase_encoding",
                                        "day_cor",
                                        "use_lavaan",
                                        "drop_vars_list",
                                    ],
                                    include_nofa_model=True,
                                )
                                # specify model
                                if trait_type is not None:
                                    folder = op.join(
                                        get_scale_name_from_trait(trait_type),
                                        scale_name,
                                    )
                                    scale_suffix = f" in {scale_name} of {folder}"

                                param_position_dict = get_param_position_dict(
                                    control,
                                    vis_model_type,
                                    cov_cor,
                                    phase_encoding,
                                    day_cor,
                                    use_lavaan,
                                    fc_unit,
                                    trait_type,
                                    scale_name,
                                    drop_vars_list,
                                )
                                fc_param, trait_param, cor_param = get_parameters(
                                    params_dict,
                                    vis_model_type,
                                    param_position_dict,
                                    get_load=get_load,
                                    target=target,
                                )
                                fc_param_pd = pd.DataFrame(
                                    fc_param, columns=fc_column_names
                                )
                                if (
                                    vis_model_type in ["model_trait", "model_both"]
                                    and drop_vars_list is not None
                                ):
                                    if trait_type == "personality":
                                        trait_column_names_used = [
                                            i
                                            for i in trait_column_names
                                            if NEO_FFI_DICT_REVERSED.get(
                                                scale_name
                                            ).get(i)
                                            not in drop_vars_list
                                        ]
                                    elif trait_type == "cognition":
                                        trait_column_names_used = [
                                            i
                                            for i in trait_column_names
                                            if COG_SCALES_PUB_DICT_REVERSED.get(i)
                                            not in drop_vars_list
                                        ]
                                else:
                                    trait_column_names_used = trait_column_names
                                trait_param_pd = pd.DataFrame(
                                    trait_param, columns=trait_column_names_used
                                )
                                for key, value in {
                                    "gsr_type": gsr_type,
                                    "model": vis_model_type,
                                    "drop": drop,
                                    "scale_name": scale_name,
                                }.items():
                                    fc_param_pd[key] = value
                                    trait_param_pd[key] = value

                                fc_param_pd = fc_param_pd.reset_index().rename(
                                    columns={"index": "edge"}
                                )
                                if vis_model_type == "model_trait":
                                    fc_param_pd["Scan"] = "All"
                                elif vis_model_type in ["model_fc", "model_both"]:
                                    fc_param_pd = fc_param_pd.melt(
                                        id_vars=[
                                            "edge",
                                            "gsr_type",
                                            "drop",
                                            "model",
                                            "scale_name",
                                        ],
                                        value_vars=fc_column_names,
                                        var_name="Scan",
                                        value_name="estimate",
                                    )

                                trait_param_pd = trait_param_pd.reset_index().rename(
                                    columns={"index": "edge"}
                                )
                                if vis_model_type in ["model_trait", "model_both"]:
                                    trait_param_pd = trait_param_pd.melt(
                                        id_vars=[
                                            "edge",
                                            "gsr_type",
                                            "drop",
                                            "model",
                                            "scale_name",
                                        ],
                                        value_vars=trait_column_names_used,
                                        var_name="Item",
                                        value_name="estimate",
                                    )
                                elif vis_model_type == "model_fc":
                                    trait_param_pd["Item"] = "All"

                                return fc_param_pd, trait_param_pd

                            fc_params_pd, trait_params_pd = get_params()

                            fc_params_pd_all = pd.concat(
                                [fc_params_pd_all, fc_params_pd], axis=0
                            )
                            trait_params_pd_all = pd.concat(
                                [trait_params_pd_all, trait_params_pd], axis=0
                            )

    def replace_and_reorder_df(df):
        """inner function for replace and reorder columns"""
        df = replace_and_reorder_column(
            df,
            var_name_dict={
                "gsr_type": GSR_DICT,
                "drop": DROP_DICT,
                "model": MODEL_DICT,
            },
        )
        df = make_categories_in_df(df, "scale_name", subscale_level_dict.keys())
        return df

    fc_params_pd_all = replace_and_reorder_df(fc_params_pd_all)
    trait_params_pd_all = replace_and_reorder_df(trait_params_pd_all)
    # set filename for saving
    remove_input_dict = {"scale": True, "dt_time": True, "gsr_type": True}
    if filename is None:
        filename = filename_list_dict.get(gsr_type).get(scale_name).get("not_dropped")
    filename_saved = remove_strings_from_filename(
        filename, remove_prefix=True, **remove_input_dict
    )
    param_prefix = "loadings_" if get_load else "error_"
    fc_filename_saved = (
        param_prefix + "fc_params_" + filename_saved.replace(".dat", ".RData")
    )
    trait_filename_saved = (
        param_prefix + "trait_params_" + filename_saved.replace(".dat", ".RData")
    )

    # save RData file
    print(f"Saving .RData file of FC parameters ({fc_filename_saved}).")
    save_rdata_file(
        fc_params_pd_all,
        op.join(SCHAEFER_DIR, trait_scale_name, "data", "RData", fc_filename_saved),
    )
    print(f"Saving .RData file of trait parameters ({trait_filename_saved}).")
    save_rdata_file(
        trait_params_pd_all,
        op.join(SCHAEFER_DIR, trait_scale_name, "data", "RData", trait_filename_saved),
    )
    print("Complete saving.")


def vis_params_msst(
        parcellation='Schaefer',
        filenames_dict={},
        param_order_filename=None,
        xlim=[0, 1]
        ):
    """
    Visualize distributions of parameter estimates of MSST models
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    params_dir = op.join(atlas_dir, 'reliability', 'parameters', 'combined', 'split_half_cv')
    param_order_dir = op.join(FA_PARAMS_DIR, 'reliability')
    param_order = pd.read_csv(op.join(param_order_dir, param_order_filename))
    state_load_positions = param_order.query('op == "=~" & rhs.str.contains("s")').index
    trait_load_positions = param_order.query('op == "=~" & rhs.str.contains("o")').index
    bins = np.arange(0, 1 + 0.01, 0.01)
    for j, fold in enumerate(['Fold_0', 'Fold_1']):
        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.suptitle(fold)
        for k, gsr_type in enumerate(['nogs', 'gs']):
            param_filename = filenames_dict.get(fold).get(gsr_type)
            params = np.load(op.join(params_dir, param_filename))
            # get parameter estimates of state loadings
            params_state = params[:, state_load_positions, 0]
            # get parameter estimates of trait loadings
            params_trait = params[:, trait_load_positions, 0]
            # draw histograms of state loadings
            #axes[0, k].hist(params_state, alpha=0.2, bins=bins, density=True)
            #for i in range(4):
            sns.histplot(params_state, ax=axes[0, k], kde_kws={'bw_adjust': 0.1, 'clip': xlim}, kde=True, bins=bins)
            # draw_histograms of trait loadings
            #axes[1, k].hist(params_trait, alpha=0.2, bins=bins, density=True)
            #for i in range(2):
            sns.histplot(params_trait, ax=axes[1, k], kde_kws={'bw_adjust': 1, 'clip': xlim}, kde=True, bins=bins)
            for i in range(1):
                axes[i, k].set_xlim(xlim)
                axes[i, k].set_title(gsr_type)
                for x_value in [0.3, 0.4]:
                    axes[i, k].axvline(x_value, linestyle='--', color='black')
        fig.tight_layout()


def vis_fits_msst(
        parcellation='Schaefer',
        filenames_dict={},
        fit_indices=['SRMR', 'RMSEA', 'CFI']
):
    """
    Visualize distributions of fit indices
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    fits_dir = op.join(atlas_dir, 'reliability', 'fit_indices', 'combined', 'split_half_cv')
    param_order_dir = op.join(FA_PARAMS_DIR, 'reliability')
    bins = np.arange(0, 1 + 0.01, 0.01)
    fit_positions = [FIT_INDICES.index(i) for i  in fit_indices]
    for j, fold in enumerate(['Fold_0', 'Fold_1']):
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
        fig.suptitle(fold)
        for k, gsr_type in enumerate(['nogs', 'gs']):
            fit_filename = filenames_dict.get(fold).get(gsr_type)
            fits = np.load(op.join(fits_dir, fit_filename))
            # extract fit indices
            fits = fits[:, fit_positions]
            for i, fit_index in enumerate(fit_indices):
                sns.kdeplot(fits[:, i], ax=axes[i, k], bw_adjust=1, clip=[0, 1])
                axes[i, k].set_xlim([0, 1])
                axes[i, k].set_title(f'{fit_index} {gsr_type}')
                if fit_index in ['SRMR', 'RMSEA']:
                    for x_value in [0.08, 0.1]:
                        axes[i, k].axvline(x_value, linestyle='--', color='black')
                elif fit_index == 'CFI':
                    axes[i, k].axvline(0.90, linestyle='--', color='black')
        fig.tight_layout()


def visualize_params_of_indicators_gsr_slurm_arrays(
    vis_model_type: ModelFA,
    dtype_memmap="float32",
    target="std_estimates",
    get_load=False,
    fig_height=12,
    fig_width=20,
    vis_vars="method",
    scale_x_lim=[-1, 1],
    add_marker=True,
    return_plot=True,
    parcellation='Schaefer',
    n_arrays_sem=431,
    multistate_single_trait=False,
    get_std=False,
    return_array=False,
    include_day_cor=False,
    **kwargs
) -> None:
    """
    function for visualising error variance of indicators
    vis_param should be FC or trait
    target shoule be std_estimates, unstd_estimates, or unstd_se
    """
    # 3 includes Model RSFC 1, 2, and 3
    output_df = pd.DataFrame()

    for g, gsr_type in enumerate(["nogs", "gs"]):
        filename_list = get_list_of_filenames(
                data_type='parameters', 
                gsr_type=gsr_type, 
                parcellation=parcellation,
                addMarker=add_marker,
                include_day_cor=include_day_cor,
                multistate_single_trait=multistate_single_trait,
                get_std=get_std,
                **kwargs
                )
        param_array_all = combine_array_files_dat(
            filename_list, n_arrays_sem, data_type='parameter', parcellation=parcellation
        )
        if return_array:
            return param_array_all
        param_filename = sort_list_by_time(filename_list)[-1]
        filename_after_models = reduce(add, re.findall("Model_.*", param_filename))
        #params_dict = generate_params_dict(filename_after_models, n_arrays=n_arrays_sem)
        input_list = [
            "trait_type",
            "scale_name",
            "model_type",
            "control",
            "gsr_type",
            "cov_cor",
            "fc_unit",
            "phase_encoding",
            "day_cor",
            "use_lavaan",
            "order_in_day",
        ]
        (
            trait_type,
            scale_name,
            model_type,
            control,
            gsr_type,
            cov_cor,
            fc_unit,
            phase_encoding,
            day_cor,
            use_lavaan,
            order_in_day,
        ) = get_strings_from_filename(
            filename_after_models,
            input_list,
            include_nofa_model=True,
        )
        if add_marker:
            add_marker = get_strings_from_filename(
                filename_after_models, ["add_marker"], include_nofa_model=True
            )

        # specify model
        model = check_models_rsfc(cov_cor, day_cor, phase_encoding, order_in_day)
        if trait_type is not None:
            folder = op.join(get_scale_name_from_trait(trait_type), scale_name)
            scale_suffix = f" in {scale_name} of {folder}"
        else:
            folder = "reliability"
            scale_suffix = ""

        param_position_dict = get_param_position_dict(param_filename)
        fc_param, trait_param, cor_param, method_param = get_parameters(
            param_array_all, "model_onlyFC", param_position_dict
        )
        fc_param_pd = pd.DataFrame(
            fc_param, columns=["Q" + str(i) for i in range(1, 5)]
        )
        fc_param_pd["gsr_type"] = gsr_type
        fc_param_pd["model"] = model
        method_col_names = (
            ["Parameter 1", "Parameter 2"]
            if not add_marker
            else ["Loading " + str(i) for i in range(1, 5)]
        )
        if not method_param.shape[1] == 0:
            method_pd = pd.DataFrame(method_param, columns=method_col_names)
        else:
            method_pd = pd.DataFrame(
                np.nan, index=range(len(method_param)), columns=method_col_names
            )
        concat_df = pd.concat([fc_param_pd, method_pd], axis=1)
        output_df = pd.concat([output_df, concat_df], axis=0)
    # reshape df for visualization
    output_df = output_df.reset_index().rename(columns={"index": "edge"})
    model_suffix = "b" if not add_marker else "a"
    output_df = output_df.replace(
        {
            "Model RSFC 2": f"Model RSFC 2-{model_suffix}",
        }
    )
    if vis_vars == "indicators":
        color_name = "Scan"
        long_df = output_df.melt(
            id_vars=["edge", "gsr_type", "model"],
            value_vars=["Q" + str(i) for i in range(1, 5)],
            var_name=color_name,
            value_name="estimate",
        )
        xlab = "Standardized factor loading"
        title = "(I) Standardized factor loading"
        xlim = [0, 1]
    elif vis_vars == "method":
        color_name = "Error covariance"
        long_df = output_df.melt(
            id_vars=["edge", "gsr_type", "model"],
            value_vars=method_col_names,
            var_name=color_name,
            value_name="estimate",
        ).query('model not in "Model RSFC 1"')
        xlab = "Standardized covariance between errors"
        title = "(IV) Correlated uniqueness"
        xlim = scale_x_lim

    long_df = replace_and_reorder_column(long_df, "gsr_type", GSR_DICT)
    # set filename for saving
    remove_input_dict = {
        "scale": True,
        "dt_time": True,
        "gsr_type": True,
        "trait": True,
        "day_cor": True,
        "phase_encoding": True,
    }
    filename_saved = remove_strings_from_filename(param_filename, **remove_input_dict)
    # draw figure
    print("Drawing figure")
    g = (
        ggplot(long_df, aes("estimate", fill=color_name))
        + geom_density(alpha=0.2)
        + facet_grid("gsr_type ~ model")
        + scale_x_continuous(limits=xlim)
        + theme_bw()
        + labs(x=xlab, y="Scaled density (a.u.)")
        + theme(axis_text_x=element_text(angle=45))
    )
    if return_plot:
        g = g + ggtitle(title) + theme(plot_title=element_text(hjust=0))
        return g, long_df

    g.show()


def visualize_params_of_indicators_gsr(
    filename_list_dict: dict[str : list[str]],
    vis_model_type: ModelFA,
    dtype_memmap="float32",
    target="std_estimates",
    get_load=False,
    fig_height=12,
    fig_width=20,
    vis_vars="method",
    scale_x_lim=[-1, 1],
    add_marker=False,
    return_plot=True,
) -> None:
    """
    function for visualising error variance of indicators
    vis_param should be FC or trait
    target shoule be std_estimates, unstd_estimates, or unstd_se
    """
    # 3 includes Model RSFC 1, 2, and 3
    output_df = pd.DataFrame()

    for g, gsr_type in enumerate(["nogs", "gs"]):
        for f, filename in enumerate(filename_list_dict[gsr_type]):
            print(filename)
            filename_after_models = reduce(add, re.findall("Model_.*", filename))
            params_dict = generate_params_dict(filename_after_models)
            input_list = [
                "trait_type",
                "scale_name",
                "model_type",
                "control",
                "gsr_type",
                "cov_cor",
                "fc_unit",
                "phase_encoding",
                "day_cor",
                "use_lavaan",
                "order_in_day",
            ]
            (
                trait_type,
                scale_name,
                model_type,
                control,
                gsr_type,
                cov_cor,
                fc_unit,
                phase_encoding,
                day_cor,
                use_lavaan,
                order_in_day,
            ) = get_strings_from_filename(
                filename_after_models,
                input_list,
                include_nofa_model=True,
            )
            if add_marker:
                add_marker = get_strings_from_filename(
                    filename_after_models, "add_marker", include_nofa_model=True
                )
            # specify model
            model = check_models_rsfc(cov_cor, day_cor, phase_encoding, order_in_day)
            if trait_type is not None:
                folder = op.join(get_scale_name_from_trait(trait_type), scale_name)
                scale_suffix = f" in {scale_name} of {folder}"
            else:
                folder = "reliability"
                scale_suffix = ""

            param_position_dict = get_param_position_dict(filename)
            fc_param, trait_param, cor_param, method_param = get_parameters(
                params_dict,
                vis_model_type,
                param_position_dict,
                get_load=get_load,
                target=target,
                add_marker=add_marker,
            )
            fc_param_pd = pd.DataFrame(
                fc_param, columns=["Q" + str(i) for i in range(1, 5)]
            )
            fc_param_pd["gsr_type"] = gsr_type
            fc_param_pd["model"] = model
            method_col_names = (
                ["Parameter 1", "Parameter 2"]
                if not add_marker
                else ["Loading " + str(i) for i in range(1, 5)]
            )
            if not method_param.shape[1] == 0:
                method_pd = pd.DataFrame(method_param, columns=method_col_names)
            else:
                method_pd = pd.DataFrame(
                    np.nan, index=range(len(method_param)), columns=method_col_names
                )
            concat_df = pd.concat([fc_param_pd, method_pd], axis=1)
            output_df = pd.concat([output_df, concat_df], axis=0)
    # reshape df for visualization
    output_df = output_df.reset_index().rename(columns={"index": "edge"})
    model_suffix = "b" if not add_marker else "a"
    output_df = output_df.replace(
        {
            "Model RSFC 2": f"Model RSFC 2-{model_suffix}",
            "Model RSFC 3": f"Model RSFC 3-{model_suffix}",
            "Model RSFC 4": f"Model RSFC 4-{model_suffix}",
        }
    )
    if vis_vars == "indicators":
        color_name = "Scan"
        long_df = output_df.melt(
            id_vars=["edge", "gsr_type", "model"],
            value_vars=["Q" + str(i) for i in range(1, 5)],
            var_name=color_name,
            value_name="estimate",
        )
        xlab = "Standardized factor loading"
        title = "(I) Standardized factor loading"
        xlim = [0, 1]
    elif vis_vars == "method":
        color_name = "Error covariance"
        long_df = output_df.melt(
            id_vars=["edge", "gsr_type", "model"],
            value_vars=method_col_names,
            var_name=color_name,
            value_name="estimate",
        ).query('model not in "Model RSFC 1"')
        xlab = "Standardized covariance between errors"
        title = "(IV) Correlated uniqueness"
        xlim = scale_x_lim

    long_df = replace_and_reorder_column(long_df, "gsr_type", GSR_DICT)
    # set filename for saving
    remove_input_dict = {
        "scale": True,
        "dt_time": True,
        "gsr_type": True,
        "trait": True,
        "day_cor": True,
        "phase_encoding": True,
    }
    filename_saved = remove_strings_from_filename(filename, **remove_input_dict)
    # draw figure
    print("Drawing figure")
    g = (
        ggplot(long_df, aes("estimate", color=color_name))
        + geom_density()
        + facet_grid("gsr_type ~ model")
        + scale_x_continuous(limits=xlim)
        + theme_bw()
        + labs(x=xlab, y="Scaled density (a.u.)")
        + theme(axis_text_x=element_text(angle=45))
    )
    if return_plot:
        g = g + ggtitle(title) + theme(plot_title=element_text(hjust=0))
        return g

    g.show()
    # save figure
    g.save(
        op.join(
            SCHAEFER_DIR, "reliability", "figures", f"{vis_vars}_{filename_saved}.png"
        ),
        height=fig_height,
        width=fig_width,
        units="cm",
    )


def visualize_params_of_indicators(
    filename_cor: str,
    vis_param: VisParam,
    vis_model_type: ModelFA,
    add_vline: Optional[list[float]] = None,
    dtype_memmap="float32",
    target="std_estimates",
    get_load=False,
    fig_size=(12, 4),
    plt_close=False,
    ax=None,
) -> None:
    """
    function for visualising error variance of indicators
    vis_param should be FC or trait
    target shoule be std_estimates, unstd_estimates, or unstd_se
    """
    filename_after_cor_type = reduce(add, re.findall("Model_.*", filename_cor))

    params_dict = generate_params_dict(filename_after_cor_type)
    (
        trait_type,
        scale_name,
        model_type,
        control,
        gsr_type,
        cov_cor,
        phase_encoding,
        day_cor,
        use_lavaan,
        fc_unit,
    ) = get_strings_from_filename(
        filename_after_cor_type,
        [
            "trait_type",
            "scale_name",
            "model_type",
            "control",
            "gsr_type",
            "cov_cor",
            "phase_encoding",
            "day_cor",
            "use_lavaan",
            "fc_unit",
        ],
        include_nofa_model=True,
    )
    if trait_type is not None:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        folder = op.join(trait_scale_name, scale_name)
        scale_suffix = f" in {scale_name} of {trait_scale_name}"
    else:
        folder = "reliability"
        scale_suffix = ""

    param_position_dict = get_param_position_dict(
        control,
        vis_model_type,
        cov_cor,
        phase_encoding,
        day_cor,
        use_lavaan,
        fc_unit,
        trait_type,
        scale_name,
    )
    fc_param, trait_param, cor_param = get_parameters(
        params_dict,
        vis_model_type,
        param_position_dict,
        get_load=get_load,
        target=target,
    )

    gsr_suffix = generate_gsr_suffix(gsr_type)

    if not get_load:
        ax_title_top = "Error variances"
        fig_name_insert = "error_vars"
    else:
        ax_title_top = "Factor loadings"
        fig_name_insert = "loadings"

    def draw_fc_hist(ax):
        sns.histplot(fc_param, ax=ax, kde=True, alpha=0.25, binwidth=0.01)
        handler, label = ax.get_legend_handles_labels()
        if fc_unit == "session":
            legends = ["Q" + str(i) for i in range(1, 5)]
        elif fc_unit == "day":
            legends = ["day1", "day2"]
        if vis_model_type in ["model_fc", "model_both", "model_onlyFC"]:
            ax.legend(
                # handler,
                legends,
                # loc='upper right'
            )
        elif vis_model_type == "model_trait":
            ax.get_legend().remove()

    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        if ~np.all(np.isnan(fc_param)):
            # axes[0].hist(fc_param, bins=np.arange(*axes[0].get_xlim(), 0.01))
            draw_fc_hist(axes[0])
        if trait_param is not None:
            if ~np.all(np.isnan(trait_param)):
                axes[1].hist(trait_param, bins=np.arange(*axes[1].get_xlim(), 0.01))
        if cor_param is not None:
            if ~np.all(np.isnan(cor_param)):
                axes[2].hist(cor_param, bins=np.arange(*axes[2].get_xlim(), 0.01))

        axes[0].set_title(f"{ax_title_top} of\n indicators of FC in {vis_model_type}")
        axes[1].set_title(
            f"{ax_title_top} of\n indicators of Trait in {vis_model_type}"
        )
        axes[2].set_title("Inter-factor covariance")

        filename = f"{vis_model_type}_{fig_name_insert}_hist_{filename_after_cor_type.replace('.dat', '')}_{target}.png"
        if target == "std_estimates":
            sup_title = f"Standardized estimates"
        elif target == "unstd_se":
            sup_title = f"Unstandardized standard error of parameters"
        elif target == "unstd_estimates":
            sup_title = f"Unstandardized estimates"
        fig.suptitle(sup_title + scale_suffix + " " + gsr_suffix)
        fig.tight_layout()

        fig.savefig(
            op.join(SCHAEFER_DIR, folder, "figures", filename),
            bbox_inches="tight",
        )
        if plt_close:
            plt.close()
    else:
        if vis_param == "FC":
            # ax.hist(fc_error_param, alpha=0.25)
            draw_fc_hist(ax)
        elif vis_param == "trait":
            # ax.hist(trait_error_param, alpha=0.25)
            sns.histplot(trait_param, ax=ax, kde=True, alpha=0.25, binwidth=0.01)
            if vis_model_type == "model_fc":
                ax.get_legend().remove()
            elif vis_model_type in ["model_trait", "model_both"]:
                handler, label = ax.get_legend_handles_labels()
                # n_items should be calculated
                ax.legend(handler, ["Item " + str(i) for i in range(1, n_items)])
        elif vis_param == "cor":
            # ax.hist(cor_param, alpha=0.25)
            sns.histplot(cor_param, ax=ax, kde=True, alpha=0.25, binwidth=0.01)
            ax.get_legend().remove()
        for i in add_vline:
            ax.axvline(i, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"{scale_name} {gsr_suffix}")


def loop_for_visualize_params_of_indicators(
    n_edge: int,
    sample_n: int,
    trait_type_list,
    vis_model_type: ModelFA,
    est_method: str,
    loading_or_error="error",
    vis_param_list: list[str] = ["FC", "trait"],
    target="std_estimates",
    add_vline: list[float] = [0.3, 0.4, 0.5],
    fig_size=(12, 4),
    plt_close=False,
    **kwargs,
):
    """function for looping "visualize_params_of_indicators()"""
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, est_method, "correlation"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        elif trait_type == "mental":
            ncol = 4
        for vis_param in vis_param_list:
            fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
            for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
                filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
                for i, filename_cor in enumerate(filename_list_gsr_type):
                    visualize_params_of_indicators(
                        filename_cor,
                        vis_param=vis_param,
                        vis_model_type=vis_model_type,
                        loading_or_error=loading_or_error,
                        add_vline=add_vline,
                        ax=axes[j, i],
                    )

            if loading_or_error == "error":
                fig_title_insert = "error variances of indicators of {vis_param}"
                fig_name_insert = "error_var_of_{vis_param}"
            elif loading_or_error == "loading":
                fig_title_insert = f"factor loadings of indicators of {vis_param}"
                fig_name_insert = "loading_of_{vis_param}"
            # if vis_param == "fc":
            #     if vis_param_type == 'error':
            #         param_name = "error variance of indicators of FC"
            #     elif vis_param_type == 'loading':
            #         param_name = 'factor loadings of indicators of FC'
            # elif vis_param == "trait":
            #     if vis_param_type == 'error':
            #         param_name = "error variance of indicators of trait"
            #     elif vis_param_type == 'loading':

            #  elif vis_param_type == "inter_factor_covariance":
            #      param_name = "inter-factor covariance"
            fig.suptitle(
                f"Histogram of {fig_title_insert} in {trait_scale_name} in {vis_model_type} with {est_method} (N = {sample_n}, number of edges is {n_edge})"
            )
            fig.tight_layout()

            fig_name = f"hist_{fig_name_insert}_{vis_model_type}_{trait_scale_name}_{est_method}_sampleN_{sample_n}_edgeN_{n_edge}.png"
            fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
            fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def map_chisq_test_passed_edge_to_networks(
    filename_fit: str,
    p_value: float,
    node_summary_path: str,
) -> pd.DataFrame:
    """
    function for mapping misfit edges to networks
    """

    edges_df = get_edge_summary(node_summary_path)
    misfit_edges_df = edges_df.query("edge in @misfit_set")

    return misfit_edges_df


def add_laterality_of_nodes(edges_df: pd.DataFrame) -> pd.DataFrame:
    """add laterality of nodes to dataframe"""
    edges_df["ipsi_right"] = np.where(
        (edges_df["node1_hem"] == "RH") & (edges_df["node2_hem"] == "RH"), True, False
    )
    edges_df["ipsi_left"] = np.where(
        (edges_df["node1_hem"] == "LH") & (edges_df["node2_hem"] == "LH"), True, False
    )
    edges_df["contra"] = np.where(
        edges_df["node1_hem"] != edges_df["node2_hem"], True, False
    )
    return edges_df


def rename_net_summary_df(net_summary_pd: pd.DataFrame) -> pd.DataFrame:
    """
    rename columns and indices of dataframe on network summary for publication
    """
    net_summary_pd.columns = NETWORK_ORDER_FOR_PUB_LIST
    net_summary_pd.index = NETWORK_ORDER_FOR_PUB_LIST
    return net_summary_pd


def draw_hmap_on_prop_of_removed_edges(
    filename_fit_list: list[str, str],
    error_vars_dict,
    fit_indices_thresholds_dict,
    node_summary_path,
    save_fig=False,
):
    """
    draw heatmap on proportion of removed edges per networks with and without GSR
    """
    # generate long dataframe of edge summary
    edges_df = get_edge_summary(node_summary_path)
    edges_df = generate_set_of_networks(edges_df)
    edges_df.sort_values("edge", inplace=True)
    edges_count_wide_df = get_n_of_edges_per_network(node_summary_path)
    # set figure
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 6))
    data_dict = defaultdict(dict)
    # loop for gsr type
    for i, gsr_type in enumerate(["nogs", "gs"]):
        filename_fit = filename_fit_list[i]
        model_type, phase_encoding, day_cor, gsr_type = get_strings_from_filename(
            filename_fit, ["model_type", "phase_encoding", "day_cor", "gsr_type"]
        )
        misfit_set = get_set_of_locally_globlly_misfit_edges(
            filename_fit, error_vars_dict, fit_indices_thresholds_dict, model_type
        )
        # specify misfit edges
        # count number of removed edges
        if len(model_type) == 1:
            model_type = reduce(add, model_type)
        remove_edges = misfit_set[model_type]
        remove_edges_df = edges_df.query("edge in @remove_edges")
        remove_edges_count_wide_df = get_counts_of_edges_per_network(remove_edges_df)
        # draw heatmap of percentage of removed edges
        prop_edges_df_remove = remove_edges_count_wide_df / edges_count_wide_df
        prop_edges_df_remove = rename_net_summary_df(prop_edges_df_remove)
        data_dict[gsr_type] = prop_edges_df_remove
    # loop for visualize
    max_list = []
    for key in data_dict.keys():
        df_max = data_dict[key].max()
        max_list.append(df_max)
    max_prop = np.max(max_list)
    for i, gsr_type in enumerate(["nogs", "gs"]):
        sns.heatmap(
            data_dict[gsr_type],
            annot=True,
            fmt=".1%",
            ax=axes[i],
            vmin=0,
            vmax=1,
        )
        # set title
        gsr_suffix = generate_gsr_suffix(gsr_type, capitalize=True)
        axes[i].set_title(gsr_suffix)
        axes[i].set_xticklabels(labels=axes[i].get_xticklabels(), rotation=45)
    fig.tight_layout()

    if save_fig:
        fig_folder = op.join(SCHAEFER_DIR, "reliability", "figures")
        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, fit_indices_thresholds_dict, cor_min_max_dict=None
        )
        str_day_cor = "_Daycor" if day_cor else ""
        str_pe = "_PE" if phase_encoding else ""
        model_str = str_day_cor + str_pe
        fig_name = f"percentage_of_removed_edges_{model_str}_{thresholds_suffix}.png"
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def calc_omega_rsfc(
    filename,
    fit_indices_thresholds_dict=None,
    node_summary_path: str = NODE_SUMMARY_PATH,
):
    """
    Calculate omega coefficients of RSFCs focusing on the Model RSFC 2
    """
    params_dict = generate_params_dict(filename)
    loadings = params_dict.get("model_onlyFC")[:, 0:4, 0]
    covariances = params_dict.get("model_onlyFC")[:, 4:6, 0]
    return calc_omega_2d_array(loadings, covariances)


def get_load_cov_from_filename(
    filename=None,
    params=None,
    PE_exist=False,
    day_exist=False,
    order_exist=False,
    get_cov_load=True,
    get_model_type=False,
    parcellation='Schaefer',
    msst=False,
    param_order_filename=None,
    family_cv=False,
    trait_type=None,
    scale_name=None
):
    """
    Get factor loadings and error covariances from filename
    """
    if params is None:
        params = generate_params_dict(filename, parcellation=parcellation, family_cv=family_cv).get("model_onlyFC")
    trait_scale_name = get_scale_name_from_trait(trait_type)
    if param_order_filename:
        param_order = pd.read_csv(op.join(FA_ORDER_DIR, trait_scale_name, param_order_filename))
    if not msst:
        loadings = params[:, 0:4, 0]
    else:
        load_o1_positions = param_order.query('lhs == "o1" & (rhs == "s1" | rhs == "s2")').index
        load_o2_positions = param_order.query('lhs == "o2" & (rhs == "s3" | rhs == "s4")').index
        load_tf_positions = param_order.query('lhs == "ff" & (rhs == "o1" | rhs == "o2")').index
        loadings_o1, loadings_o2, loadings_tf = params[:, load_o1_positions, 0], params[:, load_o2_positions, 0], params[:, load_tf_positions, 0]
        if trait_type in TRAIT_TYPES:
            # get trait loadings (necessary when investigating loadings of trait scales)
            pass

    if filename:
        PE_exist, day_exist, order_exist = (
            ("PE" in filename),
            ("DayCor" in filename),
            ("OrderInDay" in filename),
        )

    if (not PE_exist) and (not day_exist) and (not order_exist):
        covariances = None
        model_type = "errorNone"

    elif day_exist and not PE_exist and not order_exist:
        covariances = params[:, 4:6, 0]
        model_type = "errorDay"
    elif PE_exist and not day_exist and not order_exist:
        covariances = params[:, 4:6, 0]
        model_type = "errorPE"
    elif order_exist and not day_exist and not PE_exist:
        covariances = params[:, 4:6, 0]
        model_type = "errorOrder"

    elif day_exist and PE_exist and not order_exist:
        covariances = params[:, 4:8, 0]
        model_type = "errorPEDay"

    elif day_exist and order_exist and not PE_exist:
        covariances = params[:, 4:8, 0]
        model_type = "errorOrderDay"
    
    if msst:
        return np.concatenate([loadings_o1, loadings_o2], axis=1), loadings_tf
    else:
        if get_cov_load:
            return loadings, covariances
        elif get_model_type:
            return model_type


def calc_omega_2d_array(loadings, covariances=None):
    """
    Calculate omega coefficient using a 2 dimensional array
    """
    uniqueness = 1 - loadings**2
    if covariances is not None:
        denominator = (
            np.sum(loadings, axis=1) ** 2
            + np.sum(uniqueness, axis=1)
            + 2 * np.sum(covariances, axis=1)
        )
    else:
        denominator = np.sum(loadings, axis=1) ** 2 + np.sum(uniqueness, axis=1) + 2
    numerator = np.sum(loadings, axis=1) ** 2
    omega = numerator / denominator
    return omega


def calc_omega_2d_array_from_filename(
        filename=None, 
        params=None,
        parcellation='Schaefer'
        ):
    """
    Calculate omega coefficnents from filename
    """
    loadings, covariances = get_load_cov_from_filename(filename, params=params, parcellation=parcellation)
    omegas = calc_omega_2d_array(loadings, covariances)
    return omegas


def z_transform(r, n):
    z = np.log((1 + r) / (1 - r)) * (np.sqrt(n - 3) / 2)
    return z


def wrapper_of_compare_correspondence_between_folds(
    n_arrays_dict, save_filename, return_gglist=False, **kwargs
):
    """
    Wrapper function of compare_correspondence_between_folds
    """
    g_list = []
    trait_type_list = ["cognition", "mental", "personality"]

    for trait_type in trait_type_list:
        title = get_scale_name_from_trait(trait_type, publication=True)
        g = compare_correspondence_between_folds(
            trait_type, n_arrays_dict, title=title, return_plot=True, **kwargs
        )
        g_list.append(g)

    if return_gglist:
        return g_list

    combine_gg_list(g_list, fig_height=8, filename_fig=save_filename)


#def read_trait_data(trait_type):
#    """
#    Read trait data based on inpuy
#    """
#    if trait_type == 'cognition':
#        df = read_cog_data() 
#    elif trait_type == 'mental':
#        df = read_asr_data()
#    elif trait_type == 'personality':
#        df = read_ffi_data()
#    return df


def vis_fscore_raw_trait(**kwargs):
    """
    Examine relationshiops between raw scores and factor scores of trait measures
    """
    output_df = pd.DataFrame()
    subjects_lists = train_test_split_family()
    for trait_type in TRAIT_TYPES:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        fscore_dir = op.join(FA_PARAMS_DIR, trait_scale_name, 'tables')
        subscales = get_subscale_list(trait_scale_name)
        for i, fold in enumerate(['Fold_0', 'Fold_1']):
            # rad raw data
            subjects_list = subjects_lists[i]
            trait_df_raw = read_trait_data(trait_type, subjects_list=subjects_list)
            trait_df_raw['score_type'] = 'raw'
            # read fscore data
            fscore_filename = [i for i in os.listdir(fscore_dir) if fold.lower() in i]
            if len(fscore_filename) == 1:
                trait_df_fscore = pd.read_csv(op.join(fscore_dir, fscore_filename[0]))
                trait_df_fscore['score_type'] = 'fscore'
                trait_df_fscore['Subject'] = trait_df_raw['Subject'].tolist()
            else:
                raise Exception('fscore_filename includes more than 1 file names.')
            common_columns = ['score_type', 'Subject'] + subscales
            merged_df = pd.concat([trait_df_raw[common_columns], trait_df_fscore[common_columns]], axis=0) 
            merged_df = merged_df.melt(id_vars=['score_type', 'Subject'], value_vars=subscales, var_name='scale', value_name='score')
            merged_df = merged_df.pivot(index=['Subject', 'scale'], columns='score_type', values='score').reset_index()
            merged_df['Fold'] = fold
            output_df = pd.concat([output_df, merged_df], axis=0)
    return output_df


def get_fscores_dict(
        parcellation='Schaefer',
        gsr_types=['nogs', 'gs'],
        controls:list=None,
        msst=True,
        diff_load=True,
        random_seed=None
        ):
    """
    Get dictionary including factor score files to use cross-validation frameworks
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    fscore_dir = op.join(atlas_dir, 'reliability', 'factor_scores', 'combined', 'split_half_cv')
    output_dict = defaultdict(dict)
    if controls is not None:
        control_suffix = 'controlling_' + '_'.join(controls)
    if random_seed is not None:
        seed_str = f'seed{random_seed}'
    else:
        seed_str = ''
    for gsr_type in gsr_types:
        for fold in ['Fold_0', 'Fold_1']:
            filenames = [i for i in os.listdir(fscore_dir) if f'_{gsr_type}_' in i and fold in i]
            if controls is not None:
                filenames = [i for i in filenames if control_suffix in i]
            if msst:
                filenames = [i for i in filenames if 'MSST' in i]
            if diff_load:
                filenames = [i for i in filenames if 'DL' in i]
            if random_seed is not None:
                filenames = [i for i in filenames if seed_str in i]
            filename = sort_list_by_time(filenames)[-1]
            output_dict[gsr_type][fold] = filename
    return output_dict


def get_fscore_validity_edges_df(
        parcellation='Schaefer',
        fc_filename_dict={},
        fscores_dict={},
        invalid_edge_file_dict={},
        gsr_types=['nogs', 'gs'],
        msst=True,
        param_order_filename=None,
        calc_cor=False
        ):
    """
    Examine relationships between factor scores of FC and raw FC
    fscores_dict can be get by get_fscores_dict() function
    """
    subjects_lists = train_test_split_family()
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    fscore_dir = op.join(atlas_dir, 'reliability', 'factor_scores')
    
    params_dir = op.join(atlas_dir, 'reliability', 'parameters')
    param_order = pd.read_csv(op.join(FA_PARAMS_DIR, 'reliability', param_order_filename))
    p_vf_positions_o1 = param_order.query('op == "=~" & lhs.str.contains("o1")').index
    p_vf_positions_o2 = param_order.query('op == "=~" & lhs.str.contains("o2")').index
    p_fh_positions = param_order.query('op == "=~" & rhs.str.contains("o")').index

    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    edge_index = edges_df.index
    invalid_edges_dir = op.join(atlas_dir, 'reliability', 'invalid_edges')
    
    subjects_list = os.listdir(op.join(HCP_ROOT_DIR, 'data'))
    fold0_ids, fold1_ids = train_test_split_family()
    fold0_ids_idx, fold1_ids_idx = sorted([subjects_list.index(i) for i in fold0_ids]), sorted([subjects_list.index(i) for i in fold1_ids])
    subject_idx_dict = {'Fold_0': fold0_ids_idx, 'Fold_1': fold1_ids_idx}
    fold_list = ['Fold_0', 'Fold_1']

    for gsr_type in gsr_types:
        gsr_str = gsr_type.replace('_', '')
        print(f'Processing {gsr_type}')
        fc_filename = fc_filename_dict.get(gsr_str)
        fc_array = np.load(op.join(atlas_dir, fc_filename))[edge_index]
        fc_array_mean = np.mean(fc_array, axis=2)
        for fig_row, fold in enumerate([fold_list[0]]):
            valid_fold = [i for i in fold_list if i != fold][0]
            print(f'Processing {fold}')
            train_fscore_filename = fscores_dict.get(gsr_str).get(fold)
            train_fscores = np.load(op.join(fscore_dir, 'combined', 'split_half_cv', train_fscore_filename))[edge_index]
            valid_fscore_filename = fscores_dict.get(gsr_str).get(valid_fold)
            valid_fscores = np.load(op.join(fscore_dir, 'combined', 'split_half_cv', valid_fscore_filename))[edge_index]
            if msst:
                train_fscores, valid_fscores = train_fscores[:, :, 2].T, valid_fscores[:, :, 2].T
            train_subject_idx = subject_idx_dict.get(fold)
            train_mscores = fc_array_mean[:, train_subject_idx]
            valid_subject_idx = subject_idx_dict.get(valid_fold)
            valid_mscores = fc_array_mean[:, valid_subject_idx]
            train_mscores, valid_mscores = train_mscores.T, valid_mscores.T
#            invalid_edges_train = np.where(np.isnan(train_scores))[1]
#            invalid_edges_valid = np.where(np.isnan(valid_scores))[1]
#            invalid_edges = set(invalid_edges_train) | set(invalid_edges_valid)
            # remove invalid edges
            train_params = np.load(op.join(params_dir, 'combined', 'split_half_cv', train_fscore_filename.replace('factor_scores', 'params')))
            valid_params = np.load(op.join(params_dir, 'combined', 'split_half_cv', valid_fscore_filename.replace('factor_scores', 'params')))
           # invalid_edges_train_dir = op.join(invalid_edges_dir, fold)
           # invalid_edges_valid_dir = op.join(invalid_edges_dir, valid_fold) 
           # invalid_edge_file = invalid_edge_file_dict.get(gsr_str) 
           # invalid_edges_train = np.loadtxt(op.join(invalid_edges_train_dir, invalid_edge_file)).astype(int)
           # invalid_edges_valid = np.loadtxt(op.join(invalid_edges_valid_dir, invalid_edge_file)).astype(int)
        #    invalid_edges = set(invalid_edges_train) | set(invalid_edges_valid)
        #    train_fscores, train_mscores = np.delete(train_fscores, list(invalid_edges), axis=1), np.delete(train_mscores, list(invalid_edges), axis=1)
        #    valid_fscores, valid_mscores = np.delete(valid_fscores, list(invalid_edges), axis=1), np.delete(valid_mscores, list(invalid_edges), axis=1)
            # remove edges with nan
        #    nan_edge_bool = np.isnan(train_fscores).any(axis=0) | np.isnan(valid_fscores).any(axis=0)
        #    train_fscores, train_mscores = train_fscores[:, ~nan_edge_bool], train_mscores[:, ~nan_edge_bool]
        #    valid_fscores, valid_mscores = valid_fscores[:, ~nan_edge_bool], valid_mscores[:, ~nan_edge_bool]
            # calculate correlation between factor scores and mean scores
            n_valid_rsfc = valid_fscores.shape[1]
            if calc_cor:
                cor_array_train, cor_array_valid = np.empty(shape=(n_valid_rsfc)), np.empty(shape=(n_valid_rsfc))
                cor_array_train[:], cor_array_valid[:] = np.nan, np.nan
                cor_array_train_rho, cor_array_valid_rho = np.empty(shape=(n_valid_rsfc)), np.empty(shape=(n_valid_rsfc))
                cor_array_train_rho[:], cor_array_valid_rho[:] = np.nan, np.nan
                print_str = 'correlation and '
            else:
                print_str = ''
            valid_array_train, valid_array_valid = np.empty(shape=(n_valid_rsfc)), np.empty(shape=(n_valid_rsfc))
            valid_array_train[:], valid_array_valid[:] = np.nan, np.nan
            
            print(f'Calculating {print_str}validity coefficients.')
            for i in range(n_valid_rsfc):
                # calculate correlations between factor scores and raw FCs
                if calc_cor:
                    cor_array_train[i] = np.corrcoef(train_mscores[:, i], train_fscores[:, i])[0, 1]
                    cor_array_valid[i] = np.corrcoef(valid_mscores[:, i], valid_fscores[:, i])[0, 1]
                    cor_array_train_rho[i] = spearmanr(train_mscores[:, i], train_fscores[:,i]).correlation
                    cor_array_valid_rho[i] = spearmanr(valid_mscores[:, i], valid_fscores[:,i]).correlation
                ### calculate validity coefficients according to Grice (2001)
                # get Rkk (original item correlations)
                r_kk_train = np.corrcoef(fc_array[i, train_subject_idx, :], rowvar=False)
                r_kk_valid = np.corrcoef(fc_array[i, valid_subject_idx, :], rowvar=False)
                ## calculate Pvh (in Gorsuch) Skf (in Grice) (correlation between items and higher-order factors)
                # get Pvf (First order factor loadings)
                p_vf_train_o1, p_vf_valid_o1 = train_params[i, p_vf_positions_o1, 0], valid_params[i, p_vf_positions_o1, 0]
                p_vf_train_o2, p_vf_valid_o2 = train_params[i, p_vf_positions_o2, 0], valid_params[i, p_vf_positions_o2, 0]
                p_vf_train_o1, p_vf_valid_o1 = np.pad(p_vf_train_o1, (0, 2)), np.pad(p_vf_valid_o1, (0, 2))
                p_vf_train_o2, p_vf_valid_o2 = np.insert(p_vf_train_o2, [0, 0], [0, 0]), np.insert(p_vf_valid_o2, [0, 0], [0, 0])
                p_vf_train, p_vf_valid = np.hstack([p_vf_train_o1[:, np.newaxis], p_vf_train_o2[:, np.newaxis]]), np.hstack([p_vf_valid_o1[:, np.newaxis], p_vf_valid_o2[:, np.newaxis]])
                # get Pfh (second order factor loadings)
                p_fh_train, p_fh_valid = train_params[i, p_fh_positions, 0], valid_params[i, p_fh_positions, 0]
                p_fh_train, p_fh_valid = p_fh_train[:, np.newaxis], p_fh_valid[:, np.newaxis]
                # calculate Pvh
                p_vh_train, p_vh_valid = p_vf_train.dot(p_fh_train), p_vf_train.dot(p_fh_valid)
                ## calculate Wkf in Grice
                w_kf_train, w_kf_valid = inv(r_kk_train).dot(p_vh_train), inv(r_kk_valid).dot(p_vh_valid)
                ## calculate Css in Grice
                c_ss_train, c_ss_valid = w_kf_train.T.dot(r_kk_train).dot(w_kf_train), w_kf_valid.T.dot(r_kk_valid).dot(w_kf_valid)
                # calculate Lss in Grice
                l_ss_train, l_ss_valid = np.sqrt(np.diag(c_ss_train)), np.sqrt(np.diag(c_ss_valid))
                ## calculate Rfs in Grice
                validity_train, validity_valid = p_vh_train.T.dot(w_kf_train)[0, 0] / l_ss_train[0], p_vh_valid.T.dot(w_kf_valid)[0, 0] / l_ss_valid[0]
                valid_array_train[i], valid_array_valid[i] = validity_train, validity_valid
            if calc_cor:
                edges_df[f'{gsr_type}_train_r'] = cor_array_train
                edges_df[f'{gsr_type}_valid_r'] = cor_array_valid
                edges_df[f'{gsr_type}_train_rho'] = cor_array_train_rho
                edges_df[f'{gsr_type}_valid_rho'] = cor_array_valid_rho
            edges_df[f'{gsr_type}_train_validity'] = valid_array_train
            edges_df[f'{gsr_type}_valid_validity'] = valid_array_valid
    print('Finished.')
    return edges_df


def split_half_family_cv(
        parcellation='Schaefer',
        n_arrays=431,
        estimator=Ridge,
        grid_search=False,
        grid_n_splits=5,
        grid_n_repeats=5,
        msst=True,
        fscores_dict={},
        fc_filename_dict={},
        target_fc='fscore',
        target_trait='fscore',
        target_net_list=None,
        within_network=False,
        metric_func_dict={'mae':median_absolute_error, 'r2':r2_score},
        grid_params_dict={'alpha': np.logspace(3, 7, num=10)},
        grid_scoring='neg_median_absolute_error',
        estimator_tol=0.0001,
        estimator_selection=None,
        trait_list=None,
        draw_fig=False,
        fig_size=(12, 6),
        grid_get_train_score=True,
#        separate_x_scaling=False,
#        separate_y_scaling=False,
        grid_n_jobs=50,
        permutation=False,
        p_thres=None,
        y_std=False,
        invalid_edge_file_dict=None,
        gsr_types=['nogs', 'gs'],
        covariates=['age', 'gender', 'MeanRMS'],
        item_level_pred=False,
        stacking_item=False,
        grid_params_dict_meta={'alpha': np.logspace(3, 7, num=10)},
        param_order_filename=None,
        trait_equal_loadings=False,
        random_seed=0,
        save_filename=None,
        get_pred_values=False,
        output_pred_value_filename='actual_pred',
        multivariate_items=False,
        **kwargs
        ):
    """
    Apply Ridge regression to split-half cross-validation data
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    fscore_dir = op.join(atlas_dir, 'reliability', 'factor_scores')
    params_dir = op.join(atlas_dir, 'reliability', 'parameters')
    if param_order_filename:
        param_order = pd.read_csv(op.join(FA_PARAMS_DIR, 'reliability', param_order_filename))
        p_vf_positions = param_order.query('op == "=~" & rhs.str.contains("s")').index
        p_fh_positions = param_order.query('op == "=~" & rhs.str.contains("o")').index
    
    subjects = np.loadtxt(SUBJECT_ANALYSIS_PATH, dtype=str) 
    subjects_list = os.listdir(op.join(HCP_ROOT_DIR, 'data'))

    # Get trait data
    cog_df = read_cog_data() 
    asr_df = read_asr_data()
    ffi_df = read_ffi_data()

    trait_df = pd.merge(cog_df, asr_df, on='Subject')
    trait_df = pd.merge(trait_df, ffi_df, on='Subject')
    trait_df.query('Subject in @subjects', inplace=True)
    # read data including confounds
    cov_df = pd.read_csv(COVARIATES_PATH)
    cov_df['Subject'] = cov_df['Subject'].astype(str)
    trait_df = pd.merge(trait_df, cov_df, on='Subject')
    # add modified scores
    trait_df['CogFluidComp_Unadj_dropped'] = trait_df[[i for i in FLUID_COLUMNS if not i in ['PicSeq_Unadj', 'ListSort_Unadj']]].mean(axis=1)
    trait_df['NEOFAC_Openness_dropped'] = trait_df[['NEORAW_13', 'NEORAW_23', 'NEORAW_28', 'NEORAW_43', 'NEORAW_48', 'NEORAW_53', 'NEORAW_58']].mean(axis=1)
    fold0_ids, fold1_ids = train_test_split_family(random_seed=random_seed)
    # these idx are necessary for subsetting from fc data
    fold0_ids_idx, fold1_ids_idx = sorted([subjects_list.index(i) for i in fold0_ids]), sorted([subjects_list.index(i) for i in fold1_ids])
    subject_idx_dict = {'Fold_0': fold0_ids_idx, 'Fold_1': fold1_ids_idx}
    subjects = set(fold0_ids) | set(fold1_ids)
    trait_df.query('Subject in @subjects', inplace=True)
    # standardization
    if y_std:
        for trait in trait_list:
            scaler = StandardScaler()
            trait_df[trait] = scaler.fit_transform(trait_df[trait].to_numpy().reshape(-1, 1))
    fold0_trait_df, fold1_trait_df = trait_df.query('Subject in @fold0_ids'), trait_df.query('Subject in @fold1_ids')
    trait_cols = [i for i in trait_df.columns if i != 'Subject']
    if trait_list:
        trait_cols = [i for i in trait_cols if i in trait_list]
    # Get edge summary
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    edge_index = edges_df.index
    if target_net_list is not None:
        edge_index = edges_df.query('(node1_net in @target_net_list) & (node2_net in @target_net_list)').index
    if within_network:
        edge_index_add = edges_df.query("node1_net == node2_net").index
        edge_index = [i for i in edge_index if i in edge_index_add]
    fold_list = ['Fold_0', 'Fold_1']
    
    result_df = pd.DataFrame()
    
#    if get_pred_values:
#        # gsr * actual vs pred * trait * subjects
#        output_values_fold0 = np.empty(shape=(2, 2, len(trait_list), len(fold0_ids)))
#        output_values_fold1 = np.empty(shape=(2, 2, len(trait_list), len(fold1_ids)))
#        output_values_fold0[:], output_values_fold1[:] = np.nan, np.nan 

    invalid_edges_dir = op.join(atlas_dir, 'reliability', 'invalid_edges')
    
    for gsr_int, gsr_type in enumerate(gsr_types):
        gsr_str = gsr_type.replace('_', '')
        print(f'Processing {gsr_type}')
        if draw_fig:
            fig, axes = plt.subplots(len(fold_list), len(trait_cols), figsize=fig_size)
        fc_filename = fc_filename_dict.get(gsr_str)
        fc_array = np.load(op.join(atlas_dir, fc_filename))
        fc_array_mean = np.mean(fc_array, axis=2)
        for fig_row, fold in enumerate(fold_list):
            valid_fold = [i for i in fold_list if i != fold][0]
            print(f'Processing {fold}')
            train_subject_idx = subject_idx_dict.get(fold)
            valid_subject_idx = subject_idx_dict.get(valid_fold)

            if get_pred_values:
                # actual vs pred * trait * subjects
                output_values = np.empty(shape=(2, len(trait_list), len(valid_subject_idx)))
                output_values[:] = np.nan
            invalid_edges_train_dir = op.join(invalid_edges_dir, fold)
            invalid_edges_valid_dir = op.join(invalid_edges_dir, valid_fold) 
            if invalid_edge_file_dict:
                invalid_edge_file = invalid_edge_file_dict.get(gsr_str) 
                invalid_edges_train = np.loadtxt(op.join(invalid_edges_train_dir, invalid_edge_file)).astype(int)
                invalid_edges_valid = np.loadtxt(op.join(invalid_edges_valid_dir, invalid_edge_file)).astype(int)
            if target_fc == 'fscore':
                train_fscore_filename = fscores_dict.get(gsr_str).get(fold)
                train_scores = np.load(op.join(fscore_dir, 'combined', 'split_half_cv', train_fscore_filename))
                valid_fscore_filename = fscores_dict.get(gsr_str).get(valid_fold)
                valid_scores = np.load(op.join(fscore_dir, 'combined', 'split_half_cv', valid_fscore_filename))
                train_params = np.load(op.join(params_dir, 'combined', 'split_half_cv', train_fscore_filename.replace('factor_scores', 'params')))
                valid_params = np.load(op.join(params_dir, 'combined', 'split_half_cv', valid_fscore_filename.replace('factor_scores', 'params')))
                if msst:
                    train_scores, valid_scores = train_scores[:, :, 2].T, valid_scores[:, :, 2].T
            elif target_fc == 'mean':
                train_scores = fc_array_mean[:, train_subject_idx]
                valid_scores = fc_array_mean[:, valid_subject_idx]
                train_scores, valid_scores = train_scores.T, valid_scores.T
            print('Applying model.')
            n_keys = len(metric_func_dict.keys())
            lists = [[] for _ in range(n_keys)]
            # Missing value imputation
            if invalid_edge_file_dict:
                train_scores[:, invalid_edges_train] = np.nan
                valid_scores[:, invalid_edges_valid] = np.nan
            imputer_train = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
            imputer_valid = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
            train_scores = imputer_train.fit_transform(train_scores)
            valid_scores = imputer_valid.fit_transform(valid_scores)
            # edge selection
            train_scores = train_scores[:, edge_index]
            valid_scores = valid_scores[:, edge_index]
    #        # standardization

            #        if target_fc == 'mean':
    #            train_scores
            scaler_x = StandardScaler()
            train_scores_std = scaler_x.fit_transform(train_scores)
            valid_scores_std = scaler_x.transform(valid_scores)
            for fig_col, trait_col in enumerate(trait_cols):
                if trait_col in ['CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'CogTotalComp_Unadj']:
                    trait_type = 'cognition'
                elif trait_col in [ 'All', 'Internalizing', 'Externalizing', 'Others']:
                    trait_type = 'mental'
                elif trait_col in ['NEOFAC_Neuroticism', 'NEOFAC_Extraversion', 'NEOFAC_Openness', 'NEOFAC_Agreeableness', 'NEOFAC_Conscientiousness']:
                    trait_type = 'personality'
                # add processing item level predictions
                if target_trait == 'mean':
                    y_valid = fold1_trait_df[trait_col] if fold == 'Fold_0' else fold0_trait_df[trait_col]
                    if not item_level_pred:
                        y = fold0_trait_df[trait_col] if fold == 'Fold_0' else fold1_trait_df[trait_col]
                    else:
                        if trait_col in COMPOSITE_COLUMNS:
                            trait_scale_name = 'NIH_Cognition'
                            subscale_col = NIH_COGNITION_SCALES_MAPPING.get(trait_col)
                        elif trait_col in ASR_SUBSCALES:
                            trait_scale_name = 'ASR'
                            subscale_col = trait_col
                        elif trait_col in ['NEOFAC_' + i for i in NEO_FFI_SCALES]:
                            trait_scale_name = 'NEO_FFI'
                            subscale_col = trait_col.replace('NEOFAC_', '')
                        scale_names = SCALES_DICT.get('both').get(trait_scale_name).get(subscale_col)
                        y = fold0_trait_df[scale_names] if fold == 'Fold_0' else fold1_trait_df[scale_names]
                elif target_trait == 'fscore':
                    if trait_col in COMPOSITE_COLUMNS:
                        trait_scale_name = 'NIH_Cognition'
                        subscale_col = NIH_COGNITION_SCALES_MAPPING.get(trait_col)
                    elif trait_col in ASR_SUBSCALES:
                        trait_scale_name = 'ASR'
                        subscale_col = trait_col
                    elif trait_col in ['NEOFAC_' + i for i in NEO_FFI_SCALES]:
                        trait_scale_name = 'NEO_FFI'
                        subscale_col = trait_col.replace('NEOFAC_', '')
                    ## get (sub)scale data
                    # training data (modification may be necessary when including covariate in model)
                    filename = f'Seed{random_seed}__{fold.lower()}_sampleN_{kwargs.get("sample_n").get(fold)}_fscore.csv'
                    if trait_equal_loadings:
                        filename = 'EqualLoadings_' + filename
                    y = pd.read_csv(op.join(FA_PARAMS_DIR, trait_scale_name, 'tables', filename), index_col=False)[subscale_col]
                    # validation data
                    fold_valid = reduce(add, [i for i in fold_list if not i == fold])
                    filename_valid = f'Seed{random_seed}__{fold_valid.lower()}_sampleN_{kwargs.get("sample_n").get(fold_valid)}_fscore.csv'
                    if trait_equal_loadings:
                        filename_valid = 'EqualLoadings_' + filename_valid
                    y_valid = pd.read_csv(op.join(FA_PARAMS_DIR, trait_scale_name, 'tables', filename_valid), index_col=False)[subscale_col]
                model = estimator()
                # convert to z score (modification may be necessary)
                y, y_valid = y.to_numpy(dtype=float), y_valid.to_numpy(dtype=float)
                if not item_level_pred:
                    y = y.reshape(-1, 1)
                y_valid = y_valid.reshape(-1, 1)
                if y_std:
                    train_scaler = StandardScaler()
                    y = train_scaler.fit_transform(y) 
                    valid_scaler = StandardScaler()
                    y_valid = valid_scaler.fit_transform(y_valid)
                if np.isnan(y).sum() > 0:
                    y.fillna(y.median(), inplace=True)
                if np.isnan(y_valid).sum() > 0:
                    y_valid.fillna(y_valid.median(), inplace=True)
                if grid_search:
                    cv = RepeatedKFold(n_splits=grid_n_splits, n_repeats=grid_n_repeats, random_state=0) 
                    model = GridSearchCV(
                            estimator(
                                #random_state=0, 
                                #tol=estimator_tol, 
                                #selection=estimator_selection
                                ), 
                            grid_params_dict,
                            scoring=grid_scoring, 
                            n_jobs=grid_n_jobs, 
                            cv=cv,
                            return_train_score=grid_get_train_score
                            )
                if p_thres is not None:
                    n_rsfc = train_scores_std.shape[1]
                    n_sample = len(y)
                    cor_array, p_array = np.empty(shape=n_rsfc), np.empty(shape=n_rsfc)
                    cor_array[:], p_array[:] = np.nan, np.nan
                    for i in range(n_rsfc):
                        cor = np.corrcoef(train_scores_std[:, i], y)[0, 1]
                        t_value = cor * np.sqrt(n_sample - 2) / np.sqrt(1 - cor**2)
                        p = t.sf(np.abs(t_value), n_sample - 1) * 2
                        cor_array[i], p_array[i] = cor, p
                    p_index = np.where(p_array < p_thres)
                    print(f'Number of selected edges is {len(p_index[0])}.')
                    train_scores_std = train_scores_std[:, p_index[0]]
                    valid_scores_std = valid_scores_std[:, p_index[0]]
                if not item_level_pred:
                    # become effects of invalid edges null
                    model.fit(train_scores_std, y)
                    # alpha
                    alpha = model.best_params_.get("alpha")
                    print(f'alpha is {alpha:.3f}')
                    # best score
                    best_grid_score = model.best_score_
                    # become effects of invalied edges null
                    y_pred = model.predict(valid_scores_std)
                else:
                    y_pred_array_train, y_pred_array_valid = np.empty(shape=y.shape), np.empty(shape=(len(y_valid), y.shape[1]))
                    y_pred_array_train[:], y_pred_array_valid[:] = np.nan, np.nan
                    # conduct multivariate regressions
                    if multivariate_items:
                        model.fit(train_scores_std, y)
                        y_pred_array_train = model.predict(train_scores_std)
                        y_pred_array_valid = model.predict(valid_scores_std)
                    else:
                        parallel = Parallel(n_jobs=12, return_as="list", verbose=12)
                        error_count = 0
                        y = y.astype(int)

                        def model_predict(item_num, y, model, train_scores_std, valid_scores_std):
                            print(f'Processing item {item_num+1}.')
                            try:
                                model.fit(train_scores_std, y[:, item_num])
                                y_pred_train = model.predict(train_scores_std)
                                y_pred_valid = model.predict(valid_scores_std)
                            except ValueError:
                                print('ValueError occurred.')
                                y_pred_train = [np.nan] * len(train_scores_std)
                                y_pred_valid = [np.nan] * len(valid_scores_std)
                            return y_pred_valid

                        output_gen = parallel(delayed(model_predict)(item_num, y, model, train_scores_std, valid_scores_std) for item_num in range(y.shape[1]))
                        # calculate predicted score from generator or list (output of joblib)
                        for i in range(y.shape[1]):
                            y_pred_array_valid[:, i] = output_gen[i]
                        #for i in range(y.shape[1]):
                        #    print(f'Processing item number {i+1}')
                        #    y = y.astype(int)
                        #    try:
                        #        model.fit(train_scores_std, y[:, i])
                        #        y_pred_array_train[:, i] = model.predict(train_scores_std)
                        #        y_pred_array_valid[:, i] = model.predict(valid_scores_std)
                        #    except ValueError:
                        #        print('ValueError occurred.')
                        #        error_count += 1
                        #        y_pred_array_train[:, i] = [np.nan] * len(train_scores_std)
                        #        y_pred_array_valid[:, i] = [np.nan] * len(valid_scores_std)
                        print(f'Number of ValueError in modeling is {error_count}.')
                    if not stacking_item:
                        if trait_type == 'cognition':
                            y_pred = y_pred_array_valid.mean(axis=1)
                        elif trait_type in ['mental', 'personality']:
                            if np.isnan(y_pred_array_valid).sum() == 0:
                                y_pred = y_pred_array_valid.sum(axis=1)
                            else:
                                y_pred = np.nanmean(y_pred_array_valid, axis=1) * y.shape[1]
                    else:
                        # stacking
                        cv_meta = RepeatedKFold(n_splits=grid_n_splits, n_repeats=grid_n_repeats, random_state=0) 
                        model_meta = GridSearchCV(
                                estimator(
                                    random_state=0, 
                                    tol=estimator_tol, 
                                    #selection=estimator_selection
                                    ), 
                                grid_params_dict_meta,
                                scoring=grid_scoring,
                                n_jobs=grid_n_jobs, 
                                cv=cv_meta,
                                return_train_score=grid_get_train_score
                                )
                        if trait_scale_name in ['ASR', 'NEO_FFI']:
                            y_summary = y.sum(axis=1)
                        elif trait_scale_name == 'NIH_Cognition':
                            y_summary = y.mean(axis=1)
                        model_meta.fit(y_pred_array_train, y_summary)
                        alpha_meta = model_meta.best_params_.get("alpha")
                        print(f'alpha of meta model is {alpha_meta:.3f}')
                        y_pred = model_meta.predict(y_pred_array_valid)
                        #y_pred = y_pred_array.mean(axis=1)
                # permutation test
                if permutation:
                    p = permutation_test_score(model, train_scores_std, y_valid, cv=None)
                # fit in train data
#                if grid_get_train_score:
#                    mean_train_score = np.mean(model.cv_results_['mean_test_score'])
#                    sd_train_score = np.std(model.cv_results_['mean_test_score'])
                for i, (metric_key, metric_func) in enumerate(metric_func_dict.items()):
                    metric = metric_func(y_valid, y_pred)
                    lists[i].append(metric)
                    print(f'{metric_key}: {metric:.3f} in {trait_col}')
                if get_pred_values:
                    output_values[0, fig_col, :] = np.squeeze(y_valid)
                    output_values[1, fig_col, :] = np.squeeze(y_pred)
                if draw_fig:
                    target_ax = axes[fig_row, fig_col]
                    target_ax.scatter(y_pred, y_valid, s=0.5)
                    target_ax.set_title(f'{trait_col}\nin {fold}', fontsize=10)
                    target_ax.set_xlabel('Predicted')
                    target_ax.set_ylabel('Actual')
                    # linear regression line
                    b, a = np.polyfit(np.squeeze(y_pred), np.squeeze(y_valid), deg=1)
                    start, stop = target_ax.get_xlim()
                    xseq = np.linspace(start, stop, num=100)
                    target_ax.plot(xseq, a + b * xseq, color="k", lw=1)
                    # diagonal line
                    target_ax.plot(xseq, xseq, color='k', lw=1, linestyle='--')
                    #alpha_str = f'alpha = {alpha:.3f}'
                    #best_inner_score_str = f'best inner score = {best_grid_score:.3f}'
                    metric_str = f'r2 = {r2_score(y_valid, y_pred):.3f}\ntest_score = {mean_squared_error(y_valid, y_pred):.3f}'
                    # vis_str can  be arranged
                    vis_str = metric_str
                 #   if grid_get_train_score:
                 #       vis_str += f'\nmean train score = {mean_train_score:.3f}\nSD of train score = {sd_train_score:.3f}'
                    # metric
                    target_ax.text(
                            x=0.01, y=0.99,
                            s=vis_str, 
                            va='top', ha='left', 
                            fontsize=6, 
                            transform=target_ax.transAxes
                            )
                    target_ax.set_aspect('equal', adjustable='datalim')
            for i, metric_key in enumerate(metric_func_dict.keys()):
                result_df[f'{fold}_{gsr_type}_{metric_key}'] = lists[i]
            if get_pred_values:
                np.save(
                    op.join(
                        atlas_dir, 
                        'prediction', 
                        f'Fold{fold}_{gsr_str}_RSFC_{target_fc}_trait_{target_trait}_seed{random_seed}_{output_pred_value_filename}.npy'
                    ),
                    output_values
                    )
        result_df.index = trait_cols
        if draw_fig:
            fig.suptitle(f'{gsr_str} in RSFC {target_fc} and trait {target_trait}')
            if save_filename is not None:
                save_fig_dir = op.join(atlas_dir, 'prediction', 'scatterplots')
                fig.savefig(op.join(save_fig_dir, f'RSFC_{target_fc}_trait_{target_trait}_{gsr_str}_{save_filename}_seed{random_seed}.png'))
            #fig.tight_layout()
    result_df['fc_type'] = target_fc
    result_df['trait_type'] = target_trait
    return result_df


def visualize_cv(
        df, 
        id_vars=['trait', 'repeat', 'trait_type', 'fc_type'], 
        return_df=True,
        metrics=['r2', 'cor'],
        gsr_types=['nogs', 'gs']
        ):
    """
    Visualize results of cross-validation
    """
    metric_prefixes = reduce(add, [[f'Fold_0_{gsr_type}_', f'Fold_1_{gsr_type}_'] for gsr_type in gsr_types])   
    value_vars = [i + metric for metric in metrics for i in metric_prefixes]
    df = df.reset_index().rename(
            columns={'index': 'trait'}
            ).melt(
                    id_vars=id_vars, 
                    value_vars=value_vars,
                    var_name='condition',
                    #value_name=metric.replace('_', '')
                    )
    df['metric'] = df['condition'].str.split('_').str[-1]
    df['gsr_type'] = df['condition'].str.split('_').str[2]
    df['fold'] = df['condition'].str.split('_').str[1]
    df.drop('condition', axis=1, inplace=True)
    df = df.reset_index().pivot_table(index=id_vars + ['gsr_type', 'fold'], values='value', columns='metric').reset_index()
    
    conditions = (
        df['trait'].str.contains('|'.join(COMPOSITE_COLUMNS)),
        df['trait'].str.contains('|'.join(ASR_SUBSCALES)),
        df['trait'].str.contains('|'.join(NEO_FFI_SCALES))
    )
    choices = ['NIH toolbox', 'ASR', 'NEO-FFI']
    df['trait_scale_name'] = np.select(conditions, choices)
    df['trait_scale_name'] = pd.Categorical(df['trait_scale_name'], categories=choices)
    
    for replace_str in ['NEOFAC_', 'Comp_Unadj', 'Cog']:
        df['trait'] = df['trait'].str.replace(replace_str, '')
    
    NIH_traits = [i + ' (NIH toolbox)' for i in NIH_COGNITION_SCALES]
    ASR_traits = [i + ' (ASR)' for i in ASR_SUBSCALES]
    FFI_traits = [i + ' (NEO-FFI)' for i in NEO_FFI_SCALES]

    df['trait'] = df['trait'].astype(str) + ' (' + df['trait_scale_name'].astype(str) + ')'
    df['trait'] = pd.Categorical(df['trait'], categories=NIH_traits + ASR_traits + FFI_traits)
    
    for gsr_replaced, gsr_replace in zip(['nogs', 'gs'], ['Without GSR', 'With GSR']):
        df['gsr_type'] = df['gsr_type'].str.replace(gsr_replaced, gsr_replace)
    df['gsr_type'] = pd.Categorical(df['gsr_type'], categories=['Without GSR', 'With GSR'])
    df['pred_type'] = 'FC_' + df['fc_type'].astype(str) + '_Trait_' + df['trait_type'].astype(str)
    if return_df:
        return df
#    g = (
#        ggplot(df, aes('trait', 'r', fill='fold'))
#        + geom_col(position=position_dodge())
#        + facet_grid('~ gsr_type')
#        + theme(axis_text_x=element_text(angle=45))
#            )
#    return g


def create_combined_plot(
        long_df,
        vis_type='fscore',
        parcellation='Schaefer',
        save_filename=None,
        save_filename_df=None,
        item_pred_comp=False,
        create_long_df=False,
        include_multivariate=False
        ):
    """
    Save figure of combined plots on prediction performance
    vis_type should be 'fscore' or 'sem_op'
    If item_pred_comp is True, this functions will draw graph comparing univariate and multivariate predictions.
    """
    cog_colors = ['#00A5E3', '#8DD7BF', '#FF96C5']
    asr_colors = ['#FC6238', '#FFD872', '#F2D4CC', '#E77577']
    ffi_colors = ["#C05780", '#FF828B', '#E7C582', '#00B0BA', '#0065A2']
    
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    if vis_type == 'fscore':
        trait_scale = 'trait_scale_name'
    elif vis_type == 'sem_op':
        if not item_pred_comp:
            trait_scale = 'trait_type'
        else:
            trait_scale = 'trait_scale_name'
    if create_long_df:
        if vis_type == 'fscore':
            long_df['pred_type'] = 'FC_' + long_df['fc_type'].astype(str) + '_Trait_' + long_df['trait_type'].astype(str)
            long_df['pred_type'] = long_df['pred_type'].replace(
                {
                    'FC_fscore_Trait_fscore': 'RSFC: Factor score\nPhenotype: Factor score',
                    'FC_mean_Trait_fscore': 'RSFC: Average score\nPhenotype: Factor score',
                    'FC_mean_Trait_mean': 'RSFC: Average score\nPhenotype: Sum score',
                    'FC_fscore_Trait_mean': 'RSFC: Factor score\nPhenotype: Sum score'
                }
            )
            categories = [
                'RSFC: Average score\nPhenotype: Sum score',
                'RSFC: Factor score\nPhenotype: Sum score',
                'RSFC: Average score\nPhenotype: Factor score',
                'RSFC: Factor score\nPhenotype: Factor score'
            ]
        elif vis_type == 'sem_op':
            cat_list = [i for i in long_df['trait'].cat.categories if not 'NIH toolbox' in i]
            long_df['trait'] = pd.Categorical(long_df['trait'], categories=cat_list)
            if not item_pred_comp:
                long_df = long_df.rename(columns={'type': 'pred_type'})
                sem_str = 'SEM-based operative prediction\n + univariate ridge regression'
                ridge_str = 'Univariate ridge regression'
                multi_ridge_str = 'Multivariate ridge regression'
                long_df['pred_type'] = long_df['pred_type'].replace(
                    {
                        'metric_sem': sem_str,
                        'metric_mean': ridge_str,
                        'metric_mean_items': multi_ridge_str
                    }
                )
                categories = [ridge_str, sem_str, multi_ridge_str]
            else:
                long_df = long_df.drop('pred_type', axis=1)
                long_df = long_df.rename(columns={'item_level_pred': 'pred_type'})
                long_df['pred_type'] = long_df['pred_type'].replace(False, 'Univariate').replace(True, 'Multivariate')
                categories = ['Univariate', 'Multivariate']

        long_df[trait_scale] = pd.Categorical(long_df[trait_scale], categories=['ASR', 'NEO-FFI'])
        long_df['pred_type'] = pd.Categorical(long_df['pred_type'], categories=categories)
        long_df['gsr_type_lines'] = long_df['gsr_type'].replace({'With GSR': 'With\nGSR', 'Without GSR': 'Without\nGSR'})
        if save_filename_df is not None:
            long_df.to_pickle(op.join(atlas_dir, 'prediction', f'{save_filename_df}.pkl'))
    
    long_df['z'] = np.arctanh(long_df['cor'])

    # create legend
    long_df_legend = long_df.groupby(['trait']).sample(1)[['trait', 'cor', 'pred_type']]
    long_df_legend['cor'] = np.nan
    
    def get_color_list(vis_type):
        if vis_type == 'fscore':
            color_list = cog_colors + asr_colors + ffi_colors
        elif vis_type == 'sem_op':
            color_list = asr_colors + ffi_colors
        return color_list

    color_list = get_color_list(vis_type)
    g_legend = (
        ggplot(long_df_legend, aes('pred_type', 'cor', color='trait'))
        + geom_point(aes(color='trait'), alpha=0.5)
        + theme_bw()
        + theme(
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_title_y=element_blank(),
            axis_title_x=element_blank(),
            legend_title=element_blank(),
            panel_border=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            axis_ticks_major_x=element_blank(),
            axis_ticks_major_y=element_blank(),
            axis_ticks_minor_y=element_blank(),
            legend_position=(0.5, 0.5),
            legend_direction='vertical',
            legend_background=element_rect(fill='white'),
            legend_box_background=element_rect(fill='white'),
            figure_size=(1, 4)
        )
        + scale_color_manual(values=color_list)
    )
    
    def create_ggplot(long_df, y, title, y_label, vis_type=None):
        g = (
            ggplot(long_df, aes('pred_type', y))
            + geom_boxplot(outlier_color='')
            + geom_jitter(aes(color='trait'), width=0.1, alpha=0.5)
            + facet_grid(f'gsr_type_lines ~ {trait_scale}')
            + theme_bw()
            + theme(
                axis_text_x=element_text(angle=45),
                axis_title_y=element_blank(),
                legend_title=element_blank(),
                legend_position='none'
            )
            + labs(
                y=y_label,
                title=title
                  )
            + coord_flip()
            + scale_color_manual(values=color_list)
        )
        return g
    # create panel a (correlation)
    if not include_multivariate:
        long_df = long_df.query('pred_type != "Multivariate ridge regression"')
    g_z = create_ggplot(long_df, y='z', title=r"$\bf{a}$ Correlation (z-value)", y_label=r'z', vis_type=vis_type)
#    g_cor = (
#        ggplot(long_df, aes('pred_type', 'cor'))
#        + geom_boxplot(outlier_color='')
#        + geom_jitter(aes(color='trait'), width=0.1, alpha=0.5)
#        + facet_grid(f'gsr_type_lines ~ {trait_type}')
#        + theme_bw()
#        + theme(
#            axis_text_x=element_text(angle=45),
#            axis_title_y=element_blank(),
#            legend_title=element_blank(),
#            legend_position='none'
#        )
#        + labs(
#            y=r'r',
#            title=
#              )
#        + coord_flip()
#    )
    
    if vis_type == 'fscore':
        # create panel b
        long_df_mean = long_df.query('trait_type == "mean"')
        g_r2_trait_mean = create_ggplot(
                long_df_mean,
                y='r2',
                title=r"$\bf{b}$" + " Coeffcient of determination\n on sum score of phenotype",
                y_label=r'$R^2$',
                vis_type=vis_type
                )
        # create panel c
        long_df_fscore = long_df.query('trait_type == "fscore"')
        g_r2_trait_fscore = create_ggplot(
                long_df_fscore,
                y='r2',
                title=r"$\bf{c}$" + " Coeffcient of determination\n on factor score of phenotype",
                y_label=r'$R^2$',
                vis_type=vis_type
                )
        
#        g_r2_trait_mean = (
#            ggplot(
#                ,
#                aes('pred_type', 'r2')
#            )
#            + geom_boxplot(outlier_color='')
#            + geom_jitter(aes(color='trait'), width=0.1, alpha=0.5)
#            + facet_grid(f'gsr_type_lines ~ {trait_scale}')
#            + theme_bw()
#            + theme(
#                axis_text_x=element_text(angle=45),
#                axis_title_y=element_blank(),
#                legend_title=element_blank(),
#                legend_position='none'
#            )
#            + labs(
#                y=r'$R^2$',
#                title=r"$\bf{b}$" + " Coeffcient of determination\n on average score of phenotype"
#                  )
#            + coord_flip()
#        )

        # create panel c
#        g_r2_trait_fscore = (
#            ggplot(
#                long_df.query('trait_type == "fscore"'),
#                aes('pred_type', 'r2')
#            )
#            + geom_boxplot(outlier_color='')
#            + geom_jitter(aes(color='trait'), width=0.1, alpha=0.5)
#            + facet_grid(f'gsr_type_lines ~ {trait_scale}')
#            + theme_bw()
#            + theme(
#                axis_text_x=element_text(angle=45),
#                axis_title_y=element_blank(),
#                legend_title=element_blank(),
#                legend_position='none'
#            )
#            + labs(
#                y=r'$R^2$',
#                title=r"$\bf{c}$" + " Coeffcient of determination\n on factor score of phenotype"
#                  )
#            + coord_flip()
#        )

        # merge plots
        g1 = load_ggplot(g_z, figsize=(4, 3))
        g2 = load_ggplot(g_r2_trait_mean, figsize=(4, 1.5))
        g3 = load_ggplot(g_r2_trait_fscore, figsize=(4, 1.5))
        g4 = load_ggplot(g_legend)
        g_combined = (g1/g2/g3)|g4
    
    elif vis_type == 'sem_op':
        #g_cor = g_cor + scale_color_manual(values=[i for i in range(3, 12)])
        g_r2 = create_ggplot(long_df, y='r2', title=r"$\bf{b}$" + " Coeffcient of determination", y_label=r'$R^2$', vis_type=vis_type)
        #g_r2 = g_r2 + scale_color_manual(values=[i for i in range(3, 12)])
        g1 = load_ggplot(g_z, figsize=(4, 3))
        g2 = load_ggplot(g_r2, figsize=(4, 3))
        g3 = load_ggplot(g_legend)
        g_combined = (g1/g2)|g3
    # save plot
    g_combined.savefig(op.join(atlas_dir, 'figures', f'{save_filename}.png'))


def compare_correspondence_between_folds(
    trait_type,
    n_arrays_dict,
    node_summary_path=NODE_SUMMARY_PATH,
    get_fix_cov=True,
    p_value=0.05,
    cor_range=[-0.3, 0.3],
    compare="cor",
    save_filename=None,
    full_fill_na=True,
    return_plot=False,
    title=None,
    **kwargs,
):
    """
    Compare correspondence of RSFC-trait correlation between folds
    """
    folder = get_scale_name_from_trait(trait_type)
    scale_name_list = get_subscale_list(folder)
    output_df = pd.DataFrame()

    for i, scale_name in enumerate(scale_name_list):
        file_folder = "/".join(
            filter(None, (SCHAEFER_DIR, folder, scale_name, "correlation"))
        )

        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            print(gsr_type, scale_name)
            gsr_type_key_str = gsr_type.replace("_", "")
            gsr_suffix = "without GSR" if gsr_type == "_nogs_" else "with GSR"

            for fold_n in [0, 1]:
                fold_str = "Fold_" + str(fold_n)

                filename_sem_list_fold_n = [
                    i
                    for i in os.listdir(file_folder)
                    if (str(kwargs.get("n_edge").get("cor").get(gsr_type_key_str)) in i)
                    and (gsr_type in i)
                    and (str(kwargs.get("sample_n").get(fold_str)) in i)
                    and (kwargs.get("est_method") in i)
                    and ("FixedLoad" in i)
                    and ("DayCor" in i)
                    and ("both" in i)
                    and (fold_str in i)
                ]

                filename_list_free_cov = [
                    i for i in filename_sem_list_fold_n if not "Cov0" in i
                ]
                n_arrays_sem = n_arrays_dict.get("sem")
                cor_array_all_free_cov = combine_array_files_dat(
                    filename_list_free_cov, n_arrays_sem, "cor"
                )
                # Read data of fit indices
                if get_fix_cov:
                    filename_list_fix_cov = [
                        i for i in filename_sem_list_fold_n if "Cov0" in i
                    ]
                    fit_fix_cov = combine_array_files_dat(
                        [
                            i.replace("pearson", "fit_indices")
                            for i in filename_list_fix_cov
                        ],
                        n_arrays_sem,
                        "fit",
                    )
                    fit_free_cov = combine_array_files_dat(
                        [
                            i.replace("pearson", "fit_indices")
                            for i in filename_list_free_cov
                        ],
                        n_arrays_sem,
                        "fit",
                    )
                    cor_fix_cov = combine_array_files_dat(
                        filename_list_fix_cov, n_arrays_sem, "cor"
                    )
                    # Calculate chi-square and associated p values
                    delta_df = fit_fix_cov[:, 0] - fit_free_cov[:, 0]
                    # Check all of df is equal to one
                    # if not all([i == 1 for i in delta_df if not (np.isnan(i) or i == 0)]):
                    #     raise ValueError('Values of delta df is different from 1.')
                    # Calculate chi-square difference values
                    delta_chi2 = fit_fix_cov[:, 2] - fit_free_cov[:, 2]
                    p_chi2 = 1 - chi2.cdf(delta_chi2, delta_df)
                    adjusted_p_chi2 = p_chi2 * len(p_chi2)
                    adjusted_p_chi2[adjusted_p_chi2 > 1] = 1
                # Create df for drawing heatmaps
                edges_df = get_edge_summary(node_summary_path, network_hem_order=True)
                edges_df = generate_set_of_networks(edges_df)
                edges_df.sort_values("edge", inplace=True)
                invalid_edge = np.loadtxt(
                    op.join(
                        SCHAEFER_DIR,
                        "reliability",
                        "invalid_edges",
                        kwargs.get("invalid_edge_file").get(gsr_type_key_str),
                    )
                ).astype(int)
                valid_edges = np.delete(np.arange(93096), invalid_edge)

                edges_df["gsr_type"] = gsr_type
                edges_df["scale_name"] = scale_name
                cor_sem = np.squeeze(cor_array_all_free_cov)
                cor_sem[invalid_edge] = np.nan
                edges_df["cor_sem"] = cor_sem
                edges_df["fold"] = fold_n

                # get file on aggregated scores
                filename_list_full = [
                    i
                    for i in os.listdir(
                        file_folder.replace("fit_indices", "correlation")
                    )
                    if (str(kwargs.get("n_edge").get("cor").get(gsr_type_key_str)) in i)
                    and (gsr_type in i)
                    and (str(kwargs.get("sample_n").get(fold_str)) in i)
                    and ("full" in i)
                    and (fold_str in i)
                    and not ("drop" in i)
                ]
                n_arrays_full = n_arrays_dict.get("full")
                cor_array_all_full = combine_array_files_dat(
                    filename_list_full, n_arrays_full, "cor"
                )

                cor_full = np.squeeze(cor_array_all_full)
                if full_fill_na:
                    cor_full[invalid_edge] = np.nan
                n = kwargs.get("sample_n").get(fold_str)
                t_value = cor_full * np.sqrt(n - 2) / np.sqrt(1 - cor_full**2)
                p_full = t.sf(np.abs(t_value), n - 1) * 2
                adjusted_p_full = p_full * len(p_full)
                adjusted_p_full[adjusted_p_full > 1] = 1
                additional_values_df = pd.DataFrame(
                    {
                        "cor_full": cor_full,
                        "n": n,
                        "t": t_value,
                        "p_full": p_full,
                        "adjusted_p_full": adjusted_p_full,
                    }
                )

                if get_fix_cov:
                    p_chi2[invalid_edge], adjusted_p_chi2[invalid_edge] = np.nan, np.nan
                    delta_chi2[invalid_edge], delta_df[invalid_edge] = np.nan, np.nan
                    additional_values_df["p_chi2"] = p_chi2
                    additional_values_df["adjusted_p_chi2"] = adjusted_p_chi2
                    additional_values_df["delta_chi2"] = delta_chi2
                    additional_values_df["delta_df"] = delta_df
                    additional_values_df["delta_p"] = (
                        additional_values_df["p_full"] - additional_values_df["p_chi2"]
                    )
                edges_df = pd.concat(
                    [
                        edges_df.reset_index(),
                        additional_values_df.reset_index(),
                    ],
                    axis=1,
                )
                output_df = pd.concat([output_df, edges_df], axis=0)

    output_df.replace({"_nogs_": "Without GSR", "_gs_": "With GSR"}, inplace=True)
    output_df["gsr_type"] = pd.Categorical(
        output_df["gsr_type"], categories=["Without GSR", "With GSR"]
    )
    if trait_type in ["cognition", "mental"]:
        output_df.replace("All", "Overall", inplace=True)
    output_df = recat_df_from_trait(output_df, trait_type)
    wide_output_df = output_df.pivot(
        index=[
            "edge",
            "gsr_type",
            "scale_name",
            "node1",
            "node2",
            "node1_net",
            "node2_net",
        ],
        columns="fold",
        values=["cor_sem", "p_chi2", "cor_full", "p_full"],
    ).reset_index()
    wide_output_df.columns = [
        "_".join(map(str, i)) if not i[1] == "" else i[0]
        for i in wide_output_df.columns
    ]
    p_value_summary_list = ["both_sig", "fold0_sig", "fold1_sig", "both_insig"]

    def examine_p_values(df, new_column_name, fold0_name, fold1_name, p_value):
        df[new_column_name] = np.select(
            [
                (df[fold0_name] < p_value) & (df[fold1_name] < p_value),
                (df[fold0_name] < p_value) & (df[fold1_name] >= p_value),
                (df[fold0_name] >= p_value) & (df[fold1_name] < p_value),
                (df[fold0_name] >= p_value) & (df[fold1_name] >= p_value),
            ],
            p_value_summary_list,
        )
        return df

    wide_output_df = examine_p_values(
        wide_output_df, "sem_p_corr", "p_chi2_0", "p_chi2_1", p_value
    )
    wide_output_df = examine_p_values(
        wide_output_df, "full_p_corr", "p_full_0", "p_full_1", p_value
    )

    def create_summary_df_p(df, column_name, model_name):
        summary_df = (
            df.groupby(["gsr_type", "scale_name"])[column_name]
            .value_counts()
            .reset_index()
            .rename(columns={column_name: "p_summary", "count": f"count_{model_name}"})
        )
        return summary_df

    summary_full_p = create_summary_df_p(wide_output_df, "full_p_corr", "full")
    summary_sem_p = create_summary_df_p(wide_output_df, "sem_p_corr", "sem")

    summary_full_sem_p = pd.merge(summary_full_p, summary_sem_p, how="left")
    sem_edges_df = (
        summary_full_sem_p.groupby(["gsr_type", "scale_name"])["count_sem"]
        .agg("sum")
        .reset_index()
        .rename(columns={"count_sem": "count_sem_total"})
    )
    summary_full_sem_p = pd.merge(summary_full_sem_p, sem_edges_df)
    summary_full_sem_p["count_full_ratio"] = summary_full_sem_p["count_full"] / 93096
    summary_full_sem_p["count_sem_ratio"] = (
        summary_full_sem_p["count_sem"] / summary_full_sem_p["count_sem_total"]
    )
    summary_full_sem_p["p_summary"] = pd.Categorical(
        summary_full_sem_p["p_summary"], categories=p_value_summary_list
    )

    def lm_reg(df, x, y, cor_range):
        """
        Conduct linear regression and returns improvement factors
        """
        y_min, y_max = cor_range[0], cor_range[1]
        df_filtered = df.query(
            "cor_sem_0 > @y_min & cor_sem_0 < @y_max & cor_sem_1 > @y_min & cor_sem_1 < @y_max"
        )

        removed_n = len(df) - len(df_filtered)

        mask = ~np.isnan(df_filtered[x]) & ~np.isnan(df_filtered[y])
        df_x, df_y = df_filtered[x][mask], df_filtered[y][mask]
        slope, intercept, r_value, p_value, std_error = linregress(df_x, df_y)
        spearman_r = spearmanr(df_x, df_y)
        rho = spearman_r.correlation
        return pd.Series(
            {
                "slope": slope,
                "intercept": intercept,
                "r": r_value,
                "rho": rho,
                "p": p_value,
                "std_error": std_error,
                "removed_n": removed_n,
            }
        )

    def get_cor_labels(df, x, y):
        df_labels = df.groupby(["gsr_type", "scale_name"], observed=True).apply(
            lm_reg, x=x, y=y, cor_range=cor_range
        )
        df_labels["other_txt"] = [
            f"Pearson's r = {r:.3f}\nSpearman's rho = {rho:.3f}"
            for r, rho in zip(df_labels["r"], df_labels["rho"])
        ]
        df_labels.reset_index(inplace=True)
        return df_labels

    df_labels_sem = get_cor_labels(wide_output_df, "cor_sem_0", "cor_sem_1")
    df_labels_full = get_cor_labels(wide_output_df, "cor_full_0", "cor_full_1")

    # summary_edges_n = output_df.groupby(group_var_list, observed=False)[
    #     "edge"
    # ].count()
    # df_labels = pd.merge(
    #     df_labels, summary_edges_n, left_index=True, right_index=True
    # )
    # df_labels["removed_percentage"] = (
    #     df_labels["removed_n"] / df_labels["edge"]
    # )

    def draw_scatterplots(df, x, y, cor_range, cor_or_p, df_labels, title_name):
        if cor_or_p == "cor":
            df.query(
                f"{x} > {cor_range[0]} & {x} < {cor_range[1]} & {y} > {cor_range[0]} & {y} < {cor_range[1]}",
                inplace=True,
            )
        g = (
            ggplot(df, aes(x, y))
            + geom_point(alpha=0.1, size=0.05)
            + facet_grid("gsr_type ~ scale_name")
            + geom_smooth(method="lm", se=False)
            + geom_abline(slope=1, intercept=0)
            + theme_bw()
            + coord_cartesian(xlim=cor_range, ylim=(cor_range[0], cor_range[1] + 0.1))
            + geom_text(
                aes(label="other_txt", x=-np.Inf, y=cor_range[1] + 0.1),
                data=df_labels,
                va="top",
                ha="left",
                size=8,
            )
            + theme(figure_size=(len(scale_name_list) * 5 / 2.54, 8 / 2.54))
            + labs(
                x="Correlation estimated in split-half sample",
                y="Correlation estimated in \nanother split-half sample",
            )
            + ggtitle(title_name)
        )
        return g

    print("Generating scatterplots")
    if compare == "cor":
        sem_x, sem_y = "cor_sem_0", "cor_sem_1"
        full_x, full_y = "cor_full_0", "cor_full_1"
        cor_or_p = "cor"
    elif compare == "p_value":
        sem_x, sem_y = "p_chi2_0", "p_chi2_1"
        full_x, full_y = "p_full_0", "p_full_1"
        cor_or_p = "p"
    trait_scale_name = get_scale_name_from_trait(trait_type, publication=True)
    g_sem = draw_scatterplots(
        wide_output_df,
        sem_x,
        sem_y,
        cor_range,
        cor_or_p,
        df_labels_sem,
        f"{trait_scale_name} (SEM)",
    )
    g_full = draw_scatterplots(
        wide_output_df,
        full_x,
        full_y,
        cor_range,
        cor_or_p,
        df_labels_full,
        f"{trait_scale_name} (analyses on aggregate scores)",
    )

    print("Loading plots to combine")
    print("... SEM")
    g_sem_pw = load_ggplot(g_sem)
    print("... Full score analyses")
    g_full_pw = load_ggplot(g_full)

    print("Combining plots")
    g_full_sem = g_sem_pw / g_full_pw

    if return_plot:
        return g_full_sem

    if save_filename:
        print("Saving figure")
        g_full_sem.savefig(
            op.join(SCHAEFER_DIR, folder, "figures", f"{save_filename}.png")
        )
        print("Saving completed")
    # g_sem.show()
    # g_full.show()


def compare_r_invalid_and_valid(
    trait_type, invalid_edge_filename_dict, n_arrays_dict, **kwargs
):
    """
    Compare RSFC-trait correlations between invalid and valid edges
    """
    folder = get_scale_name_from_trait(trait_type)
    scale_name_list = get_subscale_list(folder)

    fc_files = get_latest_nogs_gs_files(True, False, False)

    df = pd.DataFrame()

    for i, gsr_type in enumerate(["nogs", "gs"]):
        fc_filename = fc_files[i]
        for scale_name in scale_name_list:
            file_folder = op.join(SCHAEFER_DIR, folder, scale_name, "correlation")
            filename_list_full = [
                i
                for i in os.listdir(file_folder)
                if (str(kwargs.get("n_edge")) in i)
                and (gsr_type in i)
                and (str(kwargs.get("sample_n")) in i)
                and ("full" in i)
            ]
            #                if kwargs.get("drop_vars_list_dict") is not None:
            filename_list_full = [i for i in filename_list_full if not "drop" in i]
            n_arrays_full = n_arrays_dict.get("full")
            cor_array_all_full = combine_array_files_dat(
                filename_list_full, n_arrays_full, "cor"
            )

            cor_full = np.squeeze(cor_array_all_full)

            invalid_edges_dir = op.join(SCHAEFER_DIR, "reliability", "invalid_edges")
            invalid_edges = np.loadtxt(
                op.join(invalid_edges_dir, invalid_edge_filename_dict.get(gsr_type))
            )
            invalid_array = np.zeros(shape=(kwargs.get("n_edge")))
            invalid_array[invalid_edges.astype(int)] = 1
            inner_df = pd.DataFrame(
                {
                    "scale_name": [scale_name] * kwargs.get("n_edge"),
                    "cor": cor_full,
                    "invalid": invalid_array,
                    "gsr_type": [gsr_type] * kwargs.get("n_edge"),
                    "edge": list(range(kwargs.get("n_edge"))),
                }
            )
            df = pd.concat([df, inner_df], axis=0)
    df["invalid"] = df["invalid"].replace({0: "Valid", 1: "Invalid"})
    df["invalid"] = pd.Categorical(df["invalid"], categories=["Valid", "Invalid"])
    df["gsr_type"] = df["gsr_type"].replace({"nogs": "Without GSR", "gs": "With GSR"})
    df["gsr_type"] = pd.Categorical(
        df["gsr_type"], categories=["Without GSR", "With GSR"]
    )
    df["scale_name"] = pd.Categorical(df["scale_name"], categories=scale_name_list)

    n = kwargs.get("sample_n")
    df["t_value"] = df["cor"] * np.sqrt(n - 2) / np.sqrt(1 - df["cor"] ** 2)
    df["p_value"] = t.sf(np.abs(df["t_value"]), n - 1) * 2

    return df


def wrapper_of_draw_fig_invalid_and_valid(
    invalid_edge_filename_dict,
    n_arrays_dict,
    y_var="cor",
    fig_height=5,
    filename_fig="valid_invalid_rsfc",
    coord_ymax=3,
    **kwargs,
):
    """
    Wrapper function of draw_fig_invalid_and_valid
    """
    g_list = []
    for trait_type in ["cognition", "mental", "personality"]:
        print(f"Processing {trait_type}")
        g = draw_fig_invalid_and_valid(
            trait_type,
            invalid_edge_filename_dict,
            n_arrays_dict,
            y_var=y_var,
            add_title=True,
            coord_ymax=coord_ymax,
            **kwargs,
        )
        g_list.append(g)
    print("Combining ggplots")
    combine_gg_list(
        g_list,
        fig_height=fig_height,
        filename_fig=filename_fig,
        legend_comp=True,
        comp_target="rsfc",
    )


def draw_fig_invalid_and_valid(
    trait_type,
    invalid_edge_filename_dict,
    n_arrays_dict,
    y_var="cor",
    add_title=False,
    coord_ymax=3,
    **kwargs,
):
    """
    Draw figures comparing RSFC-trait associations considering invalid and valid RSFC
    """
    df = compare_r_invalid_and_valid(
        trait_type, invalid_edge_filename_dict, n_arrays_dict, **kwargs
    )
    #    if y_var == 'cor':
    #        lab_title = 'RSFC-trait correlation (r)'
    #        g = (
    #            ggplot(df, aes(x='invalid', y=y_var))
    #            + geom_hline(yintercept=0, linetype='dashed')
    #            + geom_violin(style='left')
    #            + geom_boxplot(outlier_size=0, width=0.6, outlier_stroke=0)
    #            + geom_jitter(alpha=0.001, size=0.001)
    #            + facet_grid('gsr_type ~ scale_name')
    #            + theme_bw()
    #            + theme(axis_title_x=element_blank())
    #            #+ stat_summary(fun_y=np.mean, geom='point', color='red', size=2)
    #            + labs(y=lab_title)
    #            )
    if y_var == "p_value":
        lab_title = "Unadjusted p value"
    elif y_var == "cor":
        lab_title = "RSFC-trait correlation(r)"
    g = (
        ggplot(df, aes(x=y_var))
        + geom_density(aes(fill="invalid", linetype="invalid"), alpha=0.1)
        + facet_grid("gsr_type ~ scale_name")
        + theme_bw()
        + theme(legend_title=element_blank(), axis_text_x=element_text(angle=45))
        + labs(x=lab_title, y="Scaled density (a.u.)")
        + coord_cartesian(ylim=[0, coord_ymax])
        + scale_linetype_manual(values=["solid", "dashed"])
    )

    if y_var == "cor":
        g = g + geom_vline(xintercept=0, linetype="dashed")

    trait_scale_name = get_scale_name_from_trait(
        trait_type,
        publication=True,
    )
    if add_title:
        g = g + ggtitle(trait_scale_name) + theme(legend_position="none")
    g.draw()
    return g


def calc_r_fscore_fc_trait(
    trait_type,
    invalid_edge_filename_dict,
    n_arrays,
    fc_type="score",
    return_df=False,
    validity_threshold=None,
    include_day_cor=True,
    save_df=False,
    parcellation='Schaefer',
    param_filename_dict={},
    fscore_filename_dict={},
    msst=False,
    **kwargs,
):
    """
    Calculate correlation between factor score estimates of both trait and RSFC
    """
    folder = get_scale_name_from_trait(trait_type)
    scale_name_list = get_subscale_list(folder)
    
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    # get factor scores of trait
    fscore_df = pd.read_csv(op.join(atlas_dir, folder, "tables", "fscore.csv")).iloc[
        :, 1:
    ]

    rel_fs_dir = op.join(atlas_dir, "reliability", "factor_scores")

    day_cor_bool = True if include_day_cor else False

    df = pd.DataFrame()

    use_subjects = np.loadtxt(
        op.join(atlas_dir, "spreg_0.25_N_861_rms_percentage_0.1.csv")
    )

    for i, gsr_type in enumerate(["nogs", "gs"]):

        if gsr_type == "nogs":
            gsr_str, gsr_str_param = "", "_nogs_"
        elif gsr_type == "gs":
            gsr_str, gsr_str_param = "_gs", "_gs_"

#        def get_list_of_filenames(data_type):
#            param_filenames = os.listdir(
#                op.join(atlas_dir, "reliability", data_type)
#            )
#            param_filenames = [
#                i
#                for i in param_filenames
#                if not ("addMarker" in i)
#                and (str(kwargs.get("sample_n")) in i)
#                and (str(kwargs.get("edge_n")) in i)
#                and not ("PE" in i)
#                and not ("OrderInDay" in i)
#                and (gsr_str_param in i)
#            ]
#            if include_day_cor:
#                param_filenames = [i for i in param_filenames if "DayCor" in i]
#            else:
#                param_filenames = [i for i in param_filenames if not "DayCor" in i]
#            return param_filenames
#        get_list_of_filenames(
#            data_type, 
#            gsr_type, 
#            parcellation='Schaefer', 
#            addMarker=False,
#            include_day_cor=True,
#            multistate_single_trait=False,
#            get_std=False,
#            **kwargs
#            )
#        param_filenames = get_list_of_filenames("parameters")
#        param_filename = sort_list_by_time(param_filenames)[-1]
#        print(f"Processing {param_filename}")
#        param_position_dict = get_param_position_dict(param_filename)
#        params_array = combine_array_files_dat(
#            param_filenames, n_arrays=n_arrays, data_type="parameter", parcellation=parcellation
#        )
#        # Get factor loadings (structural coefficients)
#        s_kf_array, _, _, _ = get_parameters(
#            params_array, "model_onlyFC", param_position_dict
#        )
        # get parameters
        param_filename = param_filename_dict.get(gsr_type)
        params_array = np.load(op.join(atlas_dir, "reliability", "parameters", 'combined', param_filename))
        # get factor loadings
        if msst and validity_threshold:
            s_kf_array = params_array[:, 0:6, 0]

        invalid_edges_dir = op.join(atlas_dir, "reliability", "invalid_edges")
        invalid_edges = np.loadtxt(
            op.join(invalid_edges_dir, invalid_edge_filename_dict.get(gsr_type))
        ).astype(int)

        fc_filename = (
            f"ordered_fc_spreg_0.25{gsr_str}_demeaned_{parcel_name}.npy"
        )
        print(f"Loading FC data {gsr_type}")
        raw_fcs = np.load(op.join(atlas_dir, fc_filename))
        raw_fcs = raw_fcs[:, np.array(use_subjects, dtype=bool), :]

        if fc_type == "score":
            fscore_fc_filename = fscore_filename_dict.get(gsr_type)
            #fscore_fc_filenames = get_list_of_filenames("factor_scores")
        #    fc_scores = combine_array_files_dat(
        #        fscore_fc_filenames, n_arrays=n_arrays, data_type="factor_scores", parcellation=parcellation
        #    )
            print('Loading factor scores.')
            fc_scores = np.load(op.join(rel_fs_dir, 'combined', fscore_fc_filename))
            # get factor scores of trait factor
            if msst:
                fc_scores = fc_scores[:, :, 2]
            # fscore_fc_filename = fscore_fc_filenames[]
            # print(f'Processing {fscore_fc_filename}')

            def calc_validity_of_fscores():
                validity_array = np.empty(shape=len(raw_fcs))
                validity_array[invalid_edges.astype(int)] = np.nan
                for j in range(len(raw_fcs)):
                    if j not in invalid_edges:
                        s_kf = s_kf_array[j]
                        # Calculate W_kf in Grice (2001)
                        r_kk = np.corrcoef(raw_fcs[j], rowvar=False)
                        w_kf = inv(r_kk) @ s_kf.T
                        c_ss = w_kf.T @ r_kk @ w_kf
                        l_ss = np.sqrt(c_ss)
                        r_fs = s_kf.T @ w_kf / l_ss
                        validity_array[j] = r_fs
                return validity_array

            if validity_threshold:
                print("Calculate validity coefficients of factor scores")
                validity_fscore = calc_validity_of_fscores()
                print(
                    f"Number of edges passing validity coefficients was {len(validity_fscore[validity_fscore > validity_threshold])}"
                )
        if fc_type == "full":
            fc_filename = f"fc_spreg_0.25{gsr_str}_demeaned_{parcel_name}_all_combine_full_data_exist_ordered.npy"
            print("Loading FC data")
            fc_scores = np.load(op.join(atlas_dir, fc_filename))
            fc_scores = fc_scores[:, np.array(use_subjects, dtype=bool)]
        if fc_type == "average":
            fc_scores = np.mean(raw_fcs, axis=2)
        fc_scores[invalid_edges] = np.nan

        for scale_name in scale_name_list:
            print(f"Calculating RSFC-trait correlations with {scale_name} {gsr_type}")
            fscore_trait = fscore_df[scale_name]
            # Create array for storing correlation between FC and trait
            n_fc = fc_scores.shape[0]
            cor_array = np.empty(shape=(n_fc))
            cor_array[:] = np.nan
            # Take long time and should be modified using parallel processing

            def calc_r(i):
                return np.corrcoef(fscores_fc[i, :], fscore_trait)[0, 1]

            start = time()
            # cor_array = Parallel(n_jobs=-1)([delayed(calc_r)(n) for n in range(n_fc)])
            for k in range(n_fc):
                if validity_threshold:
                    if (k not in invalid_edges) and (
                        validity_fscore[k] > validity_threshold
                    ):
                        cor_array[k] = np.corrcoef(fc_scores[k, :], fscore_trait)[0, 1]
                else:
                    cor_array[k] = np.corrcoef(fc_scores[k, :], fscore_trait)[0, 1]

            end = time()
            print(f"It took {end - start:.2f} sec.")
            subset_df = pd.DataFrame(
                {
                    "gsr_type": [gsr_type] * n_fc,
                    "r": cor_array,
                    "scale_name": [scale_name] * n_fc,
                    "edge": list(range(n_fc)),
                    "model": [fc_type] * n_fc,
                }
            )
            df = pd.concat([df, subset_df], axis=0)

    df.replace({"nogs": "Without GSR", "gs": "With GSR"}, inplace=True)
    df["gsr_type"] = pd.Categorical(
        df["gsr_type"], categories=["Without GSR", "With GSR"]
    )
    if trait_type in ["cognition", "mental"]:
        df.replace("All", "Overall", inplace=True)
    df = recat_df_from_trait(df, trait_type)

    if save_df:
        pass

    if return_df:
        return df

    print("Creating ggplot object")
    g = (
        ggplot(df, aes(x="r"))
        + geom_density(alpha=0.1)
        + facet_grid("gsr_type ~ scale_name")
        + theme_bw()
    )
    g.draw()
    print("Finished processing")
    return g


def wrapper_of_compare_fits_on_str_cor(
    drop_bool,
    hmap_vmin=None,
    hmap_vmax=None,
    draw_heat=False,
    value_name_list=["cor_full", "cor_sem"],
    get_fix_cov=True,
    adjusted_p=False,
    vis_chi2_cor=True,
    scatter_or_hist="hist",
    n_arrays_dict=None,
    full_fill_na=True,
    fig_height=4,
    x_range_chi2_diff=[0, 8],
    y_range_p=[0, 2],
    filename_fig=None,
    include_day_cor=True,
    title_font_size=8,
    **kwargs,
):
    """
    Wrapper function of compare_fits_on_str_cor to create figures for publication of histogram on unadjusted and adjusted p values
    """
    g_list = []
    trait_type_list = (
        ["cognition", "mental", "personality"]
        if not drop_bool
        else ["cognition", "personality"]
    )
    for trait_type in trait_type_list:
        print(f"Processing {trait_type}")
        title = get_scale_name_from_trait(trait_type, publication=True, drop=drop_bool)
        g = compare_fits_on_str_cor(
            trait_type,
            drop_bool=drop_bool,
            hmap_vmin=hmap_vmin,
            hmap_vmax=hmap_vmax,
            draw_heat=draw_heat,
            value_name_list=value_name_list,
            get_fix_cov=get_fix_cov,
            adjusted_p=adjusted_p,
            vis_chi2_cor=vis_chi2_cor,
            n_arrays_dict=n_arrays_dict,
            full_fill_na=full_fill_na,
            title=title,
            x_range_chi2_diff=x_range_chi2_diff,
            y_range_p=y_range_p,
            return_plot=True,
            include_day_cor=include_day_cor,
            **kwargs,
        )
        if draw_heat:
            g.suptitle(title, size=title_font_size, horizontalalignment="left", x=0.05)
        g_list.append(g)
    if not draw_heat:
        combine_gg_list(g_list, fig_height, filename_fig, drop_bool, legend_comp=True)
    else:
        return g_list


def compare_fits_on_str_cor(
    trait_type,
    node_summary_path=NODE_SUMMARY_PATH,
    drop_bool=False,
    hmap_vmin=None,
    hmap_vmax=None,
    thres_chi_p=None,
    z_func="median",
    p_chi2_thres=0.05,
    p_full_thres=0.05,
    draw_heat=False,
    save_filename=None,
    value_name_list=["cor_full", "cor_sem"],
    get_fix_cov=True,
    adjusted_p=True,
    scatter_or_hist="hist",
    show_hist="all",
    vis_chi2_cor=False,
    cmap="bwr",
    n_arrays_dict=None,
    full_fill_na=True,
    title=None,
    x_range_chi2_diff=[0, 8],
    y_range_p=[0, 2],
    return_plot=False,
    return_df=False,
    include_day_cor=True,
    **kwargs,
):
    """
    Compare chi-square statistics between nested models
    whose differneces are structural correlation
    """
    folder = get_scale_name_from_trait(trait_type)
    scale_name_list = get_subscale_list(folder)
    if drop_bool:
        if not title:
            scale_name_list = kwargs.get("drop_vars_list_dict").keys()
        else:
            scale_name_list = kwargs.get("drop_vars_list_dict").get(trait_type).keys()
    output_df = pd.DataFrame()
    set_font()
    if draw_heat:
        fig_sem_full, axes_sem_full = plt.subplots(
            len(scale_name_list),
            len(value_name_list),
            figsize=(6 * len(value_name_list), len(scale_name_list) * 2.5),
            sharex=True,
            sharey=True,
        )
        cbar_ax = (
            fig_sem_full.add_axes([0.93, 0.3, 0.01, 0.4]) if title is None else None
        )

    #    if get_fix_cov:
    #        fig_p, axes_p = plt.subplots(
    #                len(scale_name_list),
    #                2,
    #                figsize=(10, len(scale_name_list) * 2.5),
    #                sharex=True,
    #                sharey=True,
    #            )

    for i, scale_name in enumerate(scale_name_list):
        file_folder = "/".join(
            filter(None, (SCHAEFER_DIR, folder, scale_name, "correlation"))
        )

        for drop in [drop_bool]:
            # For heatmap of FC
            mat_df = get_empty_df_for_hmap(parcellation)
            nodes_df = get_nodes_df(node_summary_path)

            for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
                print(gsr_type, scale_name)
                gsr_type_key_str = gsr_type.replace("_", "")
                gsr_suffix = "without GSR" if gsr_type == "_nogs_" else "with GSR"

                filename_list = [
                    i
                    for i in os.listdir(file_folder)
                    if (str(kwargs.get("n_edge").get(gsr_type_key_str)) in i)
                    and (gsr_type in i)
                    and (str(kwargs.get("sample_n")) in i)
                    and (kwargs.get("est_method") in i)
                    and ("FixedLoad" in i)
                    and ("both" in i)
                ]
                if include_day_cor:
                    filename_list = [i for i in filename_list if "DayCor" in i]
                else:
                    filename_list = [i for i in filename_list if not "DayCor" in i]

                filename_list_free_cov = [i for i in filename_list if not "Cov0" in i]
                if kwargs.get("drop_vars_list_dict") is not None:
                    # if kwargs.get("drop_vars_list_dict").get(scale_name) is not None:
                    filename_list_free_cov = (
                        [i for i in filename_list_free_cov if not "drop" in i]
                        if not drop
                        else [i for i in filename_list_free_cov if "drop" in i]
                    )
                    drop_str_list = (
                        kwargs.get("drop_vars_list_dict")
                        .get(trait_type)
                        .get(scale_name)
                    )
                    if len(drop_str_list) > 1:
                        drop_str = "_".join(drop_str_list)
                    elif len(drop_str_list) == 1:
                        drop_str = "drop_" + drop_str_list[0] + "_2024"
                    filename_list_free_cov = [
                        i for i in filename_list_free_cov if drop_str in i
                    ]
                n_arrays_sem = n_arrays_dict.get(drop).get("sem")
                cor_array_all_free_cov = combine_array_files_dat(
                    filename_list_free_cov, n_arrays_sem, "cor"
                )
                # Read data of fit indices
                if get_fix_cov:
                    filename_list_fix_cov = [i for i in filename_list if "Cov0" in i]
                    if kwargs.get("drop_vars_list_dict") is not None:
                        # if kwargs.get("drop_vars_list_dict").get(scale_name) is not None:
                        drop_str_list = (
                            kwargs.get("drop_vars_list_dict")
                            .get(trait_type)
                            .get(scale_name)
                        )
                        filename_list_fix_cov = (
                            [i for i in filename_list_fix_cov if not "drop" in i]
                            if not drop
                            else [i for i in filename_list_fix_cov if "drop" in i]
                        )
                        if len(drop_str_list) > 1:
                            drop_str = "_".join(drop_str_list)
                        elif len(drop_str_list) == 1:
                            drop_str = "drop_" + drop_str_list[0] + "_2024"
                        filename_list_fix_cov = [
                            i for i in filename_list_fix_cov if drop_str in i
                        ]
                    fit_fix_cov = combine_array_files_dat(
                        [
                            i.replace("pearson", "fit_indices")
                            for i in filename_list_fix_cov
                        ],
                        n_arrays_sem,
                        "fit",
                    )
                    fit_free_cov = combine_array_files_dat(
                        [
                            i.replace("pearson", "fit_indices")
                            for i in filename_list_free_cov
                        ],
                        n_arrays_sem,
                        "fit",
                    )
                    cor_fix_cov = combine_array_files_dat(
                        filename_list_fix_cov, n_arrays_sem, "cor"
                    )
                    # Calculate chi-square and associated p values
                    delta_df = fit_fix_cov[:, 0] - fit_free_cov[:, 0]
                    # Check all of df is equal to one
                    # if not all([i == 1 for i in delta_df if not (np.isnan(i) or i == 0)]):
                    #     raise ValueError('Values of delta df is different from 1.')
                    # Calculate chi-square difference values
                    delta_chi2 = fit_fix_cov[:, 2] - fit_free_cov[:, 2]
                    p_chi2 = 1 - chi2.cdf(delta_chi2, delta_df)
                    adjusted_p_chi2 = p_chi2 * len(p_chi2)
                    adjusted_p_chi2[adjusted_p_chi2 > 1] = 1
                # Create df for drawing heatmaps
                edges_df = get_edge_summary(node_summary_path, network_hem_order=True)
                edges_df = generate_set_of_networks(edges_df)
                edges_df.sort_values("edge", inplace=True)
                invalid_edge = np.loadtxt(
                    op.join(
                        SCHAEFER_DIR,
                        "reliability",
                        "invalid_edges",
                        kwargs.get("invalid_edge_file").get(gsr_type_key_str),
                    )
                ).astype(int)
                valid_edges = np.delete(np.arange(93096), invalid_edge)

                edges_df["gsr_type"] = gsr_type
                edges_df["scale_name"] = scale_name
                cor_sem = np.squeeze(cor_array_all_free_cov)
                cor_sem[invalid_edge] = np.nan
                edges_df["cor_sem"] = cor_sem

                # Get file on aggregate scores
                filename_list_full = [
                    i
                    for i in os.listdir(
                        file_folder.replace("fit_indices", "correlation")
                    )
                    if (str(kwargs.get("n_edge").get(gsr_type_key_str)) in i)
                    and (gsr_type in i)
                    and (str(kwargs.get("sample_n")) in i)
                    and ("full" in i)
                ]
                if kwargs.get("drop_vars_list_dict") is not None:
                    # if kwargs.get("drop_vars_list_dict").get(scale_name) is not None:
                    filename_list_full = (
                        [i for i in filename_list_full if not "drop" in i]
                        if not drop
                        else [i for i in filename_list_full if "drop" in i]
                    )
                    drop_str_list = (
                        kwargs.get("drop_vars_list_dict")
                        .get(trait_type)
                        .get(scale_name)
                    )
                    if len(drop_str_list) > 1:
                        drop_str = "_".join(drop_str_list)
                    elif len(drop_str_list) == 1:
                        drop_str = "drop_" + drop_str_list[0] + "_2024"
                    filename_list_full = [
                        i for i in filename_list_full if drop_str in i
                    ]
                n_arrays_full = n_arrays_dict.get(drop).get("full")
                cor_array_all_full = combine_array_files_dat(
                    filename_list_full, n_arrays_full, "cor"
                )

                cor_full = np.squeeze(cor_array_all_full)
                if full_fill_na:
                    cor_full[invalid_edge] = np.nan
                n = kwargs.get("sample_n")
                t_value = cor_full * np.sqrt(n - 2) / np.sqrt(1 - cor_full**2)
                p_full = t.sf(np.abs(t_value), n - 1) * 2
                adjusted_p_full = p_full * len(p_full)
                adjusted_p_full[adjusted_p_full > 1] = 1
                additional_values_df = pd.DataFrame(
                    {
                        "cor_full": cor_full,
                        "n": n,
                        "t": t_value,
                        "p_full": p_full,
                        "adjusted_p_full": adjusted_p_full,
                    }
                )

                if get_fix_cov:
                    p_chi2[invalid_edge], adjusted_p_chi2[invalid_edge] = np.nan, np.nan
                    delta_chi2[invalid_edge], delta_df[invalid_edge] = np.nan, np.nan
                    additional_values_df["p_chi2"] = p_chi2
                    additional_values_df["adjusted_p_chi2"] = adjusted_p_chi2
                    additional_values_df["delta_chi2"] = delta_chi2
                    additional_values_df["delta_df"] = delta_df
                    additional_values_df["delta_p"] = (
                        additional_values_df["p_full"] - additional_values_df["p_chi2"]
                    )
                edges_df = pd.concat(
                    [
                        edges_df.reset_index(),
                        additional_values_df.reset_index(),
                    ],
                    axis=1,
                )
                output_df = pd.concat([output_df, edges_df], axis=0)
                edges_df["z_full"] = np.arctanh(edges_df["cor_full"])
                edges_df["z_sem"] = np.arctanh(edges_df["cor_sem"])
                edges_df["delta_z"] = np.abs(edges_df["z_sem"] - edges_df["z_full"])

                # Get data of factor scores
                filename_list_fs = [
                    i
                    for i in os.listdir(
                        file_folder.replace("fit_indices", "factor_scores")
                    )
                    if (str(kwargs.get("n_edge").get(gsr_type_key_str)) in i)
                    and (gsr_type in i)
                    and (str(kwargs.get("sample_n")) in i)
                    and ("both" in i)
                ]
                if kwargs.get("drop_vars_list_dict") is not None:
                    # if kwargs.get("drop_vars_list_dict").get(scale_name) is not None:
                    filename_list_full = (
                        [i for i in filename_list_full if not "drop" in i]
                        if not drop
                        else [i for i in filename_list_full if "drop" in i]
                    )
                n_arrays_full = n_arrays_dict.get(drop).get("full")
                cor_array_all_full = combine_array_files_dat(
                    filename_list_full, n_arrays_full, "cor"
                )

                cor_full = np.squeeze(cor_array_all_full)
                if full_fill_na:
                    cor_full[invalid_edge] = np.nan
                n = kwargs.get("sample_n")
                t_value = cor_full * np.sqrt(n - 2) / np.sqrt(1 - cor_full**2)
                p_full = t.sf(np.abs(t_value), n - 1) * 2
                adjusted_p_full = p_full * len(p_full)
                adjusted_p_full[adjusted_p_full > 1] = 1
                additional_values_df = pd.DataFrame(
                    {
                        "cor_full": cor_full,
                        "n": n,
                        "t": t_value,
                        "p_full": p_full,
                        "adjusted_p_full": adjusted_p_full,
                    }
                )

                if draw_heat:
                    for value_name_i, value_name in enumerate(value_name_list):
                        iteration = i if title is None else None
                        wide_df = get_wide_df_hmap(edges_df, value_col_name=value_name)
                        mat_df = fill_mat_df(mat_df, wide_df, gsr_type.replace("_", ""))
                        if axes_sem_full.ndim == 2:
                            target_axes_fc_hmap = axes_sem_full[i, value_name_i]
                        elif axes_sem_full.ndim == 1:
                            target_axes_fc_hmap = axes_sem_full[value_name_i]

                        draw_hmaps_fcs(
                            mat_df,
                            nodes_df,
                            cmap=cmap,
                            save_dir=folder,
                        #    save_filename=None,
                            ax=target_axes_fc_hmap,
                            cbar_ax=cbar_ax,
                            vmin=hmap_vmin,
                            vmax=hmap_vmax,
                            iteration=iteration,
                        )
                        if value_name == "cor_full":
                            title_suffix = "(analyses on aggregate scores)"
                        elif value_name == "cor_sem":
                            title_suffix = "(SEM)"
                        target_axes_fc_hmap.set_title(f"{scale_name} {title_suffix}")
    if draw_heat:
        if title is None:
            fig_sem_full.tight_layout(rect=[0, 0, 0.9, 1])
        if save_filename is not None:
            fig_sem_full.savefig(
                op.join(SCHAEFER_DIR, folder, "figures", f"{save_filename}.png")
            )
        if return_plot:
            return fig_sem_full
        plt.close(fig_sem_full)
    if thres_chi_p is not None:
        fig_2.tight_layout()

    output_df.replace({"_nogs_": "Without GSR", "_gs_": "With GSR"}, inplace=True)
    output_df["gsr_type"] = pd.Categorical(
        output_df["gsr_type"], categories=["Without GSR", "With GSR"]
    )
    if trait_type in ["cognition", "mental"]:
        output_df.replace("All", "Overall", inplace=True)
    output_df = recat_df_from_trait(output_df, trait_type)

    if return_df:
        return output_df

    if save_filename:
        save_filepath = op.join(SCHAEFER_DIR, folder, "figures", save_filename)

    if get_fix_cov:
        if scatter_or_hist is None:
            pass
        suffix = "adjusted_" if adjusted_p else ""

        if scatter_or_hist == "scatter":
            g = (
                ggplot(output_df, aes(f"{suffix}p_full", f"{suffix}p_chi2"))
                + geom_point(size=0.001, alpha=0.1)
                + facet_grid("gsr_type ~ scale_name")
                + theme_bw()
            )
            try:
                g.show()
            except:
                pass

        elif scatter_or_hist == "hist":
            long_df = output_df.melt(
                id_vars=["scale_name", "gsr_type", "edge"],
                value_vars=[f"{suffix}p_full", f"{suffix}p_chi2"],
                value_name="p",
                var_name="model_type",
            )
            long_df.replace(
                {
                    f"{suffix}p_full": "Analyses on aggregate scores",
                    f"{suffix}p_chi2": "SEM",
                },
                inplace=True,
            )
            xlabel = "Adjusted p value" if adjusted_p else "Unadjusted p value"

            if show_hist == "all":
                if trait_type == "cognition" and drop_bool:
                    figure_size = (8 / 2.54, 8 / 2.54)
                else:
                    figure_size = (len(scale_name_list) * 4 / 2.54, 8 / 2.54)
                legend_position = "none" if title else "bottom"
                g = (
                    ggplot(long_df, aes("p", fill="model_type", linetype="model_type"))
                    + geom_density(alpha=0.2)
                    + facet_grid("gsr_type ~ scale_name")
                    + theme_bw()
                    + theme(
                        legend_title=element_blank(),
                        legend_position=legend_position,
                        axis_text_x=element_text(angle=45),
                        figure_size=figure_size,
                    )
                    + labs(x=xlabel, y="Scaled density (a.u.)")
                    + scale_linetype_manual(values=["dashed", "solid"])
                    + coord_cartesian(ylim=y_range_p)
                )
                if title:
                    g = g + ggtitle(title)
                try:
                    g.show()
                except:
                    pass
                if save_filename:
                    g.save(filename=save_filepath)
            elif show_hist == "full":
                p_type, xlim = "p_full"
            elif show_hist == "sem":
                p_type = "p_chi2"
            if show_hist in ["full", "sem"]:
                g = (
                    ggplot(long_df.query(f"model_type == {p_type}"), aes("p"))
                    + geom_density(alpha=0.2)
                    + facet_grid("gsr_type ~ scale_name")
                    + theme_bw()
                )
                try:
                    g.show()
                except:
                    pass
        if vis_chi2_cor:
            output_df["abs_cor_sem"] = np.abs(output_df["cor_sem"])
            output_df["sqroot_delta_chi2"] = np.sqrt(output_df["delta_chi2"])
            g = (
                ggplot(output_df, aes(f"sqroot_delta_chi2", "abs_cor_sem"))
                + geom_point(size=0.001, alpha=0.1)
                + facet_grid("gsr_type ~ scale_name")
                + theme_bw()
                + labs(
                    y="Absolute correlation",
                    x="Square root of difference of chi-square values",
                )
                + theme(figure_size=(len(scale_name_list) * 4 / 2.54, 8 / 2.54))
                + coord_cartesian(xlim=x_range_chi2_diff, ylim=[0, 0.25])
            )
            if title:
                g = g + ggtitle(title)
            try:
                g.show()
            except:
                if save_filename:
                    g.save(filename=save_filepath)

    if return_plot:
        return g
    return output_df


def check_covriate_params(filename):
    """
    Visualize distributions of standardised regression coefficient of covariates
    This function is only applicable to Model RSFC 2
    """
    params_dict = generate_params_dict(filename)
    regs = params_dict.get("model_onlyFC")[:, 6:8, 0]
    return regs


def get_nodes_df(parcellation='Schaefer'):
    """
    Get df of nodes
    """
    node_summary_path = NODE_SUMMARY_PATH_DICT.get(parcellation)
    nodes_df = pd.read_csv(node_summary_path)
    nodes_df["net"] = nodes_df["net"].replace("Limbic_tian", "LimbicTian")
    network_order_dict = NETWORK_ORDER_NESTED_DICT.get(parcellation)
    #nodes_df.replace(np.nan, 'NA', inplace=True)
    if parcellation == 'Gordon':
        nodes_df['net'].fillna('No', inplace=True)
    nodes_df["net"] = pd.Categorical(
        nodes_df["net"], categories=network_order_dict.keys(), ordered=True
    )
    network_agg_pub_dict = NETWORK_AGG_PUB_NESTED_DICT.get(parcellation)
    
    nodes_df["net"] = nodes_df["net"].cat.rename_categories(network_agg_pub_dict)
    nodes_df.sort_values(["net", "hem"], inplace=True)
    nodes_df.reset_index(inplace=True)
    nodes_df.rename(columns={'index': 'old_index'}, inplace=True)
    nodes_df["index"] = nodes_df.index
    
    return nodes_df


def get_empty_df_for_hmap(parcellation):
    """
    Get empty dataframe for drawing heatmap
    """
    nodes_df = get_nodes_df(parcellation)
    # Create empty dataframe
    mat_df = pd.DataFrame(index=nodes_df["node"], columns=nodes_df["node"])
    return mat_df


def get_wide_df_hmap(input_edges_df, value_col_name):
    """
    Get wide dataframe for drawing heatmap
    """
    wide_edges_df = input_edges_df.pivot(
        index="node2", values=value_col_name, columns="node1"
    )
    add_categories = list(set(wide_edges_df.index) - set(wide_edges_df.columns))
    remove_categories = list(set(wide_edges_df.columns) - set(wide_edges_df.index))

    index_list = input_edges_df["node1"].unique().tolist() + add_categories
    index_list = [i for i in index_list if i not in remove_categories]
    wide_edges_df.index = pd.CategoricalIndex(
        index_list, categories=index_list, ordered=True
    )
    return wide_edges_df


def fill_mat_df(mat_df, wide_df, gsr_type):
    """
    Fill mat df with values
    Lower diagonal ->
    Upper diagonal ->
    """
    mask = np.ones(mat_df.shape, dtype="bool")

    if gsr_type in ["nogs", 'Without GSR']:
        tri_func = np.triu_indices
    elif gsr_type in ["gs", 'With GSR']:
        tri_func = np.tril_indices
        wide_df = wide_df.T

    mask[tri_func(len(mat_df))] = False
    mat_df[mask] = wide_df
    mat_df = mat_df.astype(float)

    return mat_df


def draw_hmaps_fcs(
    mat_df,
    nodes_df,
    cmap,
    hvline_color="k",
    save_dir="reliability",
#    save_filename=None,
    ax=None,
    cbar_ax=None,
    cbar=False,
    vmin=None,
    vmax=None,
    iteration=None,
    num_iter=1,
    parcellation='Schaefer',
    rotate_x_deg=45,
    cbar_width=0.5,
    fig_direction='horizontal',
    add_custom_cbar=True
):
    """
    Draw heatmap by FC level
    """
    if ax is None:
        fig, ax = plt.subplots()
    if num_iter > 1:
        sns.heatmap(
            mat_df,
            ax=ax,
            cbar_ax=cbar_ax if iteration else None,
            mask=mat_df.isnull(),
            cmap=cmap,
            cbar=iteration == 1,
            vmin=vmin,
            vmax=vmax,
        )
    elif num_iter == 1:
        sns.heatmap(
            mat_df,
            ax=ax,
            #cbar_ax=cbar_ax,
            mask=mat_df.isnull(),
            cmap=cmap,
            cbar=cbar,
            vmin=vmin,
            vmax=vmax,
            #cbar_kws={"shrink": 0.5, 'location': 'bottom'}
        )

    sep_lines_positions = nodes_df.groupby("net", observed=False).index.max() + 1
    sep_lines_positions = np.insert(sep_lines_positions.values, 0, 0)
    for i in sep_lines_positions:
        ax.axvline(i, color=hvline_color, linewidth=0.5)
        ax.axhline(i, color=hvline_color, linewidth=0.5)

    ticks_positions = nodes_df.groupby("net", observed=False).index.median()
    ax.set_xticks(ticks_positions)
    ax.set_yticks(ticks_positions)
    network_order_for_pub_list = NETWORK_ORDER_FOR_PUB_NESTED_LIST.get(parcellation)
    if parcellation == 'Gordon':
        rotate_x_deg = 90
    elif parcellation == 'Schaefer':
        rotate_x_deg = 45
    ax.set_xticklabels(network_order_for_pub_list, fontsize=8, rotation=rotate_x_deg)
    ax.set_yticklabels(network_order_for_pub_list, fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")

    xpoints = ypoints = ax.get_xlim()

    ax.plot(
        xpoints, ypoints, linestyle="-", color="k", lw=0.5, scalex=False, scaley=False
    )
    
    # draw colorbar with specified positions
    if add_custom_cbar:
        left_cbar_position = (1 - cbar_width) / 2
        if fig_direction == 'horizontal':
            add_axes_list = [left_cbar_position, 0.01, cbar_width, 0.02]
        elif fig_direction == 'vertical':
            add_axes_list = [0.95, 0.3, 0.015, 0.4]
        
        #cbar_ax = fig.add_axes(add_axes_list)
       # if ax:
       #     cbar_ax = inset_axes(ax, width='50%', height='1%', loc='lower center')

        mappable = mpl.cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(vmin, vmax)
        if fig_direction == 'horizontal':
            orientation = 'horizontal'
        elif fig_direction == 'vertical':
            orientation = None
        plt.colorbar(
                mappable,
                ax=ax,
                #cax=cbar_ax,
                orientation=orientation,
                shrink=0.7
                )

    if ax is None:
        fig.tight_layout()
#    if save_filename is not None:
#        fig.tight_layout()
#        atlas_dir = ATLAS_DIR_DICT.get(parcellation)
#        fig.savefig(op.join(atlas_dir, save_dir, "figures", f"{save_filename}.png"))


def get_ax_and_model_name(axes, filename_list, num_iter):
    """
    Get ax and model_name for visualization for publication
    """
    if all(not 'PE' in i and not 'Order' in i and not 'Day' in i for i in filename_list):
        if num_iter > 2:
            target_ax = axes[0, 0]
        elif num_iter == 2:
            target_ax = axes[0]
        model_name = "Model RSFC 1"
        method_effect = "No state and method effects"
    elif all(not 'PE' in i and not 'Order' in i and 'Day' in i for i in filename_list):
        if num_iter > 2:
            target_ax = axes[1, 0]
        elif num_iter == 2:
            target_ax = axes[1]
        model_name = "Model RSFC 2-b"
        method_effect = "State effects of measurement day"
    elif all('PE' in i and not 'Order' in i and 'Day' in i for i in filename_list):
        if num_iter > 2:
            target_ax = axes[0, 1]
        elif num_iter == 2:
            target_ax = axes[0]
        model_name = "Model RSFC 3"
        method_effect = "Adding method effects of phase encoding directions"
    elif all('PE' in i and not 'Order' in i and 'Day' in i for i in filename_list):
        if num_iter > 2:
            target_ax = axes[1, 1]
        elif num_iter == 2:
            target_ax = axes[1]
        model_name = "Model RSFC 4-b"
        method_effect = "Method effects of measurement order"
    return target_ax, model_name, method_effect


def wrapper_for_draw_hmaps_invalid_edges(
    invalid_edge_file_nested_list, 
    fig_size=(10, 10), 
    save_filename=None,
    parcellation='Schaefer'
):
    """
    Wrapper function of draw_hmaps_invalid_edges()
    """
    num_iter = len(invalid_edge_file_nested_list)
    if num_iter == 4:
        n_row, n_col = 2, 2
    elif num_iter == 2:
        n_row, n_col = 1, 2
    size_x, size_y = 5 * n_row, 5 * n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(size_y, size_x), sharex=True, sharey=True)

    for i, invalid_edge_file_list in enumerate(invalid_edge_file_nested_list):
        target_ax, model_name, method_effect = get_ax_and_model_name(axes, invalid_edge_file_list, num_iter)
        draw_hmaps_invalid_edges(invalid_edge_file_list, ax=target_ax, num_iter=num_iter, parcellation=parcellation)
        target_ax.set_title(f"{model_name}\n({method_effect})")
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    if save_filename:
        fig.tight_layout()
        fig.savefig(
            op.join(atlas_dir, "reliability", "figures", f"{save_filename}.png")
        )


def get_fscore_validity_nodes_df(
        fc_filename_dict=None,
        fscores_dict=None,
        param_order_filename=None,
        invalid_edge_file_dict={},
        parcellation='Schaefer',
        family_cv=True,
        fold=None,
        edges_validity_df=None,
        **kwargs
        ):
    """
    Get dataframe with valid coefficients
    if edge_validity_df is None, the first three arguments should be specified
    **kwargs may include invalid_edge_file_list
    """
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    edges_df.sort_values("edge", inplace=True)
    nodes_df = get_nodes_df(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    if edges_validity_df is None:
        edges_validity_df = get_fscore_validity_edges_df(
            parcellation=parcellation,
            fc_filename_dict=fc_filename_dict,
            fscores_dict=fscores_dict,
            invalid_edge_file_dict=invalid_edge_file_dict,
            gsr_types=['nogs', 'gs'],
            msst=True,
            param_order_filename=param_order_filename
            )    
    #    for invalid_edge_file in invalid_edge_file_list:
#        if '_nogs_' in invalid_edge_file:
#            gsr_str = 'nogs'
#        elif '_gs_' in invalid_edge_file:
#            gsr_str = 'gs'
#        invalid_edges = np.loadtxt(
#            op.join(atlas_dir, "reliability", "invalid_edges", invalid_edge_file)
#        ).astype(int)
    for gsr_type in ['nogs', 'gs']:
        # train and valid represents fold_0 and fold_1, respectively
        for dtype in ['train', 'valid']:
            fold = 'Fold_0' if dtype == 'train' else 'Fold_1'
            invalid_edges = np.loadtxt(op.join(atlas_dir, 'reliability', 'invalid_edges', fold, invalid_edge_file_dict.get(gsr_type))).astype(int)
            summary_index_list = []
            colname = f'{gsr_type}_{dtype}_validity'
            for node in nodes_df['node']:
                edges_df_subset = edges_validity_df[['node1', 'node2', colname]]
                edges_df_subset.loc[invalid_edges, colname] = np.nan
                edges_df_subset = edges_df_subset.query('node1 == @node | node2 == @node')[colname]
                summary_index = np.nanmean(edges_df_subset)
                summary_index_list.append(summary_index)
            nodes_df[f'{dtype}_{gsr_type}'] = summary_index_list
    return nodes_df


def get_summary_rel_lst_df(
        filename_param_list,
        param_order_filename,
        parcellation='Schaefer',
        rel_type='rel',
        family_cv=True,
        fold=None,
        **kwargs
        ):
    """
    Get dataframe with invalid edges
    **kwargs may include invalid_edge_file_list
    """
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    edges_df.sort_values("edge", inplace=True)
    nodes_df = get_nodes_df(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    edges_df = calc_rel_lst_network(
            filename_param_list,
            return_df=True,
            rel_type=rel_type,
            param_order_filename=param_order_filename,
            family_cv=family_cv,
            fold=fold,
            **kwargs
            )
#    for invalid_edge_file in invalid_edge_file_list:
#        if '_nogs_' in invalid_edge_file:
#            gsr_str = 'nogs'
#        elif '_gs_' in invalid_edge_file:
#            gsr_str = 'gs'
#        invalid_edges = np.loadtxt(
#            op.join(atlas_dir, "reliability", "invalid_edges", invalid_edge_file)
#        ).astype(int)
    for gsr_type in ['nogs', 'gs']:
        summary_index_list = []
        for node in nodes_df['node']:
            edges_df_subset = edges_df.query('node1 == @node | node2 == @node')[f'index_mean_{gsr_type}']
            summary_index = edges_df_subset.mean()
            summary_index_list.append(summary_index)
        nodes_df[f'{rel_type}_{gsr_type}'] = summary_index_list
    return nodes_df


def get_invalid_edges_df(
        invalid_edge_file_list,
        parcellation='Schaefer',
        family_cv=True,
        fold=0
        ):
    """
    Get dataframe with invalid edges
    """
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    edges_df.sort_values("edge", inplace=True)
    nodes_df = get_nodes_df(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)

    for invalid_edge_file in invalid_edge_file_list:
        if '_nogs_' in invalid_edge_file:
            gsr_str = 'nogs'
        elif '_gs_' in invalid_edge_file:
            gsr_str = 'gs'
        target_dir = op.join(atlas_dir, "reliability", "invalid_edges")
        if family_cv:
            target_dir = op.join(target_dir, f'Fold_{fold}') 
        invalid_edges = np.loadtxt(
            op.join(target_dir, invalid_edge_file)
        ).astype(int)
        edges_df[gsr_str] = [i in invalid_edges for i in edges_df["edge"]]
        edges_df[gsr_str] = edges_df[gsr_str].astype(float)
    for gsr_type in ['nogs', 'gs']:
        summary_prop_list = []
        for node in nodes_df['node']:
            edges_df_subset = edges_df.query('node1 == @node | node2 == @node')[gsr_type]
            summary_prop = edges_df_subset.mean()
            summary_prop_list.append(summary_prop)
        nodes_df[f'prop_{gsr_type}'] = summary_prop_list
    return nodes_df


def draw_hmaps_invalid_edges(
    invalid_edge_file_list,
    parcellation='Schaefer',
    save_filename=None,
    cmap="Oranges",
    ax=None,
    num_iter=None,
    family_cv=True,
    fold=None
):
    """
    Draw heatmap representing invalid or valid edges
    """
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    edges_df.sort_values("edge", inplace=True)
    nodes_df = get_nodes_df(parcellation)
    mat_df = get_empty_df_for_hmap(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    # Loop for GSR type
    for invalid_edge_file in invalid_edge_file_list:
        target_dir = op.join(atlas_dir, "reliability", "invalid_edges")
        if family_cv:
            target_dir = op.join(target_dir, f'Fold_{fold}')
        invalid_edges = np.loadtxt(
            op.join(target_dir, invalid_edge_file)
        ).astype(int)
        edges_df["invalid"] = [i in invalid_edges for i in edges_df["edge"]]
        edges_df["invalid"] = edges_df["invalid"].astype(float)
        # Shape of wide_edges_df is number of parcellation
        wide_edges_df = get_wide_df_hmap(edges_df, value_col_name="invalid")
        if "nogs" in invalid_edge_file:
            gsr_type = "nogs"
        elif "gs_" in invalid_edge_file:
            gsr_type = "gs"
        mat_df = fill_mat_df(mat_df, wide_edges_df, gsr_type)
    draw_hmaps_fcs(
        mat_df, 
        nodes_df, 
        cmap=cmap, 
    #    save_filename=save_filename, 
        cbar_ax=False, 
        ax=ax, 
        num_iter=num_iter,
        parcellation=parcellation,
        add_custom_cbar=False
    )


def get_latest_combined_files_cv(
        parcellation='Schaefer',
        trait_scale_name=None,
        data_type='parameters'
        ):
    """
    Get latest combined .npy file in cross-validation framework
    data_type should be 'fit_indices', 'parameters', or 'factor_scores'
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    if trait_scale_name is None:
        trait_scale_name = 'reliability'
    target_dir = op.join(atlas_dir, trait_scale_name, data_type, 'combined', 'split_half_cv')
    output_dict = defaultdict(dict)
    for fold in ['Fold_0', 'Fold_1']:
        for gsr_type in ['nogs', 'gs']:
            filenames = [i for i in os.listdir(target_dir) if fold in i and f'_{gsr_type}_' in i]
            filename = sort_list_by_time(filenames)[-1]
            output_dict[fold][gsr_type] = filename
    return output_dict


def get_fit_indices_filenames(
        parcellation='Schaefer',
        trait_type=None,
        scale_name=None,
        family_cv=True,
        msst=True,
        diff_load=True,
        mean_str=True,
        controls=['none'],
        model_type='both'
        ):
    """
    Get list of filenames including nogs and gs conditions
    Output will be used in wrapper_of_select_edges_model_onlyFC()
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    trait_scale_name = get_scale_name_from_trait(trait_type)
    parent_dir = op.join(atlas_dir, trait_scale_name)
    if scale_name is not None:
        parent_dir = op.join(parent_dir, scale_name)
    target_dir = op.join(parent_dir, 'fit_indices', 'combined')
    control_suffix = 'controlling_' + '_'.join(controls)
    output_dict = defaultdict(dict)
    if family_cv:
        target_dir = op.join(target_dir, 'split_half_cv')
    for fold in ['Fold_0', 'Fold_1']:
        filenames = [i for i in os.listdir(target_dir) if fold in i and control_suffix in i]
        if msst:
            filenames = [i for i in filenames if 'MSST' in i]
        else:
            filenames = [i for i in filenames if not 'MSST' in i]

        if diff_load:
            filenames = [i for i in filenames if 'DL' in i]
        else:
            filenames = [i for i in filenames if not 'DL' in i]

        if mean_str:
            filenames = [i for i in filenames if 'MeanStr' in i]
        else:
            filenames = [i for i in filenames if not 'MeanStr' in i]
        filename_gsr = sort_list_by_time([i for i in filenames if '_gs_' in i])[-1]
        filename_nogsr = sort_list_by_time([i for i in filenames if '_nogs_' in i])[-1]
        filename_list = [filename_nogsr, filename_gsr]
        output_dict[fold] = filename_list
    return output_dict


def wrapper_of_select_edges_model_onlyFC(
    filename_fit_dict,
    fit_indices_thresholds_dict,
    loading_minimum,
    loading_maximum,
    omega_minimum=None,
    method_cor_min=None,
    method_cor_max=None,
    fig_size=(8, 20),
    parcellation='Schaefer',
    msst_list=[True],
    loading_tf_minimum=None,
    loading_tf_maximum=None,
    param_filename=None,
    param_order_filename_list=None,
    nrow_fig=1,
    family_cv=False,
    trait_type=None,
    scale_name=None,
    random_seed_inp=None,
    save_filename_venn=None
):
    """
    Draw combined figures from outputs of select_edges_model_onlyFC()
    """
    fig_venn, axes_venn = plt.subplots(nrow_fig, 2, figsize=fig_size)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    trait_scale_name = get_scale_name_from_trait(trait_type)
    if random_seed_inp is None:
        random_seed = 0
        seed_str = ''    
    else:
        random_seed = random_seed_inp
        seed_str = f'seed{random_seed}'

    for i, fold_n in enumerate(filename_fit_dict.keys()):
        filename_list = filename_fit_dict.get(fold_n)
        check_filenames_in_list(filename_list)
        if len(param_order_filename_list) > 1:
            param_order_filename = param_order_filename_list[i] 
        else:
            param_order_filename = param_order_filename_list[0]
        if len(msst_list) > 1:
            msst = msst_list[i]
        else:
            msst = msst_list[0]
        for filename in filename_list:
            print(filename)
            if "_gs_" in filename:
                gsr_title = "with GSR"
                gsr_prefix = "gs_"
                j = 1
            elif "_nogs_" in filename:
                gsr_title = "without GSR"
                gsr_prefix = "nogs_"
                j = 0
            
            if 'SelectCB' in filename:
                cb_prefix = 'SelectCB_'
            else:
                cb_prefix = ''
            controls = get_strings_from_filename(filename, ["control"])[0]
            control_suffix = "_".join(controls) if controls else ''
            mean_str_suffix = '_MeanStr' if 'MeanStr' in filename else '' 
            dl_suffix = '_DL' if 'DL' in filename else ''

            # read parameter file
            param_filename = filename.replace('fit_indices', 'params')
            params_dir = op.join(atlas_dir, trait_scale_name, 'parameters', 'combined')
            if family_cv:
                params_dir = op.join(params_dir, 'split_half_cv')
            params = np.load(op.join(params_dir, param_filename))
            if not msst:
                # this part may be necessary to be modified
                #loadings, covariances = params[:, :4, 0], None
                loadings, covariances = get_load_cov_from_filename(filename, parcellation=parcellation, param_order_filename=param_order_filename, params=params)
            else:
                loadings, loadings_tf = get_load_cov_from_filename(filename, parcellation=parcellation, msst=True, param_order_filename=param_order_filename, params=params)
            # Check criteria of loadings
            if not msst:
                boolean_loading = np.all(
                    np.logical_and(loadings >= loading_minimum, loadings <= loading_maximum),
                    axis=1,
                )
            else:
                loadings_boolean = np.logical_and(loadings >= loading_minimum, loadings <= loading_maximum).all(axis=1) 
                loadings_tf_boolean = np.logical_and(loadings_tf >= loading_tf_minimum, loadings_tf <= loading_tf_maximum).all(axis=1)
                boolean_loading = np.logical_and(loadings_boolean, loadings_tf_boolean)
            invalid_edges_loading = np.where(~boolean_loading)[0]
            combined = 'combine' in filename
            fits = np.squeeze(copy_memmap_output_data(filename, parcellation=parcellation, family_cv=family_cv))
            # Check criteria of fit indices
            fit_indices, fit_thresholds = get_names_and_thresholds_of_fit_indices(
                fit_indices_thresholds_dict
            )
            fit_indices_order = [
                list(FIT_INDICES_OP_DICT.keys()).index(i) for i in fit_indices
            ]
            # Get operations of fit indices
            ops = list(itemgetter(*fit_indices)(FIT_INDICES_OP_DICT))

            fits_interested = fits[:, fit_indices_order]
            boolean_global_each = np.empty(shape=fits_interested.shape)
            for f, fit_threshold in enumerate(fit_thresholds):
                boolean_global_each[:, f] = ops[f](
                    fits_interested[:, f], fit_thresholds[f]
                )
            boolean_global = np.all(boolean_global_each, axis=1)
            invalid_edges_fit = np.where(~boolean_global)[0]

            venn_dict = {
                "Standardized factor loadings": set(invalid_edges_loading),
                "Global fit indices": set(invalid_edges_fit),
            }

            if omega_minimum:
                omegas = calc_omega_2d_array(loadings, covariances)
                # Check criteria of composite reliability
                boolean_omega = omegas > omega_minimum
                invalid_edges_omega = np.where(~boolean_omega)[0]
                venn_dict['Omega coefficients'] = set(invalid_edges_omega)
            n_edges = len(boolean_global)

#            # Check criteria of method effects
#            if (
#                ("PE" in filename)
#                or ("DayCor" in filename)
#                or ("OrderInDay" in filename)
#            ):
#                boolean_cov = np.all(
#                    np.logical_and(
#                        covariances > method_cor_min, covariances < method_cor_max
#                    ),
#                    axis=1,
#                )
#                invalid_edges_cov = np.where(~boolean_cov)[0]
#                venn_dict["Correlated uniqueness"] = set(invalid_edges_cov)
#            else:
#                invalid_edges_cov = []
                
            # Select invalid edges overall
            invalid_edges_overall = set()
            for values in venn_dict.values():
                invalid_edges_overall = invalid_edges_overall | set(values)
           # invalid_edges_overall = (
           #     set(invalid_edges_loading) | set(invalid_edges_fit)
           # )
           # if omega_minimum:
           #     invalid_edges_overall = invalid_edges_overall | set(inavlid_edges_omega)
           # if 

            print("Percentage of invalid edges is following")
            print(
                f"Total: {len(invalid_edges_overall)} ({len(invalid_edges_overall) / n_edges * 100:.2f}%)"
            )
            print(
                f"Parameter estimates: {len(invalid_edges_loading)} ({len(invalid_edges_loading) / n_edges * 100:.2f}%)"
            )
            print(
                f"Global fit indices: {len(invalid_edges_fit)} ({len(invalid_edges_fit) / n_edges * 100:.2f}%)"
            )
            if omega_minimum:
                print(
                    f"Omega coefficient: {len(invalid_edges_omega)} ({len(invalid_edges_omega) / n_edges * 100:.2f}%)"
                )

            # Save invalid edges to file
            invalid_edges_array = np.array(list(invalid_edges_overall))
            model_type = get_load_cov_from_filename(
                filename, get_cov_load=False, get_model_type=True, parcellation=parcellation, family_cv=family_cv
            )
            global_fit_suffix = generate_filename_suffix_on_global_fit(
                fit_indices_thresholds_dict
            )
            if not msst:
                save_filename = f"{model_type}_{gsr_prefix}invalid_edges{global_fit_suffix}_loadings_from_{loading_minimum}_to_{loading_maximum}_omega_greater_than{omega_minimum}_methodCor_{method_cor_min}_to_{method_cor_max}_controlling_{control_suffix}_{seed_str}.csv"
            else:
                save_filename = f"{cb_prefix}{model_type}_{gsr_prefix}invalid_edges{global_fit_suffix}_loadings_from_{loading_minimum}_to_{loading_maximum}_2ndLoadings_from_{loading_tf_minimum}_to_{loading_tf_maximum}_controlling_{control_suffix}{mean_str_suffix}{dl_suffix}_{seed_str}.csv"
            save_dir = op.join(atlas_dir, trait_scale_name, "invalid_edges")
            if family_cv:
                save_dir = op.join(save_dir, fold_n)
                os.makedirs(save_dir, exist_ok=True)
            np.savetxt(
                op.join(
                    save_dir,
                    save_filename,
                ),
                invalid_edges_array,
                fmt="%i",
            )
#            model_type = get_load_cov_from_filename(
#                filename, get_cov_load=False, get_model_type=True, parcellation=parcellation, family_cv=family_cv
#            )

#            if model_type == "errorNone":
#                model = "Model RSFC 1"
#            elif model_type == "errorDay":
#                model = "Model RSFC 2-b"
#            elif model_type == "errorPE":
#                model = "Model RSFC 3-b"
#            elif model_type == "errorOrder":
#                model = "Model RSFC 4-b"
#            elif model_type == 'errorPEDay':
#                model = 'Model RSFC 3'
#            elif model_type == 'errorOrderDay':
#                model = 'Model RSFC 4'
            draw_venn(
                axes_venn,
                i,
                j,
                filename,
                venn_dict,
                model_type,
                gsr_title,
                invalid_edges_overall,
                n_edges,
            )
    kwargs_dict = {"alpha": 0.5}
    red_patch = mpatches.Patch(
        color="red", label="Standardized factor loadings", **kwargs_dict
    )
    blue_patch = mpatches.Patch(color="blue", label="Global fit indices", **kwargs_dict)
    if not msst:
        green_patch = mpatches.Patch(
            color="green", label="Omega coefficients", **kwargs_dict
        )
        yellow_patch = mpatches.Patch(
            color="yellow", label="Correlated uniqueness", **kwargs_dict
        )
        handles_list = [red_patch, blue_patch, green_patch, yellow_patch]
    else:
        handles_list = [red_patch, blue_patch]
    fig_venn.legend(
        handles=handles_list,
        bbox_to_anchor=(0.5, -0.05),
        loc="lower center",
        fontsize=8,
        ncol=2,
    )
    fig_venn.tight_layout()
    if save_filename_venn:
        fig_venn.savefig(
            op.join(atlas_dir, "reliability", "figures", f"{save_filename}.png"),
            bbox_inches="tight",
        )


def draw_venn(
    axes_venn,
    i,
    j,
    filename,
    venn_dict,
    model_type,
    gsr_title,
    invalid_edges_overall,
    n_edges,
    venn_annotate_fontsize=8,
    venn_fontsize=8,
    title_fontsize=10,
    cmap=["red", "blue", "green", "yellow"],
):
    """
    Draw venn diagram
    """
    label_format_func = (
        lambda x: f"{x}\n({x/n_edges:1.2%})"
        if make_newline_in_fig1
        else f"{x} ({x/n_edges:1.2%})"
    )
    if len(venn_dict.keys()) == 2:
        func = venn2
    elif len(venn_dict.keys()) == 3:
        func = venn3
    
    if len(venn_dict.keys()) in [2, 3]:
        func(
            [venn_dict.get(key) for key in venn_dict.keys()],
            ax=axes_venn[i, j],
            set_labels=venn_dict.keys()
                )

    if len(venn_dict.keys()) > 3:
        venn(
            venn_dict,
            fmt="{percentage:.1f}%",
            fontsize=venn_fontsize,
            cmap=cmap,
            legend_loc=None,
            ax=axes_venn[i, j],
        )
    axes_venn[i, j].set_title(f"{model_type} {gsr_title}", fontsize=title_fontsize)
    axes_venn[i, j].annotate(
        f"{len(invalid_edges_overall) / n_edges * 100:.2f}% in total",
        xy=(0.00, 0.00),
        xycoords="axes points",
        fontsize=venn_annotate_fontsize,
    )


#def select_edges_model_onlyFC(
#    filename_list,
#    fit_indices_thresholds_dict,
#    loading_minimum,
#    loading_maximum,
#    omega_minimum,
#    draw_fig=False,
#    fig_size1=(12, 4),
#    save_filename1=None,
#    make_newline_in_fig1=False,
#    node_summary_path=NODE_SUMMARY_PATH,
#    fig_size2=(12, 6),
#    invalid_edge_file_dict=None,
#    save_filename2=None,
#    title_fontsize=12,
#    venn_annotate_fontsize=12,
#    venn_fontsize=12,
#    make_new_line_annot=False,
#    method_cor_min=None,
#    method_cor_max=None,
#    method_thres=None,
#    msst=False,
#    higher_loading_min=None,
#    higher_loading_max=None
#):
#    """
#    Select edges using
#    (I) Thresholds of fit indices,
#    (II) Parameter estimates,
#    and (III) Omega coefficient (old version)
#    This function focused on the Model RSFC 2 but now focuses on MSST
#    """
#    fig_venn, axes_venn = plt.subplots(1, 2, figsize=fig_size1)
#    fig_heat, axes_heat = plt.subplots(2, 1, figsize=fig_size2, sharex=True)
#
#    for i, filename in enumerate(filename_list):
#        print(filename)
#        if "_gs_" in filename:
#            gsr_title = "With GSR"
#            gsr_prefix = "gs_"
#        elif "_nogs_" in filename:
#            gsr_title = "Without GSR"
#            gsr_prefix = "nogs_"
#        controls = get_strings_from_filename(filename, ["control"])[0]
#        control_suffix = "_".join(controls)
#        if not msst:
#            loadings, covariances = get_load_cov_from_filename(filename, msst=msst)
#        else:
#            loadings, higher_loadings = get_load_cov_from_filename(filename, msst=msst)
#
#        # Check criteria of loadings
#        def check_loading(loadings_input, loading_min, loading_max):
#            boolean_loading = np.all(
#                np.logical_and(loadings_input > loading_min, loadings_input < loading_max),
#                axis=1,
#            )
#            invalid_edges_loading = np.where(~boolean_loading)[0]
#            return invalid_edges_loading
#
#        invalid_edges_loading = check_loading(loadings, loading_minimum, loading_maximum)
#        if msst:
#            invalid_edges_higher_loading = check_loading(higher_loadings, higher_loading_min, higher_loading_max)
#
#        fits = np.squeeze(copy_memmap_output_data(filename))
#        # Check criteria of fit indices
#        fit_indices, fit_thresholds = get_names_and_thresholds_of_fit_indices(
#            fit_indices_thresholds_dict
#        )
#        fit_indices_order = [
#            list(FIT_INDICES_OP_DICT.keys()).index(i) for i in fit_indices
#        ]
#        # Get operations of fit indices
#        ops = list(itemgetter(*fit_indices)(FIT_INDICES_OP_DICT))
#
#        fits_interested = fits[:, fit_indices_order]
#        boolean_global_each = np.empty(shape=fits_interested.shape)
#        for f, fit_threshold in enumerate(fit_thresholds):
#            boolean_global_each[:, f] = ops[f](fits_interested[:, f], fit_thresholds[f])
#        boolean_global = np.all(boolean_global_each, axis=1)
#        invalid_edges_fit = np.where(~boolean_global)[0]
#        
#        if not msst:
#            omegas = calc_omega_2d_array(loadings, covariances)
#            # Check criteria of composite reliability
#            boolean_omega = omegas > omega_minimum
#            invalid_edges_omega = np.where(~boolean_omega)[0]
#            n_edges = len(omegas)
#        else:
#            invalid_edges_omega = []
#
#        # Check criteria of method effects
#        if ("PE" in filename) or ("DayCor" in filename) or ("OrderInDay" in filename):
#            boolean_cov = np.all(
#                np.logical_and(
#                    covariances > method_cor_min, covariances < method_cor_max
#                ),
#                axis=1,
#            )
#            invalid_edges_cov = np.where(~boolean_cov)[0]
#        else:
#            invalid_edges_cov = []
#
#        # Select invalid edges overall
#        invalid_edges_overall = (
#            set(invalid_edges_loading)
#            | set(invalid_edges_fit)
#            | set(invalid_edges_omega)
#            | set(invalid_edges_cov)
#        )
#        if msst:
#            invalid_edges_overall = invalid_edges_overall | set(invalid_edges_higher_loading)
#        print("Percentage of invalid edges is following")
#        print(
#            f"Total: {len(invalid_edges_overall)} ({len(invalid_edges_overall) / n_edges * 100:.2f}%)"
#        )
#        print(
#            f"Parameter estimates: {len(invalid_edges_loading)} ({len(invalid_edges_loading) / n_edges * 100:.2f}%)"
#        )
#        print(
#            f"Global fit indices: {len(invalid_edges_fit)} ({len(invalid_edges_fit) / n_edges * 100:.2f}%)"
#        )
#        if not msst:
#            print(
#                f"Omega coefficient: {len(invalid_edges_omega)} ({len(invalid_edges_omega) / n_edges * 100:.2f}%)"
#            )
#
#        # Save invalid edges to file
#        invalid_edges_array = np.array(list(invalid_edges_overall))
#
#        global_fit_suffix = generate_filename_suffix_on_global_fit(
#            fit_indices_thresholds_dict
#        )
#        model_type = get_load_cov_from_filename(
#            filename, get_cov_load=False, get_model_type=True
#        )
#        # invalid_edges_array = [int(i) for i in invalid_edges_array]
#        msst_suffix = '_msst' if msst else ''
#        higher_loadings_suffix = f'_higherLoadings_from_{higher_loading_min}_to_{higher_loading_max}' if msst else ''
#        omega_suffix = f'omega_greater_than{omega_minimum}' if not msst else ''
#        methodCor_suffix = f'methodCor_{method_cor_min}_to_{method_cor_max}'if ("PE" in filename) or ("DayCor" in filename) or ("OrderInDay" in filename) else ''
#        np.savetxt(
#            op.join(
#                atlas_dir,
#                "reliability",
#                "invalid_edges",
#                f"{model_type}_{gsr_prefix}invalid_edges{global_fit_suffix}_loadings_from_{loading_minimum}_to_{loading_maximum}_{higher_loadings_suffix}_{omega_suffix}_{methodCor_suffix}_controlling_{control_suffix}.csv",
#            ),
#            invalid_edges_array,
#            fmt="%i",
#        )
#
#        if draw_fig:
#            label_format_func = (
#                lambda x: f"{x}\n({x/n_edges:1.2%})"
#                if make_newline_in_fig1
#                else f"{x} ({x/n_edges:1.2%})"
#            )
#            # Draw venn diagrams
#            if (
#                ("PE" in filename)
#                or ("DayCor" in filename)
#                or ("OrderInDay" in filename)
#            ):
#                venn_dict = {
#                    "Standardized factor loadings": set(invalid_edges_loading),
#                    "Global fit indices": set(invalid_edges_fit),
#                    "Omega coefficients": set(invalid_edges_omega),
#                    "Correlated uniqueness": set(invalid_edges_cov),
#                }
#            else:
#                venn_dict = {
#                    "Standardized Factor loadings": set(invalid_edges_loading),
#                    "Global fit indices": set(invalid_edges_fit),
#                    "Omega coefficients": set(invalid_edges_omega),
#                }
#            venn(
#                venn_dict,
#                fmt="{percentage:.1f}%",
#                fontsize=venn_fontsize,
#                legend_loc="upper right",
#                ax=axes_venn[i],
#            )
#            # venn3(
#            #    [set(invalid_edges_loading), set(invalid_edges_fit), set(invalid_edges_omega)],
#            #    set_labels=['Factor loadings', 'Global fit', 'Omega coefficient'],
#            #    set_colors=['red', 'yellow', 'green'],
#            #    subset_label_formatter=label_format_func,
#            #    ax=axes1[i]
#            #    )
#            # axes1[i].get_legend().remove()
#            axes_venn[i].set_title(gsr_title, fontsize=title_fontsize)
#            new_line_annot = "\n" if make_new_line_annot else ""
#            axes_venn[i].annotate(
#                f"{len(invalid_edges_overall) / n_edges * 100:.2f}%{new_line_annot} in total",
#                xy=(0.00, 0.00),
#                xycoords="axes points",
#                fontsize=venn_annotate_fontsize,
#            )
#            if i == 0:
#                axes_venn[i].get_legend().remove()
#            # Draw heatmap of removed edges
#            edges_count_wide_df = get_n_of_edges_per_network(node_summary_path)
#            remove_edges_df = get_edge_summary(
#                node_summary_path, network_hem_order=True
#            ).query("edge in @invalid_edges_array")
#            remove_edges_count_wide_df = get_counts_of_edges_per_network(
#                remove_edges_df
#            )
#            prop_removed_edges_df = remove_edges_count_wide_df / edges_count_wide_df
#            prop_removed_edges_df = rename_net_summary_df(prop_removed_edges_df)
#
#            sns.set(font_scale=1)
#            sns.heatmap(
#                prop_removed_edges_df,
#                annot=True,
#                fmt=".1%",
#                ax=axes_heat[i],
#                vmin=0,
#                vmax=1,
#                annot_kws={"fontsize": 6},
#            )
#            axes_heat[i].set_xticklabels(
#                labels=axes_heat[i].get_xticklabels(), rotation=45
#            )
#            axes_heat[i].set_title(gsr_title)
#        fig_dir = op.join(SCHAEFER_DIR, "reliability", "figures")
#        fig_venn.tight_layout()
#        if save_filename1 is not None:
#            fig_venn.savefig(op.join(fig_dir, save_filename1))
#
#        fig_heat.tight_layout()
#        if save_filename2 is not None:
#            fig_heat.savefig(op.join(fig_dir, save_filename2))


def wrapper_of_calc_omega_per_network(
    filename_fit_nested_list, fig_size=(10, 10), save_filename=None, **kwargs
):
    """
    Wrapper function of calc_omega_per_network
    for generating figures for publication
    """
    fig, axes = plt.subplots(2, 2, figsize=fig_size, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])

    for i, filename_fit_list in enumerate(filename_fit_nested_list):
        target_ax, model_name, method_effect = get_ax_and_model_name(axes, i)
        kwargs_model = kwargs.get(model_name)
        calc_omega_per_network(
            filename_fit_list,
            ax=target_ax,
            cbar_ax=cbar_ax,
            iteration=i,
            **kwargs_model,
        )
        target_ax.set_title(f"{model_name}\n({method_effect})")

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    if save_filename:
        fig.savefig(
            op.join(SCHAEFER_DIR, "reliability", "figures", f"{save_filename}.png")
        )


def get_cor(X, y):
    cor = np.corrcoef(np.squeeze(X), np.squeeze(y))[0, 1]
    return cor


def out_of_sample_sem(
    parcellation='Schaefer',
#    fold_sample_n_dict={},
    controls=None,
    msst=True,
    mean_structure=True,
    grid_params_dict=None,
    grid_scoring='neg_mean_absolute_error',
    grid_n_jobs=50,
    grid_get_train_score=True,
    grid_n_splits=10,
    grid_n_repeats=1,
    estimator=Ridge,
    metric_func_dict={'mae':median_absolute_error, 'r2':r2_score},
    trait_types=TRAIT_TYPES,
    covariates=None,
    std_y=False,
    target_net_list=None,
    drop_scale_name_list=None,
    invalid_edge_file_dict=None,
    p_thres=None,
    t_thres=2.5,
    stacking_sem=False,
    diff_load=False,
    trait_equal_loadings=False,
    model_type='both',
    mean_output=True,
    calculate_fimp=False,
    result_filename=None,
    fold=0,
    fold_dict_pickle=None,
    random_seed_inp=None,
    fc_same_edge=False,
    draw_fig=False,
    gsr_types=['_nogs', '_gs'],
    ordered=False,
    fc_1st_load_thres_list=[0.4, 0.9],
    fc_2nd_load_thres_list=[0.7, 1],
    trait_load_thres_list=[0, 1],
    common_op_edges=True,
    save_sem_op_output=False
        ):
    """
    Try out-of-sample prediction using reflective measuremnt model according to Rooij et al. (2023)
    covariates represents covariates in machine learning pipeline while control represents covariates in SEM
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    parcel_name = PARCEL_NAME_DICT.get(parcellation)
    n_rsfc_all = N_RSFC_DICT.get(parcellation) 
    if random_seed_inp is None:
        random_seed = 0
        seed_str = ''
    else:
        random_seed = random_seed_inp
        seed_str = f'seed{random_seed}'
#    fold0_id, fold1_id = train_test_split_family(random_seed=random_seed)
    if fold_dict_pickle is not None:
        fold_dict = pd.read_pickle(fold_dict_pickle)
        train_id = fold_dict.get(fold).get('train')
        test_id = fold_dict.get(fold).get('test')
    else:
        raise Exception('fold_dict_pickle should be specified')
    subject_id_dict = {'train': train_id, 'test': test_id}
    gsr_list, trait_type_list, scale_name_list, fold_list, metric_list, metric_type_list, metric_list_fc, metric_list_fc_items = [], [], [], [], [], [], [], []
    # read data including confounds
    cov_df = pd.read_csv(COVARIATES_PATH)
    cov_df['Subject'] = cov_df['Subject'].astype(str)
    # these idx are necessary for subsetting from fc data
    subjects_list = os.listdir(op.join(HCP_ROOT_DIR, 'data'))
    train_id_idx, test_id_idx = sorted([subjects_list.index(i) for i in train_id]), sorted([subjects_list.index(i) for i in test_id])
    subject_idx_dict = {'train': train_id_idx, 'test': test_id_idx}
    train_cov_df, test_cov_df = cov_df.query('Subject in @train_id'), cov_df.query('Subject in @test_id')

    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    if target_net_list:
        net_edge_index = edges_df.query('(node1_net in @target_net_list) & (node2_net in @target_net_list)').index
    else:
        net_edge_index = edges_df.index
    # output of feature importances
    #fimp_output_array = np.empty(shape=)
    n_rsfc = len(net_edge_index)
    
    # array for storing add_rsfc_predict
    #add_rsfc_predict_array = np.empty(shape=(2, len(trait_types), ))

    for gsr_n, gsr_type in enumerate(gsr_types):
        print(f'Processing {gsr_type}')
        if invalid_edge_file_dict:
            invalid_edges = np.loadtxt(op.join(
                atlas_dir, 'reliability', 'invalid_edges', f'Fold_{fold}', invalid_edge_file_dict.get(gsr_type.replace('_', ''))
                )).astype(int)
        else:
            invalid_edges = set()
        # get predictors (FCs)
        gsr_str = 'gs_' if gsr_type == '_gs' else ''
        include_edge_index = [i for i in net_edge_index]
        if invalid_edge_file_dict is not None:
            include_edge_index = [i for i in include_edge_index if i not in invalid_edges]
        #if mean_output:
        print('Loading FC data.')
        fc_data = np.load(op.join(atlas_dir, f'ordered_fc_spreg_0.25_{gsr_str}demeaned_{parcel_name}.npy'))
        fc_array_mean = np.mean(fc_data, axis=2)
        #fc_array_mean[exclude_edge_index] = np.nan
        # get predictors (FCs)
        fc_dict = {
                'train': subset_fc_data_from_subjects(fc_data, train_id),
                'test': subset_fc_data_from_subjects(fc_data, test_id)
                }

        for trait_type in trait_types:
            trait_scale_name = get_scale_name_from_trait(trait_type)
            scale_names = get_subscale_list(trait_scale_name)
            if drop_scale_name_list:
                scale_names = [i for i in scale_names if not i in drop_scale_name_list]
            if draw_fig:
                fig, axes = plt.subplots(ncols=len(scale_names), figsize=(4*len(scale_names), 6))
            param_order_dir = op.join(FA_ORDER_DIR, trait_scale_name)
            trait_df = read_trait_data(trait_type, subjects_filename=SUBJECT_ANALYSIS_PATH)
            for col, scale_name in enumerate(scale_names):
                subscales = get_subscale_list(trait_scale_name=trait_scale_name, get_all_subscales=True, scale_name=scale_name)
                n_subscales = len(subscales)
                target_dir = op.join(atlas_dir, trait_scale_name, scale_name, 'parameters', 'combined')
                model_vcov_dir = op.join(atlas_dir, trait_scale_name, scale_name, 'model_vcov', 'combined')
                # get param order file
                control_suffix = '_'.join(controls)
                param_order_files = [i for i in os.listdir(param_order_dir) if scale_name in i and control_suffix in i]
                param_order_files = [i for i in param_order_files if 'MSST' in i if msst]
                param_order_files = [i for i in param_order_files if 'MeanStr' in i and 'FCToTrait' in i if mean_structure]
                param_order_files = [i for i in param_order_files if not i.startswith('.')]
                param_order_files = [i for i in param_order_files if model_type in i]
                if len(param_order_files) == 1:
                    param_order = pd.read_csv(op.join(param_order_dir, param_order_files[0]))
                else:
                    raise Exception('param_order_files should include only one filename.')
                # get positions of model implied variance-covariance matrix
                if model_type == 'both':
                    n_items = SUBSCALE_N_DICT.get(trait_scale_name).get(scale_name)
                    trait_start, trait_end = 0, n_items
                    fc_start, fc_end = n_items, n_items + 4
                elif model_type == 'fc':
                    trait_idx = 0
                    fc_start, fc_end = 1, 5
                # get scales
                scales = SCALES_DICT.get(model_type).get(trait_scale_name).get(scale_name)
                filenames = [i for i in os.listdir(target_dir) if gsr_type in i]
                filenames_vcov = [i for i in os.listdir(model_vcov_dir) if gsr_type in i]
                filenames_vcov = [i for i in filenames_vcov if seed_str in i]

#                for row, fold in enumerate(fold_sample_n_dict.keys()):
#                train_fold = f'Fold_{fold}'
#                valid_fold = reduce(add, [i for i in ['Fold_0', 'Fold_1'] if not i == train_fold])
#                print(f'Processing {train_fold} as a training dataset')
                # index for FC
#                train_subject_idx = subject_idx_dict.get(train_fold)
#                valid_subject_idx = subject_idx_dict.get(valid_fold)
#                train_sample_n = fold_sample_n_dict.get(fold)
              #  valid_sample_n = fold_sample_n_dict.get(reduce(add, [i for i in [0, 1] if not i == fold]))
                  #  train_filenames_vcov = [i for i in os.listdir(model_vcov_dir) if gsr_type in i and if train_fold in i]
                  #  valid_filenames_vcov = [i for i in os.listdir(model_vcov_dir) if gsr_type in i and if valid_fold in i]

                def get_latest_filename(filenames):
                    """
                    Get latest filename in cross-validation
                    filtering based on controls may be desireble
                    """
                   # train_sample_n_str = f'sampleN_{train_sample_n}'
                   # train_filenames = [i for i in filenames if train_fold in i and train_sample_n_str in i]
                   # valid_filenames = [i for i in filenames if valid_fold in i and sample_n_str in i]
                    train_filenames = [i for i in filenames if 'MeanStr' and 'FCToTrait' in i if mean_structure]
                    if diff_load:
                        train_filenames = [i for i in train_filenames if 'DL' in i]
                    else:
                        train_filenames = [i for i in train_filenames if not 'DL' in i]
                    if trait_equal_loadings:
                        train_filenames = [i for i in train_filenames if 'TEL' in i]
                    else:
                        train_filenames = [i for i in train_filenames if not 'TEL' in i]

                    if ordered:
                        train_filenames = [i for i in train_filenames if 'ordered' in i]
                    else:
                        train_filenames = [i for i in train_filenames if not 'ordered' in i]
                    train_filenames = [i for i in train_filenames if model_type in i]
                    train_filenames = [i for i in train_filenames if seed_str in i]
                    train_filenames = [i for i in train_filenames if f'Fold_{fold}' in i]
                   # valid_filenames = [i for i in valid_filenames if 'MeanStr' in i if mean_structure]
                    # this line should be modified
                   # filenames = [i for i in filenames if not 'FixedLoad' in i]
                    train_filenames = sort_list_by_time(train_filenames)
                   # valid_filenames = sort_list_by_time(valid_filenames)
                    return train_filenames[-1]
                #, valid_filenames[-1]
                    #if len(filenames) == 1:
                    #    return filenames
                    #else:
                    #    raise Exception('filename object should include only one filename')
                ### conduct SEM-based prediction
                # get positions of edges significantly associated with traits
                train_filename = get_latest_filename(filenames)
                print(f'Processing {train_filename}.')
                train_params = np.load(op.join(target_dir, train_filename))
                if len(train_params) != n_rsfc_all:
                    warn('Input file does not have enough RSFC. Some bug has occurred.')
                    continue
                # remove additional edges from parameters
                removed_edges = get_additionally_excluded_edges(
                        param_order, train_params, controls, fc_1st_load_thres_list, fc_2nd_load_thres_list, trait_load_thres_list
                        )
                #train_params = train_params[include_edge_index]
                # valid_params = np.load(op.join(target_dir, valid_filename))
                # read model implied variance covariance matrix
                train_filename_vcov = get_latest_filename(filenames_vcov)
                train_model_vcov = np.load(op.join(model_vcov_dir, train_filename_vcov))
                #valid_model_vcov = np.load(op.join(model_vcov_dir, valid_filename_vcov))
                # subset matrix
                if model_type == 'both':
                    train_sigma_xy = train_model_vcov[:, fc_start:fc_end, trait_start:trait_end]
                elif model_type == 'fc':
                    train_sigma_xy = train_model_vcov[:, fc_start:fc_end, trait_idx]
                train_sigma_xx = train_model_vcov[:, fc_start:fc_end, fc_start:fc_end]
                # get estimated mean values of responses (trait measure)
                if model_type == 'fc' and trait_type == 'personality':
                    scales = ['Total']
                trait_mean_positions = param_order.query('(lhs.isin(@scales)) & (op == "~1")').index
                train_mu_y_hat = train_params[:, trait_mean_positions, 0]
                # get estimated mean values of predictors (FCs)
                fc_mean_positions = param_order.query('(lhs.isin(["s1", "s2", "s3", "s4"])) & (op == "~1")').index
                train_mu_x_hat = train_params[:, fc_mean_positions, 0]
              #  valid_mu_x_hat = valid_params[:, fc_mean_positions, 0]
                # get predictors
                train_fc = fc_dict.get('train')
                test_fc = fc_dict.get('test')
                # create array to store predictions
                train_y_hats = np.empty(shape=(train_fc.shape[0], train_fc.shape[1], len(trait_mean_positions)))
                test_y_hats = np.empty(shape=(test_fc.shape[0], test_fc.shape[1], len(trait_mean_positions)))
                # loop for edges
                num_train_error = 0
                num_valid_error = 0
                include_edge_index = [i for i in include_edge_index if i not in removed_edges]
                exclude_edge_index = [i for i in net_edge_index if i not in include_edge_index]
                for i, edge in enumerate(include_edge_index):
                    train_sigma_xy_edge, train_sigma_xx_edge = train_sigma_xy[edge, ...], train_sigma_xx[edge, ...]
                    train_mu_x_hat_edge, train_mu_y_hat_edge = train_mu_x_hat[edge, ...], train_mu_y_hat[edge, ...]
                   # valid_mu_x_hat_edge, valid_mu_y_hat_edge = valid_mu_x_hat[edge, ...], valid_mu_y_hat[edge, ...]
                    test_fc_edge, train_fc_edge = test_fc[edge, ...], train_fc[edge, ...]
                    # calculate gamma
                    #gamma_hat_edge = np.dot(sigma_xy_edge.T, np.linalg.inv(sigma_xx_edge))
                    if model_type == 'fc':
                        train_mu_y_hat_edge = train_mu_y_hat_edge[0]
                    try:
                        add_rsfc_predict_train = (train_fc_edge - train_mu_x_hat_edge) @ np.linalg.inv(train_sigma_xx_edge) @ train_sigma_xy_edge
                        train_y_hat = train_mu_y_hat_edge + add_rsfc_predict_train 
                    except LinAlgError:
                        train_y_hat = np.empty(shape=(len(train_fc_edge), len(train_mu_y_hat_edge)))
                        train_y_hat[:] = np.nan
                        num_train_error += 1
                        add_rsfc_predict_train = np.nan
                    # SEM-based out-of-sample prediction
                    try:
                        add_rsfc_predict_test = (test_fc_edge - train_mu_x_hat_edge) @ np.linalg.inv(train_sigma_xx_edge) @ train_sigma_xy_edge
                        test_y_hat = train_mu_y_hat_edge + add_rsfc_predict_test 
                    except LinAlgError:
                        test_y_hat = np.empty(shape=(len(test_fc_edge), len(train_mu_y_hat_edge)))
                        test_y_hat[:] = np.nan
                        num_valid_error += 1
                        add_rsfc_predict_test = np.nan
                    if model_type == 'fc':
                        train_y_hat, test_y_hat = train_y_hat[:, np.newaxis], test_y_hat[:, np.newaxis]
                    train_y_hats[edge, ...] = train_y_hat
                    test_y_hats[edge, ...] = test_y_hat
                print(f'Numbers of errors in training set and validation set are {num_train_error} and {num_valid_error}, respectively.')
                # get x values of means of FCs
                train_scores = fc_array_mean[:, train_id_idx]
                test_scores = fc_array_mean[:, test_id_idx]
                # get train and true y values
#                if std_y:
#                    subjects = subject_id_dict.get(train_fold) | subject_id_dict.get(valid_fold)
#                    trait_df.query('Subject in @subjects', inplace=True)
#                    trait_df[scale_name] = StandardScaler().fit_transform(trait_df[scale_name].to_numpy().reshape(-1, 1))
                scale_name_items = get_subscale_list(trait_scale_name=trait_scale_name, scale_name=scale_name, get_all_subscales=True, name_type_pub=False)
                y_train_items = trait_df.query('Subject in @train_id')[scale_name_items]
                y_train = trait_df.query('Subject in @train_id')[scale_name]
                y_true = trait_df.query('Subject in @test_id')[scale_name]
                # feature selection based on p values of FC
                n_sample, n_rsfc = train_fc.shape[1], train_fc.shape[0]
                str_cor_index = param_order.query('op == "~"').index
                if p_thres is not None or t_thres is not None:
                    cor_array, p_array, t_array = np.empty(shape=n_rsfc), np.empty(shape=n_rsfc), np.empty(shape=n_rsfc)
                    cor_array[:], p_array[:] = np.nan, np.nan
                    for i in range(n_rsfc):
                        cor = np.corrcoef(train_scores[i, :], y_train)[0, 1]
                        t_value = cor * np.sqrt(n_sample - 2) / np.sqrt(1 - cor**2)
                        p = t.sf(np.abs(t_value), n_sample - 1) * 2
                        cor_array[i], p_array[i], t_array[i] = cor, p, t_value
                if p_thres is not None:
                    p_values = train_params[:, str_cor_index, 2]
                    p_index = np.where(p_values < p_thres)[0]
                    p_index, p_index_mean = [i for i in p_index if i in net_edge_index], [i for i in p_index_mean if i in net_edge_index]
                    p_index_common = set(p_index) & set(p_index_mean)
                    p_index_xor = set(p_index) ^ set(p_index_mean)
                    p_index_or = set(p_index) | set(p_index_mean)
                    p_index_mean = np.where(p_array < p_thres)[0]
                    print(f'Number of selected edges in SEM is {len(p_index)}.')
                    print(f'Number of selected edges in mean is {len(p_index_mean)}.')
                    print(f'Number of common index is {len(p_index_common)}')
                    print(f'Number of exclusive or index is {len(p_index_xor)}')
                if t_thres is not None:
                    t_values = np.squeeze(train_params[:, str_cor_index, 0] / train_params[:, str_cor_index, 1])
                    t_values[exclude_edge_index] = np.nan
                    cors = np.squeeze(train_params[:, str_cor_index, 0])
                    invalid_cor_bool = (cors < -.30) | (cors > .30)
                    t_values[invalid_cor_bool] = np.nan
                    #print('Calculating p values.')
                    t_cors = ma.corrcoef(ma.masked_invalid(t_values), ma.masked_invalid(t_array))[0, 1]
                    t_index = np.where(np.abs(t_values) > t_thres)[0]
                    print(f'Correlation of t values is {t_cors:.2f}.')
                    # this part may be removed
                    #fig, ax = plt.subplots()
                    #ax.scatter(t_array, t_values)
              #  else:
              #      p_index = [i for i in range(n_rsfc)]
#                    train_y_hats, valid_y_hats = train_y_hats[p_index, ...], valid_y_hats[p_index, ...] 
                # calculate summary score (mean score across RSFCs) from SEM-based prediction
                if model_type == 'both':
                    if trait_type in ['cognition']:
                        train_y_hats_summary = np.mean(train_y_hats, axis=2)
                        test_y_hats_summary = np.mean(test_y_hats, axis=2)
                    elif trait_type in ['mental', 'personality']:
                        train_y_hats_summary = np.sum(train_y_hats, axis=2)
                        test_y_hats_summary = np.sum(test_y_hats, axis=2)
                if save_sem_op_output:
                    save_sem_op_dir = op.join(atlas_dir, trait_scale_name, scale_name, 'prediction')
                    os.makedirs(save_sem_op_dir, exist_ok=True)
                    save_filename = f'Fold{fold}_{gsr_type}_sem_op_test.npy'
                    save_filename_fc = f'Fold{fold}_{gsr_type}_fc_test.npy'
                    np.save(op.join(save_sem_op_dir, save_filename), test_y_hats_summary)
                    np.save(op.join(save_sem_op_dir, save_filename_fc), fc_dict['test'])
                    if gsr_n == 0:
                        save_filename_true = f'Fold{fold}_sem_op_true.csv'
                        np.savetxt(op.join(save_sem_op_dir, save_filename_true), y_true)
                if model_type == 'fc':
                    train_y_hats_summary, test_y_hats_summary = np.squeeze(train_y_hats, axis=2), np.squeeze(test_y_hats, axis=2)
                if np.isnan(train_y_hats_summary).sum() > 0:
                    train_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
                    train_y_hats_summary = train_imputer.fit_transform(train_y_hats_summary)
                if np.isnan(test_y_hats_summary).sum() > 0:
                    test_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
                    test_y_hats_summary = test_imputer.fit_transform(test_y_hats_summary)
                # Standrdization on predicted values from SEM
                l1_sem_scaler = StandardScaler()
                train_y_hats_summary = l1_sem_scaler.fit_transform(train_y_hats_summary.T)
                if stacking_sem:
                    test_y_hats_summary = l1_sem_scaler.transform(test_y_hats_summary.T)
                else:
                    test_y_hats_summary = test_y_hats_summary.T
                ## predict scores using summary scores calculated from SEM-based prediction
                # impute values in invalid edges with values of FC
                l1_fc_scaler = StandardScaler()
                #exclude_edge_index = list(set(exclude_edge_index) | set(removed_edges))
               # if len(exclude_edge_index) + len(include_edge_index) != n_rsfc:
               #     raise Exception('Edge selection falied.')
                #if invalid_edge_file_dict is not None:
                train_y_hats_summary[:, exclude_edge_index] = l1_fc_scaler.fit_transform(train_scores[exclude_edge_index].T)
                test_y_hats_summary[:, exclude_edge_index] = l1_fc_scaler.transform(test_scores[exclude_edge_index].T)
#                    train_y_hats_summary = train_y_hats_summary.T
#                    test_y_hats_summary = test_y_hats_summary.T
                ### confound regressions
                if covariates is not None:
                    covs_train = fold0_cov_df[covariates] if fold == 0 else fold1_cov_df[covariates]
                    covs_test = fold1_cov_df[covariates] if fold == 0 else fold0_cov_df[covariates]
                    covs_train, covs_test = covs_train.to_numpy(), covs_test.to_numpy()
                ## X 
                # SEM
                #betas_x = inv(covs_train.T.dot(covs_train)).dot(covs_train.T).dot(train_y_hats_summary.T) 
                #train_y_hats_summary = train_y_hats_summary.T - covs_train.dot(betas_x)
                #test_y_hats_summary = test_y_hats_summary.T - covs_test.dot(betas_x)
                # Mean FC
                    betas_x_fc = inv(covs_train.T.dot(covs_train)).dot(covs_train.T).dot(train_scores.T) 
                    train_scores = train_scores.T - covs_train.dot(betas_x_fc)
                    test_scores = test_scores.T - covs_test.dot(betas_x_fc)
                    ## y
                    betas = inv(covs_train.T.dot(covs_train)).dot(covs_train.T).dot(y_train)        
                    y_train = y_train - covs_train.dot(betas)
                    y_true = y_true - covs_test.dot(betas)
                # exclude outliers
#                    y_train_std, y_train_mean = y_train.std(), y_train.mean()
#                    y_train_min, y_train_max = y_train_mean - 3 * y_train_std, y_train_mean + 3 * y_train_std
#                    train_outlier_idx = np.where(y_train < y_train_min | y_train > y_train_max)[0]
#                    print('Number of outlier in training dataset is {train_outlier_idx.sum()}.')
#                    y_train, train_y_hats_summary, train_scores = y_train[~train_outlier_idx], train_y_hats_summary[~train_outlier_idx], train_scores[~train_outlier_idx]
                ## scaling
                #SEM
                meta_scaler = StandardScaler()
                if p_thres is not None:
                    select_index = list(p_index_common)
                elif t_thres is not None:
                    select_index = t_index
                else:
                    select_index = net_edge_index
                train_y_hats_summary = meta_scaler.fit_transform(train_y_hats_summary[:, select_index])
                test_y_hats_summary = meta_scaler.transform(test_y_hats_summary[:, select_index])
                #train_y_hats_summary, test_y_hats_summary = np.nan_to_num(train_y_hats_summary), np.nan_to_num(test_y_hats_summary)
                # FC
                if mean_output:
                    #train_scores, test_scores = np.nan_to_num(train_scores), np.nan_to_num(test_scores)
                    scaler_fc = StandardScaler()
                    train_scores = scaler_fc.fit_transform(train_scores[select_index].T)
                    test_scores = scaler_fc.transform(test_scores[select_index].T)
                # conduct prediction using predicted scores from SEM
                cv = RepeatedKFold(n_splits=grid_n_splits, n_repeats=grid_n_repeats, random_state=0) 
                if stacking_sem:
                    model = GridSearchCV(
                            estimator(
                                random_state=0, 
                            #    tol=estimator_tol, 
                                #selection=estimator_selection
                                ), 
                            grid_params_dict,
                            scoring=grid_scoring, 
                            n_jobs=grid_n_jobs, 
                            cv=cv,
                            return_train_score=grid_get_train_score
                            )
                    model.fit(train_y_hats_summary, y_train)
                    y_pred = model.predict(test_y_hats_summary)
                else:
                    set_trace()
                if draw_fig:
                    target_ax = axes[col]
                    target_ax.set_title(f'{scale_name} in Fold {fold}', fontsize=10)
                    target_ax.set_xlabel('Predicted')
                    target_ax.set_ylabel('Actual')
                    target_ax.scatter(y_pred, y_true, s=0.5, color='blue', alpha=0.5, label='SEM')
                # FC
                if mean_output:
                    if train_y_hats_summary.shape[1] != train_scores.shape[1]:
                        raise Exception('Number of features does not match between FC-based prediction and SEM-based prediction.')
                    if test_y_hats_summary.shape[1] != test_scores.shape[1]:
                        raise Exception('Number of features does not match between FC-based prediction and SEM-based prediction.')
                    print(f'Number of features is {train_scores.shape[1]}.')
                    model_fc = GridSearchCV(
                        estimator(
                            random_state=0, 
                        #    tol=estimator_tol, 
                            #selection=estimator_selection
                            ), 
                        grid_params_dict,
                        scoring=grid_scoring, 
                        n_jobs=grid_n_jobs, 
                        cv=cv,
                        return_train_score=grid_get_train_score
                        )
                    model_fc.fit(train_scores, y_train)
                    #model_fc.fit(train_scores[:, p_index], y_train)
                    y_pred_fc = model_fc.predict(test_scores)

                    ## multivariate ridge regressions
                    model_fc_items = GridSearchCV(
                        estimator(
                            random_state=0, 
                        #    tol=estimator_tol, 
                            #selection=estimator_selection
                            ), 
                        grid_params_dict,
                        scoring=grid_scoring, 
                        n_jobs=grid_n_jobs, 
                        cv=cv,
                        return_train_score=grid_get_train_score
                        )
                    model_fc_items.fit(train_scores, y_train_items)
                    y_pred_fc_items = model_fc_items.predict(test_scores)
                    y_pred_fc_items = np.sum(y_pred_fc_items, axis=1)
                    #y_pred_fc = model_fc.predict(test_scores[:, p_index])
                    # visualize predictions
                    if draw_fig:
                        target_ax.scatter(y_pred_fc, y_true, s=0.5, color='green', alpha=0.5, label='Mean RSFC')
                ## linear regression line
                if draw_fig:
                    start, stop = target_ax.get_xlim()
                    xseq = np.linspace(start, stop, num=100)
                    # SEM
                    b, a = np.polyfit(np.squeeze(y_pred), np.squeeze(y_true), deg=1)
                    target_ax.plot(xseq, a + b * xseq, color="blue", lw=1)
                    if mean_output:
                        # FC 
                        b_fc, a_fc = np.polyfit(np.squeeze(y_pred_fc), np.squeeze(y_true), deg=1)
                        target_ax.plot(xseq, a_fc + b_fc * xseq, color="green", lw=1)
                        metric_str_mean = f'r2 = {r2_score(y_true, y_pred_fc):.3f}, mse = {mean_squared_error(y_true, y_pred_fc):.3f}, cor = {get_cor(y_true, y_pred_fc):.3f} (Mean)'
                    # diagonal line
                    target_ax.plot(xseq, xseq, color='k', lw=1, linestyle='--')
                    # mean of true y
                    mean_true_y = y_true.mean()
                    target_ax.axhline(mean_true_y, color='k', lw=1, linestyle=':')
                    #alpha_str = f'alpha = {alpha:.3f}'
                    #best_inner_score_str = f'best inner score = {best_grid_score:.3f}'
                    metric_str_sem = f'r2 = {r2_score(y_true, y_pred):.3f}, mse = {mean_squared_error(y_true, y_pred):.3f}, cor = {get_cor(y_true, y_pred):.3f} (SEM)'
                    # vis_str can  be arranged
                    #vis_str = metric_str_sem + '\n' + metric_str_mean
#                    if grid_get_train_score:
#                        vis_str += f'\nmean train score = {mean_train_score:.3f}\nSD of train score = {sd_train_score:.3f}'
                    # metric
                 #   target_ax.text(
                 #           x=0.01, y=0.99,
                 #           s=vis_str, 
                 #           va='top', ha='left', 
                 #           fontsize=8, 
                 #           transform=target_ax.transAxes
                 #           )
                    target_ax.set_aspect('equal', adjustable='datalim')
                print(scale_name)
                print(f'Min, Max, and SD of y_true is {y_true.min():.2f}, {y_true.max():.2f}, and {y_true.std():.2f}.')
                print(f'Min, Max, and SD of y_pred is {y_pred.min():.2f}, {y_pred.max():.2f}, and {y_pred.std():.2f}.')
                if mean_output:
                    print(f'Min, Max, and SD of y_pred_fc is {y_pred_fc.min():.2f}, {y_pred_fc.max():.2f}, and {y_pred_fc.std():.2f}.')
                # evaluate predictive values
                for i, (metric_key, metric_func) in enumerate(metric_func_dict.items()):
                    metric = metric_func(y_true, y_pred)
                  #  lists[i].append(metric)
                    print(f'{metric_key}: {metric:.3f} (SEM)')
                    metric_list.append(metric)
                    if mean_output:
                        metric_fc = metric_func(y_true, y_pred_fc)
                        metric_fc_items = metric_func(y_true, y_pred_fc_items)
                        print(f'{metric_key}: {metric_fc:.3f} (Mean)')
                        metric_list_fc.append(metric_fc)
                        metric_list_fc_items.append(metric_fc_items)
                    metric_type_list.append(metric_key)
                    trait_type_list.append(trait_type)
                    fold_list.append(fold)
                    scale_name_list.append(scale_name)
                    gsr_list.append(gsr_type)
                ## evaluate feature importance (decoding according to Haufe et al. (2014))
                if calculate_fimp:
                    fc_fimp_array, sem_fimp_array = np.empty(shape=(len(edges_df))), np.empty(shape=(len(edges_df)))
                    fc_fimp_array[:], sem_fimp_array[:] = np.nan, np.nan
                    # insert test_scores of mean RSFC
                    test_scores_inserted = np.empty(shape=(len(test_scores), len(edges_df)))
                    test_scores_inserted[:] = np.nan
                    test_scores_inserted[:, net_edge_index] = test_scores
                    # insert predictor array in testation dataset in SEM
                    test_y_hats_summary_inserted = np.empty(shape=(len(test_y_hats_summary), len(edges_df)))
                    test_y_hats_summary_inserted[:] = np.nan
                    test_y_hats_summary_inserted[:, include_edge_index] = test_y_hats_summary
#                        # insert pediction scores
#                        y_pred_inserted, y_pred_fc_inserted = np.empty(shape=(len(edges_df))), np.empty(shape=(len(edges_df)))
#                        y_pred_inserted[:], y_pred_fc_inserted[:] = np.nan, np.nan
#                        y_pred_inserted[include_edge_index], y_pred_fc_inserted[include_edge_index] = y_pred, y_pred_fc
                    for i in range(len(edges_df)):
                        # mean_fc
                        fc_fimp = np.cov(test_scores_inserted[:, i], y_pred_fc)[0, 1] 
                        fc_fimp_array[i] = fc_fimp
                        # SEM
                        try:
                            sem_fimp = np.cov(test_y_hats_summary_inserted[:, i], y_pred)[0, 1]
                        except:
                            sem_fimp = np.nan
                        sem_fimp_array[i] = sem_fimp
                    gsr_prefix = gsr_type.replace('_', '')
                    edges_df[f'Fold{fold}_{gsr_prefix}_{trait_type}_{scale_name}_fc'] = fc_fimp_array
                    edges_df[f'Fold{fold}_{gsr_prefix}_{trait_type}_{scale_name}_sem'] = sem_fimp_array
            if draw_fig:
                fig.suptitle(f'{trait_type} {gsr_type}')
                fig.tight_layout()
    output_df = pd.DataFrame(
            {
                'gsr_type': gsr_list, 
                'scale_name': scale_name_list, 
                'trait_type': trait_type_list,
                'fold': fold_list,
                'metric_type': metric_type_list, 
                'metric_sem': metric_list,
                }
            )
    if mean_output:
        output_df['metric_mean'] = metric_list_fc
        output_df['metric_mean_items'] = metric_list_fc_items
    if calculate_fimp:
        return output_df, edges_df
    if result_filename:
        output_df.to_csv()
    return output_df
                    # model fits by edges
#                    for edge in range(len(valid_y_hat)):
#                        item_hat_edge = valid_y_hat[edge, ...]
#
                    ## calculate sum values across edges
                    #valid_y_deltas_fc_sum = np.nansum(valid_y_deltas, axis=0)
                    ## calculate sum values across subscores
                    #valid_y_array_mean = np.nanmean(valid_y_deltas_fc_mean, axis=1)
                    ## get true values
                    #subject_ids = subject_id_valid_dict.get(f'Fold_{fold}')
                    #trait_df_subset = trait_df.query('Subject in @subject_ids')


def get_long_df_sem_op(df, value_vars=['metric_sem', 'metric_mean'], include_repeat=False):
    """
    Create long sumamry data from output of out_of_sample_sem
    """
    id_vars = ['trait_type', 'gsr_type', 'scale_name', 'fold']
    if include_repeat:
        id_vars.append('repeat')
    long_df = df.melt(
        id_vars=id_vars+['metric_type'],
        value_vars=value_vars
    ).pivot(
        index=id_vars+['variable'],
        columns='metric_type',
        values='value'
    ).reset_index()
    long_df['z'] = np.arctanh(long_df['cor'])
    
    choices = ['NIH toolbox', 'ASR', 'NEO-FFI']

    long_df['trait_type'] = long_df['trait_type'].str.replace('cognition', 'NIH toolbox').replace('mental', 'ASR').replace('personality', 'NEO-FFI')
    long_df['trait_type'] = pd.Categorical(long_df['trait_type'], categories=choices)
    
    NIH_traits = [i + ' (NIH toolbox)' for i in NIH_COGNITION_SCALES]
    ASR_traits = [i + ' (ASR)' for i in ASR_SUBSCALES]
    FFI_traits = [i + ' (NEO-FFI)' for i in NEO_FFI_SCALES]

    long_df['trait'] = long_df['scale_name'].astype(str) + ' (' + long_df['trait_type'].astype(str) + ')'
    long_df['trait'] = pd.Categorical(long_df['trait'], categories=NIH_traits + ASR_traits + FFI_traits)
    long_df.rename(columns={'variable': 'type'}, inplace=True)
    
    for gsr_replaced, gsr_replace in zip(['_nogs', '_gs'], ['Without GSR', 'With GSR']):
        long_df['gsr_type'] = long_df['gsr_type'].str.replace(gsr_replaced, gsr_replace)
    long_df['gsr_type'] = pd.Categorical(long_df['gsr_type'], categories=['Without GSR', 'With GSR'])
    
    return long_df


def calc_rel_lst_network(
    filename_param_list,
    parcellation='Schaefer',
    hmap_vmin=0,
    hmap_vmax=1,
    fig_size=(6, 5),
    rel_type='cons',
    output_filename=None,
    ax=None,
    cbar_ax=None,
    iteration=None,
    param_order_filename=None,
    return_df=False,
    return_all_indicators=False,
    family_cv=True,
    fold=None,
    **kwargs,
):
    """
    Calculate reliability, consistency,and specificity coefficient in a combination of networks
    rel_type should be 'cons', 'spec', 'rel', or 'error'
    'mean' in column name represents calculating means of multiple indicators
    """
    if not ax and not return_df:
        fig, ax = plt.subplots(figsize=fig_size)

    mat_df = get_empty_df_for_hmap(parcellation)
    nodes_df = get_nodes_df(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    param_order = pd.read_csv(op.join(FA_ORDER_DIR_REL, param_order_filename))
    
    if return_df:
        edges_df = get_edge_summary(parcellation, network_hem_order=True)
        edges_df.sort_values("edge", inplace=True)

    for i, filename in enumerate(filename_param_list):
        if "_gs_" in filename:
            gsr_title, gsr_type = "With GSR", "gs"
        elif "_nogs_" in filename:
            gsr_title, gsr_type = "Without GSR", "nogs"

        if 'invalid_edge_file' in kwargs.keys(): 
            if (
                invalid_edge_file := kwargs.get("invalid_edge_file").get(gsr_type)
            ) is not None:
                target_dir = op.join(atlas_dir, "reliability", "invalid_edges")
                if fold is not None:
                    target_dir = op.join(target_dir, f'Fold_{fold}')
                invalid_edges = np.loadtxt(
                    op.join(target_dir, invalid_edge_file)
                ).astype(int)
        params_dir = op.join(atlas_dir, 'reliability', 'parameters', 'combined')
        if family_cv:
            params_dir = op.join(params_dir, 'split_half_cv')
        params = np.load(op.join(params_dir, filename))
 
        ## calculate reliability, consistency, and specificity
        # get parameters
        first_order_load_position = param_order.query('lhs.str.contains("o") & rhs.str.contains("s")', engine='python').index
        second_order_load_position = param_order.query('lhs == "ff" & rhs.str.contains("o")', engine='python').index
        zeta_positions = param_order.query('lhs == rhs & lhs.str.contains("o")', engine='python').index
        # subset array
        lambda_array = params[:, first_order_load_position, 0]
        gamma_array = params[:, second_order_load_position, 0]
        zeta_array = params[:, zeta_positions, 0]
        # calculate consistency, specificty
        cons_array, spec_array, rel_array = np.empty(shape=lambda_array.shape), np.empty(shape=lambda_array.shape), np.empty(shape=lambda_array.shape)
        cons_array[:], spec_array[:], rel_array[:] = np.nan, np.nan, np.nan
        for i in range(lambda_array.shape[1]):
            # day 1
            if i in [0, 1]:
                index = 0
            # day 2
            elif i in [2, 3]:
                index = 1
            gammas, zetas = gamma_array[:, index], zeta_array[:, index]
            # 1 represents variance of indicators
            cons = (gammas * lambda_array[:, i]) ** 2 / 1
            spec = (lambda_array[:, i] ** 2 * zetas) / 1
            cons_array[:, i], spec_array[:, i], rel_array[:, i] = cons, spec, cons + spec
            error_array = 1 - rel_array
        cons_means, spec_means, rel_means, error_means = cons_array.mean(axis=1), spec_array.mean(axis=1), rel_array.mean(axis=1), error_array.mean(axis=1)

        if 'invalid_edge_file' in kwargs.keys():
            if invalid_edge_file:
                cons_means[invalid_edges] = np.nan
                spec_means[invalid_edges] = np.nan
                rel_means[invalid_edges] = np.nan
                error_means[invalid_edges] = np.nan

        if not return_df:
            edges_df = get_edge_summary(parcellation, network_hem_order=True)
            edges_df.sort_values("edge", inplace=True)
            if rel_type == 'cons':
                edges_df["index_mean"] = cons_means
            elif rel_type == 'spec':
                edges_df['index_mean'] = spec_means
            elif rel_type == 'rel':
                edges_df['index_mean'] = rel_means
            elif rel_type == 'error':
                edges_df['index_mean'] = error_means
            edges_df.loc[edges_df['index_mean'] > 1, 'index_mean'] = -100
            wide_df = get_wide_df_hmap(edges_df, value_col_name="index_mean")
            mat_df = fill_mat_df(mat_df, wide_df, gsr_type)
        else:
            if rel_type == 'cons':
                if not return_all_indicators:
                    edges_df[f'index_mean_{gsr_type}'] = cons_means
                else:
                    for i in range(4):
                        edges_df[f'session_{i+1}_{gsr_type}'] = cons_array[:, i]
            elif rel_type == 'spec':
                if not return_all_indicators:
                    edges_df[f'index_mean_{gsr_type}'] = spec_means
                else:
                    for i in range(4):
                        edges_df[f'session_{i+1}_{gsr_type}'] = spec_array[:, i]
            elif rel_type == 'rel':
                if not return_all_indicators:
                    edges_df[f'index_mean_{gsr_type}'] = rel_means
                else:
                    for i in range(4):
                        edges_df[f'session_{i+1}_{gsr_type}'] = rel_array[:, i]
            elif rel_type == 'error':
                if not return_all_indicators:
                    edges_df[f'index_mean_{gsr_type}'] = error_means
                else:
                    for i in range(4):
                        edges_df[f'session_{i+1}_{gsr_type}'] = error_array[:, i]


    if return_df:
        return edges_df

    if hmap_vmin is None:
        hmap_vmin = np.nanmin(mat_df)
    if hmap_vmax is None:
        hmap_vmax = np.nanmax(mat_df)
    draw_hmaps_fcs(
        mat_df,
        nodes_df,
        cmap="Oranges",
        ax=ax,
        vmin=hmap_vmin,
        vmax=hmap_vmax,
        cbar_ax=cbar_ax,
        iteration=iteration,
        parcellation=parcellation
    )
    if output_filename and not ax:
        fig.tight_layout()
        fig.savefig(
            op.join(atlas_dir, "reliability", "figures", f"{output_filename}.png")
        )


def calc_omega_per_network(
    filename_fit_list,
    parcellation='Schaefer',
    hmap_vmin=0.6,
    hmap_vmax=1,
    fig_size=(6, 5),
    hmap_func=None,
    output_filename=None,
    ax=None,
    cbar_ax=None,
    iteration=None,
    **kwargs,
):
    """
    Calculate omega coefficient in a combination of networks
    """
    if not ax:
        fig, ax = plt.subplots(figsize=fig_size)
    if hmap_func is not None:
        fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    mat_df = get_empty_df_for_hmap(parcellation)
    nodes_df = get_nodes_df(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)

    for i, filename in enumerate(filename_fit_list):
        if "_gs_" in filename:
            gsr_title, gsr_type = "With GSR", "gs"
        elif "_nogs_" in filename:
            gsr_title, gsr_type = "Without GSR", "nogs"

        if (
            invalid_edge_file := kwargs.get("invalid_edge_file").get(gsr_type)
        ) is not None:
            invalid_edges = np.loadtxt(
                op.join(atlas_dir, "reliability", "invalid_edges", invalid_edge_file)
            ).astype(int)
        omegas = calc_omega_2d_array_from_filename(filename, parcellation=parcellation)
        edges_df = get_edge_summary(parcellation, network_hem_order=True)
        edges_df.sort_values("edge", inplace=True)
        if invalid_edge_file:
            omegas[invalid_edges] = np.nan
        edges_df["omega"] = omegas

        if hmap_func is not None:
            edges_df_omega_summary = get_summary_of_edges_per_network(
                edges_df, "omega", hmap_func
            )
            edges_df_omega_summary = rename_net_summary_df(edges_df_omega_summary)
            if gsr_type == "nogs":
                edges_df_omega_summary = edges_df_omega_summary.T
            sns.heatmap(
                edges_df_omega_summary,
                annot=True,
                fmt=".2f",
                cmap="Oranges",
                ax=ax2[i],
                vmin=hmap_vmin,
                vmax=hmap_vmax,
            )
            ax2[i].set_title(gsr_title)
            ax2[i].set_xticklabels(labels=ax2[i].get_xticklabels(), rotation=45)

        wide_df = get_wide_df_hmap(edges_df, value_col_name="omega")
        mat_df = fill_mat_df(mat_df, wide_df, gsr_type)
    draw_hmaps_fcs(
        mat_df,
        nodes_df,
        cmap="Oranges",
        ax=ax,
        vmin=hmap_vmin,
        vmax=hmap_vmax,
        cbar_ax=cbar_ax,
        iteration=iteration,
        parcellation=parcellation
    )
    #    sns.heatmap(edges_df_omega_summary, ax=axes[i], annot=True, fmt='.2f', vmin=0.6, vmax=1)
    #    axes[i].set_title(gsr_title)
    #ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
    if output_filename and not ax:
        fig.tight_layout()
        fig.savefig(
            op.join(atlas_dir, "reliability", "figures", f"{output_filename}.png")
        )

    if hmap_func is not None:
        fig2.tight_layout()
        fig2.savefig(
            op.join(
                atlas_dir, "reliability", "figures", f"{output_filename}_summary.png"
            )
        )


def get_latest_nogs_gs_files(
    order_in_day=False,
    parcellation='Schaefer',
    data_type='fit_indices',
    combined=False,
    multistate_single_trait=False,
    family_cv=True,
    fold=None,
    random_seed=None,
    controls=None,
    **kwargs
):
    """
    Get latest files of outputs of fit indices
    """
    output_list = []
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    target_dir = op.join(atlas_dir, "reliability", data_type)
    if combined:
        target_dir = op.join(target_dir, 'combined')
        if family_cv:
            target_dir = op.join(target_dir, 'split_half_cv')
    filename_list_all = os.listdir(target_dir)
    seed_str = f'seed{random_seed}' if random_seed is not None else ''
    for gsr_type in ["_nogs_", "_gs_"]:
        filename_list = [
                i for i in filename_list_all 
                if (gsr_type in i)
                and not ("addMarker" in i)
                and (str(kwargs.get('sample_n')) in i)
                and (str(kwargs.get('edge_n')) in i)
                ]
        filename_list = [i for i in filename_list if (seed_str in i)]
        if fold is not None:
            filename_list = [i for i in filename_list if f'Fold_{fold}' in i]
        filename_list = select_filenames_from_CU_or_MSST(
                filename_list, controls=controls, order_in_day=order_in_day, multistate_single_trait=multistate_single_trait
                )
        filename = sort_list_by_time(filename_list)[-1]
        output_list.append(filename)
    return output_list


def select_filenames_from_CU_or_MSST(
    filenames,
    day_cor=False,
    PE=False,
    order_in_day=False,
    multistate_single_trait=False,
    bi_factor=False,
    family_fold=None,
    sample_n=None,
    controls=None,
    add_CU=False,
    mean_structure=False,
    diff_load=False,
    trait_equal_loadings=False
        ):
    """
    Select filenames from correlated uniqueness
    """
    if sample_n:
        filenames = [i for i in filenames if f'sampleN_{sample_n}' in i]

    if day_cor:
        filenames = [i for i in filenames if 'DayCor' in i]
    else:
        filenames = [i for i in filenames if not 'DayCor' in i]
    
    if PE:
        filenames = [i for i in filenames if 'PE' in i]
    else:
        filenames = [i for i in filenames if not 'PE' in i]
    
    if order_in_day:
        filenames = [i for i in filenames if 'OrderInDay' in i]
    else:
        filenames = [i for i in filenames if not 'OrderInDay' in i]
    
    if multistate_single_trait:
        filenames = [i for i in filenames if 'MultiStateSingleTrait' in i or 'MSST' in i]
    else:
        filenames = [i for i in filenames if not 'MultiStateSingleTrait' in i and not 'MSST' in i]
    
    if bi_factor:
        filenames = [i for i in filenames if 'Bifactor' in i]
    else:
        filenames = [i for i in filenames if not 'Bifactor' in i]

    if mean_structure:
        filenames = [i for i in filenames if 'MeanStr' in i]
    else:
        filenames = [i for i in filenames if not 'MeanStr' in i]
    
    if diff_load:
        filenames = [i for i in filenames if 'DL' in i]
    else:
        filenames = [i for i in filenames if not 'DL' in i]
    
    if trait_equal_loadings:
        filenames = [i for i in filenames if 'TEL' in i]
    else:
        filenames = [i for i in filenames if not 'TEL' in i]

    if add_CU:
        filenames = [i for i in filenames if '_CU_' in i]
    else:
        filenames = [i for i in filenames if '_CU_' not in i]

    if family_fold is not None:
        filenames = [i for i in filenames if f'Fold_{family_fold}' in i]
    
    if controls:
        control_str = 'controlling_' + '_'.join(controls)
        if multistate_single_trait:
            control_str = control_str + '_MSST'
        # should check correctly extracting covariates
        filenames = [i for i in filenames if control_str in i]
    return filenames


def mv_from_nested_trash_to_parent(
        parcellation='Schaefer',
        data_type_list=['correlation', 'fit_indices', 'parameters', 'model_vcov']
        ):
    """
    Move memmap files in trash to main folder
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    for trait_type in TRAIT_TYPES:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        subscale_list = get_subscale_list(trait_scale_name)
        for subscale in subscale_list:
            for data_type in data_type_list:
                trash_folder = op.join(atlas_dir, trait_scale_name, subscale, data_type, 'trash')
                target_trash_folder = op.join(trash_folder, 'trash')
                # move files to trash folder
                for filename in os.listdir(target_trash_folder):
                    move(op.join(target_trash_folder, filename), op.join(trash_folder, filename))


def create_combined_files_from_slurm_array(
        n_arrays=431,
        parcellation='Schaefer',
        trait_type=None,
        day_cor=False,
        PE=False,
        order_in_day=False,
        add_marker=False,
        data_type_list=['fit_indices', 'parameters'],
        single_trait=False,
        multistate_single_trait=False,
        bi_factor=False,
        mean_structure=False,
        diff_load=False,
        add_CU=False,
        family_fold: int=None,
        over_write=False,
        controls=['age', 'gender', 'MeanRMS'],
        controls_before=None,
        model_type='onlyFC',
        search_trash=False,
        remove_str_list=[],
        drop_scale_name_list=None,
        trait_equal_loadings=False,
        gsr_types=['_nogs_', '_gs_'],
        random_seed=None,
        select_cb=False,
        ordered=False,
        **kwargs
        ):
    """
    Create a single file containing results of slurm array
    data_type_list includes 'fit_indices', 'parameters', or 'factor_scores'
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    trait_scale_name = select_folder_from_trait(trait_type)
    if controls:
        control_suffix = 'controlling_' + '_'.join(controls)
    if controls_before:
        control_before_suffix = 'controllingBefore_' + '_'.join(controls_before)
    if random_seed is not None:
        seed_str = f'seed{random_seed}'
    else:
        seed_str = ''

    def combine_files(target_dir, data_type_list, n_arrays): 
        for gsr_type in gsr_types:
            for data_type in data_type_list:
                output_dir = op.join(target_dir, data_type)
                if search_trash:
                    output_dir = op.join(output_dir, 'trash')
                print(f'Processing {output_dir} in {gsr_type}.')
                filenames = [
                        i for i in os.listdir(output_dir)
                        if (gsr_type in i)
                        and (str(kwargs.get('sample_n')) in i)
                        and (str(kwargs.get('edge_n')) in i)
                        and (seed_str in i)
                        ]
                if add_marker:
                    filenames = [i for i in filenames if 'addMarker' in i]
                else:
                    filenames = [i for i in filenames if not 'addMarker' in i]

                if select_cb:
                    filenames = [i for i in filenames if 'SelectCB' in i]
                else:
                    filename = [i for i in filenames if not 'SelectCB' in i]

                if controls is not None:
                    filenames = [i for i in filenames if control_suffix in i]
                else:
                    if controls_before is None:
                        filenames = [i for i in filenames if not 'controlling' in i]

                if controls_before:
                    filenames = [i for i in filenames if control_before_suffix in i]
                if ordered:
                    filenames = [i for i in filenames if 'ordered' in i]
                else:
                    filenames = [i for i in filenames if not 'ordered' in i]
                filenames = [i for i in filenames if model_type in i]

                if single_trait:
                    filenames = [i for i in filenames if 'SingleTrait' in i]
                else:
                    filenames = [i for i in filenames if not 'SingleTrait' in i]

                filenames = select_filenames_from_CU_or_MSST(
                        filenames, 
                        day_cor, 
                        PE,
                        order_in_day, 
                        multistate_single_trait,
                        bi_factor=bi_factor,
                        mean_structure=mean_structure,
                        add_CU=add_CU,
                        diff_load=diff_load,
                        trait_equal_loadings=trait_equal_loadings,
                        controls=controls,
                        family_fold=family_fold,
                        sample_n=kwargs.get('sample_n')
                        )
                if len(filenames) > 0:
                    combined_array, filename = combine_array_files_dat(
                            filenames,
                            n_arrays=n_arrays,
                            data_type=data_type,
                            return_filename=True,
                            parcellation=parcellation,
                            msst=multistate_single_trait,
                            bi_factor=bi_factor,
                            trait_type_for_fscores=trait_type,
                            search_trash=search_trash
                            )
                    print(combined_array.shape)
                    if not search_trash:
                        save_folder = op.join(output_dir, 'combined')
                    else:
                        save_folder = op.join(target_dir, data_type, 'combined')
                    if family_fold is not None:
                        save_folder = op.join(save_folder, 'split_half_cv')
                    os.makedirs(save_folder, exist_ok=True)

                    # remove some strings
                    for string in remove_str_list:
                        filename = filename.replace(f'{string}_', '')

                    target_filepath = op.join(save_folder, filename)
                    print(f'Saving {filename}.')
                    if not over_write:
                        if not op.isfile(target_filepath):
                            np.save(target_filepath, combined_array)
                    else:
                        np.save(target_filepath, combined_array)
                    print('Saving completed.')
                    # move (or delete) old files
                    if not search_trash:
                        trash_dir = op.join(output_dir, 'trash')
                        os.makedirs(trash_dir, exist_ok=True)
                        for filename in filenames:
                            move(op.join(output_dir, filename), op.join(trash_dir, filename)) 
                else:
                    warn('Processing stopped since filenames in empty.')
    if trait_type:
        scale_name_list = get_subscale_list(trait_scale_name)
        data_type_list.append('correlation')
        if drop_scale_name_list:
            scale_name_list = [i for i in scale_name_list if not i in drop_scale_name_list]
        for scale_name in scale_name_list:
            target_dir = op.join(atlas_dir, trait_scale_name, scale_name)
            combine_files(target_dir, data_type_list, n_arrays)
    else:
        target_dir = op.join(atlas_dir, trait_scale_name)
        combine_files(target_dir, data_type_list, n_arrays)
   
    print('Completed.')


def wrapper_of_draw_hmap_on_prop_chisq_diff_test(
    filename_fit_nested_list,
    invalid_edge_file_list_nested_dict,
    fig_size=(15, 5),
    save_filename=None,
    compare='no_effect',
    vmax=320
):
    """
    Wrapper function of draw_hmap_on_prop_of_chisq_diff_test
    to create figure for publication
    """
    num_axes = len(filename_fit_nested_list)
    fig, axes = plt.subplots(1, num_axes, figsize=fig_size, sharey=True)
    cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
    if num_axes == 1:
        target_ax = axes

    for i, filename_fit_list_compared in enumerate(filename_fit_nested_list):
        if all('DayCor' in i and not 'PE' in i and not 'OrderInDay' in i for i in filename_fit_list_compared):
            model_name = "Model RSFC 2-b"
            add_method = "measurement day"
        elif all('DayCor' in i and 'PE' in i and not 'OrderInDay' in i for i in filename_fit_list_compared):
            model_name = "Model RSFC 3"
            add_method = "phase encoding direction"
        elif all('DayCor' in i and 'OrderInDay' in i and not 'PE' in i for i in filename_fit_list_compared):
            model_name = "Model RSFC 4"
            add_method = "measurement order in days"

        invalid_edge_file_list_dict = invalid_edge_file_list_nested_dict.get(model_name)
        if num_axes > 1:
            target_ax = axes[i]

        if compare == 'no_effect':
            filenames_baseline = get_latest_nogs_gs_files(False, False, False, combined=True)
        elif compare == 'day_effect':
            filenames_baseline = get_latest_nogs_gs_files(True, False, False, combined=True)

        draw_hmap_on_prop_of_chisq_diff_test(
            filename_fit_list=filenames_baseline,
            filename_fit_list2=filename_fit_list_compared,
            invalid_edge_file_list_dict=invalid_edge_file_list_dict,
            ax=target_ax,
            cbar_ax=cbar_ax,
            iteration=i,
            vmax=vmax,
            num_iter=num_axes
        )
        target_ax.set_title(f"{model_name}\n(Effects of {add_method})")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    if save_filename:
        fig.savefig(
            op.join(SCHAEFER_DIR, "reliability", "figures", f"{save_filename}.png")
        )


def check_filenames_in_list(filename_list):
    if len(filename_list) != 2:
        raise Exception('List should include two files of no GSR and GSR conditions.')
    
    nogs_file, gs_file = filename_list[0], filename_list[1]
    if not '_nogs_' in nogs_file:
        raise Exception('First element of filename_fit_list should be no GSR condition data.')
    if not '_gs_' in gs_file:
        raise Exception('Second element of filename_fit_list should be GSR condition data.')


def calc_chi2_summary_per_node(
        filename_fit_list,
        filename_fit_list2=None,
        parcellation='Schaefer',
        summary='median',
        family_cv=True
        ):
    """
    Calculate mean chi-square values per node
    """
    edges_df = get_edge_summary(network_hem_order=True, parcellation=parcellation)
    nodes_df = get_nodes_df(parcellation=parcellation)
    
    check_filenames_in_list(filename_fit_list)
#    if filename_fit_list2:
#        check_error(filename_fit_list2)

    for i, gsr_type in enumerate(['nogs', 'gs']):
        chi2 = conduct_chi_square_diff_test(filename_fit_list[i], parcellation=parcellation, family_cv=family_cv)
        if filename_fit_list2:
            chi2_2 = conduct_chi_square_diff_test(filename_fit_list2[i], parcellation=parcellation)
            chi2 = chi2 - chi2_2
        edges_df[gsr_type] = chi2
    # Calculate summary of chi-square values pre node
    nodes_df = get_nodes_df_from_edges_df(edges_df, nodes_df, 'chi2')

    return nodes_df


def get_nodes_df_from_edges_df(edges_df, nodes_df, prefix, summary='median'):
    """
    Convert edge-level df to node-level df with summary function
    """
    for gsr_type in ['nogs', 'gs']:
        summary_stat_list = []
        for node in nodes_df['node']:
            edges_df_subset = edges_df.query('node1 == @node | node2 == @node')[gsr_type]
            if summary == 'mean':
                stat_summary = edges_df_subset.mean()
            elif summary == 'median':
                stat_summary = edges_df_subset.median()
            summary_stat_list.append(stat_summary)
        nodes_df[f'{prefix}_{gsr_type}'] = summary_stat_list
    return nodes_df


def vis_stats_on_brain(
        nodes_df=None, 
        parcellation='Schaefer',
        save_filename=None,
        fig_size=(12, 5),
        vis_dlabel=False,
        prefix_col_name='',
        fig_direction='horizontal',
        add_fig_suptitle=None,
        cbar_min=0,
        cbar_max=1,
        vmax_rsfc=None,
        filename_fit_list=None,
        hmap_ratio=0.4,
        cbar_ratio=0.6,
        summary_type='medians',
        value_name='',
        n_arrays=72,
        invalid_edge_file_list=None,
        invalid_edge_file_dict=None,
        filename_param_list=None,
        fscores_files=None,
        fc_filename_dict=None,
        edges_validity_df=None,
        param_order_filename=None,
        gsr_label_y=0.925,
        comp_chi2='day',
        filename_fit_list1=None,
        filename_fit_list2=None,
        edges_df=None,
        cmap='Oranges',
        family_cv=True,
        fold=None,
        figure_indicators=['a', 'b'],
        **kwargs
        ):
    """
    Visualize mean chi-square values on brain surface
    kwargs should include invalid edge filenames
    value_name shoule be 'exclusion', 'chi-square values', or ''
    gsr_label_y should be set to 0.875 when suptitle is added
    edges_df is an optional input for drawing heatmap of edges
    """
    parcel_file = PARCEL_FILE_DICT.get(parcellation)
    
    if nodes_df is None:
        if value_name == 'exclusion':
            print('Getting df of nodes.')
            nodes_df = get_invalid_edges_df(invalid_edge_file_list, family_cv=family_cv, fold=fold)
            prefix_col_name = 'prop'
        elif value_name == 'chi-square values':
            nodes_df = calc_chi2_summary_per_node(filename_fit_list=filename_fit_list1, filename_fit_list2=filename_fit_list2, family_cv=family_cv)
        elif value_name in ['reliability', 'random error', 'common consistency', 'occasion specificity']:
            # arguments may be required to be more concised
            name_dict = {'reliability': 'rel', 'random error': 'error', 'common consistency': 'cons', 'occasion specificity': 'spec'}
            rel_type = name_dict.get(value_name)
            print('Getting df of nodes')
            nodes_df = get_summary_rel_lst_df(filename_param_list, param_order_filename, rel_type=rel_type, family_cv=family_cv, fold=fold, **kwargs)
            prefix_col_name = rel_type
        elif value_name == 'validity coefficients':
            nodes_df = get_fscore_validity_nodes_df(edges_validity_df=edges_validity_df, invalid_edge_file_dict=invalid_edge_file_dict)
            fold_dict = {0: 'train', 1: 'valid'}
            prefix_col_name = fold_dict.get(fold)
        nodes_df.rename(columns={'index': 'new_index'}, inplace=True)
    if vis_dlabel:
        dlabel_array = np.squeeze(nib.load(parcel_file).get_fdata())
    if cbar_max is None:
  #      nodes_df.loc[nodes_df[f'{prefix_col_name}_nogs'] >= 1, f'{prefix_col_name}_nogs'] = np.nan
  #      nodes_df.loc[nodes_df[f'{prefix_col_name}_gs'] >= 1, f'{prefix_col_name}_gs'] = np.nan
  #      cbar_max = np.nanmax(nodes_df[[f'{prefix_col_name}_nogs', f'{prefix_col_name}_gs']])
        cbar_max = nodes_df[[f'{prefix_col_name}_nogs', f'{prefix_col_name}_gs']].max().max() if not vis_dlabel else dlabel_array.max()
    nodes_dict = nodes_df.set_index('old_index').to_dict(orient='index')
    cifti_parcel_array = nib.load(parcel_file).get_fdata().copy()
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    bmap_ratio = 1 - hmap_ratio
    cbar_width = bmap_ratio * cbar_ratio
    if fig_direction == 'horizontal':
        left_cbar_position = hmap_ratio + (bmap_ratio - cbar_width) / 2
        nrow, ncol, add_axes_list = 2, 4, [left_cbar_position, 0.01, cbar_width, 0.02]
    elif fig_direction == 'vertical':
        bottom_cbar_position = hmap_ratio + (bmap_ratio - cbar_width) / 2
        nrow, ncol, add_axes_list = 4, 4, [0.95, bottom_cbar_position, 0.015, cbar_width]
    
    fig = plt.figure(layout='constrained', figsize=fig_size)
    if fig_direction == 'horizontal':
        subfigs = fig.subfigures(1, 2, width_ratios=[hmap_ratio, 1-hmap_ratio])
    elif fig_direction == 'vertical':
        subfigs = fig.subfigures(2, 1, height_ratios=[hmap_ratio, 1-hmap_ratio])

    axes_bmap = subfigs[1].subplots(
            nrow, ncol, 
           # layout='constrained', 
            subplot_kw={'projection':'3d'},
           # figsize=fig_size
            )
    cbar_ax = fig.add_axes(add_axes_list)
    
    if fig_direction == 'horizontal':
        axes_hmap = subfigs[0].subplots()
    elif fig_direction == 'vertical':
        axes_hmap = subfigs[1].subplots(1, 2)

    input_data_dict = {}
    for gsr_type in ['nogs', 'gs']:
        # create statistics array
        stat_array = np.empty(shape=cifti_parcel_array.shape[1])
        stat_array[:] = np.nan
        # insert values to chi-square array
        for j in nodes_dict.keys():
            # get chi-square value in old index
            stats = nodes_dict.get(j).get(f'{prefix_col_name}_{gsr_type}')
            stat_array[np.squeeze(cifti_parcel_array == j+1)] = stats
        # subcortex is filled with white in heatmap
        stat_array = np.nan_to_num(stat_array)
        half_n_vertices = int(len(hcp.mesh.inflated[0])/2)
        input_data = stat_array if not vis_dlabel else dlabel_array
        input_data_dict[gsr_type] = input_data
    # get max value to determine vmax
    if cbar_max is None:
        cbar_max = 0
        for key in input_data_dict.keys():
            max_value = np.nanmax(input_data_dict.get(key))
            if max_value > cbar_max:
                cbar_max = max_value
    # get min value to determine vmin
    if cbar_min is None:
        cbar_min = 0
        for key in input_data_dict.keys():
            min_value = np.nanmin(input_data_dict.get(key))
            if min_value < cbar_min:
                cbar_min = min_value

    for k, gsr_type in enumerate(['nogs', 'gs']):
        input_data = input_data_dict.get(gsr_type)
        for i, direction in enumerate(['left', 'right']):
            if direction == 'left':
                func = hcp.left_cortex_data
                mesh = hcp.mesh.inflated_left
            elif direction == 'right':
                func = hcp.right_cortex_data
                mesh = hcp.mesh.inflated_right
            for j, view in enumerate(['lateral', "medial"]):
                print(f'Ploting {direction} {view} in {gsr_type} condition.')
                if fig_direction == 'horizontal':
                    target_ax = axes_bmap[j, i+2*k]
                elif fig_direction == 'vertical':
                    target_ax = axes_bmap[j+2*k, i]
                surf = plot_surf_stat_map(
                        mesh,
                        func(input_data), 
                        hemi=direction, 
                        view=view,
                        colorbar=False,
                        cmap=cmap,
                        vmin=cbar_min,
                        vmax=cbar_max,
                        axes=target_ax
                        )
    mappable = mpl.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(cbar_min, cbar_max)
    if fig_direction == 'horizontal':
        orientation = 'horizontal'
    elif fig_direction == 'vertical':
        orientation = None
    fig.colorbar(
            mappable,
            cax=cbar_ax,
            orientation=orientation
            )
    if fig_direction == 'horizontal':
        x1, x2 = hmap_ratio + bmap_ratio * 0.25, hmap_ratio + bmap_ratio * 0.75
        for x, gsr_str in zip([x1, x2], ['Without GSR', 'With GSR']):
            plt.figtext(x=x, y=gsr_label_y, s=gsr_str, horizontalalignment='center', verticalalignment='top', fontsize=16)
        if value_name == 'omega coefficients':
            print('Drawing heatmap of edges.')
            calc_omega_per_network(
                    get_latest_nogs_gs_files(True, False, False, combined=True, parcellation=parcellation),
                    ax=axes_hmap,
                    parcellation=parcellation,
                    n_arrays=n_arrays,
                    **kwargs
                    )
        elif value_name == 'chi-square values':
            if filename_fit_list1 is None and filename_fit_list2 is None:
                if comp_chi2 == 'day':
                    filename_fit_list1 = get_latest_nogs_gs_files(False, False, False, combined=True, parcellation=parcellation)
                    filename_fit_list2 = get_latest_nogs_gs_files(True, False, False, combined=True, parcellation=parcellation)
                elif comp_chi2 == 'PE':
                    filename_fit_list1 = get_latest_nogs_gs_files(True, False, False, combined=True, parcellation=parcellation)
                    filename_fit_list2 = get_latest_nogs_gs_files(True, True, False, combined=True, parcellation=parcellation)
                elif comp_chi2 == 'order':
                    filename_fit_list1 = get_latest_nogs_gs_files(True, False, False, combined=True, parcellation=parcellation)
                    filename_fit_list2 = get_latest_nogs_gs_files(True, False, True, combined=True, parcellation=parcellation)
            print('Drawing heatmap of edges.')
            visualize_chi_squares(
                filename_fit_list=filename_fit_list1,
                filename_fit_list2=filename_fit_list2,
                parcellation=parcellation,
                ax=axes_hmap,
                vmax=vmax_rsfc
                #**kwargs
                )
        elif value_name == 'exclusion':
            print('Drawing heatmap of edges.')
            draw_hmaps_invalid_edges(
                invalid_edge_file_list,
                parcellation=parcellation,
                ax=axes_hmap,
                family_cv=family_cv,
                fold=fold,
                num_iter=1
                )
        elif value_name in ['reliability', 'random error', 'common consistency', 'occasion specificity']:
            print('Drawing heatmap of edges.')
            calc_rel_lst_network(
                filename_param_list=filename_param_list,
                param_order_filename=param_order_filename,
                hmap_vmin=cbar_min,
                hmap_vmax=cbar_max,
                rel_type=rel_type,
                family_cv=family_cv,
                fold=fold,
                **kwargs,
                ax=axes_hmap
                    )
        elif value_name == 'validity coefficients':
            fold_dict = {0: 'train', 1: 'valid'}
            mat_df = get_empty_df_for_hmap(parcellation)
            nodes_df = get_nodes_df(parcellation)
            for gsr_type in ['nogs', 'gs']:
                wide_df = get_wide_df_hmap(edges_validity_df, value_col_name=f"{gsr_type}_{fold_dict.get(fold)}_validity")
                mat_df = fill_mat_df(mat_df, wide_df, gsr_type)
            draw_hmaps_fcs(
                    mat_df, nodes_df, cmap=cmap, ax=axes_hmap, vmin=cbar_min, vmax=cbar_max, parcellation=parcellation
                    )

        elif ('both' in value_name) or ('mean' in value_name) or ('diff' in value_name):
            print('Drawing heatmap of edges.')
            draw_hmaps_rsfc_trait_associaiton(
                edges_df,
                outcome_type=value_name,
                parcellation=parcellation,
                save_filename=save_filename,
                cmap=cmap,
                ax=axes_hmap,
                num_iter=1,
                vmin=cbar_min,
                vmax=cbar_max
                )
    elif fig_direction == 'vertical':
        for y, gsr_str in zip([0.95, 0.45], ['Without GSR', 'With GSR']):
            plt.figtext(x=0.5, y=y, s=gsr_str, horizontalalignment='center', verticalalignment='top', fontsize=16)
    if value_name == 'chi-square values':
        value_name = "$\\Delta\\chi^2$ values"
    else:
        value_name = value_name.capitalize()
    figure_indicator1, figure_indicator2 = figure_indicators[0], figure_indicators[1]
    subfigs[0].suptitle(rf"$\mathbf{{{figure_indicator1}}}$ " + value_name + ' of edges', fontsize=16)
    subfigs[1].suptitle(rf"$\mathbf{{{figure_indicator2}}}$" + f' {summary_type.capitalize()} of ' + value_name + ' by nodes', fontsize=16)
    
    if add_fig_suptitle:
        fig.suptitle(add_fig_suptitle, fontsize=20)
    if save_filename:
        print(f'Saving file.')
        atlas_dir = ATLAS_DIR_DICT.get(parcellation)
        fig.savefig(op.join(atlas_dir, 'reliability', 'figures', f'{save_filename}.png'), bbox_inches='tight')
        print('Saving completed.')


def draw_hmaps_rsfc_trait_associaiton(
    edges_df,
    outcome_type='both_abs',
    parcellation='Schaefer',
    save_filename=None,
    cmap="Oranges",
    ax=None,
    num_iter=None,
    vmin=None,
    vmax=None
):
    """
    edges_df may includes columns of values with and without GSR
    outcome_type may be 'both_abs', 'mean_abs', or 'diff_abs'
    Draw heatmap representing invalid or valid edges
    """
#    edges_df = get_edge_summary(parcellation, network_hem_order=True)
#    edges_df.sort_values("edge", inplace=True)
    nodes_df = get_nodes_df(parcellation)
    mat_df = get_empty_df_for_hmap(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)

    for gsr_type in ['_nogs', '_gs']:
        columns = ['node1', 'node2'] + [i for i in edges_df.columns if gsr_type in i and outcome_type in i]
        edges_df_subset =  edges_df[columns]
        wide_edges_df = get_wide_df_hmap(edges_df_subset, value_col_name=f'{outcome_type}{gsr_type}')
        mat_df = fill_mat_df(mat_df, wide_edges_df, gsr_type.replace('_', ''))
    draw_hmaps_fcs(
        mat_df, 
        nodes_df, 
        cmap=cmap, 
    #    save_filename=save_filename, 
        cbar_ax=False, 
        ax=ax, 
        num_iter=num_iter,
        parcellation=parcellation,
        add_custom_cbar=False,
        vmin=vmin,
        vmax=vmax
    )


def visualize_chi_squares(
        filename_fit_list,
        filename_fit_list2=None,
        parcellation='Schaefer',
        vmax=None,
        return_df=False,
        save_filename=None,
        rotate_x_deg=45,
        fig_size=None,
        ax=None
        ):
    """
    Draw heatmaps representing chi-square values
    """
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    nodes_df = get_nodes_df(parcellation)
    
    def create_mat_df(filename_fit_list):
        concat_df = pd.DataFrame()
        for i, gsr_type in enumerate(['nogs', 'gs']):
            chi2 = conduct_chi_square_diff_test(filename_fit_list[i], parcellation=parcellation)
            inner_df = pd.DataFrame()
            inner_df['chi2'] = chi2
            inner_df['gsr_type'] = gsr_type
            inner_df['edge'] = [i for i in range(len(chi2))]
            concat_df = pd.concat([concat_df, inner_df], axis=0)
 
        mat_df = get_empty_df_for_hmap(parcellation)
    
        for gsr_type in concat_df["gsr_type"].unique():
            df = concat_df.query("gsr_type == @gsr_type")
            df = pd.merge(df, edges_df, on='edge')
            wide_df = get_wide_df_hmap(df, value_col_name="chi2")
            mat_df = fill_mat_df(mat_df, wide_df, gsr_type)
        return mat_df
    
    mat_df = create_mat_df(filename_fit_list)

    if filename_fit_list2:
        mat_df2 = create_mat_df(filename_fit_list2)
        mat_df = mat_df - mat_df2
    if return_df:
        return mat_df
    if ax is None:
        if fig_size is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=fig_size)
    if vmax is None:
        vmax = mat_df.max().max()
    draw_hmaps_fcs(
        mat_df,
        nodes_df,
        cmap="Oranges",
        #save_filename=save_filename,
        vmin=0,
        vmax=vmax,
        ax=ax,
        num_iter=1,
        parcellation=parcellation,
        rotate_x_deg=rotate_x_deg
    )
    if save_filename:
        fig.tight_layout()
        atlas_dir = ATLAS_DIR_DICT.get(parcellation)
        fig.savefig(op.join(atlas_dir, 'reliability', "figures", f"{save_filename}.png"))


def draw_hmap_on_prop_of_chisq_diff_test(
    filename_fit_list: list[str, str],
    #    p_value: float,
    filename_fit_list2: Optional[list[str, str]] = None,
    parcellation='Schaefer',
    invalid_edge_file_list_dict=None,
    laterality=False,
    ax=None,
    cbar_ax=None,
    iteration=None,
    save_fig=False,
    save_filename=None,
    vmin=0,
    vmax=320,
    num_iter=None
):
    """
    Draw heatmap on results of chi-square difference test
    which examine improvement by adding error terms of day correlation and/or phase encoding directions
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    node_summary_path = NODE_SUMMARY_PATH_DICT.get(parcellation)

    def conduct_chisq_diff_test():
        chisq_dict, ps_chisq_dict = defaultdict(dict), defaultdict(dict)
        for i, gsr_type in enumerate(["nogs", "gs"]):
            ps_chisq, chisq = conduct_chi_square_diff_test(
                filename_fit_list[i], filename_fit_list2[i]
            )
            chisq_dict[gsr_type], ps_chisq_dict[gsr_type] = chisq[:, 0], ps_chisq
        return chisq_dict, ps_chisq_dict

    def add_chisq_and_remove_misfit_in_edges_df():
        concat_edges_df = pd.DataFrame()
        for i, gsr_type in enumerate(["nogs", "gs"]):
            misfit_edges_1 = np.loadtxt(
                op.join(
                    atlas_dir,
                    "reliability",
                    "invalid_edges",
                    invalid_edge_file_list_dict.get(gsr_type)[0],
                )
            ).astype(int)
            misfit_edges_2 = np.loadtxt(
                op.join(
                    atlas_dir,
                    "reliability",
                    "invalid_edges",
                    invalid_edge_file_list_dict.get(gsr_type)[1],
                )
            ).astype(int)
            misfit_edges = set(misfit_edges_1) | set(misfit_edges_2)
            inner_edges_df = get_edge_summary(network_hem_order=True)
            inner_edges_df["chisq"] = chisq_dict[gsr_type]
            edges_df_subset = inner_edges_df.query("edge not in @misfit_edges")
            edges_df_subset["gsr_type"] = gsr_type
            concat_edges_df = pd.concat([concat_edges_df, edges_df_subset], axis=0)
        return concat_edges_df

    def summarise_chi_square():
        summary_chisq_dict = defaultdict(dict)
        for gsr_type in ["nogs", "gs"]:
            edges_df_chisq = get_summary_of_edges_per_network(
                concat_edges_df.query("gsr_type == @gsr_type"), "chisq", "median"
            )
            edges_df_chisq = rename_net_summary_df(edges_df_chisq)
            summary_chisq_dict[gsr_type] = edges_df_chisq
        return summary_chisq_dict

    # get df of edges
    edges_df = get_edge_summary(parcellation, network_hem_order=True)
    # get chi-square and associated p values
    chisq_dict, ps_chisq_dict = conduct_chisq_diff_test()
    # add chi-square values to edges_df
    concat_edges_df = add_chisq_and_remove_misfit_in_edges_df()
    mat_df = get_empty_df_for_hmap(parcellation)
    nodes_df = get_nodes_df(node_summary_path)
    
    for gsr_type in concat_edges_df["gsr_type"].unique():
        df = concat_edges_df.query("gsr_type == @gsr_type")
        df = pd.merge(edges_df, df[["edge", "chisq"]], how="left", on="edge")
        wide_df = get_wide_df_hmap(df, value_col_name="chisq")
        mat_df = fill_mat_df(mat_df, wide_df, gsr_type)

    draw_hmaps_fcs(
        mat_df,
        nodes_df,
        cmap="Oranges",
    #    save_filename=save_filename,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_ax=cbar_ax,
        iteration=iteration,
        num_iter=num_iter
    )

    # get summary of chi-square values
    # summary_chisq_dict = summarise_chi_square()


# sns.heatmap(prop_edges_df_remove, annot=True, fmt='.1%', ax=axes[0])
# axes[0].set_title('Percentage of removal edges')
# display(prop_edges_df_remove)

# draw heatmap of percentage of improved edges determined by chi-square difference test after removing misfit edges
#    fit_edges = np.where(ps_chisq < p_value)[0]
#    improve_edges_df = edges_df.reset_index(drop=True).loc[fit_edges]
#    improve_edges_df.query('edge not in @model_out_edge', inplace=True)
#    nodes_df = pd.read_csv(node_summary_path, index_col=0)
#    if subcortex_removed:
#        edges_df.query('~net_set2.str.contains("Limbic")', inplace=True)
#        nodes_df.query('~net.str.contains("Limbic")', inplace=True)
#        nodes_df.reset_index(drop=True, inplace=True)
#    mapping_dict = {node: key for node, key in zip(nodes_df['node'], nodes_df['Key'].index)}
#    edges_df['i'] = improve_edges_df['node1'].map(mapping_dict)
#    edges_df['j'] = improve_edges_df['node2'].map(mapping_dict)
#
#    def get_prop_edges(edges_df, improve_edges_df, laterality, axes, i=None, lateral=None):
#        edges_count_wide_df = get_counts_of_edges_per_network(edges_df)
#        imp_edges_count_wide_df = get_counts_of_edges_per_network(improve_edges_df)
#        prop_edges_df_imp = imp_edges_count_wide_df / edges_count_wide_df
#        # draw heatmap
#        axes_num = i + 1 if i is not None else 1
#        prop_edges_df_imp = rename_net_summary_df(prop_edges_df_imp)
#        sns.heatmap(prop_edges_df_imp, annot=True, fmt='.1%', ax=axes[axes_num])
#        suffix = f'with {lateral}' if laterality else ''
#        ax_title1 = f'Percentage of improved edges {suffix}'
#        axes[axes_num].set_title(ax_title1)
#        # plot edges
#        edges_df_selected = edges_df[['i', 'j']].dropna(axis=0).astype(int).reset_index(drop=True)
#        netplotbrain.plot(
#          #  template='MNI152NLin6Asym',
#            nodes=nodes_df[['x', 'y', 'z']].reset_index(drop=True),
#            edges=edges_df_selected,
#            view=['LSR', 'AIP']
#            )
#        # draw heatmap of mean chi-square values
#        # calculate mean chi-square values from dataframe where misfit edges are removed
# chisq_dict[gsr_type] = edges_ddf_chisq
#
#        if not save_fig:
#            axes_num2 = 4 + i if i is not None else 2
#            ax = axes[axes_num2]
#        else:
#            fig_chisq_diff, ax = plt.subplots(2, 1, figsize=(16, 4))
#        for
#        sns.heatmap(edges_df_chisq, annot=True, fmt='.1f', ax=ax)
#
#        if save_fig:
#            fig_folder = op.join(SCHAEFER_DIR, 'reliability', 'figures')
#            thresholds_suffix = generate_fig_name_suffix_from_thresholds(
#                    error_vars_dict,
#                    fit_indices_thresholds_dict,
#                    cor_min_max_dict,
#                )
#            ax.tick_params(axis='x', rotation=0)
#            model_comp_day_cor = '_compDaycor' if not day_cor and day_cor2 else ''
#            model_comp_pe = '_compPE' if not phase_encoding and phase_encoding2 else ''
#            model_add_day_cor = '_addDaycor' if day_cor and day_cor2 else ''
#            model_add_pe = '_addPE' if phase_encoding and phase_encoding2 else ''
#            if gsr_type == gsr_type2:
#                gsr_suffix = '_' + gsr_type
#            else:
#                raise GSRTypeError('GSR types are not same.')
#            model_str = model_comp_day_cor + model_comp_pe + model_add_day_cor + model_add_pe + gsr_suffix
#            fig_name = f"median_chisq_diff_{model_str}_{thresholds_suffix}.png"
#            #extent = axes[axes_num2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#            #fig.savefig('ax2_figure.png', bbox_inches=extent)
#            # Pad the saved area by 10% in the x-direction and 20% in the y-direction
#            fig_chisq_diff.tight_layout()
#            fig_chisq_diff.savefig(op.join(fig_folder, fig_name), bbox_inches='tight')
#            #axes[axes_num2].savefig(op.join(fig_folder, fig_name))
#        else:
#            ax_title2 = f'Median of chi-square values {suffix}'
#            axes[axes_num2].set_title(ax_title2)
#        display(edges_df_chisq)
#
#    if laterality:
#        for i, lateral in enumerate(['ipsi_right', 'ipsi_left', 'contra']):
#            # subset data from laterality
#            edges_df_lateral = edges_df[edges_df[lateral]]
#            improve_edges_df_lateral = improve_edges_df[improve_edges_df[lateral]]
#            draw_heatmaps_of_prop_edges(edges_df_lateral, improve_edges_df_lateral, laterality, axes, i=i, lateral=lateral)
#    else:
#        draw_heatmaps_of_prop_edges(edges_df, improve_edges_df, laterality, axes, i=None)
#
#    if laterality:
#        imp_edges_count_wide_df = get_counts_of_edges_per_network(improve_edges_df)
#        prop_edges_df_imp = imp_edges_count_wide_df / edges_count_wide_df
#        prop_edges_df_imp = rename_net_summary_df(prop_edges_df_imp)
#        sns.heatmap(prop_edges_df_imp, annot=True, fmt='.1%', ax=axes[-1])
#        axes[-1].set_title('Percentage of improved edges')
#
#    return edges_df


def conduct_chi_square_diff_test(
        filename_fit: str, 
        filename_fit2:str=None,
        parcellation='Schaefer',
        family_cv=True
        ):
    """
    function for conducting chi-square differnce test
    filename_fit2 should be nested within filename_fit1
    """
    # create indices
    chi2_index = FIT_INDICES.index("chi2")
    df_index = FIT_INDICES.index("DoF")
    chi2_df_list = [chi2_index, df_index]
    # copy fit data
    fit = copy_memmap_output_data(filename_fit, parcellation=parcellation, family_cv=family_cv)
    # get data of chi-square and p values
    chi2_df = fit.take(chi2_df_list, axis=1)
    # calculate difference of chi-square
    if filename_fit2 is None:
        return chi2_df[..., 0]
        # calculate p values
    else:
        fit2 = copy_memmap_output_data(filename_fit2, parcellation=parcellation)
        chi2_df2 = fit2.take(chi2_df_list, axis=1)
        chi2_diff = chi2_df - chi2_df2
        ps = 1 - chi2.cdf(chi2_diff[:, 0], chi2_diff[:, 1])
        return ps, chi2_diff


def compare_mean_fs_cor(filename_fs: str):
    """
    function for comparing correlation calculated from mean and factor scores
    """
    fs_array = copy_memmap_output_data(filename_fs)


def calculate_chi_squared_df_ratio(filename_fit, model_type_list):
    """
    function for visualising ratio of chi-square value and df
    """
    fit_array = copy_memmap_output_data(filename_fit)
    edge_n = get_strings_from_filename(filename_fit, ["edge_n"])
    chi2_position = FIT_INDICES.index("chi2")
    df_position = FIT_INDICES.index("DoF")
    (
        model_fa_list,
        _,
        _,
    ) = get_model_strings(filename_fit)
    chi_df_ratio = fit_array[:, chi2_position, :] / fit_array[:, df_position, :]


def get_dict_of_latest_filenames_by_gsr(trait_type, **kwargs):
    """
    get latest filenames considering gsr type
    """
    filename_list = get_latest_files_with_args(
        [trait_type],
        kwargs["n_edge"],
        kwargs["sample_n"],
        kwargs["est_method"],
        kwargs["data_type"],
        kwargs["model_type"],
        kwargs["drop_vars_list"],
    )
    # select folder
    folder = get_scale_name_from_trait(trait_type)
    # select scale names
    scale_name_list = get_subscale_list(folder)
    # create output dictionary
    out_dict = {}
    for gsr_type in ["nogs", "gs"]:
        out_dict[gsr_type] = [
            filename for filename in filename_list if f"_{gsr_type}_" in filename
        ]
    return out_dict


def check_input_dict_gsr_types(filename_list_dict: dict[str : list[str]]):
    """
    get filenames from input dictionary by gsr types
    """
    n_scales_nogs, n_scales_gs = len(filename_list_dict.get("nogs")), len(
        filename_list_dict.get("gs")
    )
    if n_scales_nogs == n_scales_gs:
        n_col = n_scales_nogs
        return n_col
    else:
        raise Exception(
            "Length of lists of input dictionary is different between nogs and gs."
        )


def get_df_cor_of_mean_pca(filename_cor_list_dict):
    """
    get results of analyses using mean and pca
    """
    output_df = pd.DataFrame()
    for g, gsr_type in enumerate(["nogs", "gs"]):
        gsr_suffix = generate_gsr_suffix(gsr_type)
        for col, scale_name in enumerate(filename_cor_list_dict.get(gsr_type)):
            for drop in ["not_dropped", "dropped"]:
                filename_cor = (
                    filename_cor_list_dict.get(gsr_type).get(scale_name).get(drop)
                )
                if filename_cor is not None:
                    cor_data = copy_memmap_output_data(filename_cor)[:, :2]
                    cor_df = array_to_df(
                        cor_data,
                        ["mean", "pca"],
                        **{
                            "scale_name": scale_name,
                            "gsr_type": gsr_type,
                            "drop": drop,
                        },
                    )
                    output_df = pd.concat([output_df, cor_df], axis=0)
    long_df = output_df.melt(
        id_vars=["edge", "scale_name", "gsr_type", "drop"],
        value_vars=["mean", "pca"],
        value_name="Correlation",
        var_name="model",
    )
    return long_df


def array_to_df(array, column_names, **kwargs):
    """
    convert numpy array to pandas dataframe in for loop
    """
    df = pd.DataFrame(array, columns=column_names)
    df = df.reset_index().rename(columns={"index": "edge"})
    for key, value in kwargs.items():
        df[key] = value
    return df


def get_df_cor_of_models(
    filename_cor_list_dict,
    remove_edge_file_dict,
    fit_indices_thresholds_dict=None,
    error_vars_dict=None,
    cor_min_max_dict=None,
):
    """
    Draw histograms reprsenting correlations.
    """
    # check input dictionary
    n_col = check_input_dict_gsr_types(filename_cor_list_dict)
    output_df = pd.DataFrame()
    filename_cor_list = []
    for g, gsr_type in enumerate(["nogs", "gs"]):
        gsr_suffix = generate_gsr_suffix(gsr_type)
        removed_edges = np.loadtxt(
            op.join(
                SCHAEFER_DIR,
                "reliability",
                "invalid_edges",
                remove_edge_file_dict.get(gsr_type),
            )
        ).astype(int)
        for col, scale_name in enumerate(filename_cor_list_dict.get(gsr_type)):
            # generate combined dataframe using dropped variables

            for drop in ["not_dropped", "dropped"]:
                filename_cor = (
                    filename_cor_list_dict.get(gsr_type).get(scale_name).get(drop)
                )
                if filename_cor is not None:
                    filename_cor_list.append(filename_cor)
                    print(filename_cor)
                    (
                        model_fa_list,
                        trait_type,
                        scale_name,
                        gsr_type,
                        num_iter,
                        phase_encoding,
                        day_cor,
                        drop_vars_list,
                    ) = get_strings_from_filename(
                        filename_cor,
                        [
                            "model_type",
                            "trait_type",
                            "scale_name",
                            "gsr_type",
                            "num_iter",
                            "phase_encoding",
                            "day_cor",
                            "drop_vars_list",
                        ],
                        include_nofa_model=False,
                    )
                    # misfit_set = get_set_of_locally_globlly_misfit_edges(
                    #    filename_cor,
                    #    error_vars_dict,
                    #    fit_indices_thresholds_dict,
                    #    model_fa_list,
                    #    cor_min_max_dict
                    #    )
                    cor_array = copy_memmap_output_data(filename_cor)
                    cor_array = np.delete(cor_array, removed_edges, axis=0)
                    cor_pd = array_to_df(
                        cor_array,
                        model_fa_list,
                        **{
                            "scale_name": scale_name,
                            "gsr_type": gsr_type,
                            "drop": drop,
                        },
                    )
                    output_df = pd.concat([output_df, cor_pd], axis=0)
    # make a long format data
    long_df = output_df.melt(
        id_vars=["edge", "scale_name", "gsr_type", "drop"],
        value_vars=model_fa_list,
        var_name="model",
        value_name="Correlation",
    )
    return long_df


#def combine_files():
#    """
#    Combine output files of slurm job arrays
#    """
#    folder = get_scale_name_from_trait(trait_type)
#    subscales = get_subscale_list(get_scale_name_from_trait(trait_type))
#    scale_name_list = get_subscale_list(folder)
#    file_dict_x, file_dict_y = defaultdict(dict), defaultdict(dict)
#
#    for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
#        gsr_type_key_str = gsr_type.replace("_", "")
#        gsr_suffix = "without GSR" if gsr_type == "_nogs_" else "with GSR"
#
#        for i, scale_name in enumerate(scale_name_list):
#            print(gsr_type, scale_name)
#            file_folder = "/".join(
#                filter(None, (SCHAEFER_DIR, folder, scale_name, "correlation"))
#            )
#            file_dict_x[gsr_type_key_str][scale_name] = defaultdict(dict)
#            file_dict_y[gsr_type_key_str][scale_name] = defaultdict(dict)
#
#            for drop in drop_bool:
#                filename_list_x = [
#                    for i in os.listdir(file_folder)
#                    if (str(kwargs.get("n_edge").get(gsr_type_key_str)) in i)
#                    and (gsr_type in i)
#                    and (str(kwargs.get("sample_n")) in i)
#                    and ("mean" in i)
#                ]


def combine_array_files_dat(
    filename_list, 
    n_arrays, 
    data_type, 
    trait_type_for_fscores=None,
    return_filename=False,
    parcellation='Schaefer',
    msst=True,
    bi_factor=False,
    search_trash=False,
    **kwargs
):
    """
    Combine data of filenames in a list
    """
    if data_type in ['cor', 'correlation']:
        array_dim = (0, 1)
    elif 'fit' in data_type:
        array_dim = (0, 15)
    elif "parameter" in data_type:
        filename_sample = sort_list_by_time(filename_list)[-1]
        param_num, var_num = get_param_num_from_filename(filename_sample)
        ncol = 5
        array_dim = (0, param_num, ncol)
    elif data_type == "factor_scores":
        fscores_ndim2 = 1 if not trait_type_for_fscores else 2
        if msst or bi_factor:
            fscores_ndim2 += 2
        sample_n = get_strings_from_filename(filename_list[0], ["sample_n"])
        array_dim = (0, int(sample_n[0]), fscores_ndim2)
    elif data_type in ['residual', 'model_vcov']:
        filename_sample = sort_list_by_time(filename_list)[-1]
        _, var_num = get_param_num_from_filename(filename_sample)
        array_dim = (0, var_num, var_num)

    array_all = np.array([]).reshape(array_dim)
    if len(filename_list) >= 1:
        array_suffix_list = ["_" + str(i) + ".dat" for i in range(n_arrays)]
        arrays_filename_dict = defaultdict(dict)

        for array_suffix in array_suffix_list:
            filename_list_array = [i for i in filename_list if array_suffix in i]
#            try:
                # get latest file
            if len(filename_list_array) >= 1:
                arrays_filename_dict[array_suffix] = sort_list_by_time(
                    filename_list_array
                )[-1]
           # except:
           #     set_trace()
            else:
                raise Exception('There does not exist filename.')
        # should check the same suffixes are included in filenames
        filename_common = reduce(
            add,
            re.findall("(.*)_\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", arrays_filename_dict.get('_0.dat'))
        )
        for filename_value in arrays_filename_dict.values():
            filename_suffix = reduce(
                add,
                re.findall("(.*)_\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", filename_value)
            )
            if filename_common != filename_suffix:
                set_trace()
                raise Exception('Filenames mismatch between files.')
            array_data = copy_memmap_output_data(
                    filename_value, 
                    n_array=n_arrays, 
                    parcellation=parcellation,
                    search_trash=search_trash
                    )
            if "fit" in data_type:
                array_data = np.squeeze(array_data)
            array_all = np.concatenate([array_all, array_data], axis=0)
    
    first_filename = arrays_filename_dict.get('_0.dat')
    combined_filename = first_filename.replace('_0.dat', '_combined.npy')
    
    if return_filename:
        return array_all, combined_filename
    return array_all


def get_additionally_excluded_edges(
        param_order, 
        params, 
        control,
        fc_1st_load_thres_list,
        fc_2nd_load_thres_list,
        trait_load_thres_list
        ):
    """
    Select removval edges from parameter estimates
    """
    sessions = [f's{i}' for i in range(1, 5)]
    days = ['o1', 'o2']
    fc_1st_load_positions = param_order.query('(rhs.isin(@sessions)) & (op == "=~")').index
    fc_1st_load_params = params[:, fc_1st_load_positions, 0]
    fc_2nd_load_positions = param_order.query('(rhs.isin(@days)) & (op == "=~")').index
    fc_2nd_load_params = params[:, fc_2nd_load_positions, 0]
    trait_load_positions = param_order.query('(lhs == "tf") & (op == "=~")').index
    trait_load_params = params[:, trait_load_positions, 0]
    cor_position = param_order.query('(lhs == "tf") & (rhs == "ff")').index
    # check parameters
    fc_1st_error_positions = param_order.query('(rhs.isin(@sessions)) & (op == "~~") & (lhs.isin(@sessions))').index
    if control is not None:
        control_query_rhs, control_query_lhs = '& (~rhs.isin(@control))', '& (~lhs.isin(@control))'
    else:
        control_query_rhs, control_query_lhs = '', ''
    trait_error_positions = param_order.query(f'(~rhs.isin(@sessions)) {control_query_rhs} & (op == "~~") & (~lhs.isin(@sessions)) {control_query_lhs}').index
    fc_1st_error_params = params[:, fc_1st_error_positions, 0]
    trait_error_params = params[:, trait_error_positions, 0]
    if not np.allclose(fc_1st_error_params, 1 - fc_1st_load_params**2, equal_nan=True):
        raise Exception('Somethind wrong might happen in .npy parameter file.')
    #cor_remove_index = np.where(np.abs(params[:, cor_position, 0]) > cor_thres)[0]
    
    def get_exclude_edges_from_param_thres(input_array, thres_min_max_list):
        min_thres, max_thres = thres_min_max_list[0], thres_min_max_list[1]
        if min_thres >= max_thres:
            raise ValueError('Minimum of thresholds is greater than maximum of thresholds.')
        booleans = (input_array < min_thres) | (input_array > max_thres)
        booleans = np.any(booleans, axis=1)
        return booleans

    fc_1st_remove_booleans = get_exclude_edges_from_param_thres(fc_1st_load_params, fc_1st_load_thres_list)
    fc_2nd_remove_booleans = get_exclude_edges_from_param_thres(fc_2nd_load_params, fc_2nd_load_thres_list)
    trait_remove_booleans = get_exclude_edges_from_param_thres(trait_load_params, trait_load_thres_list)

    remove_booleans = fc_1st_remove_booleans | fc_2nd_remove_booleans | trait_remove_booleans
    remove_edges = np.where(remove_booleans)[0]
    print(f'Number of excluded edges\nFC 1st loadings: {fc_1st_remove_booleans.sum()}\nFC 2nd loadings: {fc_2nd_remove_booleans.sum()}\nTrait loadings: {trait_remove_booleans.sum()}')
    return remove_edges


def generate_summary_df(
    trait_type,
    control=["age", "gender", "MeanRMS"],
    return_long=False,
    drop_item_dict={},
    parcellation='Schaefer',
    full_or_mean='mean',
    msst=True,
    trait_equal_loadings=True,
    split_half_family=True,
    fold=0,
    invalid_edge_file_dict={},
    sample_n=None,
    skip_scale=None,
#    cor_thres=0.3,
    fc_1st_load_thres_list=[0.4, 0.9],
    fc_2nd_load_thres_list=[0.5, 1],
    trait_load_thres_list=[0.2, 1],
    **kwargs,
):
    """
    Generate summary df of r
    """
    folder = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(get_scale_name_from_trait(trait_type))
    scale_name_list = get_subscale_list(folder)
    n_rsfc = N_RSFC_DICT.get(parcellation)
    if skip_scale is not None:
        scale_name_list = [i for i in scale_name_list if i != skip_scale]
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    output_df = pd.DataFrame()
    if fold is not None:
        fold_str = f'Fold_{fold}'
    else:
        fold_str = ''
    invalid_edges_dir = op.join(atlas_dir, "reliability", "invalid_edges")
    if split_half_family:
        invalid_edges_dir = op.join(invalid_edges_dir, fold_str)

    for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
        gsr_type_key_str = gsr_type.replace("_", "")
        gsr_suffix = "without GSR" if gsr_type == "_nogs_" else "with GSR"
        invalid_edge_filename = invalid_edge_file_dict.get(gsr_type_key_str)
#        if not include_no_method_effect:
         #   if invalid_edge_filename is not None:
        if invalid_edge_filename is not None:
            invalid_edge_file = op.join(invalid_edges_dir, invalid_edge_filename)
            invalid_edges = np.loadtxt(invalid_edge_file).astype(int)
 #           else:
 #               invalid_edges = [None]
#        else:
#            if invalid_edge_filenames is not None:
#                invalid_edges1 = np.loadtxt(
#                    op.join(invalid_edges_dir, invalid_edge_filenames[0])
#                ).astype(int)
#                invalid_edges2 = np.loadtxt(
#                    op.join(invalid_edges_dir, invalid_edge_filenames[1])
#                ).astype(int)
#                invalid_edges = set(invalid_edges1) | set(invalid_edges2)
#            else:
#                invalid_edges = [None]

        for i, scale_name in enumerate(scale_name_list):
            # get additional invalid edges
            params_dir = op.join(atlas_dir, folder, scale_name, 'parameters', 'combined')
            param_order_dir = op.join(FA_PARAMS_DIR, folder)
            file_folder = "/".join(
                filter(None, (atlas_dir, folder, scale_name, "correlation", 'combined'))
            )
#            if scale_name == 'Crystal':
#                continue
            if split_half_family:
                params_dir = op.join(params_dir, 'split_half_cv')
                file_folder = op.join(file_folder, 'split_half_cv')
#            # get filename of parameters
#            param_filenames = [
#                    i for i in os.listdir(param_dir)
#                    if gsr_type in i
#                    ]

            def select_filename_sem(filename_list):
                filename_list_y = [
                    i
                    for i in filename_list
                    if ("Model_both" in i)
                ]
                if msst:
                    filename_list_y = [i for i in filename_list_y if 'MultiStateSingleTrait' in i or 'MSST' in i]
                else:
                    filename_list_y = [i for i in filename_list_y if not 'MultiStateSingleTrait' in i and not 'MSST' in i]

                if trait_equal_loadings:
                    filename_list_y = [i for i in filename_list_y if 'TEL' in i]
                else:
                    filename_list_y = [i for i in filename_list_y if not 'TEL' in i]
                return filename_list_y

            def select_elements_in_list(file_folder, control, gsr_type=None, sem=True, param_order=False, **kwrags):
                """
                Inner function for selecting elements in a list
                """
                filename_list =  [
                    i
                    for i in os.listdir(file_folder)
                    if all(j in i for j in control)
                    ]
                if param_order:
                    return filename_list
                filename_list = [
                    i for i in filename_list
                   # if (str(kwargs.get("n_edge")) in i)
                    if (gsr_type in i)
                    and (str(sample_n) in i)
                    and (fold_str in i)
                ]
                if not sem:
                    return [i for i in filename_list if 'mean' in i and 'session' in i]
                else:
                    return select_filename_sem(filename_list)

#            for drop in drop_bool_list:
            print(
                    scale_name,
#                        drop
                    )
            filename_list_x = select_elements_in_list(file_folder, control, gsr_type, sem=False, **kwargs)
            filename_list_y = select_elements_in_list(file_folder, control, gsr_type, sem=True, **kwargs)
            param_filenames = select_elements_in_list(params_dir, control, gsr_type, sem=True, **kwargs)
            # subset filenames based on covariates
            param_order_filenames = select_elements_in_list(param_order_dir, control, param_order=True)
            param_order_filenames = [
                    i for i in param_order_filenames 
                    if ('both' in i) 
                    and (scale_name in i)
                    and (not 'MeanStr' in i)
                    and ('MSST' in i)
                    and (not 'drop' in i)
                    and (not i.startswith('.~'))
                    ]

            params = np.load(op.join(params_dir, sort_list_by_time(param_filenames)[-1]))
            if len(param_order_filenames) == 1:
                param_order = pd.read_csv(op.join(param_order_dir, param_order_filenames[0]))
            else:
                print(param_order_filenames)
                raise Exception('param_order_filename could not be specified.')
            sessions = [f's{i}' for i in range(1, 5)]
            days = ['o1', 'o2']
            fc_1st_load_positions = param_order.query('(rhs.isin(@sessions)) & (op == "=~")').index
            fc_1st_load_params = params[:, fc_1st_load_positions, 0]
            fc_2nd_load_positions = param_order.query('(rhs.isin(@days)) & (op == "=~")').index
            fc_2nd_load_params = params[:, fc_2nd_load_positions, 0]
            trait_load_positions = param_order.query('(lhs == "tf") & (op == "=~")').index
            trait_load_params = params[:, trait_load_positions, 0]
            cor_position = param_order.query('(lhs == "tf") & (rhs == "ff")').index
            # check parameters
            fc_1st_error_positions = param_order.query('(rhs.isin(@sessions)) & (op == "~~") & (lhs.isin(@sessions))').index
            trait_error_positions = param_order.query('(~rhs.isin(@sessions)) & (~rhs.isin(@control)) & (op == "~~") & (~lhs.isin(@sessions)) & (~lhs.isin(@control))').index
            fc_1st_error_params = params[:, fc_1st_error_positions, 0]
            trait_error_params = params[:, trait_error_positions, 0]
            if not np.allclose(fc_1st_error_params, 1 - fc_1st_load_params**2, equal_nan=True):
                raise Exception('Somethind wrong might happen in .npy parameter file.')
            #cor_remove_index = np.where(np.abs(params[:, cor_position, 0]) > cor_thres)[0]
            
            def get_exclude_edges_from_param_thres(input_array, thres_min_max_list):
                min_thres, max_thres = thres_min_max_list[0], thres_min_max_list[1]
                if min_thres >= max_thres:
                    raise ValueError('Minimum of thresholds is greater than maximum of thresholds.')
                booleans = (input_array < min_thres) | (input_array > max_thres)
                booleans = np.any(booleans, axis=1)
                return booleans

            if fc_1st_load_thres_list is not None:
                fc_1st_remove_booleans = get_exclude_edges_from_param_thres(fc_1st_load_params, fc_1st_load_thres_list)
            else:
                fc_1st_remove_booleans = np.array([False] * n_rsfc)
            if fc_2nd_load_thres_list is not None:
                fc_2nd_remove_booleans = get_exclude_edges_from_param_thres(fc_2nd_load_params, fc_2nd_load_thres_list)
            else:
                fc_2nd_remove_booleans = np.array([False] * n_rsfc)
            if trait_load_thres_list is not None:
                trait_remove_booleans = get_exclude_edges_from_param_thres(trait_load_params, trait_load_thres_list)
            else:
                trait_remove_booleans = np.array([False] * n_rsfc)
            remove_booleans = fc_1st_remove_booleans | fc_2nd_remove_booleans | trait_remove_booleans
            remove_edges = np.where(remove_booleans)[0]
            print(f'Number of excluded edges\nFC 1st loadings: {fc_1st_remove_booleans.sum()}\nFC 2nd loadings: {fc_2nd_remove_booleans.sum()}\nTrait loadings: {trait_remove_booleans.sum()}')
#            filename_list = select_elements_in_list(
#                file_folder, control, gsr_type, **kwargs
#            )
#            if full_or_mean == 'mean':
#                filename_list_x = [i for i in filename_list if ("mean" in i) and ('session' in i)]
#                # Process filename when drop variables are specified
#                filename_list_x = (
#                    [i for i in filename_list_x if not "drop" in i]
#                    if not drop
#                    else [i for i in filename_list_x if "drop" in i]
#                )
#                if kwargs.get("drop_vars_list_dict") is not None:
#                    if drop:
#                        drop_str_list = (
#                            kwargs.get("drop_vars_list_dict")
#                            .get(trait_type)
#                            .get(scale_name)
#                        )
#                        if trait_type == "cognition":
#                            filename_list_x = [
#                                i
#                                for i in filename_list_x
#                                if all(j in i for j in drop_str_list)
#                            ]
#                        elif trait_type == "personality" and scale_name == "Openness":
#                            filename_list_x = [
#                                i
#                                for i in filename_list_x
#                                if "_".join(drop_str_list) in i
#                            ]
#                        elif trait_type == "mental":
#                            pass

#                if include_no_method_effect:
#                    filename_list_no_method_effect = [
#                        i for i in filename_list_y if not "DayCor" in i
#                    ]

#                filename_list_y = (
#                    [i for i in filename_list_y if not "drop" in i]
#                    if not drop
#                    else [i for i in filename_list_y if "drop" in i]
#                )
#                if drop:
#                    if kwargs.get("drop_vars_list_dict") is not None:
#                        if trait_type == "cognition":
#                            filename_list_y = [
#                                i
#                                for i in filename_list_y
#                                if all(j in i for j in drop_str_list)
#                            ]
#                        elif trait_type == "personality" and scale_name == "Openness":
#                            filename_list_y = [
#                                i
#                                for i in filename_list_y
#                                if "_".join(drop_str_list) in i
#                            ]
#                        elif trait_type == "mental":
#                            pass
#                    if include_no_method_effect:
#                        filename_list_no_method_effect = (
#                            [
#                                i
#                                for i in filename_list_no_method_effect
#                                if not "drop" in i
#                            ]
#                            if not drop
#                            else [
#                                i for i in filename_list_no_method_effect if "drop" in i
#                            ]
#                        )
#                n_arrays_sem = n_arrays_dict.get(drop).get("sem")
#                cor_array_all_y = combine_array_files_dat(
#                    filename_list_y, n_arrays_sem, "cor", parcellation=parcellation
#                )
#                if include_no_method_effect:
#                    cor_array_all_no_method_effect = combine_array_files_dat(
#                        filename_list_no_method_effect, n_arrays_sem, "cor", parcellation=parcellation
#                    )
#
#                if add_model_fc:
#                    filename_list_model_fc = [
#                        i
#                        for i in filename_list
#                        if (kwargs.get("est_method").get("y") in i)
#                        and ("Model_fc" in i)
#                        and not ("Cov0" in i)
#                    ]
#                    if day_cor:
#                        filename_list_model_fc = [
#                            i for i in filename_list_model_fc if "DayCor" in i
#                        ]
#                    if kwargs.get("drop_vars_list_dict") is not None:
#                        filename_list_model_fc = (
#                            [i for i in filename_list_model_fc if not "drop" in i]
#                            if not drop
#                            else [i for i in filename_list_model_fc if "drop" in i]
#                        )
#                    cor_array_all_model_fc = combine_array_files_dat(
#                        filename_list_model_fc, n_arrays_sem, "cor", parcellation=parcellation
#                    )
#                if not add_model_fc:
#                    if not include_no_method_effect:
            if len(filename_list_x) == 1:
                filename_x = filename_list_x[0]
            else:
                filename_x = sort_list_by_time(filename_list_x)[-1]
            if len(filename_list_y) == 1:
                filename_y = filename_list_y[0]
            else:
                filename_y = sort_list_by_time(filename_list_y)[-1]
            print(f'Processing {filename_y}')
            cor_array_all_x = np.load(op.join(file_folder, filename_x))
            cor_array_all_y = np.load(op.join(file_folder, filename_y))
            cor_array_concat = np.concatenate(
                [cor_array_all_x, cor_array_all_y], axis=1
            )
            column_name_list = ["mean", "both"]
#                    else:
#                        cor_array_concat = np.concatenate(
#                            [
#                                cor_array_all_x,
#                                cor_array_all_y,
#                                cor_array_all_no_method_effect,
#                            ],
#                            axis=1,
#                        )
#                        column_name_list = ["mean", "both", "no_method_effect"]
#
#                else:
#                    cor_array_concat = np.concatenate(
#                        [cor_array_all_x, cor_array_all_y, cor_array_all_model_fc],
#                        axis=1,
#                    )
#                    column_name_list = ["mean", "both", "fc"]
            # Concatenate df
            cor_df = array_to_df(
                cor_array_concat,
                column_name_list,
                **{
                    "scale_name": scale_name,
                    "gsr_type": gsr_type_key_str,
            #        "drop": drop,
                },
            )
            # remove edges in both SEM and control conditions
            if invalid_edge_filename is not None:
                cor_df.query("edge not in @invalid_edges", inplace=True)
                remove_edges_add = [i for i in remove_edges if i not in invalid_edges]
                print(f'Number of additional removal edges is {len(remove_edges_add)}.')
                cor_df.query("edge not in @remove_edges_add", inplace=True)
            else:
                cor_df.query("edge not in @remove_edges", inplace=True)
                print(f'Number of removal edges is {len(remove_edges)}.')
            output_df = pd.concat([output_df, cor_df], axis=0)
    output_df = replace_and_reorder_column(
        output_df,
        var_name_dict={
            "gsr_type": GSR_DICT,
#            "drop": DROP_DICT2,
        },
    )
    output_df = make_categories_in_df(output_df, "scale_name", subscales)
#    output_df["scale_name"] = output_df["scale_name"].cat.rename_categories(
#        {"All": "Total"}
#    )

    def melt_df(df, column_name_list):
        """
        Inner function for creating long dataframe
        """
        return df.melt(
            id_vars=[
                "edge", 
                "gsr_type",
                "scale_name",
               # "drop"
                ],
            value_vars=column_name_list,
            var_name="model",
            value_name="r",
        )

    if return_long:
        return melt_df(output_df, column_name_list)

    return output_df


def replace_model_sem_or_full(df):
    """
    Replace values for visualization
    """
    return df.replace(
        {
            "both": "SEM",
            "mean": "Analyses on aggregate scores",
            "score": "Analyses on factor scores of both RSFC and trait",
            "average": "Analyses on average scores of RSFC and factor scores of trait",
            "full": "Analyses on full scores of RSFC and factor scores of trait",
        }
    )


def lm_reg(df, x, y, comp=None, include_sem=True, **kwargs):
    """
    Conduct linear regression and returns improvement factors
    """
    try:
        y_min, y_max = kwargs.get("y_min"), kwargs.get("y_max")
    except:
        y_min, y_max = -np.Inf, np.Inf

    if include_sem:
        if comp is None:
            df_filtered = df
        elif comp == "drop":
            df_filtered = df.query(
                "Original > @y_min & Original < @y_max & Modified > @y_min & Modified < @y_max"
            )
        elif comp == "full_sem":
            df_filtered = df
            #df_filtered = df.query("both > @y_min & both < @y_max")
        elif comp == "method_effect":
            df_filtered = df.query(
                "both > @y_min & both < @y_max & no_method_effect > @y_min & no_method_effect < @y_max"
            )
        elif comp == "nogsr_gsr":
            df_filtered = df.query(
                "`Without GSR` > @y_min & `Without GSR` < @y_max & `With GSR` > @y_min & `With GSR` < @y_max"
            )

        removed_n = len(df) - len(df_filtered)
        mask = ~np.isnan(df_filtered[x]) & ~np.isnan(df_filtered[y])
        df_x, df_y = df_filtered[x][mask], df_filtered[y][mask]
    else:
        mask = ~np.isnan(df[x]) & ~np.isnan(df[y])
        df_x, df_y = df[x][mask], df[y][mask]
    slope, intercept, r_value, p_value, std_error = linregress(df_x, df_y)

    if include_sem:
        return pd.Series(
            {
                "slope": slope,
                "intercept": intercept,
                "r": r_value,
                "p": p_value,
                "std_error": std_error,
                "removed_n": removed_n,
            }
        )
    else:
        return pd.Series(
            {
                "slope": slope,
                "intercept": intercept,
                "r": r_value,
                "p": p_value,
                "std_error": std_error,
            }
        )


def wrapper_of_visualize_imp_factors(
    y_min,
    y_max,
    x_min=None,
    x_max=None,
    drop_bool_list=[False],
    filename_fig=None,
    other_new_line=True,
    comp="method_effect",
    add_legend=True,
    custom_fig_width=10,
    n_arrays_dict={True: {"full": 216, "sem": 72}, False: {"full": 216, "sem": 72}},
    include_no_method_effect=True,
    trait_equal_loadings=True,
    msst=True,
    visualize="full_and_sem",
    fig_height=8 / 2.54,
    n_arrays_fscores=431,
    parcellation='Schaefer',
    full_or_mean='mean',
    **kwargs,
):
    """
    Wrapper function of visualize_imp_factors to create figures for publication
    """
    g_list = []
    if any(drop_bool_list):
        trait_type_list = ["cognition", "personality"]
    else:
        trait_type_list = ["cognition", "mental", "personality"]
    for trait_type in trait_type_list:
        print(f"Processing {trait_type}.")
        trait_scale_name = get_scale_name_from_trait(
            trait_type, publication=True, drop_subscale=any(drop_bool_list)
        )
        g = visualize_imp_factors(
            trait_type,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            drop_bool_list=drop_bool_list,
            comp=comp,
            other_new_line=other_new_line,
            n_arrays_dict=n_arrays_dict,
            include_no_method_effect=include_no_method_effect,
            return_plot=True,
            save_file=False,
            add_title=True,
            title=trait_scale_name,
            visualize=visualize,
            n_arrays_fscores=n_arrays_fscores,
            parcellation=parcellation,
            full_or_mean=full_or_mean,
            **kwargs,
        )
        g_list.append(g)

    combine_gg_list(
        g_list,
        fig_height,
        filename_fig,
        legend_comp=add_legend,
        custom_fig_width=custom_fig_width,
        drop_bool=any(drop_bool_list),
        parcellation=parcellation
    )


def create_additional_plot(legend_comp, comp_target="model"):
    """
    Create empty or legend plot for publication
    """
    if comp_target == "model":
        comp1 = "Analyses on aggregate scores"
        comp2 = "SEM"
    elif comp_target == "drop":
        comp1 = "Original measurement model"
        comp2 = "Modified measurement model"
    elif comp_target == "cor_est":
        comp1 = "Correlation estimated in analyes on aggregate scores"
        comp2 = "$\it{True}$" + " correlation estimated in SEM"
    elif comp_target == "rsfc":
        comp1 = "RSFC where valid measurment model was established (RSFC+)"
        comp2 = "RSFC where valid measurment model was not established (RSFC-)"

    line_values = ["dashed", "solid"] if comp_target != "rsfc" else ["solid", "dashed"]

    add_df = pd.DataFrame({"x": [0] * 20, "Model": [comp1] * 10 + [comp2] * 10})
    add_df["Model"] = pd.Categorical(add_df["Model"], categories=[comp1, comp2])

    if not legend_comp:
        add_ggplot = ggplot(add_df, aes(x="x")) + theme_void()
    else:
        add_ggplot = (
            ggplot(add_df, aes(x="x", fill="Model", linetype="Model"))
            + geom_density(alpha=0.2)
            + geom_rect(
                aes(xmax=np.Inf, xmin=-np.Inf, ymin=-np.Inf, ymax=np.Inf),
                fill="white",
                inherit_aes=False,
            )
            + theme(
                plot_background=element_blank(),
                axis_text=element_blank(),
                axis_ticks=element_blank(),
                axis_title=element_blank(),
                legend_background=element_rect(color="none", fill="white"),
                legend_position=(0.5, 0.5),
                panel_background=element_blank(),
                panel_grid=element_blank(),
                rect=element_rect(color="white", size=0, fill="white"),
                figure_size=(1e-4, 1e-4),
                legend_title=element_blank(),
                legend_direction="vertical",
                legend_text=element_text(size=12),
            )
            + labs(x="", y="")
            + scale_linetype_manual(values=line_values)
        )

    return add_ggplot


def combine_gg_list(
    gg_list: list,
    fig_height: float,
    filename_fig: str,
    drop_bool=False,
    legend_comp=False,
    custom_fig_width=None,
    n_ffi_plots=3,
    comp_target="model",
    parcellation='Schaefer'
):
    """
    Combine ggplots for publications
    """
    print("Loading ggplot objects")

    add_ggplot = create_additional_plot(legend_comp, comp_target=comp_target)
    add_width = 5 / 2.54 if drop_bool else 5 * 2 / 2.54
    g_add = load_ggplot(add_ggplot, figsize=(add_width, fig_height))
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    if not drop_bool:
        print("... Cognition")
        g_cog = load_ggplot(gg_list[0], figsize=(5 * 3 / 2.54, fig_height))
        print("... Mental")
        g_mental = load_ggplot(gg_list[1], figsize=(5 * 4 / 2.54, fig_height))
        print("... personality")
        g_personality = load_ggplot(gg_list[2], figsize=(5 * 5 / 2.54, fig_height))
        print("Combining ggplot objects")
        g_combined = (g_cog | g_mental) / (g_personality | g_add)
    else:
        print("... Cognition")
        fig_width = custom_fig_width / 2.54 if custom_fig_width else 7.5 / 2.54
        g_cog = load_ggplot(gg_list[0], figsize=(fig_width, fig_height))
        print("... Personality")
        if n_ffi_plots:
            g_personality = load_ggplot(
                gg_list[1], figsize=(fig_width * n_ffi_plots, fig_height)
            )
        if not legend_comp:
            g_combined = g_cog | g_personality
        else:
            g_combined = g_cog | g_personality | g_add

    if filename_fig:
        print("Saving combined figure")
        g_combined.savefig(op.join(atlas_dir, "figures", f"{filename_fig}.png"))
        print("Saving completed.")

    return g_combined


def get_list_of_filenames(
        data_type, 
        gsr_type, 
        parcellation='Schaefer', 
        addMarker=False,
        include_day_cor=True,
        multistate_single_trait=False,
        get_std=False,
        **kwargs
        ):
    """
    Get a list of filenames generated by SLURM to combine
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    if gsr_type == "nogs":
        gsr_str, gsr_str_param = "", "_nogs_"
    elif gsr_type == "gs":
        gsr_str, gsr_str_param = "_gs", "_gs_"

    filenames = os.listdir(op.join(atlas_dir, "reliability", data_type))
    filenames = [
        i
        for i in filenames
        if (str(kwargs.get("sample_n")) in i)
        and (str(kwargs.get("edge_n")) in i)
        and not ("PE" in i)
        and not ("OrderInDay" in i)
        and (gsr_str_param in i)
    ]
    
    if addMarker:
        filenames = [i for i in filenames if 'addMarker' in i]
    else:
        filenames = [i for i in filenames if not 'addMarker' in i]

    if include_day_cor:
        filenames = [i for i in filenames if "DayCor" in i]
    else:
        filenames = [i for i in filenames if not "DayCor" in i]
    
    if multistate_single_trait:
        filenames = [i for i in filenames if 'MultiStateSingleTrait' in i]
    else:
        filenames = [i for i in filenames if not 'MultiStateSingleTrait' in i]
    
    if get_std:
        filenames = [i for i in filenames if 'OutStd' in i]
    else:
        filenames = [i for i in filenames if not 'OutStd' in i]

    return filenames


def calc_omega_summary_per_node(
        n_arrays=72,
        parcellation='Schaefer',
        include_day_cor=True,
        summary_func='median',
        **kwargs
        ):
    """
    Calculate summary of omega coefficients per node
    """
    omega_pd = get_edge_omega_df(
        n_arrays,
        parcellation,
        include_day_cor,
        **kwargs
        )
    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    nodes_df = get_nodes_df(parcellation)
    omega_pd = pd.merge(omega_pd, edges_df, on='edge')
    wide_omega_df = omega_pd.pivot(
            index=['edge', 'node1', 'node2'],
            columns='gsr_type',
            values='omega'
            )
    wide_omega_df.rename(columns={'Without GSR': 'nogs', 'With GSR': 'gs'}, inplace=True)
    wide_omega_df.reset_index(inplace=True)
    nodes_df = get_nodes_df_from_edges_df(wide_omega_df, nodes_df, 'omega')
    return nodes_df


def get_edge_omega_df(
        n_arrays=72,
        parcellation='Schaefer',
        include_day_cor=True,
        **kwargs
        ):
    """
    Get omega coefficnets of edges
    """
    omega_list = []
    for g, gsr_type in enumerate(["nogs", "gs"]):

        param_filenames = get_list_of_filenames(
                "parameters", 
                gsr_type, 
                parcellation, 
                include_day_cor=include_day_cor,
                **kwargs
                )
        param_filename = sort_list_by_time(param_filenames)[-1]
        print(f"Processing {param_filename}")
        param_position_dict = get_param_position_dict(param_filename)
        params_array = combine_array_files_dat(
            param_filenames, n_arrays=n_arrays, data_type="parameter", parcellation=parcellation
        )
        omegas = calc_omega_2d_array_from_filename(params=params_array)
        omega_list.append(omegas.tolist())
    omega_n = len(omegas)
    omega_pd = pd.DataFrame(
        {
            "edge": [i for i in range(omega_n)] * 2,
            "omega": add(omega_list[0], omega_list[1]),
            "gsr_type": ["Without GSR"] * omega_n + ["With GSR"] * omega_n,
        }
    )
    omega_pd.query("(omega < 1) & (omega > 0)", inplace=True)
    return omega_pd


def vis_rel_rsfc_trait_cor(
    n_arrays_dict, 
    n_arrays=431, 
    drop_bool_list=[False], 
    include_day_cor=True, 
    parcellation='Schaefer',
    **kwargs
):
    """
    Visualize correlation between reliability of RSFC and RSFC-trait associations
    """
    omega_pd = get_edge_omega_df(
            n_arrays,
            parcellation,
            include_day_cor,
            **kwargs
            )
    g_list = []
    for trait_type in ["cognition", "mental", "personality"]:
        trait_title = get_scale_name_from_trait(trait_type, publication=True)
        df = generate_summary_df(
            trait_type,
            n_arrays_dict=n_arrays_dict,
            return_long=True,
            drop_bool_list=drop_bool_list,
            **kwargs,
        )
        df = pd.merge(omega_pd, df, on=["edge", "gsr_type"])
        df["abs_r"] = np.abs(df["r"])
        df.query('model == "mean"', inplace=True)
        g = (
            ggplot(df, aes("omega", "abs_r"))
            + geom_point(size=0.001, alpha=0.1)
            + facet_grid("gsr_type ~ scale_name")
            + geom_smooth(method="lm", se=False, color="red")
        )
        g_list.append(g)
    return g_list


def generate_merged_df(
        n_arrays_dict, 
        drop_bool_list=[False], 
        parcellation='Schaefer',
        full_or_mean='mean',
        **kwargs
        ):
    """
    Generate df combining outputs and edge information
    """
    edge_df = get_edge_summary(network_hem_order=True, parcellation=parcellation)

    output_df = pd.DataFrame()
    for trait_type in ["cognition", "mental", "personality"]:
        trait_title = get_scale_name_from_trait(trait_type, publication=True)
        df = generate_summary_df(
            trait_type,
            n_arrays_dict=n_arrays_dict,
            return_long=True,
            drop_bool_list=drop_bool_list,
            parcellation=parcellation,
            full_or_mean=full_or_mean,
            **kwargs,
        )
        df["trait_type"] = trait_type
        if trait_type == "cognition":
            df.replace({"Total": "Total_cog"}, inplace=True)
        elif trait_type == "mental":
            df.replace({"Total": "Total_mental"}, inplace=True)
        merged_df = (
            pd.merge(df, edge_df, on="edge")
            .pivot(
                index=[
                    "edge",
                    "gsr_type",
                    "scale_name",
                    "node1_net",
                    "node2_net",
                    "trait_type",
                ],
                values="r",
                columns="model",
            )
            .reset_index()
        )
        output_df = pd.concat([output_df, merged_df], axis=0)
    return output_df


def get_p_values_of_cor(
        parcellation='Schaefer',
        controls=['age', 'gender', 'MeanRMS'],
        invalid_edge_file_dict=None,
        **kwargs
        ):
    """
    Get p values from (partial) correlation
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    output_df = pd.DataFrame()
    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    n_edge = len(edges_df)
    dof_cor = kwargs.get('sample_n') - 2 - len(controls)

    for trait_type in ['cognition', 'mental', 'personality']:
        folder = get_scale_name_from_trait(trait_type)
        scale_name_list = get_subscale_list(folder)
        for scale_name in scale_name_list:
            cor_combined_folder = op.join(atlas_dir, folder, scale_name, 'correlation', 'combined')
            for gsr_type in ['_nogs_', '_gs_']:
                print(f'Processing {scale_name} of {trait_type} {gsr_type}')
                filenames = [i for i in os.listdir(cor_combined_folder) if gsr_type in i and 'mean' in i]
                if len(filenames) != 1:
                    raise Exception('filenames should include only 1 filename')
                filename = sort_list_by_time(filenames)[-1]
                cors = np.squeeze(np.load(op.join(cor_combined_folder, filename)))
                # calculate p value from correlation and sample size
                t_value = cors * np.sqrt(dof_cor) / np.sqrt(1 - cors**2)
                p_value = t.sf(np.abs(t_value), df=dof_cor) * 2
                if invalid_edge_file_dict:
                    invalid_edge_file = invalid_edge_file_dict.get(gsr_type.replace('_', ''))
                    invalid_edges = np.loadtxt(op.join(atlas_dir, 'reliability', 'invalid_edges', invalid_edge_file))
                inner_df = pd.DataFrame(
                        {
                            'edge': [i for i in range(n_edge)],
                            'gsr_type': [gsr_type] * n_edge,
                            'scale_name': [scale_name] * n_edge,
                            'trait_type': [trait_type] * n_edge,
                            'r': cors,
                            't': t_value,
                            'p': p_value
                            }
                        )
                output_df = pd.concat([output_df, inner_df], axis=0)
    output_df = pd.merge(output_df, edges_df, on='edge', how='left')
    return output_df


def get_summary_of_sem_result(
        parcellation='Schaefer',
        msst=True,
        controls=['age', 'gender', 'MeanRMS'],
        get='p_value',
        invalid_edge_file_dict=None,
        **kwargs
        ):
    """
    Get df storing results of applying SEM
    """
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    output_df = pd.DataFrame()
    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    n_edge = len(edges_df) 

    if get == 'p_value':
        col_index = 3
    elif get == 'est':
        col_index = 0

    for trait_type in ['cognition', 'mental', 'personality']:
        folder = get_scale_name_from_trait(trait_type)
        scale_name_list = get_subscale_list(folder)
        for scale_name in scale_name_list:
            param_order_folder = op.join(FA_ORDER_DIR, folder)
            param_order_filenames = os.listdir(param_order_folder)
            param_order_filenames = select_filenames_from_CU_or_MSST(param_order_filenames, multistate_single_trait=msst, controls=controls)
            param_order_filenames = [i for i in param_order_filenames if scale_name in i and not i.startswith('.')]
            if len(param_order_filenames) != 1:
                raise Exception('param_order_filenames should only 1 filename')
            param_order_filename = param_order_filenames[0]
            param_order = pd.read_csv(op.join(param_order_folder, param_order_filename))
            param_index = param_order.query('lhs == "tf" & rhs == "ff"').index
            param_combined_folder = op.join(atlas_dir, folder, scale_name, 'parameters', 'combined')
            for gsr_type in ['_nogs_', '_gs_']:
                print(f'Processing {scale_name} of {trait_type} {gsr_type}')
                filenames = [i for i in os.listdir(param_combined_folder) if gsr_type in i]
                filenames = select_filenames_from_CU_or_MSST(filenames, multistate_single_trait=msst, sample_n=kwargs.get('sample_n'), controls=controls)
                filename = sort_list_by_time(filenames)[-1]
                params = np.load(op.join(param_combined_folder, filename))[:, param_index, col_index]
                if invalid_edge_file_dict:
                    invalid_edge_file = invalid_edge_file_dict.get(gsr_type.replace('_', ''))
                    invalid_edges = np.loadtxt(op.join(atlas_dir, 'reliability', 'invalid_edges', invalid_edge_file))
                inner_df = pd.DataFrame(
                        {
                            'edge': [i for i in range(n_edge)],
                            'gsr_type': [gsr_type] * n_edge,
                            'scale_name': [scale_name] * n_edge,
                            'trait_type': [trait_type] * n_edge,
                            get: np.squeeze(params)
                            }
                        )
                output_df = pd.concat([output_df, inner_df], axis=0)
    output_df = pd.merge(output_df, edges_df, on='edge', how='left')
    return output_df


def summarise_imp_networks(
    n_arrays_dict,
    drop_bool_list,
    fig_width=12,
    fig_height=4,
    hmap_vmin=0.7,
    hmap_vmax=1.42,
    parcellation='Schaefer',
    full_or_mean='mean',
    vis_level='overall',
    msst=True,
    return_df=False,
    **kwargs,
):
    """
    Summarise improvement factors in combinations of network across scales and traits
    """
    df = generate_merged_df(
            n_arrays_dict, 
            drop_bool_list,
            parcellation=parcellation,
            full_or_mean=full_or_mean,
            msst=msst,
            **kwargs
            )
    if return_df:
        return df

    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    network_order_for_pub_list = NETWORK_ORDER_FOR_PUB_NESTED_LIST.get(parcellation)
    n_net = len(network_order_for_pub_list)
    x, y = "mean", "both"
    if parcellation == 'Schaefer':
        x_rotate_deg = 45
    elif parcellation == 'Gordon':
        x_rotate_deg = 90
    if vis_level == 'trait_type':
        fig, axes = plt.subplots(
            2, 3, sharex=True, sharey=True, figsize=(fig_width, fig_height)
        )
        cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
        for h, trait_type in enumerate(["cognition", "mental", "personality"]):
            trait_title = (
                get_scale_name_from_trait(trait_type, publication=True)
                .replace("(I)", "")
                .replace("(II)", "")
                .replace("(III)", "")
            )
            subset_df = df.query("trait_type == @trait_type")
            for i, gsr_type in enumerate(["Without GSR", "With GSR"]):
                output_array = np.zeros(shape=(n_net, n_net))
                subset_df_gsr = subset_df.query("gsr_type == @gsr_type")
                subset_df_gsr = (
                    subset_df_gsr.groupby(
                        ["scale_name", "node1_net", "node2_net"], observed=False
                    )[["node1_net", "node2_net", x, y]]
                    .apply(
                        lm_reg, x, y, "full_sem", True, **{"y_min": -0.3, "y_max": 0.3}
                    )
                    .reset_index()
                )
                for j, scale_name in enumerate(subset_df_gsr["scale_name"].unique()):
                    wide_df = subset_df_gsr.query("scale_name == @scale_name").pivot(
                        index="node1_net", values="slope", columns="node2_net"
                    )
                    output_array += np.array(wide_df)
                #    subset_df = subset_df.reset_index().pivot(index='node1_net', values='slope', columns='node2_net')
                mean_output_array = output_array / subset_df["scale_name"].nunique()
                output_df = pd.DataFrame(
                    mean_output_array,
                    index=network_order_for_pub_list,
                    columns=network_order_for_pub_list,
                )
                iteration = h * i
                sns.heatmap(
                    output_df,
                    annot=True,
                    fmt=".2f",
                    ax=axes[i, h],
                    vmin=hmap_vmin,
                    vmax=hmap_vmax,
                    cbar_ax=cbar_ax if iteration else None,
                    cbar=iteration == 1,
                    annot_kws={"fontsize": 8},
                    cmap="Oranges",
                )
                axes[i, h].set_title(f"{trait_title} {gsr_type}")
                axes[i, h].tick_params(axis="x", rotation=45)
                axes[i, h].set(xlabel=None, ylabel=None)
    
    elif vis_level == 'overall':
        fig, axes = plt.subplots(
            1, 2, sharex=True, sharey=True, figsize=(fig_width, fig_height)
        )
        cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
        for i, gsr_type in enumerate(["Without GSR", "With GSR"]):
            output_array = np.zeros(shape=(n_net, n_net))
            subset_df_gsr = df.query("gsr_type == @gsr_type")
            subset_df_gsr = (
                subset_df_gsr.groupby(
                    ["scale_name", "node1_net", "node2_net"], observed=False
                )[["node1_net", "node2_net", x, y]]
                .apply(
                    lm_reg, x, y, "full_sem", True, **{"y_min": -0.3, "y_max": 0.3}
                )
                .reset_index()
            )
            for j, scale_name in enumerate(subset_df_gsr["scale_name"].unique()):
                wide_df = subset_df_gsr.query("scale_name == @scale_name").pivot(
                    index="node1_net", values="slope", columns="node2_net"
                )
                output_array += np.array(wide_df)
            #    subset_df = subset_df.reset_index().pivot(index='node1_net', values='slope', columns='node2_net')
            mean_output_array = output_array / df["scale_name"].nunique()
            output_df = pd.DataFrame(
                mean_output_array,
                index=network_order_for_pub_list,
                columns=network_order_for_pub_list,
            )
            iteration = i
            sns.heatmap(
                output_df,
                annot=True,
                fmt=".2f",
                ax=axes[i],
                vmin=hmap_vmin,
                vmax=hmap_vmax,
                cbar_ax=cbar_ax if iteration else None,
                cbar=iteration == 1,
                annot_kws={"fontsize": 8},
                cmap="Oranges",
            )
            axes[i].set_title(f"{gsr_type}")
            axes[i].tick_params(axis="x", rotation=x_rotate_deg)
            axes[i].set(xlabel=None, ylabel=None)

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(
        op.join(
            atlas_dir, "figures", "imp_factor_networks_summary_mean.png"
        ),
        bbox_inches="tight",
    )


def vis_imp_factor_networks(
    get_imp_df=False,
    parcellation='Schaefer',
    controls=['age', 'gender', 'MeanRMS'],
    invalid_edge_file_dict={},
    full_or_mean='mean',
    msst=True,
    trait_equal_loadings=True,
    save_filename=None,
    save_filename_all=None,
    split_half_family=False,
    fold=0,
    sample_n=None,
#    sem_min_max_r_dict={'cognition': 0.5, 'mental': 0.5, 'personality': 0.5},
    target_net_list:list[str]=None,
    skip_scale_dict=None,
    vmin=1.2,
    vmax=1.8,
    fc_1st_load_thres_list=[0.4, 0.9],
    fc_2nd_load_thres_list=[0.5, 1],
    trait_load_thres_list=[0.2, 1],
    figsize=(15, 6),
    figsize_all=(8, 4),
    fig_suptitle=None,
    draw_scatters=False,
    draw_imp_heat_all=True,
    draw_imp_heat_phenotypes=False,
    scatter_lim=[-0.3, 0.3],
    scatter_filename=None,
    g_scatters_filename=None,
    draw_hists_sign_error=False,
    add_annotate=True,
    add_regline=True,
    **kwargs
):
    """
    Visualize improvement factors per networks
    """
    edge_df = get_edge_summary(network_hem_order=True, parcellation=parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    if fc_1st_load_thres_list is not None:
        fc_1st_suffix = 'FC1st_' + '_'.join(map(str, fc_1st_load_thres_list)) + '_'
    else:
        fc_1st_suffix = ''
    if fc_2nd_load_thres_list is not None:
        fc_2nd_suffix = 'FC2nd_' + '_'.join(map(str, fc_2nd_load_thres_list)) + '_'
    else:
        fc_2nd_suffix = ''
    if trait_load_thres_list is not None:
        trait_load_suffix = 'Trait_' + '_'.join(map(str, trait_load_thres_list)) 
    else:
        trait_load_suffix = ''
    suffix = fc_1st_suffix + fc_2nd_suffix + trait_load_suffix

    if target_net_list is None:
        networks = NETWORK_ORDER_FOR_PUB_NESTED_LIST.get(parcellation)
    else:
        networks = target_net_list.copy()
    if draw_imp_heat_phenotypes:
        fig, axes = plt.subplots(
            nrows=2, ncols=3, sharex=True, sharey=True, figsize=figsize
        )
        cbar_ax = fig.add_axes([.91, .3, .015, .4])
    if draw_imp_heat_all:
        fig_all, axes_all = plt.subplots(
            ncols=2, sharey=True, figsize=figsize_all
        )
        cbar_ax_all = fig_all.add_axes([.91, .3, .015, .4])
        imp_wide_df_dict = defaultdict(dict)
   
    if get_imp_df:
        imp_df_nets_all = pd.DataFrame()
        imp_df_overall_all = pd.DataFrame()
    g_list = []
    for trait_col, trait_type in enumerate(["cognition", "mental", "personality"]):
        if skip_scale_dict is not None:
            skip_scale = skip_scale_dict.get(trait_type)
        else:
            skip_scale = None
        if trait_type == 'cognition':
            n_subscales, fig_header = 3, r'$\bf{a}$'
        elif trait_type == 'mental':
            n_subscales, fig_header = 4, r'$\bf{b}$'
        elif trait_type == 'personality':
            n_subscales, fig_header = 5, r'$\bf{c}$'
        trait_title = get_scale_name_from_trait(trait_type, publication=True)
        # get additional invalid edges
        df = generate_summary_df(
            trait_type,
            return_long=True,
            parcellation=parcellation,
            control=controls,
            full_or_mean=full_or_mean,
            invalid_edge_file_dict=invalid_edge_file_dict,
            fc_1st_load_thres_list=fc_1st_load_thres_list,
            fc_2nd_load_thres_list=fc_2nd_load_thres_list,
            trait_load_thres_list=trait_load_thres_list,
            msst=msst,
            trait_equal_loadings=trait_equal_loadings,
            fold=fold,
            sample_n=sample_n,
            split_half_family=split_half_family,
            skip_scale=skip_scale,
            **kwargs,
        )
        merged_df = (
            pd.merge(df, edge_df, on="edge")
            .pivot(
                index=["edge", "gsr_type", "scale_name", "node1_net", "node2_net"],
                values="r",
                columns="model",
            )
            .reset_index()
        )
        folder = select_folder_from_trait(trait_type)
        # remove edges with reversed signs
        merged_df['sign_error'] = np.sign(merged_df['both']) != np.sign(merged_df['mean'])
        sign_errored_df = merged_df[merged_df['sign_error']]
        sign_correct_df = merged_df[~ merged_df['sign_error']]
        # visualize sign error distributions
        if draw_hists_sign_error:
            print('Drawing histograms of sign error.')
            g = (
                ggplot(sign_errored_df, aes('both'))
                + geom_histogram(binwidth=0.01)
                + facet_grid('scale_name ~ gsr_type')
                + theme_bw()
                + coord_cartesian(xlim=[-0.2, 0.2])
                + scale_x_continuous(limits=[-0.2, 0.2])
                    )
            g.save(filename=op.join(atlas_dir, folder, 'figures', f'sign_error_hist.png'))
            #g.show()
        # check distribution of correlation
       # fig, ax = plt.subplots()
       # ax.hist(merged_df['both'], bins=np.arange(-1, 1, 0.01))
       # scale_n = df["scale_name"].nunique()
#        sem_max_cor = sem_min_max_r_dict.get(trait_type)
#        sem_min_cor = -sem_max_cor
        n_scales = merged_df['scale_name'].nunique()

        for j, gsr_type in enumerate(merged_df["gsr_type"].unique()):
            sum_imp_df = pd.DataFrame(0, index=networks, columns=networks)
            for i, scale_name in enumerate(merged_df["scale_name"].unique()):
                x, y = "mean", "both"
                subset_df = merged_df.query("scale_name == @scale_name & gsr_type == @gsr_type")
                mask = ~np.isnan(subset_df['mean']) & ~np.isnan(subset_df['both'])
                df_x, df_y = subset_df['mean'][mask], subset_df['both'][mask]
                slope, intercept, r_value, p_value, std_error = linregress(df_x, df_y)
                summary_imp = pd.DataFrame(
                    {
                        "slope": slope,
                        "intercept": intercept,
                        "r": r_value,
                        "p": p_value,
                        "std_error": std_error,
                        'trait_type': trait_type,
                        "scale_name": scale_name, 
                        'gsr_type': gsr_type
                    },
                    index=[f'{trait_type}_{scale_name}_{gsr_type}']
                )
                if get_imp_df:
                    imp_df_overall_all = pd.concat([imp_df_overall_all, summary_imp], axis=0)
                imp_df_nets = (
                    subset_df.groupby(["node1_net", "node2_net"], observed=False)[
                        ["node1_net", "node2_net", x, y]
                    ]
                    .apply(
                        lm_reg,
                        x,
                        y,
                        "full_sem",
                        True,
                   #     **{"y_min": sem_min_cor, "y_max": sem_max_cor},
                    )
                    .reset_index()
                )
                if get_imp_df:

                    def add_imp_df_cols(df, gsr_type, scale_name, trait_type):
                        df['gsr_type'] = gsr_type
                        df['scale_name'] = scale_name
                        df['trait_type'] = trait_type
                        return df

                    imp_df_nets = add_imp_df_cols(imp_df_nets, gsr_type, scale_name, trait_type)
                    imp_df_nets_all = pd.concat([imp_df_nets_all, imp_df_nets], axis=0)
#                    imp_df_overall = add_imp_df_cols(imp_df_overall, gsr_type, scale_name, trait_type)
#                    imp_df_overall_all = pd.concat([imp_df_overall_all, imp_df_overall], axis=0)

                subset_wide_df = imp_df_nets.pivot(index="node1_net", values="slope", columns="node2_net")
                if target_net_list:
                    subset_wide_df = subset_wide_df.loc[target_net_list, target_net_list]
                sum_imp_df += subset_wide_df
            mean_wide_df = sum_imp_df / n_scales
            if draw_imp_heat_all:
                imp_wide_df_dict[trait_type][gsr_type] = sum_imp_df
            if draw_imp_heat_phenotypes:
                target_ax = axes[j, trait_col]
                print("Drawing heatmap.")
                sns.heatmap(
                    mean_wide_df,
                    annot=True,
                    fmt=".2f",
                    ax=target_ax,
                    vmin=vmin,
                    vmax=vmax,
                    annot_kws={"fontsize": 8},
                    cmap="Oranges",
                    cbar=trait_col == 0,
                    cbar_ax=None if trait_col else cbar_ax
                    )
                target_ax.set_title(f'{trait_title} ({gsr_type.replace("With", "with")})')
                target_ax.tick_params(axis="x", rotation=45)
                target_ax.set(xlabel=None, ylabel=None)
    #fig.suptitle(trait_title, x=0.05, ha="left", size=16)
        if draw_scatters:
            if target_net_list:
                merged_df = merged_df.query('(node1_net in @target_net_list) & (node2_net in @target_net_list)')
            print('Drawing scatterplots.')
            g = (
                ggplot(merged_df, aes('mean', 'both'))
                + geom_point(size=0.25, alpha=0.5)
                + facet_grid('gsr_type ~ scale_name')
                + theme_bw()
                + geom_abline(slope=1, intercept=0, linetype='dashed')
                + labs(
                    y='Corrected association', 
                    x='Uncorrected association',
                    title=rf'{fig_header} {trait_title}'
                    )
                )
            if add_regline:
                g = g + geom_smooth(method='lm', color='red', size=1.25, se=False)
            if add_annotate:
                df_labels = merged_df.groupby(['gsr_type', 'scale_name'], observed=True).apply(
                    lm_reg,
                    x='mean',
                    y='both',
                )
                df_labels["imp_txt"] = [
                    f" Improvement factor =\n {slope:.3f}"
                    for slope in df_labels["slope"]
                ]
                g = (
                    g
                    + geom_text(
                        aes(label="imp_txt", x=-np.Inf, y=np.Inf),
                        data=df_labels.reset_index(),
                        va="top",
                        ha="left",
                        size=8,
                    )
                    )

            if scatter_lim is not None:
                g = (g + coord_cartesian(ylim=scatter_lim, xlim=scatter_lim))
            if scatter_filename is None:
                scatter_filename = 'scatter_each_scale'
            g_list.append(g)
            g.save(filename=op.join(atlas_dir, folder, 'figures', f'{scatter_filename}_{suffix}.png'),
                    height=4.5, width=n_subscales*2)
        # end of for loop of trait_type
    if draw_scatters: 
        (g_cog, g_mental, g_ffi) = (
                load_ggplot(g_list[0], figsize=(4.5, 2)), 
                load_ggplot(g_list[1], figsize=(6, 2)), 
                load_ggplot(g_list[2], figsize=(7.5, 2))
                )
        g_empty = (
            ggplot()
            + geom_blank() 
            + theme(panel_background=element_rect(fill='white'))
        )
        g_add = load_ggplot(g_empty, figsize=(3, 2))
        g_combined = (g_cog|g_mental)/(g_ffi|g_add)
        g_combined.savefig(op.join(atlas_dir, 'figures', f'{g_scatters_filename}.png'))

    if get_imp_df:
        return imp_df_nets_all, imp_df_overall_all
    
    if draw_imp_heat_phenotypes:
        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(
            op.join(
                atlas_dir, "figures", f"{save_filename}.png"
            )
        )
    if draw_imp_heat_all:
        for fig_col, gsr_type in enumerate(['Without GSR', 'With GSR']):
            summary_imp_df = pd.DataFrame(0, index=networks, columns=networks)
            for _, value in imp_wide_df_dict.items():
                summary_imp_df += value.get(gsr_type)
            summary_imp_df /= 12
            target_ax = axes_all[fig_col]
            sns.heatmap(
                summary_imp_df,
                annot=True,
                fmt=".2f",
                ax=target_ax,
                vmin=vmin,
                vmax=vmax,
                annot_kws={"fontsize": 8},
                cmap="Oranges",
                cbar=fig_col == 0,
                cbar_ax=None if fig_col else cbar_ax_all
                )
            target_ax.set_title(gsr_type)
            target_ax.tick_params(axis="x", rotation=45)
            target_ax.set(xlabel=None, ylabel=None)
        if fig_suptitle is not None:
            fig_all.suptitle(fig_suptitle)
        fig_all.tight_layout(rect=[0, 0, .9, 1])
        fig_all.savefig(
            op.join(
                atlas_dir, "figures", f"{save_filename_all}.png"
            )
        )


def visualize_imp_factors(
    trait_type,
    y_min,
    y_max,
    x_min=None,
    x_max=None,
    drop_bool_list=[False],
    imp_new_line=False,
    other_new_line=False,
    filename_fig="imp_factor_scatterplots.png",
    fig_height=8,
    scatter_or_hist=None,
    network_imp=False,
    comp="full_sem",
    visualize="full",
    control=["age", "gender", "MeanRMS"],
    n_arrays_dict=None,
    day_cor=True,
    add_model_fc=False,
    include_no_method_effect=False,
    return_plot=False,
    save_file=False,
    hmap_vmin=1,
    hmap_vmax=1.2,
    title=None,
    add_fscore=False,
    fc_type="full",
    validity_threshold=None,
    n_arrays_fscores=431,
    return_df=False,
    parcellation='Schaefer',
    full_or_mean='mean',
    msst=True,
    param_filename_dict={},
    fscore_filename_dict={},
    **kwargs,
):
    """
    Visualize improvement factors according to Teeuw et al. (2021)
    In some cases, model_x represents mean and model_y represents FAModel
    """
    folder = get_scale_name_from_trait(trait_type)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    if scatter_or_hist == "hist" or network_imp:
        return_long = True
    elif scatter_or_hist == "scatter":
        if comp in ["full_sem", "method_effect"]:
            return_long = False
        elif comp in ["nogsr_gsr", "drop"]:
            return_long = True
    output_df = generate_summary_df(
        trait_type,
        control=control,
        n_arrays_dict=n_arrays_dict,
        return_long=return_long,
        drop_bool_list=drop_bool_list,
        day_cor=day_cor,
        add_model_fc=add_model_fc,
        include_no_method_effect=include_no_method_effect,
        parcellation=parcellation,
        full_or_mean=full_or_mean,
        msst=msst,
        **kwargs,
    )
    output_df.replace({"Total": "Overall"}, inplace=True)
    if add_fscore:
        print("Calculating correlation between FC and trait")
        add_df = calc_r_fscore_fc_trait(
            trait_type,
            kwargs.get("invalid_edge_file"),
            return_df=True,
            fc_type=fc_type,
            validity_threshold=validity_threshold,
            n_arrays=n_arrays_fscores,
            include_day_cor=day_cor,
            parcellation=parcellation,
            msst=msst,
            param_filename_dict=param_filename_dict,
            fscore_filename_dict=fscore_filename_dict,
            **kwargs,
        )
        output_df = pd.concat([output_df, add_df], axis=0)
        valid_edges = add_df.dropna(subset=["r"]).groupby("gsr_type")["edge"].unique()
        valid_edges_nogsr, valid_edges_gsr = valid_edges[0], valid_edges[1]
        output_df.query(
            '((gsr_type == "Without GSR") & (edge in @valid_edges_nogsr)) | ((gsr_type == "With GSR") & (edge in @valid_edges_gsr))',
            inplace=True,
        )

    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    output_df = pd.merge(output_df, edges_df, on="edge", how="left")

    if return_df:
        return output_df

    if not comp == "drop":
        fig_width = 5 * output_df["scale_name"].nunique()
    else:
        fig_width = 16

    def get_x_y_facet(comp, facet_x="gsr_type"):
        if comp == "drop":
            query_str1, query_str2 = "Original", "Modified"
            if visualize == "full_and_sem":
                facet_str = f"{facet_x} ~ model"
            else:
                facet_str = f"{facet_x} ~ scale_name"
        elif comp == "nogsr_gsr":
            query_str1, query_str2, facet_str = (
                "`Without GSR`",
                "`With GSR`",
                f"{facet_x} ~ scale_name",
            )
        elif comp == "full_sem":
            query_str1, query_str2, facet_str = (
                "mean",
                "both",
                f"{facet_x} ~ scale_name",
            )
            if add_model_fc:
                query_str3 = "fc"
        elif comp == "method_effect":
            query_str1, query_str2, facet_str = (
                "no_method_effect",
                "both",
                f"{facet_x} ~ scale_name",
            )

        return query_str1, query_str2, facet_str

    if scatter_or_hist == "scatter":

        def replace_model_str(df):
            df = df.replace({"both": "SEM", "mean": "Analyses on aggregate scores"})
            df["model"] = pd.Categorical(
                df["model"], categories=["Analyses on aggregate scores", "SEM"]
            )
            return df

        if comp == "full_sem":
            imp_column_x, imp_column_y, include_sem, group_var = (
                "mean",
                "both",
                True,
                "gsr_type",
            )

        if comp == "method_effect":
            imp_column_x, imp_column_y, include_sem, group_var = (
                "no_method_effect",
                "both",
                True,
                "gsr_type",
            )

        elif comp == "nogsr_gsr":
            imp_column_x, imp_column_y, include_sem, group_var = (
                "Without GSR",
                "With GSR",
                True,
                "model",
            )
            output_df = output_df.pivot(
                index=["edge", "scale_name", "model", "drop"],
                values="r",
                columns="gsr_type",
            ).reset_index()
            output_df = replace_model_str(output_df)

        elif comp == "drop":
            imp_column_x, imp_column_y, group_var = (
                "Original",
                "Modified",
                ["gsr_type", "model"],
            )
            if "sem" in visualize:
                include_sem = True
            elif visualize == "full":
                include_sem = False
            if trait_type == "cognition":
                target_scale_list = ["Fluid"]
            elif trait_type == "personality":
                target_scale_list = ["Openness"]
            output_df.query("scale_name in @target_scale_list", inplace=True)
            output_df = output_df.pivot(
                columns="drop",
                index=["edge", "gsr_type", "scale_name", "model"],
                values="r",
            ).reset_index()
            if visualize == "full":
                output_df = output_df.query('model == "mean"')
            elif visualize == "sem":
                output_df = output_df.query('model == "both"')
            elif visualize == "full_and_sem":
                output_df = replace_model_str(output_df)

        if comp == "drop":
            group_var_list = ["scale_name", "gsr_type", "model"]
        else:
            group_var_list = ["scale_name", group_var]
        df_labels = output_df.groupby(group_var_list, observed=True).apply(
            lm_reg,
            x=imp_column_x,
            y=imp_column_y,
            comp=comp,
            include_sem=include_sem,
            **{"y_min": y_min, "y_max": y_max},
        )
        new_line_marker = "\n" if imp_new_line else ""
        df_labels["imp_txt"] = [
            f" Improvement factor {new_line_marker} = {slope:.3f}"
            for slope in df_labels["slope"]
        ]

        if include_sem:
            summary_edges_n = output_df.groupby(group_var_list, observed=False)[
                "edge"
            ].count()
            df_labels = pd.merge(
                df_labels, summary_edges_n, left_index=True, right_index=True
            )
            df_labels["removed_percentage"] = df_labels["removed_n"] / df_labels["edge"]
        #    if drop_bool:
        #        df_labels_merged.dropna(inplace=True)
        if other_new_line:
            if include_sem:
                if comp in ["full_sem", "drop", "method_effect"]:
                    df_labels["other_txt"] = [
                        f"Pearson's r = {r:.3f} \nNumber of removed edges is {removed_n} \n ({removed_percentage:.2%} in selected edges) "
                        for r, removed_n, removed_percentage in zip(
                            df_labels["r"],
                            df_labels["removed_n"].astype(int),
                            df_labels["removed_percentage"],
                        )
                    ]
                if comp == "nogsr_gsr":
                    df_labels["other_txt"] = [
                        f"Pearson's r = {r:.3f}" for r in df_labels["r"]
                    ]
            else:
                r_values = df_labels["r"].values
                df_labels["other_txt"] = [f"Pearson's r = {r:.3f}" for r in r_values]

        df_labels.reset_index(inplace=True)

        def common_gg_func(gg_object, df_labels, df, facet_x="gsr_type", title=None):
            """
            Add common elements to gg object
            """
            query_str1, query_str2, facet_str = get_x_y_facet(comp, facet_x)
            smooth_query = f"{query_str1} > @y_min & {query_str1} < @y_max & {query_str2} > @y_min & {query_str2} < @y_max"
            if add_model_fc:
                smooth_query += f" & {query_str3} > @y_min & {query_str3} < @y_max"

            g = (
                gg_object
                + geom_point(size=0.001, alpha=0.1)
                + facet_grid(facet_str)
                + theme_bw()
                + geom_abline(intercept=0, slope=1, linetype="dashed")
                + scale_y_continuous(limits=[y_min, y_max])
                + scale_y_continuous(limits=[x_min, x_max])
                + coord_cartesian(xlim=[x_min, x_max], ylim=[y_min, y_max])
                + geom_smooth(
                    data=df.query(smooth_query), method="lm", se=False, color="red"
                )
            )
            if comp in ["full_sem", "drop", "method_effect"]:
                g = (
                    g
                    + geom_text(
                        aes(label="imp_txt", x=-np.Inf, y=y_max),
                        data=df_labels,
                        va="top",
                        ha="left",
                        size=8,
                    )
                    + geom_text(
                        aes(label="other_txt", x=np.Inf, y=y_min),
                        data=df_labels,
                        va="bottom",
                        ha="right",
                        size=6,
                    )
                )

            if comp == "nogsr_gsr":
                g = (
                    g
                    + geom_text(
                        aes(label="other_txt", x=-np.Inf, y=y_max),
                        data=df_labels,
                        va="top",
                        ha="left",
                        size=8,
                    )
                    + geom_text(
                        aes(label="imp_txt", x=np.Inf, y=y_min),
                        data=df_labels,
                        va="bottom",
                        ha="right",
                        size=6,
                    )
                )

            if title:
                g = g + ggtitle(title)

            return g

        if comp == "full_sem":
            g = ggplot(output_df, aes(x="mean", y="both"))
            g = common_gg_func(g, df_labels, output_df, facet_x="gsr_type", title=title)
            g = g + labs(
                x="Correlation estimated from aggregate scores",
                y="Structural correlation estimated in SEM",
            )

        elif comp == "nogsr_gsr":
            g = ggplot(output_df, aes(x="Without GSR", y="With GSR"))
            g = common_gg_func(g, df_labels, output_df, facet_x="model", title=title)
            g = g + labs(
                x="RSFC-trait correlation estimated without GSR",
                y="RSFC-trait correlation estimated after GSR",
            )

        elif comp == "method_effect":
            g = ggplot(output_df, aes(x="no_method_effect", y="both"))
            g = common_gg_func(g, df_labels, output_df, facet_x="gsr_type", title=title)
            g = g + labs(
                x="RSFC-trait correlation estimated without method effect",
                y="RSFC-trait correlation estimated \nwith error terms on the same day",
            )

        elif comp == "drop":
            if visualize == "full":
                xlabel = "Correlation between sum scores of the original model\n of trait measure and aggreated score of RSFC"
                ylabel = "Correlation between sum scores of \nthe modified model\n of trait measure and aggreated score of RSFC"
            elif visualize == "sem":
                xlabel = "Correlation estimated from the original model\n of trait measure and RSFC"
                ylabel = "Correlation estimated from \nthe modified model\n of trait measure and RSFC"
            elif visualize == "full_and_sem":
                xlabel = "RSFC-trait correlation estimated from \noriginal model"
                ylabel = "RSFC-trait correlation estimated from \nmodified model"
            g = ggplot(output_df, aes(x="Original", y="Modified"))
            g = common_gg_func(g, df_labels, output_df, title=title)
            g = (
                g
                + labs(x=xlabel, y=ylabel)
                + theme(
                    axis_title_y=element_text(ha="center", va="center"),
                    axis_title_x=element_text(ha="center", va="center")
                    #  plot_margin_left=0.0375,
                    #  plot_margin_bottom=0.025
                )
            )

    elif scatter_or_hist == "hist":

        def draw_hists(df, color_var, title=None):
            """
            Inner function for drawing histograms
            """
            legend_position = "none" if title else "bottom"
            if df["model"].nunique() == 2:
                linetype_values = ["dashed", "solid"]
            elif df["model"].nunique() == 3:
                linetype_values = ["dashed", "solid", "solid"]

            g = (
                ggplot(df, aes(x="r", fill=color_var, linetype=color_var))
                + geom_density(alpha=0.2)
                + facet_grid("gsr_type ~ scale_name")
                + theme_bw()
                + theme(
                    legend_title=element_blank(),
                    legend_position=legend_position,
                    axis_text_x=element_text(angle=45),
                )
                + scale_x_continuous(limits=[x_min, x_max])
                + geom_vline(xintercept=0, linetype="dashed", size=0.2)
                + coord_cartesian(xlim=[x_min, x_max], ylim=[0, 15])
                + labs(x="RSFC-trait correlation (r)", y="Scaled density (a.u.)")
                + scale_linetype_manual(values=linetype_values)
            )
            if title:
                g = g + ggtitle(title)
            return g

        output_df = replace_model_sem_or_full(output_df)
        if comp == "full_sem":
            g = draw_hists(output_df, "model", title)
        elif comp == "drop":
            g = draw_hists(output_df, "drop", title)

    if scatter_or_hist:
        if comp == "nogsr_gsr":
            width_extend_ratio = 1.2
        else:
            width_extend_ratio = 1
        g = g + theme(
            figure_size=(fig_width / 2.54 * width_extend_ratio, fig_height / 2.54)
        )

        try:
            g.show()
        except:
            pass

        if save_file:
            save_filepath = op.join(atlas_dir, folder, "figures", filename_fig)
            if op.isfile(save_filepath):
                os.remove(save_filepath)
            g.save(
                filename=save_filepath,
                height=fig_height,
                width=fig_width * width_extend_ratio,
                units="cm",
            )

        if return_plot:
            return g

    if network_imp:

        def vis_imp_factor_networks(df, trait_type, comp, filename_fig, save_file):
            """
            Inner function for drawing heatmap representing network-level improvement factors
            """
            edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
            merged_df = (
                pd.merge(df, edges_df, on="edge")
                .pivot(
                    index=["edge", "gsr_type", "scale_name", "node1_net", "node2_net"],
                    values="r",
                    columns="model",
                )
                .reset_index()
            )

            folder = select_folder_from_trait(trait_type)
            scale_n = df["scale_name"].nunique()

            fig, axes = plt.subplots(
                scale_n, 2, sharex=True, sharey=True, figsize=(12, 2.4 * scale_n)
            )
            cbar_ax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
            x, y, _ = get_x_y_facet(comp)
            for i, scale_name in enumerate(merged_df["scale_name"].unique()):
                for j, gsr_type in enumerate(merged_df["gsr_type"].unique()):
                    subset_wide_df = (
                        merged_df.query(
                            "scale_name == @scale_name & gsr_type == @gsr_type"
                        )
                        .groupby(["node1_net", "node2_net"], observed=False)[
                            ["node1_net", "node2_net", x, y]
                        ]
                        .apply(
                            lm_reg, x, y, comp, True, **{"y_min": -0.25, "y_max": 0.25}
                        )
                        .reset_index()
                        .pivot(index="node1_net", values="slope", columns="node2_net")
                    )
                    iteration = i * j
                    sns.heatmap(
                        subset_wide_df,
                        annot=True,
                        fmt=".2f",
                        ax=axes[i, j],
                        vmin=hmap_vmin,
                        vmax=hmap_vmax,
                        cbar_ax=cbar_ax if iteration else None,
                        cbar=iteration == 1,
                        annot_kws={"fontsize": 8},
                        cmap="Oranges",
                    )
                    axes[i, j].set_title(f"{scale_name} {gsr_type}")
                    axes[i, j].tick_params(axis="x", rotation=45)
                    axes[i, j].set(xlabel=None, ylabel=None)

            fig.tight_layout(rect=[0, 0, 0.9, 1])
            if save_file:
                print("Saving figure")
                fig.savefig(
                    op.join(atlas_dir, folder, "figures", f"{filename_fig}.png")
                )

        vis_imp_factor_networks(output_df, trait_type, comp, filename_fig, save_file)


def conduct_power_analysis(
    trait_type_list,
    alpha_list,
    n_list=None,
    power_list=None,
    control=["age", "gender", "MeanRMS"],
    parcellation='Schaefer',
    trait_equal_loadings=False,
    msst=True,
    invalid_edge_file_dict={'nogs': None, 'gs': None},
    sample_n=None,
    fold=None,
    skip_scale=None,
#    cor_thres=0.3,
    fc_1st_load_thres_list=[0.4, 0.9],
    fc_2nd_load_thres_list=[0.5, 1],
    trait_load_thres_list=[0, 1],
    vmin=0,
    vmax=3,
    figsize=(12, 4),
    fig_suptitle=None,
    save_filename=None,
    #drop_bool_list=[False],
    **kwargs,
):
    """
    Calculate power or required sample size from estimates of correlation
    """
    output_df = pd.DataFrame()
    for trait_type in trait_type_list:
        folder = get_scale_name_from_trait(trait_type)
        trait_scale_name = get_scale_name_from_trait(trait_type, publication=True)
        # Essential data
        (
            gsr_type_list,
            scale_name_list,
            n_list_out,
            alpha_list_out,
            power_list_out,
            model_list,
            edge_list,
            r_list,
        ) = ([], [], [], [], [], [], [], [])

        for alpha in alpha_list:
            if n_list is not None:
                iterator_list, print_str = n_list, "N"
            elif power_list is not None:
                iterator_list, print_str = power_list, "Power"
            for element in iterator_list:
                print(f"Processing {print_str}: {element} and Alpha: {alpha}")
                df = generate_summary_df(
                    trait_type,
                    control=["age", "gender", "MeanRMS"],
                    return_long=False,
                    drop_item_dict={},
                    parcellation=parcellation,
                    full_or_mean='mean',
                    msst=True,
                    trait_equal_loadings=trait_equal_loadings,
                    split_half_family=False,
                    fold=fold,
                    invalid_edge_file_dict=invalid_edge_file_dict,
                    sample_n=sample_n,
                    skip_scale=skip_scale,
     #               cor_thres=cor_thres,
                    fc_1st_load_thres_list=fc_1st_load_thres_list,
                    fc_2nd_load_thres_list=fc_2nd_load_thres_list,
                    trait_load_thres_list=trait_load_thres_list,
                    **kwargs,
                )
                df = df.melt(
                        id_vars=['edge', 'scale_name', 'gsr_type'], 
                        value_vars=['mean', 'both'], 
                        value_name='r', 
                        var_name='model'
                        )
                df_length = len(df)

                if n_list is not None:
                    powers = calculate_power_from_r(element, df["r"], alpha)
                elif power_list is not None:
                    ns = calculate_n_from_power(element, df["r"], alpha)

                alpha_list_out.append([alpha] * df_length)

                if n_list is not None:
                    power_list_out.append(powers.tolist())
                elif power_list is not None:
                    power_list_out.append([element] * df_length)

                if power_list is not None:
                    n_list_out.append(ns.values.tolist())
                elif n_list is not None:
                    n_list_out.append([element] * df_length)
                gsr_type_list.append(df["gsr_type"].astype(str).values.tolist())
                scale_name_list.append(df["scale_name"].astype(str).values.tolist())
                model_list.append(df["model"].values.tolist())
                edge_list.append(df["edge"].values.tolist())
                r_list.append(df["r"].values.tolist())
        all_df = pd.DataFrame(
            {
                "n": sum(n_list_out, []),
                "gsr_type": sum(gsr_type_list, []),
                "scale_name": sum(scale_name_list, []),
                "alpha": sum(alpha_list_out, []),
                "power": sum(power_list_out, []),
                "model": sum(model_list, []),
                "edge": sum(edge_list, []),
                "r": sum(r_list, []),
            }
        )
        all_df.replace(
            {"both": "Corrected correlation", "mean": "Uncorrected correlation"}, inplace=True
        )

        def set_categories(df):
            """
            Inner function for setting categories
            """
            df["gsr_type"] = pd.Categorical(
                df["gsr_type"], categories=["Without GSR", "With GSR"]
            )
            df["model"] = pd.Categorical(
                df["model"], categories=["Uncorrected correlation", "Corrected correlation"]
            )
            return df

        all_df = recat_df_from_trait(all_df, trait_type)
        all_df = set_categories(all_df)
        all_df['trait_scale_name'] = trait_scale_name
        output_df = pd.concat([output_df, all_df], axis=0)
    edges_df = get_edge_summary(parcellation=parcellation, network_hem_order=True)
    output_df = pd.merge(output_df, edges_df, on='edge')
    output_df = output_df.pivot(
            index=['edge', 'node1_net', 'node2_net', 'gsr_type', 'trait_scale_name', 'scale_name', 'alpha', 'power'], 
            values='n', 
            columns='model'
            ).reset_index()
    output_df['n_ratio'] = output_df['Uncorrected correlation'] / output_df['Corrected correlation']
    output_df = output_df.groupby(['gsr_type', 'node1_net', 'node2_net'])['n_ratio'].median().reset_index()
    
#    mat_df = get_empty_df_for_hmap(parcellation)
    atlas_dir = ATLAS_DIR_DICT.get(parcellation)
    
    # Loop for GSR type
    fig, axes = plt.subplots(ncols=2, figsize=figsize, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .015, .4])
    for i, gsr_type in enumerate(['Without GSR', 'With GSR']):
        output_df_gsr = output_df.query('gsr_type == @gsr_type').pivot(index='node1_net', columns='node2_net', values='n_ratio')
        target_ax = axes[i]
        sns.heatmap(
            output_df_gsr,
            annot=True,
            fmt=".2f",
            ax=target_ax,
            vmin=vmin,
            vmax=vmax,
            annot_kws={"fontsize": 8},
            cmap="Oranges",
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax
            )
        target_ax.set_title(gsr_type)
        target_ax.tick_params(axis="x", rotation=45)
        target_ax.set(xlabel=None, ylabel=None)
    if fig_suptitle is not None:
        fig.suptitle(fig_suptitle)
    fig.tight_layout(rect=[0, 0, .9, 1])
       # wide_edges_df = get_wide_df_hmap(output_df_gsr, value_col_name="n_ratio")
       # mat_df = fill_mat_df(mat_df, wide_edges_df, gsr_type)
#    nodes_df = get_nodes_df(parcellation)
#    mat_df[mat_df > 100] = np.nan
#    draw_hmaps_fcs(
#        mat_df, 
#        nodes_df, 
#        cmap='Oranges', 
#        #save_filename=save_filename, 
#        cbar_ax=False,
#        cbar=True,
#        ax=ax, 
#        #num_iter=num_iter,
#        parcellation=parcellation,
#        add_custom_cbar=False
#    )
    if save_filename:
        fig.savefig(op.join(atlas_dir, 'figures', f'{save_filename}.png'))
    return output_df


def recat_df_from_trait(df, trait_type):
    """
    Reset order of categories considering trait_type
    """
    if trait_type == "personality":
        cat_list = NEO_FFI_SCALES
    elif trait_type == "mental":
        cat_list = ["All", "Internalizing", "Externalizing", "Others"]
#        df = df.replace("Total", "Overall")
    elif trait_type == "cognition":
        cat_list = NIH_COGNITION_SCALES 
#        cat_list = [
#            i.replace("All", "Overall") if i == "All" else i
#            for i in NIH_COGNITION_SCALES
#        ]
#        df = df.replace("Total", "Overall")

    df["scale_name"] = pd.Categorical(df["scale_name"], categories=cat_list)

    return df


def wrapper_of_visualize_power_analysis(
    alpha_list,
    n_list=None,
    n_arrays_dict=None,
    fig_type="boxplot",
    save_filename=None,
    fig_height=5,
    **kwargs,
):
    """
    Wrapper function of visualize_power_analysis() for publication
    """
    g_list = []
    for trait_type in ["cognition", "mental", "personality"]:
        trait_title = get_scale_name_from_trait(trait_type, publication=True)
        g = visualize_power_analysis(
            trait_type,
            alpha_list,
            n_list=n_list,
            n_arrays_dict=n_arrays_dict,
            return_plot=True,
            title=trait_title,
            fig_type=fig_type,
            **kwargs,
        )
        g_list.append(g)
    combine_gg_list(
        g_list,
        fig_height=fig_height,
        filename_fig=save_filename,
        legend_comp=True,
        comp_target="cor_est",
    )


def visualize_power_analysis(
    trait_type,
    alpha_list,
    n_list=None,
    power_list=None,
    control=["age", "gender", "MeanRMS"],
    n_arrays_dict=None,
    drop_bool_list=[False],
    fig_filename=None,
    fig_type="summary",
    summary_func="median",
    return_plot=False,
    title=None,
    **kwargs,
):
    """
    Draw plots representing differences of power between statistical methods
    """
    power_df = conduct_power_analysis(
        trait_type,
        alpha_list,
        n_list,
        power_list,
        control=control,
        n_arrays_dict=n_arrays_dict,
        drop_bool_list=drop_bool_list,
        **kwargs,
    )
    fig_width = 4 * power_df["scale_name"].nunique()
    print("Drawing figure")

    if summary_func == "median":
        apply_func = np.median
    elif summary_func == "mean":
        apply_func = np.mean

    if n_list:
        legend_position = "none" if title else "bottom"
        if fig_type == "boxplot":
            label_y = "Power"
            g = (
                ggplot(
                    power_df,
                    aes(
                        x="factor(n)",
                        y="power",
                    ),
                )
                #    + geom_violin(aes(fill='model'), position=position_dodge(0.9), alpha=0.2, style='left')
                + geom_boxplot(
                    aes(fill="model"),
                    alpha=0.1,
                    width=0.75,
                    outlier_color="green",
                    outlier_size=0.1,
                    outlier_alpha=0.1,
                    outlier_stroke=0.1,
                )
                + stat_summary(aes(color="model"), fun_y=apply_func, geom="point")
                + stat_summary(
                    aes(color="model", group="model"), fun_y=apply_func, geom="line"
                )
                + geom_hline(yintercept=[0.80, 0.90], linetype="dashed", size=0.25)
                + scale_y_continuous(breaks=[0, 0.5, 0.8, 0.9, 1.0])
                #                + geom_jitter(aes(color='model'), width=0.2, size=.0001, stroke=.00001, alpha=0.1)
            )

        elif fig_type == "summary":
            label_y = f"{summary_func.title()} of power"
            g = (
                ggplot(
                    power_df,
                    aes(
                        x="factor(n)",
                        y="power",
                    ),
                )
                + stat_summary(aes(color="model"), fun_y=apply_func, geom="point")
                + stat_summary(
                    aes(color="model", group="model"), fun_y=apply_func, geom="line"
                )
            )

        g = (
            g
            + facet_grid("gsr_type ~ scale_name")
            + theme_bw()
            + scale_x_discrete(labels=n_list)
            + labs(x="Hypothetical sample size", y=label_y)
            + theme(
                legend_title=element_blank(),
                legend_position=legend_position,
                figure_size=(fig_width * 1.2 / 2.54, 10 / 2.54),
                axis_text_x=element_text(angle=45),
            )
        )

    elif power_list:
        power_df.replace(np.Inf, np.nan, inplace=True)
        power_df.dropna(subset="n", inplace=True)
        g = (
            ggplot(
                power_df,
                aes(
                    x="model",
                    y="n",
                ),
            )
            + geom_violin(aes(fill="power"), alpha=0.2, width=0.75)
            + geom_boxplot(
                aes(fill="power"),
                alpha=0.2,
                width=0.75,
                outlier_color=None,
                outlier_size=0,
            )
            + geom_jitter(
                aes(color="power"), width=0.2, size=0.0001, stroke=0.00001, alpha=0.1
            )
            + facet_grid("gsr_type ~ scale_name")
            + theme_bw()
            + labs(x="Model", y="Required sample size")
            + theme(
                legend_title=element_blank(),
                legend_position="bottom",
                figure_size=(fig_width * 1.2 / 2.54, 10 / 2.54),
            )
            + coord_cartesian(ylim=[0, 100000])
        )

    if title:
        g = g + ggtitle(title)

    try:
        g.show()
    except:
        pass

    if return_plot:
        return g

    if fig_filename is not None:
        folder = get_scale_name_from_trait(trait_type)
        save_path = op.join(SCHAEFER_DIR, folder, "figures", f"{fig_filename}.png")
        if op.isfile(save_path):
            os.remove(save_path)
        g.save(
            save_path,
            height=10,
            width=fig_width * 1.2,
            units="cm",
        )


def make_categories_in_df(df, column, categories):
    df[column] = pd.Categorical(df[column], categories=categories)
    return df


def vis_hist_of_correlation(
    filename_cor_list_dict_mean_pca,
    filename_cor_list_dict_models,
    removed_edges_dict,
    trait_type,
    fig_height=12,
    fig_width=20,
):
    """
    draw histograms of correlation of mean, pca and FA models
    """
    long_df_mean_pca = get_df_cor_of_mean_pca(filename_cor_list_dict_mean_pca)
    long_df_models = get_df_cor_of_models(
        filename_cor_list_dict_models, removed_edges_dict
    )
    long_df = pd.concat([long_df_mean_pca, long_df_models], axis=0)
    long_df = replace_and_reorder_column(
        long_df,
        var_name_dict={
            "gsr_type": GSR_DICT,
            "drop": DROP_DICT,
            "model": MODEL_DICT_ADD_MEAN_PCA,
        },
    )

    trait_scale_name = get_scale_name_from_trait(trait_type)
    subscales = get_subscale_list(get_scale_name_from_trait(trait_type))
    long_df["scale_name"] = pd.Categorical(long_df["scale_name"], categories=subscales)

    filename_sample = (
        filename_cor_list_dict_models.get("nogs").get(subscales[0]).get("not_dropped")
    )
    # draw figures
    input_params_ggplot_list = [
        "Correlation",
        "model",
        "drop",
        "gsr_type",
        "scale_name",
    ]
    if trait_type == "mental":
        input_params_ggplot_list = [
            None if i == "drop" else i for i in input_params_ggplot_list
        ]

    g = draw_ggplot_density(long_df, *input_params_ggplot_list)
    g.show()
    remove_str_dict = {"scale": True, "dt_time": True, "gsr_type": True}
    save_ggplot(
        g,
        op.join(SCHAEFER_DIR, trait_scale_name, "figures"),
        filename_sample,
        remove_str_dict,
        fig_height,
        fig_width,
    )


def draw_ggplot_density(
    long_df, x_axis_value_str, color_str, linetype_str, facet_y_str, facet_x_str
):
    """draw density plot using ggplot"""
    if linetype_str is not None:
        g_base = ggplot(
            long_df, aes(x_axis_value_str, color=color_str, linetype=linetype_str)
        )
    else:
        g_base = ggplot(long_df, aes(x_axis_value_str, color=color_str))

    g = (
        g_base
        + geom_density()
        + facet_grid(f"{facet_y_str} ~ {facet_x_str}")
        + theme_bw()
        + labs(y="Density")
        + theme(legend_title=element_blank(), axis_text_x=element_text(angle=45))
    )
    return g


def investigate_factor_scores(model_type_list, **kwargs):
    """
    investigate correlations and predictive powers using factor scores
    """
    # get factor score estimates
    output_cor_df = pd.DataFrame()
    subscales = get_subscale_list(
        get_scale_name_from_trait(kwargs["trait_type_list"][0])
    )
    model_type_list_joined_str = "_".join(model_type_list)
    for model_str in model_type_list:
        file_dict = get_latest_files_with_args(
            kwargs["trait_type_list"],
            kwargs["edge_n"],
            kwargs["sample_n"],
            kwargs["est_method"],
            kwargs["data_type"],
            [model_str],
            kwargs.get("drop_vars_list"),
        )
        model_type = "model_" + model_str
        for g, gsr_type in enumerate(["nogs", "gs"]):
            gsr_suffix = generate_gsr_suffix(gsr_type)
            for col, scale_name in enumerate(file_dict.get(gsr_type)):
                # generate combined dataframe using dropped variables
                for drop in ["not_dropped", "dropped"]:
                    filename_fs = file_dict.get(gsr_type).get(scale_name).get(drop)
                    if filename_fs is not None:
                        fs_array = copy_memmap_output_data(filename_fs)
                        misfit_set = get_set_of_locally_globlly_misfit_edges(
                            filename_fs.replace(model_str, model_type_list_joined_str),
                            kwargs["error_vars_dict"],
                            kwargs["fit_indices_thresholds_dict"],
                            [model_type],
                            kwargs["cor_min_max_dict"],
                        )
                        # calculate correlations between factor score estimates
                        out_cor_array = np.empty(shape=fs_array.shape[0])
                        out_cor_array[:] = np.nan
                        print(
                            model_str,
                            scale_name,
                            gsr_type,
                            len(misfit_set.get(model_type)),
                        )
                        print(misfit_set.get(model_type))
                        for i in range(fs_array.shape[0]):
                            if i not in misfit_set.get(model_type):
                                target_array = np.squeeze(fs_array[i, ...])
                                if kwargs["cor_type"] == "pearson":
                                    cor = np.corrcoef(target_array.T)[0, 1]
                                elif kwargs["cor_type"] == "spearman":
                                    cor = spearmanr(target_array).correlation
                                out_cor_array[i] = cor
                        cor_df = array_to_df(
                            out_cor_array,
                            ["Correlation"],
                            **{
                                "scale_name": scale_name,
                                "gsr_type": gsr_type,
                                "drop": drop,
                                "model": model_type,
                            },
                        )
                        output_cor_df = pd.concat([output_cor_df, cor_df], axis=0)
    output_cor_df = replace_and_reorder_column(
        output_cor_df, var_name_dict={"gsr_type": GSR_DICT, "model": MODEL_DICT}
    )
    output_cor_df["scale_name"] = pd.Categorical(
        output_cor_df["scale_name"], categories=subscales
    )
    g = draw_ggplot_density(
        output_cor_df, "Correlation", "model", "drop", "gsr_type", "scale_name"
    )
    g.show()


class LoopDrawHistgrams:
    def __init__(self, input_filename_dict):
        self.input_dict = input_filename_dict

    def make_fig(self, input_filename_dict):
        n_col = check_input_dict_gsr_types(filename_cor_list_dict)
        fig, axes = plt.subplots(2, n_col, figsize=fig_size, sharex=True, sharey=True)


def multiple_replace(replacements, text):
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacements[mo.group()], text)


def draw_hist_of_omega_onlyFC(
    filename_list_dict, fig_height=12, fig_width=20, return_plot=True, add_marker=False
):
    """
    Draw density distributions of omega coefficients estimated in model onlyFC
    """
    # Create output df
    output_df = pd.DataFrame()
    for g, gsr_type in enumerate(["nogs", "gs"]):
        for f, filename in enumerate(filename_list_dict[gsr_type]):
            omegas = calc_omega_2d_array_from_filename(filename)
            omega_pd = pd.DataFrame()

            omega_pd["omega"] = omegas
            omega_pd["gsr_type"] = gsr_type

            dayCor = True if "DayCor" in filename else False
            PE = True if "PE" in filename else False
            orderCor = True if "OrderInDay" in filename else False
            model = check_models_rsfc(True, dayCor, PE, orderCor)
            # if PE and not dayCor:
            #    model = 'Model RSFC 3'
            # elif dayCor and not PE:
            #    model = 'Model RSFC 2'
            # else:
            #    model = 'Model RSFC 1'
            omega_pd["model"] = model
            output_df = pd.concat([output_df, omega_pd], axis=0)
    output_df = replace_and_reorder_column(output_df, "gsr_type", GSR_DICT)
    model_suffix = "b" if not add_marker else "a"
    output_df = output_df.replace(
        {
            "Model RSFC 2": f"Model RSFC 2-{model_suffix}",
            "Model RSFC 3": f"Model RSFC 3-{model_suffix}",
            "Model RSFC 4": f"Model RSFC 4-{model_suffix}",
        }
    )
    print("Drawing figure")
    g = (
        ggplot(output_df, aes("omega"))
        + geom_density()
        + facet_grid("gsr_type ~ model")
        + scale_x_continuous(limits=[0, 1])
        + theme_bw()
        + labs(x="Omega coefficient", y="Scaled density (a.u.)")
        + theme(axis_text_x=element_text(angle=45))
    )

    if return_plot:
        g = (
            g
            + ggtitle("(III) Omega composite reliability")
            + theme(plot_title=element_text(hjust=0))
        )
        return g

    g.show()
    g.save(
        op.join(SCHAEFER_DIR, "reliability", "figures", f"omega_dist.png"),
        height=fig_height,
        width=fig_width,
        units="cm",
    )

    return output_df


def draw_hist_of_fit_indices_onlyFC(
    filename_fit_list_dict: dict[str : list[str]],
    fit_indices_dict_with_xlim: dict[FitIndicesList : list[float, float, float]] = None,
    dtype_memmap="float32",
    save_fig=True,
    fig_height=12,
    fig_width=20,
    filename_prefix=None,
    return_plot=True,
    add_marker=False,
):
    # output df
    output_df = pd.DataFrame()
    # loop for fit indices
    for i, fit_index in enumerate(fit_indices_dict_with_xlim.keys()):
        # get index of fit index
        if fit_indices_dict_with_xlim is not None:
            fit_index_index = [
                FIT_INDICES.index(i) for i in fit_indices_dict_with_xlim.keys()
            ]
        if fit_index is not None:
            fit_index_index = FIT_INDICES.index(fit_index)
        lower, upper, binwidth = fit_indices_dict_with_xlim.get(fit_index)

        # create an empty dataframe
        filename_fit_list_stored = []
        # loop for gsr types
        for g, gsr_type in enumerate(["nogs", "gs"]):
            gsr_suffix = generate_gsr_suffix(gsr_type)
            filename_fit_list = filename_fit_list_dict.get(gsr_type)
            if filename_fit_list is not None:
                for filename_fit in filename_fit_list:
                    print(filename_fit)
                    filename_fit_list_stored.append(filename_fit)
                    (
                        model_fa_list,
                        trait_type,
                        scale_name,
                        gsr_type,
                        num_iter,
                        phase_encoding,
                        day_cor,
                        order_in_day,
                    ) = get_strings_from_filename(
                        filename_fit,
                        [
                            "model_type",
                            "trait_type",
                            "scale_name",
                            "gsr_type",
                            "num_iter",
                            "phase_encoding",
                            "day_cor",
                            "order_in_day",
                        ],
                        include_nofa_model=False,
                    )
                    fit_data = copy_memmap_output_data(
                        filename_fit,
                    )
                    fit_data = fit_data[:, fit_index_index, :]
                    fit_data = fit_data[(fit_data > lower) & (fit_data < upper)]
                    # create df to merge
                    fit_data_pd = pd.DataFrame(fit_data, columns=["value"])
                    fit_data_pd["gsr_type"] = gsr_type
                    fit_data_pd["fit_index"] = fit_index

                    if day_cor and not phase_encoding:
                        model = "Model RSFC 2"
                    elif phase_encoding and not day_cor:
                        model = "Model RSFC 3"
                    elif order_in_day and not day_cor and not phase_encoding:
                        model = "Model RSFC 4"
                    elif "Time" in filename_fit:
                        model = "Model RSFC 5"
                    else:
                        model = "Model RSFC 1"
                    fit_data_pd["model"] = model
                    # concatenate df to produce output for visualization
                    output_df = pd.concat([output_df, fit_data_pd], axis=0)
    # reorder and replace columns in df
    output_df = replace_and_reorder_column(output_df, "gsr_type", GSR_DICT)
    output_df["fit_index"] = pd.Categorical(
        output_df["fit_index"], categories=FIT_INDICES_PUB
    )
    model_suffix = "b" if not add_marker else "a"
    output_df = output_df.replace(
        {
            "Model RSFC 2": f"Model RSFC 2-{model_suffix}",
            "Model RSFC 3": f"Model RSFC 3-{model_suffix}",
            "Model RSFC 4": f"Model RSFC 4-{model_suffix}",
        }
    )
    # draw figure
    print("Drawing figure")
    g = (
        ggplot(output_df, aes("value", color="model"))
        + geom_density()
        + facet_grid("gsr_type ~ fit_index", scales="free")
        + theme_bw()
        + theme(legend_title=element_blank(), axis_text_x=element_text(angle=45))
        + labs(x="Value", y="Scaled density (a.u.)")
    )

    if return_plot:
        g = (
            g
            + ggtitle("(II) Global fit indices")
            + theme(plot_title=element_text(hjust=0))
        )
        return g

    g.show()
    # set filename for saving figure
    remove_input_dict = {
        "scale": True,
        "dt_time": True,
        "gsr_type": True,
        "trait": True,
        "day_cor": True,
        "phase_encoding": True,
    }
    filename = remove_strings_from_filename(filename_fit, **remove_input_dict)
    g.save(
        filename=op.join(
            SCHAEFER_DIR, "reliability", "figures", f"{filename_prefix}{filename}.png"
        ),
        height=fig_height,
        width=fig_width,
        units="cm",
    )


def save_ggplot(plot, save_dir, filename, remove_str_dict, fig_height, fig_width):
    filename_mod = remove_strings_from_filename(filename, **remove_str_dict)
    plot.save(
        filename=op.join(save_dir, f"{filename_mod}.png"),
        height=fig_height,
        width=fig_width,
        units="cm",
    )


def save_rdata_file(pd_df, filename):
    with (ro.default_converter + pandas2ri.converter).context():
        r_from_pd_df = ro.conversion.get_conversion().py2rpy(pd_df)
    ro.r.assign("my_df", r_from_pd_df)
    ro.r(f"save(my_df, file='{filename}')")


def remove_strings_from_filename(filename: str, remove_prefix=False, **kwargs):
    if kwargs.get("scale"):
        filename = re.sub("Scale_[A-Za-z]+_", "", filename)
    if kwargs.get("dt_time"):
        filename = re.sub(" \d{2}:\d{2}:\d{2}", "", filename)
    if kwargs.get("trait"):
        filename = re.sub("Trait_[A-Za-z]+_", "", filename)
    if kwargs.get("day_cor"):
        filename = filename.replace("_DayCor", "")
    if kwargs.get("phase_encoding"):
        filename = filename.replace("_PE", "")
    if kwargs.get("gsr_type"):
        filename = re.sub("_nogs|_gs", "", filename)

    if remove_prefix:
        for prefix in ["pearson_", "fit_indices"]:
            filename = filename.replace(prefix, "")

    return filename


def save_df_for_fit_indices_gsr(
    filename_fit_list_dict: dict[str : list[str]],
    fit_indices_list,
    dtype_memmap="float32",
    **kwargs,
) -> None:
    """
    function for drawing histograms or kernel density plots of fit indices
    fit_indices_dict_with_xlim should be specified when using this function alone
    """
    # check input dictionary
    n_col = check_input_dict_gsr_types(filename_fit_list_dict)
    # loop for fit index
    for i, fit_index in enumerate(fit_indices_list):
        # get index of fit index
        if fit_index is not None:
            fit_index_index = FIT_INDICES.index(fit_index)

        # create an empty dataframe
        output_df = pd.DataFrame()
        filename_fit_list = []
        # loop for gsr types
        for g, gsr_type in enumerate(["nogs", "gs"]):
            gsr_suffix = generate_gsr_suffix(gsr_type)
            subscale_list = filename_fit_list_dict.get(gsr_type)
            for col, scale_name in enumerate(subscale_list):
                # generate combined dataframe using dropped variables

                for drop in ["not_dropped", "dropped"]:
                    filename_fit = (
                        filename_fit_list_dict.get(gsr_type).get(scale_name).get(drop)
                    )
                    if filename_fit is not None:
                        print(filename_fit)
                        filename_fit_list.append(filename_fit)
                        (
                            model_fa_list,
                            trait_type,
                            scale_name,
                            gsr_type,
                            num_iter,
                            phase_encoding,
                            day_cor,
                        ) = get_strings_from_filename(
                            filename_fit,
                            [
                                "model_type",
                                "trait_type",
                                "scale_name",
                                "gsr_type",
                                "num_iter",
                                "phase_encoding",
                                "day_cor",
                            ],
                            include_nofa_model=False,
                        )

                        fit_data = copy_memmap_output_data(
                            filename_fit,
                        )
                        fit_data = fit_data[:, fit_index_index, :]
                        fit_pd_data_exist = pd.DataFrame(
                            fit_data, columns=model_fa_list
                        )

                        def add_columns_to_df(df):
                            """add columns to df in a loop"""
                            df["drop"] = drop
                            df["scale_name"] = scale_name
                            df["gsr_type"] = gsr_type
                            return df

                        fit_pd_data_exist = add_columns_to_df(fit_pd_data_exist)

                        trait_scale_name = get_scale_name_from_trait(trait_type)
                        output_df = pd.concat([output_df, fit_pd_data_exist], axis=0)

        output_df = output_df.reset_index().rename(columns={"index": "edge"})
        long_df = output_df.melt(
            id_vars=["edge", "gsr_type", "scale_name", "drop"],
            value_vars=model_fa_list,
            var_name="model",
        )
        # replace and reorder columns
        long_df = replace_and_reorder_column(long_df, "gsr_type", GSR_DICT)
        long_df = replace_and_reorder_column(long_df, "drop", DROP_DICT)
        long_df = replace_and_reorder_column(long_df, "model", MODEL_DICT)

        long_df["scale_name"] = pd.Categorical(
            long_df["scale_name"], categories=subscale_list
        )
        save_dir = op.join(SCHAEFER_DIR, trait_scale_name, "data", "RData")
        os.makedirs(save_dir, exist_ok=True)

        if any("drop" in i for i in filename_fit_list):
            filename_fit_list = [i for i in filename_fit_list if "drop" in i]
        filename_fit = filename_fit_list[0]
        replace_dict = {
            "fit_indices_": "",
            "nogs_": "",
            "gs_": "",
        }
        filename_fit_replaced = multiple_replace(replace_dict, filename_fit)
        filename_fit_replaced = re.sub("Scale_[A-Za-z]+_", "", filename_fit_replaced)
        filename_fit_replaced = re.sub(" \d{2}:\d{2}:\d{2}", "", filename_fit_replaced)

        filename = f"{fit_index}_{filename_fit_replaced}.RData"
        print(f"Saving {filename}")
        save_rdata_file(long_df, op.join(save_dir, filename))
        print("Complete saving.")


def draw_ggplot_fit_indices():
    # reorder scales
    # make figure
    if trait_scale_name == "ASR":
        mapping_density = aes(y=after_stat("density"), color="model")
    else:
        mapping_density = aes(y=after_stat("density"), linetype="drop", color="model")

    if common_y_axis:
        facet_ggplot = facet_grid(rows="gsr_type", cols="scale_name")
    else:
        facet_ggplot = facet_wrap(
            "~ gsr_type + scale_name", scales="free_y", ncol=len(subscale_list)
        )

    g = (
        ggplot(long_df, aes(x="value"))
        + geom_density(mapping=mapping_density)
        + facet_ggplot
        + theme_classic()
        + labs(x=fit_index, y="Density")
        + theme(
            axis_text_x=element_text(angle=45),
            # axis_text_y=element_text(size=0),
            legend_title=element_blank(),
            # axis_ticks_y=element_blank()
        )
        + scale_x_continuous(limits=[lower, upper])
    )
    g.show()

    if save_fig:
        kde_suffix = "_kde" if add_kde else ""
        common_y_suffix = "_commonY" if common_y_axis else ""
        if any("drop" in i for i in filename_fit_list):
            filename_fit_list = [i for i in filename_fit_list if "drop" in i]
        filename_fit = filename_fit_list[0]
        replace_dict = {
            "fit_indices_": "",
            "nogs_": "",
            "gs_": "",
        }
        filename_fit_replaced = multiple_replace(replace_dict, filename_fit)
        filename_fit_replaced = re.sub("Scale_[A-Za-z]+_", "", filename_fit_replaced)
        filename_fit_replaced = re.sub(" \d{2}:\d{2}:\d{2}", "", filename_fit_replaced)

        fig_filename = f"{fit_index}_{filename_fit_replaced}_From_{lower}_To_{upper}{common_y_suffix}{kde_suffix}.png"

        if trait_scale_name is not None:
            folder = trait_scale_name
        else:
            folder = "reliability"
        fig_folder = op.join(SCHAEFER_DIR, folder, "figures")
        os.makedirs(fig_folder, exist_ok=True)
        g.save(
            filename=op.join(fig_folder, fig_filename), width=20, height=8, units="cm"
        )


def loop_for_investigate_residuals_of_cfa_only_trait(
    trait_scale_name_list: list[str],
    sample_n: int,
    est_method: str,
    covariates: list[str],
    use_lavaan=True,
    drop_vars=None,
    vmin=-0.30,
    vmax=0.30,
):
    """
    draw heatmaps representing residual correlations with all scales
    """
    for trait_scale_name in trait_scale_name_list:
        scale_name_list = get_subscale_list(trait_scale_name)
        for scale_name in scale_name_list:

            def draw_results_of_residual_correlations(
                trait_scale_name: str,
                scale_name: str,
            ):
                """
                investigate model-implied residuals in the CFAs
                """
                # get residual data
                dir_path = op.join(
                    SCHAEFER_DIR,
                    trait_scale_name,
                    scale_name,
                    "residuals",
                    "model_only_trait",
                )
                file_name = generate_res_cor_filename(
                    scale_name,
                    sample_n,
                    est_method,
                    covariates,
                    drop_vars,
                    use_lavaan=use_lavaan,
                )
                res_cor = pd.read_csv(op.join(dir_path, file_name[0]), index_col=0)
                var_names = res_cor.columns
                var_names = [i.replace("cov.", "") for i in var_names]
                if trait_scale_name == "NIH_Cognition":
                    var_names = [
                        COG_SCALES_PUB_DICT.get(i) if i not in covariates else i
                        for i in var_names
                    ]
                elif trait_scale_name == "ASR":
                    var_names = [
                        ASR_SCALES_PUB_DICT.get(i) if i not in covariates else i
                        for i in var_names
                    ]

                # draw a hetamap
                fig, ax = plt.subplots()
                res_cor = pd.DataFrame(
                    generate_lower_triangle_array(res_cor)[1:][:, :-1],
                    index=var_names[1:],
                    columns=var_names[:-1],
                )
                sns.heatmap(
                    res_cor,
                    annot=True,
                    fmt=".2f",
                    ax=ax,
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                )
                # ax.set_title(f'Heatmap of residual correlations in {scale_name} of {trait_scale_name}')
                ax.tick_params(axis="x", rotation=60)
                ax.tick_params(axis="y", rotation=0)

                res_cor

            draw_results_of_residual_correlations(trait_scale_name, scale_name)


def draw_hist_of_ncp(filename_fit: str, use_sns=False, binwidth: float = 0.1):
    """
    calculate noncentrality parameter from chi-square and degree of freedom
    """
    fit_array = copy_memmap_output_data(filename_fit)
    dof_index, chi2_index = FIT_INDICES.index("DoF"), FIT_INDICES.index("chi2")
    # calculate noncentrality parameters
    ncp_array = fit_array[:, chi2_index, :] - fit_array[:, dof_index, :]
    model_type_list = get_strings_from_filename(filename_fit, ["model_type"])[0]
    # draw histgrams of ncp
    fig, axes = plt.subplots(1, len(model_type_list))
    for i, model in enumerate(model_type_list):
        ncp_value = ncp_array[:, i]
        if use_sns:
            sns.histplot(ncp_value, ax=axes[i], binwidth=binwidth)
        else:
            axes[i].hist(
                ncp_value,
                bins=np.arange(min(ncp_value), max(ncp_value) + binwidth, binwidth),
            )
        axes[i].set_title(model)


def generate_lower_triangle_array(array: NDArray) -> NDArray:
    """
    generate lower triangular array for drawing a heatmap
    """
    array = np.tril(array, -1)
    array[array == 0] = np.nan
    return array


def loop_investigating_residuals_trait_fc(
    trait_type,
    **kwargs
    # filename_rescor_dict_by_gsr: dict[str: list[str]]
):
    """
    visualize mean and sd of residuals of total scale and subscales
    """
    filename_rescor_dict_by_gsr = get_dict_of_latest_filenames_by_gsr(
        trait_type,
        **{
            "n_edge": kwargs["n_edge"],
            "sample_n": kwargs["sample_n"],
            "est_method": kwargs["est_method"],
            "data_type": kwargs["data_type"],
            "model_type": kwargs["model_type"],
        },
    )
    trait_scale_name = get_scale_name_from_trait(trait_type)
    main_subscale_num = len(get_subscale_list(trait_scale_name))
    # make figures
    fig_mean, axes_mean = plt.subplots(
        main_subscale_num,
        2,
        figsize=(12, 4 * main_subscale_num),
        # sharey=True,
        # sharex=True
    )
    fig_median, axes_median = plt.subplots(
        main_subscale_num,
        2,
        figsize=(12, 4 * main_subscale_num),
        # sharey=True,
        # sharex=True
    )
    fig_sd, axes_sd = plt.subplots(
        main_subscale_num,
        2,
        figsize=(12, 4 * main_subscale_num),
        # sharey=True,
        # sharex=True
    )

    # get variables
    model_type = kwargs["model_type"].lower()
    for gsr_j, gsr_type in enumerate(["nogs", "gs"]):
        gsr_suffix = generate_gsr_suffix(gsr_type)
        # get a list of filenames
        filename_rescor_list = filename_rescor_dict_by_gsr.get(gsr_type)

        # draw a heatmap of mean and sd of residuals
        for fig_i, filename in enumerate(filename_rescor_list):

            def investigate_residuals_of_cfa_trait_fc(
                residual_filename: str, annot_fontsize: int = 8
            ):
                """
                investigate model-implied residuals in the CFAs
                """
                # get residual data as dictionary
                residual_dict = generate_params_dict(residual_filename, residuals=True)
                residual_array = np.array(*residual_dict.values())

                # calculate mean of residuals
                residual_mean_array = np.nanmean(residual_array, axis=0)
                # calculate mean of residuals
                residual_median_array = np.nanmedian(residual_array, axis=0)
                # calculate sd of residuals
                residual_sd_array = np.nanstd(residual_array, axis=0)

                # get names of variables
                scale_name, control, fc_unit = get_strings_from_filename(
                    residual_filename, ["scale_name", "control", "fc_unit"]
                )
                # get a list of subscales and associated list for reordering array
                if model_type in ["model_trait", "model_both"]:
                    subscale_list, reordered_list = get_subscale_list(
                        trait_scale_name,
                        get_all_subscales=True,
                        get_reordered_list=True,
                        scale_name=scale_name,
                    )
                elif model_type == "model_fc":
                    subscale_list = get_subscale_list(
                        trait_scale_name,
                        get_all_subscales=False,
                        get_reordered_list=True,
                        scale_name=scale_name,
                    )
                # specify variable names of FC
                if fc_unit == "session":
                    fc_vars = ["s" + str(i) for i in range(1, 5)]
                elif fc_unit == "day":
                    fc_vars = ["day1", "day2"]
                # specify variable names
                if model_type == "model_fc":
                    var_names = ["Total"] + fc_vars + control
                elif model_type == "model_trait":
                    var_names = subscale_list + ["FC_total"] + control
                elif model_type == "model_both":
                    var_names = subscale_list + fc_vars + control

                # reorder array when modeling trait
                if (
                    model_type in ["model_trait", "model_both"]
                    and trait_type != "personality"
                ):
                    reordered_list_added = np.array(
                        reordered_list
                        + [
                            i
                            for i in range(
                                len(reordered_list), residual_mean_array.shape[0]
                            )
                        ]
                    )

                    def reorder_array(input_array, reorder_list):
                        """get reordered array for visualization"""
                        input_array = input_array[:, reorder_list][reorder_list]
                        return input_array

                    # reorder array
                    residual_mean_array = reorder_array(
                        residual_mean_array, reordered_list_added
                    )
                    residual_sd_array = reorder_array(
                        residual_sd_array, reordered_list_added
                    )

                def draw_heatmap(array, ax, title, summary_type, **kwargs):
                    array = generate_lower_triangle_array(array)
                    # array = np.tril(array)
                    # array[array == 0] = np.nan
                    sns.heatmap(
                        pd.DataFrame(array, index=var_names, columns=var_names),
                        annot=True,
                        fmt=".3f",
                        annot_kws={"fontsize": annot_fontsize},
                        ax=ax,
                        vmin=kwargs["cmap_range"][summary_type]["vmin"],
                        vmax=kwargs["cmap_range"][summary_type]["vmax"],
                        center=0,
                    )
                    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)
                    ax.set_title(title)

                # heatmap of mean
                draw_heatmap(
                    residual_mean_array,
                    axes_mean[fig_i, gsr_j],
                    f"{scale_name} {gsr_suffix}",
                    summary_type="mean",
                    **kwargs,
                )
                # heatmap of median
                draw_heatmap(
                    residual_median_array,
                    axes_median[fig_i, gsr_j],
                    f"{scale_name} {gsr_suffix}",
                    summary_type="median",
                    **kwargs,
                )
                # heatmap of sd
                draw_heatmap(
                    residual_sd_array,
                    axes_sd[fig_i, gsr_j],
                    f"{scale_name} {gsr_suffix}",
                    summary_type="sd",
                    **kwargs,
                )

            # investigate residuals by drawing heatmaps
            investigate_residuals_of_cfa_trait_fc(filename)

    title_suffix = f" of residual correlations in {trait_scale_name} using {model_type}"
    fig_mean.suptitle(f"Mean {title_suffix}")
    fig_sd.suptitle(f"SD {title_suffix}")
    fig_mean.tight_layout()
    fig_sd.tight_layout()
    fig_median.suptitle(f"Median {title_suffix}")
    fig_median.tight_layout()


def draw_hist_of_fit_indices(
    # list includes mimimum of x axis, maximum of x axis, and binwidth of plot
    # fit_indices_dict_with_xlim argument moved to loop_for_draw_hist_of_fit_indices()
    filename_fit_list: list[str],
    vis_model_type: str,
    fit_index_xlim: list[float, float, float] = None,
    fit_index: str = None,
    fit_indices_dict_with_xlim: dict[FitIndicesList : list[float, float, float]] = None,
    dtype_memmap="float32",
    fig_size=(12, 4),
    save_fig=True,
    plt_close=False,
    ax=None,
    use_sns=False,
    **kwargs,
) -> None:
    """
    function for drawing histograms or kernel density plots of fit indices
    fit_indices_dict_with_xlim should be specified when using this function alone
    """
    if ax is None:
        for i, fit_index in enumerate(fit_indices_dict_with_xlim.keys()):
            # get index of fit index
            if fit_indices_dict_with_xlim is not None:
                fit_index_index = [
                    FIT_INDICES.index(i) for i in fit_indices_dict_with_xlim.keys()
                ]
            if fit_index is not None:
                fit_index_index = FIT_INDICES.index(fit_index)
            # create figures
            fig, axes = plt.subplots(
                # len(model_fa_list),
                figsize=fig_size,
                # sharex=True,
                sharey=True,
            )
            legend_str_list = []
            # fig.suptitle(
            #     f"Histogram of {fit_index} in {scale_name} of {trait_scale_name} {gsr_suffix}"
            # )
            lower, upper, binwidth = fit_indices_dict_with_xlim.get(fit_index)

            for filename_fit in filename_fit_list:
                (
                    model_fa_list,
                    trait_type,
                    scale_name,
                    gsr_type,
                    phase_encoding,
                    day_cor,
                ) = get_strings_from_filename(
                    filename_fit,
                    [
                        "model_type",
                        "trait_type",
                        "scale_name",
                        "gsr_type",
                        "phase_encoding",
                        "day_cor",
                    ],
                    include_nofa_model=False,
                )
                pe_label = "_PE" if phase_encoding else ""
                day_cor_label = "_DayCor" if day_cor else ""
                legend_str = "Base" + pe_label + day_cor_label
                legend_str_list.append(legend_str)

                fit_data = copy_memmap_output_data(
                    filename_fit,
                )
                fit_data = fit_data[:, fit_index_index, :]

                trait_scale_name = get_scale_name_from_trait(trait_type)
                gsr_suffix = generate_gsr_suffix(gsr_type)

                def draw_hist_fit(ax, data):
                    if not use_sns:
                        ax.hist(
                            data,
                            bins=np.arange(lower, upper + binwidth, binwidth),
                            alpha=0.5,
                            label=legend_str,
                        )
                    else:
                        sns.histplot(
                            data,
                            binwidth=binwidth,
                            alpha=0.5,
                            # kde=True,
                            # kde_kws={'bw_adjust': 5},
                            ax=ax,
                        )
                    if j is not None:
                        ax.set_title(model)
                    ax.set_xlim(left=lower, right=upper)
                    ax.set_ylabel("Frequency")
                    ax.set_xlabel(fit_index)
                    # ax.set_title(fit_index)
                    # if not use_sns:
                    #    plt.legend()
                    fig.tight_layout()

                if len(model_fa_list) > 1:
                    for j, model in enumerate(model_fa_list):
                        draw_hist_fit(axes[j], fit_data[:, i, j])
                elif len(model_fa_list) == 1:
                    j = None
                    draw_hist_fit(axes, fit_data[:, 0])
            handler, label = axes.get_legend_handles_labels()
            replace_dict = {
                "Base": "Model RSFC 1",
                #'Baseline model\n(Model RSFC 1)',
                "Base_DayCor": "Model RSFC 2",
                #'Model with correlated error terms in the same days\n(Model RSFC 2)',
                "Base_PE_DayCor": "Model RSFC 3"
                #'Model with correlated error terms in the same days\nand the same phase encoding directions\n(Model RSFC 3)'
            }
            axes.legend([replace_dict.get(i) for i in legend_str_list])
            if save_fig:
                fig_filename = f"{fit_index}_{filename_fit.replace('.dat', '')}_From_{lower}_To_{upper}.png"
                if trait_scale_name is not None:
                    folder = op.join(trait_scale_name, scale_name)
                else:
                    folder = "reliability"
                fig_folder = op.join(SCHAEFER_DIR, folder, "figures")
                os.makedirs(fig_folder, exist_ok=True)
                fig.savefig(op.join(fig_folder, fig_filename), bbox_inches="tight")
                if plt_close:
                    plt.close()

    else:
        model_index = model_fa_list.index(vis_model_type)
        lower, upper, binwidth = fit_index_xlim
        plt.rcParams["font.size"] = kwargs["plt_fsize"]
        plt.rcParams["xtick.labelsize"] = kwargs["plt_tick_lsize"]
        plt.rcParams["ytick.labelsize"] = kwargs["plt_tick_lsize"]
        ax.hist(
            fit_data[:, model_index], bins=np.arange(lower, upper + binwidth, binwidth)
        )
        ax.set_xlim(left=lower, right=upper)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(fit_index)
        ax.set_title(f"{scale_name} {gsr_suffix}")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45)


def loop_for_draw_hist_of_fit_indices(
    fit_indices_dict_with_xlim: dict[FitIndicesList : list[float, float, float]],
    n_edge: int,
    sample_n: int,
    trait_type_list,
    vis_model_type: str,
    est_method: str,
    dtype_memmap="float32",
    fig_size=(12, 4),
    save_fig=True,
    plt_close=False,
    **kwargs,
):
    """
    function for looping for draw_hist_of_fit_indices
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, est_method, "fit_indices"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        elif trait_type == "mental":
            ncol = 4
        for fit_index, fit_index_xlim in fit_indices_dict_with_xlim.items():
            fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
            for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
                filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
                for i, filename_fit in enumerate(filename_list_gsr_type):
                    draw_hist_of_fit_indices(
                        # list includes mimimum of x axis, maximum of x axis, and binwidth of plot
                        filename_fit,
                        vis_model_type,
                        fit_index_xlim,
                        fit_index,
                        fit_indices_dict_with_xlim,
                        dtype_memmap="float32",
                        ax=axes[j, i],
                        **kwargs,
                    )
            fig.suptitle(
                f"{fit_index} in {vis_model_type} in {trait_scale_name} with {est_method} (N = {sample_n}, number of edge = {n_edge})"
            )
            fig.tight_layout()
            # filename_fit_list = get_latest_files_with_args(
            fig_name = f"{fit_index}_{vis_model_type}_{trait_scale_name}_{est_method}_SampleN_{sample_n}_edgeN_{n_edge}.png"
            fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
            fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def calculate_power_from_r(n, r, alpha, use_root=False, power_input=None):
    """
    Calculate power based on two-side test
    """
    # https://github.com/cran/pwr/blob/master/R/pwr.r.test.R
    ttt = t.ppf(alpha / 2, n - 2)
    rc = np.sqrt(ttt**2 / (ttt**2 + n - 2))
    zr = np.arctanh(r) + r / (2 * (n - 1))
    zrc = np.arctanh(rc)
    power = norm.cdf((zr - zrc) * np.sqrt(n - 3)) + norm.cdf(
        (-zr - zrc) * np.sqrt(n - 3)
    )

    if use_root:
        return power - power_input

    return power


def calculate_n_from_power(power, r, alpha):
    """
    Calculate sample size to achieve power
    """
    # https://www2.ccrb.cuhk.edu.hk/stat/other/correlation.html
    C_r = 1 / 2 * np.log((1 + r) / (1 - r))
    Z_alpha = norm.ppf(1 - alpha / 2)
    Z_beta = norm.ppf(power)
    N = ((Z_alpha + Z_beta) / C_r) ** 2 + 3
    #    # https://github.com/cran/pwr/blob/master/R/pwr.r.test.R
    #    n = root(
    #            # calculate p value from correlation and sample size
#    calculate_power_from_r,
    #            [4 + 1e-10, 1e+09],
    #            args=(r, alpha, False, power)
    #            )
    return np.ceil(N)


def get_boolean_fit_locally_globally(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    model_type_list,
) -> NDArray[Shape["Num_iter, Num_modelfa"], Bool]:
    """function for getting boolean of fit edges"""
    boolean_locally_specified_array = select_models_from_local_fits(
        filename_cor, error_vars_dict, cor_min_max_dict, model_type_list
    )
    boolean_globally_specified_array = select_models_from_global_fits(
        filename_cor, fit_indices_thresholds_dict, model_type_list
    )[0]
    bool_array_inclusion = select_edges_locally_and_globally(
        boolean_locally_specified_array,
        boolean_globally_specified_array,
    )
    return bool_array_inclusion


def rename_fit_indices_memmep():
    """function for removing 'mean_pca_' from filenames of memory map on fit indices"""
    for trait_scale_name in ["NEO_FFI", "NIH_Cognition", "ASR"]:
        subscale_list = get_subscale_list(trait_scale_name)
        for subscale_name in subscale_list:
            memmap_list = os.listdir(
                op.join(SCHAEFER_DIR, trait_scale_name, subscale_name, "fit_indices")
            )
            memmap_list_renamed = [i.replace("mean_pca_", "") for i in memmap_list]
            for memmap_file, memmap_file_renamed in zip(
                memmap_list, memmap_list_renamed
            ):
                memmap_file_abs_path = op.join(
                    SCHAEFER_DIR,
                    trait_scale_name,
                    subscale_name,
                    "fit_indices",
                    memmap_file,
                )
                memmap_file_abs_path_renamed = op.join(
                    SCHAEFER_DIR,
                    trait_scale_name,
                    subscale_name,
                    "fit_indices",
                    memmap_file_renamed,
                )
                os.rename(memmap_file_abs_path, memmap_file_abs_path_renamed)


def get_cor_data_passing_thresholds(
    filename_cor: str, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
):
    """
    function for getting correlation data which passed thresholds
    """
    cor_data = copy_memmap_output_data(filename_cor)
    model_nofa_list = get_model_strings(filename_cor)[2]

    bool_array_inclusion = get_boolean_fit_locally_globally(
        filename_cor, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    bool_array_inclusion = get_boolean_array_of_selected_edges(
        bool_array_inclusion, model_nofa_list
    )
    cor_data[~bool_array_inclusion] = np.nan

    return cor_data


def get_array_with_power_of_edges(
    filename_cor,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas=[1 * 10**-i for i in range(3, 8)],
) -> NDArray[Shape["Num_iter, Num_models, Num_alphas"], Float]:
    """
    function for getting array storing power
    """
    # read data
    cor_data = get_cor_data_passing_thresholds(
        filename_cor, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )

    sample_n = int(reduce(add, get_strings_from_filename(filename_cor, ["sample_n"])))

    power_alphas = np.empty(shape=list(cor_data.shape) + [len(alphas)])
    power_alphas[:] = np.nan
    for i in range(len(alphas)):
        power_alphas[:, :, i] = calculate_power_from_r(
            n=sample_n, r=cor_data, alpha=alphas[i]
        )
    return power_alphas


def get_n_of_edges_over_power_thresholds(
    filename_cor,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas=[1 * 10**-i for i in range(3, 8)],
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
):
    """
    function for getting number of edges over thresholds of power
    """
    power_alphas = get_array_with_power_of_edges(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
    )
    array_over_power_thresholds = np.zeros(
        shape=(power_alphas.shape[1], len(alphas), len(power_thresholds))
    )
    for j, _ in enumerate(alphas):
        for k in range(power_alphas.shape[1]):
            power_array = power_alphas[:, k, j]
            for i, power in enumerate(power_thresholds):
                n_over_power = np.sum(power_array > power)
                array_over_power_thresholds[k, j, i] = n_over_power
    return array_over_power_thresholds


def visualize_set_of_edges_over_power_thresholds_with_networks(
    filename_cor,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    alphas=[1 * 10**-i for i in range(3, 8)],
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
    fig_size=(12, 4),
    plt_close=True,
) -> pd.DataFrame:
    """
    function for getting set of edges over thresholds of power
    """
    power_alphas = get_array_with_power_of_edges(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
    )
    edges_df = get_edge_summary(node_summary_path)
    model_type = get_strings_from_filename(filename_cor, ["model_type"])
    model_type = reduce(add, model_type)
    edges_over_power_df_all = pd.DataFrame()
    edges_count_wide_df = get_n_of_edges_per_network(node_summary_path)

    for j, alpha in enumerate(alphas):
        for k, model in enumerate(model_type):
            power_array = power_alphas[:, k, j]
            for power in power_thresholds:
                edge_set_over_power = np.where(power_array > power)[0]
                edges_over_power_df = edges_df.query(
                    "edge in @edge_set_over_power"
                ).assign(model=model, alpha=alpha, power=power)
                if edges_over_power_df.shape[0] != 0:
                    edges_over_power_wide_df = get_counts_of_edges_per_network(
                        long_edges_df=edges_over_power_df
                    )
                    prop_edges_df = edges_over_power_wide_df / edges_count_wide_df
                    (
                        fig_title_suffix,
                        fig_name_suffix,
                        save_fig_folder,
                    ) = generate_suffixes_and_saving_folder(filename_cor)
                    fig_title = f"Heatmap of proportion of edges over power of {power} in {model} with alpha being {alpha:.3g} {fig_title_suffix}"

                    fig, ax = plt.subplots(figsize=fig_size)
                    sns.heatmap(prop_edges_df, annot=True, fmt=".1%", ax=ax)
                    caption = generate_caption_of_figure_on_parameter_thresholds(
                        error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
                    )
                    fig.text(0, -0.125, caption)
                    fig.suptitle(fig_title)
                    fig.tight_layout()

                    fig_filename_suffix_thresholds = (
                        generate_fig_name_suffix_from_thresholds(
                            error_vars_dict,
                            cor_min_max_dict,
                            fit_indices_thresholds_dict,
                        )
                    )
                    fig_filename = f"heatmap_prop_over_power_{power}_edges_{model}_alpha_{alpha}_{fig_name_suffix}_{fig_filename_suffix_thresholds}.png"
                    fig.savefig(
                        op.join(save_fig_folder, fig_filename), bbox_inches="tight"
                    )
                    if plt_close:
                        plt.close()
    return edges_over_power_df_all


def get_df_with_n_of_edges_over_power_thresholds(
    filename_cor,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas=[1 * 10**-i for i in range(3, 8)],
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
) -> pd.DataFrame:
    """
    function for getting pandas dataframe for visualising n of edges over thresholds of power
    """
    array_over_power_thresholds = get_n_of_edges_over_power_thresholds(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
        power_thresholds=power_thresholds,
    )
    shapes_array = array_over_power_thresholds.shape
    reshaped_array = array_over_power_thresholds.reshape(
        shapes_array[0] * shapes_array[1], shapes_array[2]
    )
    model_type = get_strings_from_filename(filename_cor, ["model_type"])

    alphas = [f"{i:.3g}" for i in alphas]
    len_alphas = len(alphas)
    model_type_values = reduce(add, [[i] * len_alphas for i in reduce(add, model_type)])
    alphas_values = alphas * len(reduce(add, model_type))
    n_edge_over_power_df = pd.DataFrame(reshaped_array, columns=power_thresholds)
    n_edge_over_power_df["model_type"] = model_type_values
    n_edge_over_power_df["alpha"] = alphas_values
    n_edge_over_power_long_df = n_edge_over_power_df.melt(
        id_vars=["model_type", "alpha"],
        value_vars=power_thresholds,
        var_name="power",
        value_name="n",
    )
    return n_edge_over_power_long_df


def visualise_n_of_edges_over_power_thresholds(
    filename_cor,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas=[1 * 10**-i for i in range(3, 8)],
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
    plt_close=False,
    height=4,
    aspect=3,
    y_upper_lim=10000,
) -> None:
    n_edge_over_power_long_df = get_df_with_n_of_edges_over_power_thresholds(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
        power_thresholds=power_thresholds,
    )
    scale_name, trait_type, gsr_type = get_strings_from_filename(
        filename_cor, ["scale_name", "trait_type", "gsr_type"]
    )
    trait_scale_name = get_scale_name_from_trait(trait_type)
    gsr_suffix = generate_gsr_suffix(gsr_type)

    g = sns.FacetGrid(
        data=n_edge_over_power_long_df,
        col="model_type",
        row="alpha",
        margin_titles=True,
        height=height,
        aspect=aspect,
    )
    g.map(sns.barplot, "power", "n")
    g.set(ylim=(0, y_upper_lim))
    fig_title = f"Count of edges which passed thresholds of power in {scale_name} of {trait_scale_name} {gsr_suffix}"
    g.fig.suptitle(fig_title)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.tight_layout()
    (
        fig_title_suffix,
        fig_name_suffix,
        save_fig_folder,
    ) = generate_suffixes_and_saving_folder(filename_cor)

    thresholds_suffix = generate_fig_name_suffix_from_thresholds(
        error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    fig_name = f"count_n_of_edges_over_power_{fig_name_suffix}_{thresholds_suffix}.png"

    g.fig.savefig(op.join(save_fig_folder, fig_name))
    if plt_close:
        plt.close()


def loop_for_visualise_n_of_edges_over_power_thresholds(
    trait_type_list,
    n_edge: int,
    sample_n: int,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas=[1 * 10**-i for i in range(3, 8)],
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
    plt_close=False,
    height=4,
    aspect=0.8,
    y_upper_lim=None,
) -> None:
    """
    function for looping visualise_n_of_edges_over_power_thresholds()
    """
    filename_cor_list = get_latest_files_with_args(
        trait_type_list=trait_type_list,
        n_edge=n_edge,
        sample_n=sample_n,
        data_type="correlation",
    )
    for filename_cor in filename_cor_list:
        visualise_n_of_edges_over_power_thresholds(
            filename_cor,
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
            alphas=alphas,
            power_thresholds=power_thresholds,
            plt_close=plt_close,
            height=height,
            aspect=aspect,
            y_upper_lim=y_upper_lim,
        )


class NameDifferentError(Exception):
    pass


class GSRTypeError(Exception):
    pass


def calc_r_ci(r: float, alpha: float, n: int):
    """
    function for calculating confidence intervals of r
    """
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_critical = norm.ppf(1 - alpha / 2)

    low = z - z_critical * se
    high = z + z_critical * se

    return (np.tanh(low), np.tanh(high))


def get_pd_on_correlation_of_r_between_split_samples(
    file1_path,
    file2_path,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
) -> pd.DataFrame:
    """
    function for investigating correlation of r between split samples
    """
    cor_array1 = get_cor_data_passing_thresholds(
        file1_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    cor_array2 = get_cor_data_passing_thresholds(
        file2_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type1,
        trait_scale_name1,
        scale_name1,
        gsr_type1,
        sample_n1,
    ) = get_strings_from_filename(
        file1_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    (
        model_type2,
        trait_scale_name2,
        scale_name2,
        gsr_type2,
        sample_n2,
    ) = get_strings_from_filename(
        file2_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )

    for variable_str in ["model_type", "trait_scale_name", "scale_name", "gsr_type"]:
        variable1_str, variable2_str = variable_str + "1", variable_str + "2"
        variable1, variable2 = locals()[variable1_str], locals()[variable2_str]
        if variable1 != variable2:
            raise NameDifferentError(
                "file1_path and file2_path should be matched excluding sample size"
            )

    cor_store_array = np.empty(shape=(len(model_type1) * 2, 3))
    for i, model in enumerate(model_type1):
        cor_data1, cor_data2 = cor_array1[:, i], cor_array2[:, i]
        cor_array = np.vstack((cor_data1, cor_data2)).T
        cor_array_na_removed = cor_array[~np.isnan(cor_array).any(axis=1)]
        cor = np.corrcoef(cor_array_na_removed.T)[0, 1]
        cor_ci = calc_r_ci(cor, 0.05, int(sample_n1))

        rho = spearmanr(cor_array_na_removed).correlation
        rho_ci = calc_r_ci(cor, 0.05, int(sample_n1))
        cor_store_array[i, :] = (cor, *cor_ci)
        cor_store_array[len(model_type1) + i, :] = (rho, *rho_ci)

    cor_store_df = pd.DataFrame(
        cor_store_array, columns=["r", "ci_low", "ci_high"], index=model_type1 * 2
    )
    cor_store_df = cor_store_df.assign(
        trait_scale=trait_scale_name1,
        scale=scale_name1,
        gsr_type=gsr_type1,
        cor_type=["Pearson"] * len(model_type1) + ["Spearman"] * len(model_type1),
    )
    return cor_store_df


def get_concat_pd_on_correlation_of_r_between_split_samples(
    trait_type_list,
    n_edge,
    sample_n,
    split_ratio,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
) -> pd.DataFrame:
    """
    getting dataframe including combinations of trait scale, subscale, and gsr type
    """
    filename_list = get_latest_files_with_args(
        trait_type_list, n_edge, sample_n, "correlation", split_ratio
    )
    concat_df = pd.DataFrame()
    for filename_pair in filename_list:
        file1_path, file2_path = filename_pair
        cor_df = get_pd_on_correlation_of_r_between_split_samples(
            file1_path,
            file2_path,
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
        )
        concat_df = pd.concat([concat_df, cor_df], axis=0)
    concat_df = concat_df.reset_index(names="model")

    for trait_scale in concat_df["trait_scale"].unique():
        trait_df = concat_df.query("trait_scale == @trait_scale")
        trait_df["model"] = pd.Categorical(
            trait_df["model"], categories=concat_df.model.unique()
        )
        trait_df["gsr_type"] = pd.Categorical(
            trait_df["gsr_type"], categories=concat_df.gsr_type.unique()
        )
        trait_df["scale"] = pd.Categorical(
            trait_df["scale"], categories=concat_df.scale.unique()
        )
        p = (
            ggplot(trait_df, aes("model", "r", fill="cor_type"))
            + geom_bar(stat="identity", position=position_dodge())
            + geom_errorbar(
                aes(ymin="ci_low", ymax="ci_high"),
                width=0.2,
                position=position_dodge(width=0.9),
            )
            + facet_grid("gsr_type ~ scale")
            + theme(axis_text_x=element_text(angle=45))
        )

        if trait_scale == "NEO_FFI":
            save_fig_folder = op.join(SCHAEFER_DIR, "NEO_FFI", "figures")
        elif trait_scale == "NIH_Cognition":
            save_fig_folder = op.join(SCHAEFER_DIR, "NIH_Cognition", "figures")

        thresholds_suffix = (
            thresholds_suffix
        ) = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )

        filename = f"Correlation of r in split samples in {trait_scale}_{thresholds_suffix}.png"
        p.save(filename=op.join(save_fig_folder, filename))

    return concat_df


def get_common_edges_over_power_thresholds_across_split_samples(
    file1_path: str,
    file2_path: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas: list[float],
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
) -> dict:
    """
    function for getting common edges which is over power thresholds across split samples
    """
    cor_array1 = get_cor_data_passing_thresholds(
        file1_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    cor_array2 = get_cor_data_passing_thresholds(
        file2_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type1,
        trait_scale_name1,
        scale_name1,
        gsr_type1,
        sample_n1,
    ) = get_strings_from_filename(
        file1_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    (
        model_type2,
        trait_scale_name2,
        scale_name2,
        gsr_type2,
        sample_n2,
    ) = get_strings_from_filename(
        file2_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )

    for variable_str in ["model_type", "trait_scale_name", "scale_name", "gsr_type"]:
        variable1_str, variable2_str = variable_str + "1", variable_str + "2"
        variable1, variable2 = locals()[variable1_str], locals()[variable2_str]
        if variable1 != variable2:
            raise NameDifferentError(
                "file1_path and file2_path should be matched excluding sample size"
            )

    power_alphas1 = get_array_with_power_of_edges(
        file1_path,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
    )
    power_alphas2 = get_array_with_power_of_edges(
        file2_path,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
    )
    edge_set_dict = defaultdict(lambda: defaultdict(dict))
    for j, alpha in enumerate(alphas):
        for k, model in enumerate(model_type1):
            power_array1 = power_alphas1[:, k, j]
            power_array2 = power_alphas2[:, k, j]
            for power in power_thresholds:
                edge_set1_over_power = set(np.where(power_array1 > power)[0])
                edge_set2_over_power = set(np.where(power_array2 > power)[0])
                common_edges = edge_set1_over_power & edge_set2_over_power
                all_edges = edge_set1_over_power | edge_set2_over_power
                edge_set_dict[f"{alpha:.3g}"][model][power] = {
                    "common_edges_n": len(common_edges),
                    "all_edges_n": len(all_edges),
                    "common_edges": common_edges,
                    "all_edges": all_edges,
                }
    return edge_set_dict


def loop_for_get_common_edges_over_power_thresholds_across_split_samples(
    trait_type_list,
    n_edge,
    sample_n,
    split_ratio,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas,
    power_thresholds: list[float] = [0.80, 0.90, 0.95],
):
    """
    function for looping get_common_edges_over_power_thresholds_across_split_samples()
    """
    filename_list = get_latest_files_with_args(
        trait_type_list, n_edge, sample_n, "correlation", split_ratio
    )
    common_edge_dict_all = defaultdict(lambda: defaultdict(dict))
    for filename_pair in filename_list:
        file1_path, file2_path = filename_pair
        trait_scale_name1, scale_name1, gsr_type1 = get_strings_from_filename(
            file1_path, ["trait_scale_name", "scale_name", "gsr_type"]
        )
        trait_scale_name2, scale_name2, gsr_type2 = get_strings_from_filename(
            file2_path, ["trait_scale_name", "scale_name", "gsr_type"]
        )
        for variable_str in ["trait_scale_name", "scale_name", "gsr_type"]:
            variable1_str, variable2_str = variable_str + "1", variable_str + "2"
            variable1, variable2 = locals()[variable1_str], locals()[variable2_str]
            if variable1 != variable2:
                raise NameDifferentError(
                    "file1_path and file2_path should be matched excluding sample size"
                )

        common_edge_dict = get_common_edges_over_power_thresholds_across_split_samples(
            file1_path,
            file2_path,
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
            alphas=alphas,
            power_thresholds=power_thresholds,
        )
        common_edge_dict_all[trait_scale_name1][scale_name1][
            gsr_type1
        ] = common_edge_dict
    return common_edge_dict_all


def visualize_mean_of_z_values_of_networks(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    vis_model_type,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    display_df=True,
    ax=None,
    **kwargs,
):
    """
    function for visualising mean z values of correlation between fc and trait based on networks
    """
    cor_array = get_cor_data_passing_thresholds(
        filename_cor, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type,
        trait_scale_name,
        scale_name,
        gsr_type,
        sample_n,
    ) = get_strings_from_filename(
        filename_cor,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    edges_df = get_edge_summary(node_summary_path)
    np_index = edges_df.index
    cor_array = cor_array[np_index]
    for i, model in enumerate(model_type):
        edges_df[model] = cor_array[:, i]
    edges_df["net_comb"] = list(zip(edges_df.node1_net, edges_df.node2_net))
    edges_df["net_set"] = edges_df["net_comb"].apply(lambda x: frozenset(x))
    edges_df["net_set2"] = edges_df["net_set"].apply(
        lambda x: sorted([i for i in x], key=lambda y: NETWORK_ORDER_DICT[y])
    )
    edges_df["net_set2"] = edges_df.apply(lambda x: "_".join(x["net_set2"]), axis=1)
    if type(vis_model_type) is str:
        vis_model_type = [vis_model_type]
    edges_mean_df = (
        edges_df.melt(
            id_vars=["edge", "net_set2"],
            value_vars=model_type,
            value_name="r",
            var_name="model",
        )
        .assign(z=lambda x: np.arctanh(x.r))
        .groupby(["model", "net_set2"])["z"]
        .mean()
        .reset_index()
        .query("model in @vis_model_type")
        .assign(node_list=lambda x: x["net_set2"].str.split("_"))
    )
    edges_mean_df["node1"] = pd.Categorical(
        edges_mean_df["node_list"].apply(lambda x: x[0]),
        categories=NETWORK_ORDER_DICT.keys(),
    )
    edges_mean_df["node2"] = pd.Categorical(
        edges_mean_df["node_list"].apply(lambda x: x[-1]),
        categories=NETWORK_ORDER_DICT.keys(),
    )
    edges_mean_wide_df = edges_mean_df.pivot_table(
        index="node1",
        columns="node2",
        values="z",
    )
    gsr_suffix = generate_gsr_suffix(gsr_type)
    plt.rcParams["font.size"] = kwargs["plt_fsize"]
    plt.rcParams["xtick.labelsize"] = kwargs["plt_tick_lsize"]
    plt.rcParams["ytick.labelsize"] = kwargs["plt_tick_lsize"]

    if ax is None:
        fig, ax = plt.subplots()
        sns.heatmap(data=edges_mean_wide_df, annot=True, fmt=".2f", ax=ax)
        ax.set_title(
            f"Heaitmap of z values in {scale_name} of {trait_scale_name} {gsr_suffix}"
        )
    else:
        sns.heatmap(
            data=edges_mean_wide_df,
            annot=True,
            fmt=".2f",
            ax=ax,
            vmax=kwargs["vmax"],
            vmin=kwargs["vmin"],
            annot_kws={"fontsize": kwargs["annot_fsize"], "fontfamily": "serif"},
        )
        ax.set_title(f"{scale_name} {gsr_suffix}")


def loop_for_visualize_mean_of_z_values_of_networks(
    trait_type_list,
    n_edge,
    sample_n,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    vis_model_type,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    ax=None,
    **kwargs,
):
    """function for looping visualize_mean_of_z_values_of_networks()"""
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, "correlation"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        elif trait_type == "mental":
            ncol = 4
        fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
            for i, filename_cor in enumerate(filename_list_gsr_type):
                visualize_mean_of_z_values_of_networks(
                    filename_cor,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    node_summary_path,
                    vis_model_type,
                    save_fig=save_fig,
                    fig_size=fig_size,
                    plt_close=plt_close,
                    ax=axes[j, i],
                    **kwargs,
                )
        fig.suptitle(
            f"Heatmap of mean z values of correlation in {trait_scale_name} with total sample size of {sample_n} in {vis_model_type}"
        )
        if "model" in vis_model_type:
            caption = generate_caption_of_figure_on_parameter_thresholds(
                error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
            )
            fig.text(0, -0.05, caption, fontsize=kwargs["caption_fsize"])
        fig.tight_layout()

        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = f"heatmap_mean_z_{vis_model_type}_{trait_scale_name}_{thresholds_suffix}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def visualize_proportions_of_edges_with_alpha_and_power(
    filename_cor: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    vis_model_type,
    alpha: float,
    power: float = 0.80,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    display_df=True,
    ax=None,
    **kwargs,
):
    """
    function for drawing heatmaps representing proportions of edges with specified power and alpha without splitting samples
    """
    cor_array = get_cor_data_passing_thresholds(
        filename_cor, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type,
        trait_scale_name,
        scale_name,
        gsr_type,
        sample_n,
    ) = get_strings_from_filename(
        filename_cor,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    power_alphas = get_array_with_power_of_edges(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=list(alpha),
    )
    power_alphas = np.squeeze(power_alphas)
    edges_df = get_edge_summary(node_summary_path)
    all_edges_wide_df = get_counts_of_edges_per_network(edges_df)

    for k, model in enumerate(model_type):
        power_array = power_alphas[:, k]
        edge_set_over_power = set(np.where(power_array > power)[0])
        print(
            trait_scale_name,
            scale_name,
            gsr_type,
            f"{alpha[0]:.3g}",
            power,
            model,
            len(edge_set_over_power),
            f"{len(edge_set_over_power) / power_array.shape[0]:.2%}",
        )
        if len(edge_set_over_power) > 10:
            selected_edges_df = edges_df.query("edge in @edge_set_over_power")
            selected_edges_wide_df = get_counts_of_edges_per_network(selected_edges_df)
            proportions_wide_df = selected_edges_wide_df / all_edges_wide_df
            if display_df:
                display(selected_edges_wide_df)
                display(proportions_wide_df * 100)

            if ax is None:
                fig, ax = plt.subplots(figsize=fig_size)
                sns.heatmap(proportions_wide_df, annot=True, fmt=".2%", ax=ax)
                (
                    fig_title_suffix,
                    fig_name_suffix,
                    save_fig_folder,
                ) = generate_suffixes_and_saving_folder(file1_path)

                fig_title = f"Heatmap of proportions of edges with power {power} and alpha {reduce(add, alpha):.2g} {fig_title_suffix}"
                fig.suptitle(fig_title)
                fig.tight_layout()
                thresholds_suffix = generate_fig_name_suffix_from_thresholds(
                    error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
                )
                fig_name = f"Heatmap_of_proportions_of_edges_{fig_name_suffix}_{thresholds_suffix}_alpha_{reduce(add, alpha):.3g}_power_{power}.png"
                if save_fig:
                    fig.savefig(op.join(save_fig_folder, fig_name))
                if plt_close:
                    plt.close()
            else:
                if model == vis_model_type:
                    plt.rcParams["font.size"] = kwargs["plt_fsize"]
                    plt.rcParams["xtick.labelsize"] = kwargs["plt_tick_lsize"]
                    plt.rcParams["ytick.labelsize"] = kwargs["plt_tick_lsize"]
                    sns.heatmap(
                        proportions_wide_df,
                        annot=True,
                        fmt=".1%",
                        ax=ax,
                        vmax=kwargs["vmax"],
                        annot_kws={
                            "fontsize": kwargs["annot_fsize"],
                            "fontfamily": "serif",
                        },
                    )
                    gsr_suffix = generate_gsr_suffix(gsr_type)
                    ax.set_title(f"{scale_name} {gsr_suffix}")


def loop_for_visualize_proportions_of_edges_with_alpha_and_power(
    trait_type_list,
    n_edge,
    sample_n,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path,
    vis_model_type: str,
    alpha,
    power,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    display_df=False,
    **kwargs,
) -> None:
    """
    function for looping visualize_common_edges_with_specified_alpha_and_power()
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, "correlation"
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [i for i in filename_list if (gsr_type in i)]
            for i, filename_cor in enumerate(filename_list_gsr_type):
                visualize_proportions_of_edges_with_alpha_and_power(
                    filename_cor,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    node_summary_path,
                    vis_model_type,
                    alpha,
                    power,
                    save_fig,
                    fig_size,
                    plt_close,
                    ax=axes[j, i],
                    display_df=display_df,
                    **kwargs,
                )
        fig.suptitle(
            f"Heatmap of proportions of edges with power being greater than {power} and alpha being {reduce(add, alpha):.2g}\n in {trait_scale_name} with total sample size of {sample_n}"
        )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.05, caption, fontsize=kwargs["caption_fsize"])
        fig.tight_layout()

        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = f"heatmap_cor_{trait_scale_name}_{thresholds_suffix}_alpha{reduce(add, alpha):.2g}_power{power}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def visualize_venn_diagram_of_common_edges_with_specified_alpha_and_power(
    file1_path: str,
    file2_path: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type,
    alpha: float,
    power: float = 0.80,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    ax=None,
    **kwargs,
) -> Optional[plt.axes]:
    """
    function for visualising count of common edges with specified alpha and power
    """
    cor_array1 = get_cor_data_passing_thresholds(
        file1_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    cor_array2 = get_cor_data_passing_thresholds(
        file2_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type1,
        trait_scale_name1,
        scale_name1,
        gsr_type1,
        sample_n1,
    ) = get_strings_from_filename(
        file1_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    (
        model_type2,
        trait_scale_name2,
        scale_name2,
        gsr_type2,
        sample_n2,
    ) = get_strings_from_filename(
        file2_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )

    for variable_str in ["model_type", "trait_scale_name", "scale_name", "gsr_type"]:
        variable1_str, variable2_str = variable_str + "1", variable_str + "2"
        variable1, variable2 = locals()[variable1_str], locals()[variable2_str]
        if variable1 != variable2:
            raise NameDifferentError(
                "file1_path and file2_path should be matched excluding sample size"
            )
    power_alphas1 = get_array_with_power_of_edges(
        file1_path,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=list(alpha),
    )
    power_alphas2 = get_array_with_power_of_edges(
        file2_path,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=list(alpha),
    )
    power_alphas1, power_alphas2 = np.squeeze(power_alphas1), np.squeeze(power_alphas2)
    for k, model in enumerate(model_type1):
        power_array1 = power_alphas1[:, k]
        power_array2 = power_alphas2[:, k]
        edge_set1_over_power = set(np.where(power_array1 > power)[0])
        edge_set2_over_power = set(np.where(power_array2 > power)[0])
        common_edges = edge_set1_over_power & edge_set2_over_power
        all_edges = edge_set1_over_power | edge_set2_over_power
        print(
            trait_scale_name1,
            scale_name1,
            gsr_type1,
            f"{alpha[0]:.3g}",
            power,
            model,
            len(common_edges),
        )
        if (len(common_edges) > 0) and (model == vis_model_type):
            gsr_suffix = generate_gsr_suffix(gsr_type1)
            if ax is None:
                fig, ax = plt.subplots()
            else:
                venn2(
                    [edge_set1_over_power, edge_set2_over_power],
                    (f"sample 1\n(N = {sample_n1})", f"sample 2\n(N = {sample_n2})"),
                    ax=ax,
                    **kwargs,
                )
                ax.set_title(f"{scale_name1} {gsr_suffix}")


def loop_for_visualize_venn_diagram_of_common_edges_with_specified_alpha_and_power(
    trait_type_list,
    n_edge,
    sample_n,
    split_ratio,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type,
    alpha,
    power,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    **kwargs,
):
    """
    function for looping visualize_venn_diagram_of_common_edges_with_specified_alpha_and_power()
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, "correlation", split_ratio
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [
                sublist
                for sublist in filename_list
                if (gsr_type in sublist[0]) and (gsr_type in sublist[1])
            ]
            for i, filename_pair in enumerate(filename_list_gsr_type):
                file1_path, file2_path = filename_pair
                visualize_venn_diagram_of_common_edges_with_specified_alpha_and_power(
                    file1_path,
                    file2_path,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    vis_model_type,
                    alpha,
                    power,
                    save_fig,
                    fig_size,
                    plt_close,
                    ax=axes[j, i],
                    **kwargs,
                )
                fig.suptitle(
                    f"Venn diagram of edges with power being greater than {power} and alpha being {reduce(add, alpha):.2g}\n in {trait_scale_name} across split samples with total sample size of {sample_n}"
                )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.05, caption, fontsize=kwargs["caption_fsize"])
        fig.tight_layout()

        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = f"venn_diagram_edges_{trait_scale_name}_{thresholds_suffix}_alpha_{reduce(add, alpha):.2g}_power_{power}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def visualize_common_edges_with_specified_alpha_and_power(
    file1_path: str,
    file2_path: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path: str,
    alpha: float,
    power: float = 0.80,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    display_df=True,
    ax=None,
    **kwargs,
) -> Optional[plt.axes]:
    """
    function for visualising count of common edges with specified alpha and power
    """
    cor_array1 = get_cor_data_passing_thresholds(
        file1_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    cor_array2 = get_cor_data_passing_thresholds(
        file2_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type1,
        trait_scale_name1,
        scale_name1,
        gsr_type1,
        sample_n1,
    ) = get_strings_from_filename(
        file1_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    (
        model_type2,
        trait_scale_name2,
        scale_name2,
        gsr_type2,
        sample_n2,
    ) = get_strings_from_filename(
        file2_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )

    for variable_str in ["model_type", "trait_scale_name", "scale_name", "gsr_type"]:
        variable1_str, variable2_str = variable_str + "1", variable_str + "2"
        variable1, variable2 = locals()[variable1_str], locals()[variable2_str]
        if variable1 != variable2:
            raise NameDifferentError(
                "file1_path and file2_path should be matched excluding sample size"
            )
    power_alphas1 = get_array_with_power_of_edges(
        file1_path,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=list(alpha),
    )
    power_alphas2 = get_array_with_power_of_edges(
        file2_path,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=list(alpha),
    )
    power_alphas1, power_alphas2 = np.squeeze(power_alphas1), np.squeeze(power_alphas2)
    edges_df = get_edge_summary(node_summary_path)
    all_edges_wide_df = get_counts_of_edges_per_network(edges_df)
    for k, model in enumerate(model_type1):
        power_array1 = power_alphas1[:, k]
        power_array2 = power_alphas2[:, k]
        edge_set1_over_power = set(np.where(power_array1 > power)[0])
        edge_set2_over_power = set(np.where(power_array2 > power)[0])
        common_edges = edge_set1_over_power & edge_set2_over_power
        all_edges = edge_set1_over_power | edge_set2_over_power
        print(
            trait_scale_name1,
            scale_name1,
            gsr_type1,
            f"{alpha[0]:.3g}",
            power,
            model,
            len(common_edges),
        )
        if len(common_edges) > 0:
            selected_edges_df = edges_df.query("edge in @common_edges")
            selected_edges_wide_df = get_counts_of_edges_per_network(selected_edges_df)
            proportions_wide_df = selected_edges_wide_df / all_edges_wide_df
            if display_df:
                display(selected_edges_wide_df)
                display(proportions_wide_df * 100)

            if ax is None:
                fig, ax = plt.subplots(figsize=fig_size)
                sns.heatmap(proportions_wide_df, annot=True, fmt=".2%", ax=ax)
                (
                    fig_title_suffix,
                    fig_name_suffix,
                    save_fig_folder,
                ) = generate_suffixes_and_saving_folder(file1_path)

                fig_title = f"Heatmap of proportions of common edges across split samples with power {power} and alpha {reduce(add, alpha):.2g} {fig_title_suffix}"
                fig.suptitle(fig_title)
                fig.tight_layout()
                thresholds_suffix = generate_fig_name_suffix_from_thresholds(
                    error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
                )
                fig_name = f"Heatmap_of_proportions_of_common_edges_{fig_name_suffix}_{thresholds_suffix}.png"
                if save_fig:
                    fig.savefig(op.join(save_fig_folder, fig_name))
                if plt_close:
                    plt.close()
            else:
                plt.rcParams["font.size"] = kwargs["plt_fsize"]
                plt.rcParams["xtick.labelsize"] = kwargs["plt_tick_lsize"]
                plt.rcParams["ytick.labelsize"] = kwargs["plt_tick_lsize"]
                sns.heatmap(
                    proportions_wide_df,
                    annot=True,
                    fmt=".1%",
                    ax=ax,
                    vmax=kwargs["vmax"],
                    annot_kws={
                        "fontsize": kwargs["annot_fsize"],
                        "fontfamily": "serif",
                    },
                    **kwargs,
                )
                gsr_suffix = generate_gsr_suffix(gsr_type1)
                ax.set_title(f"{scale_name1} {gsr_suffix}")


def loop_for_visualize_common_edges_with_specified_alpha_and_power(
    trait_type_list,
    n_edge,
    sample_n,
    split_ratio,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    node_summary_path,
    alpha,
    power,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    display_df=False,
    **kwargs,
) -> None:
    """
    function for looping visualize_common_edges_with_specified_alpha_and_power()
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, "correlation", split_ratio
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        fig, axes = plt.subplots(2, ncol, figsize=(ncol * 3, 6))
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [
                sublist
                for sublist in filename_list
                if (gsr_type in sublist[0]) and (gsr_type in sublist[1])
            ]
            for i, filename_pair in enumerate(filename_list_gsr_type):
                file1_path, file2_path = filename_pair
                visualize_common_edges_with_specified_alpha_and_power(
                    file1_path,
                    file2_path,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    node_summary_path,
                    alpha,
                    power,
                    save_fig,
                    fig_size,
                    plt_close,
                    ax=axes[j, i],
                    display_df=display_df,
                    **kwargs,
                )
                fig.suptitle(
                    f"Heatmap of proportions of common edges with power being greater than {power} and alpha being {reduce(add, alpha):.2g}\n in {trait_scale_name} across split samples with total sample size of {sample_n}"
                )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.05, caption, fontsize=kwargs["caption_fsize"])
        fig.tight_layout()

        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = f"heatmap_cor_prop_common_edges_{trait_scale_name}_{thresholds_suffix}_alpha_{reduce(add, alpha):.2g}_power_{power}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def visualize_correspondence_cor_between_split_samples(
    file1_path: str,
    file2_path: str,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type,
    plt_close=False,
    save_fig=True,
    fig_size=(12, 4),
    ax=None,
) -> None:
    """
    function for investigating correspondence of correlation between split samples
    """
    cor_array1 = get_cor_data_passing_thresholds(
        file1_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    cor_array2 = get_cor_data_passing_thresholds(
        file2_path, error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
    )
    (
        model_type1,
        trait_scale_name1,
        scale_name1,
        gsr_type1,
        sample_n1,
    ) = get_strings_from_filename(
        file1_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )
    (
        model_type2,
        trait_scale_name2,
        scale_name2,
        gsr_type2,
        sample_n2,
    ) = get_strings_from_filename(
        file2_path,
        ["model_type", "trait_scale_name", "scale_name", "gsr_type", "sample_n"],
    )

    for variable_str in ["model_type", "trait_scale_name", "scale_name", "gsr_type"]:
        variable1_str, variable2_str = variable_str + "1", variable_str + "2"
        variable1, variable2 = locals()[variable1_str], locals()[variable2_str]
        if variable1 != variable2:
            raise NameDifferentError(
                "file1_path and file2_path should be matched excluding sample size"
            )
    if ax is None:
        fig, axes = plt.subplots(1, len(model_type1), figsize=fig_size)
        for i, model in enumerate(model_type1):
            cor_data1, cor_data2 = cor_array1[:, i], cor_array2[:, i]
            axes[i].scatter(cor_data1, cor_data2, s=1)
            axes[i].set_title(model)
            axes[i].set_xlabel(f"sample 1 (N = {sample_n1})")
            axes[i].set_ylabel(f"sample 2 (N = {sample_n2})")
            cor = ma.corrcoef(
                ma.masked_invalid(cor_data1), ma.masked_invalid(cor_data2)
            )[0, 1]
            axes[i].text(
                0.1, 0.9, f"Pearson's r = {cor:.3f} ", transform=axes[i].transAxes
            )
            cor_array = np.vstack((cor_data1, cor_data2)).T
            cor_array_na_removed = cor_array[~np.isnan(cor_array).any(axis=1)]
            rho = spearmanr(cor_array_na_removed).correlation
            axes[i].text(
                0.1, 0.85, f"Spearman's rho = {rho:.3f}", transform=axes[i].transAxes
            )
        (
            fig_title_suffix,
            fig_name_suffix,
            save_fig_folder,
        ) = generate_suffixes_and_saving_folder(file1_path)

        fig_title = f"Scatterplot of correlations of split samples {fig_title_suffix}"
        fig.suptitle(fig_title)
        fig.tight_layout()
        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = f"scatterplot_cor_split_{fig_name_suffix}_{thresholds_suffix}.png"

        fig.savefig(op.join(save_fig_folder, fig_name))
        if plt_close:
            plt.close()
    else:
        vis_index = model_type1.index(vis_model_type)
        cor_data1, cor_data2 = cor_array1[:, vis_index], cor_array2[:, vis_index]
        ax.scatter(cor_data1, cor_data2, s=1)
        gsr_suffix = generate_gsr_suffix(gsr_type1)
        ax.set_title(f"{scale_name1} {gsr_suffix}")
        ax.set_xlabel(f"sample 1 (N = {sample_n1})")
        ax.set_ylabel(f"sample 2 (N = {sample_n2})")
        cor = ma.corrcoef(ma.masked_invalid(cor_data1), ma.masked_invalid(cor_data2))[
            0, 1
        ]
        ax.text(0.1, 0.9, f"Pearson's r = {cor:.3f} ", transform=ax.transAxes)
        cor_array = np.vstack((cor_data1, cor_data2)).T
        cor_array_na_removed = cor_array[~np.isnan(cor_array).any(axis=1)]
        rho = spearmanr(cor_array_na_removed).correlation
        ax.text(0.1, 0.85, f"Spearman's rho = {rho:.3f}", transform=ax.transAxes)
        slope, intercept = np.polyfit(
            cor_array_na_removed[:, 0], cor_array_na_removed[:, 1], deg=1
        )
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, "-", color="black", linewidth=1)


def loop_for_visualize_correspondence_cor_between_split_samples(
    trait_type_list,
    n_edge,
    sample_n,
    split_ratio,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    vis_model_type,
    save_fig=True,
    fig_size=(12, 4),
    plt_close=True,
    caption_fsize=6,
    axis_text_lsize=6,
    plt_fsize=9,
) -> None:
    """
    function for looping visualize_correspondence_cor_between_split_samples()
    """
    for trait_type in trait_type_list:
        trait_scale_name = get_scale_name_from_trait(trait_type)
        filename_list = get_latest_files_with_args(
            [trait_type], n_edge, sample_n, "correlation", split_ratio
        )
        if trait_type == "personality":
            ncol = 5
        elif trait_type == "cognition":
            ncol = 3
        fig, axes = plt.subplots(
            2, ncol, figsize=(ncol * 3, 6), sharex=True, sharey=True
        )
        for j, gsr_type in enumerate(["_nogs_", "_gs_"]):
            filename_list_gsr_type = [
                sublist
                for sublist in filename_list
                if (gsr_type in sublist[0]) and (gsr_type in sublist[1])
            ]
            for i, filename_pair in enumerate(filename_list_gsr_type):
                file1_path, file2_path = filename_pair
                visualize_correspondence_cor_between_split_samples(
                    file1_path,
                    file2_path,
                    error_vars_dict,
                    cor_min_max_dict,
                    fit_indices_thresholds_dict,
                    vis_model_type,
                    save_fig=save_fig,
                    ax=axes[j, i],
                )
                fig.suptitle(
                    f"Scatterplot of correlation in {trait_scale_name} between split samples with total sample size of {sample_n}"
                )
        caption = generate_caption_of_figure_on_parameter_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig.text(0, -0.05, caption, fontsize=caption_fsize)
        fig.tight_layout()

        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = f"scatterplot_split_sampleN_{sample_n}_{trait_scale_name}_{thresholds_suffix}.png"
        fig_folder = op.join(SCHAEFER_DIR, trait_scale_name, "figures")
        fig.savefig(op.join(fig_folder, fig_name), bbox_inches="tight")


def compare_power_models(
    filename_cor,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas=[1 * 10**-i for i in range(3, 8)],
    show_hist=True,
):
    """conduct power analysis for models"""
    power_alphas = get_array_with_power_of_edges(
        filename_cor,
        error_vars_dict,
        cor_min_max_dict,
        fit_indices_thresholds_dict,
        alphas=alphas,
    )
    for i, a in enumerate(alphas):
        fig, axes = plt.subplots(1, len(model_type), figsize=(12, 4), sharex=True)
        for j, model in enumerate(model_type):
            powers = power_alphas[:, j, i]
            sns.histplot(powers, bins=40, alpha=0.2, kde=True, ax=axes[j])
            axes[j].set_title(f"model = {model}")
            axes[j].axvline(0.80, linewidth=1, linestyle="--", color="black")
            axes[j].set_xlabel("Power")
        (
            fig_title_suffix,
            fig_name_suffix,
            save_fig_folder,
        ) = generate_suffixes_and_saving_folder(filename_cor)
        fig_title = f"Frequency of edges with power {fig_title_suffix}: alpha = {a}"
        fig.suptitle(fig_title)
        fig.tight_layout()

        thresholds_suffix = generate_fig_name_suffix_from_thresholds(
            error_vars_dict, cor_min_max_dict, fit_indices_thresholds_dict
        )
        fig_name = (
            f"Power_count_{fig_name_suffix}_{thresholds_suffix}_alpha_{a:.3g}.png"
        )

        fig.savefig(op.join(save_fig_folder, fig_name))

    return power_alphas


def loop_for_compare_power(
    trait_type_list: list[str],
    n_edge: int,
    sample_n: int,
    error_vars_dict,
    cor_min_max_dict,
    fit_indices_thresholds_dict,
    alphas: list[float],
    plt_close=False,
):
    """
    function for looping compare_power_models
    """
    filename_cor_list = get_latest_files_with_args(
        trait_type_list=trait_type_list,
        n_edge=n_edge,
        sample_n=sample_n,
        data_type="correlation",
    )
    for filename_cor in filename_cor_list:
        compare_power_models(
            filename_cor,
            error_vars_dict,
            cor_min_max_dict,
            fit_indices_thresholds_dict,
            alphas,
        )
        if plt_close:
            plt.close()
