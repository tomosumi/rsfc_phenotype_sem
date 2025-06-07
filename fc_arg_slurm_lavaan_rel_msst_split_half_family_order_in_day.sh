#!/bin/bash

n_divide=$4

# Create arrays of start and end of edges
num_edge=$2
edge_start=($(seq 0 "$(( num_edge / n_divide ))" "$(( num_edge -  $(( num_edge / n_divide )) ))" ))
edge_end=($(seq "$(( num_edge / n_divide - 1 ))" "$(( num_edge / n_divide ))" "$num_edge" ))

pyfile='/home/cezanne/t-haitani/hcp_data/code/Python/apply_model.py'
python3 $pyfile --fc_filename $1 \
                --fisher_z \
                --edge_start ${edge_start[$SLURM_ARRAY_TASK_ID]} \
                --edge_end ${edge_end[$SLURM_ARRAY_TASK_ID]} \
                --edge_total ${num_edge} \
                --edge_divide ${n_divide} \
                --array_id $SLURM_ARRAY_TASK_ID \
                --save_data \
                --save_residuals \
                --dtype_memmap "float32" `# follwoings are modeling arguments` \
                --model_type model_onlyFC \
                --cor_type pearson \
                --pca_z_scaling \
                --cov_cor \
                --use_lavaan \
                --multistate_single_trait \
                --order_in_day \
                --add_method_marker \
                --std_lv \
                --get_std_solutions \
                --model_fit_obj MLR \
                --fc_unit session \
                --se_robust  \
                --control $3 `# followings are model convergence options` \
                --controlBefore $5 \
                --iter_max 10000 `# follwoings are subject selection arguments` \
                --rms_removal \
                --rms_remove_percentage 0.1 \
                --rms_thres 0.25\
                --rms_pass_all_or_any all \
                --fmri_run_all_or_any all \
                --trait_all_or_any all \
                --cog_missing remove \
                --split_half_family \
                --select_cb
