# rsfc_phenotype_sem
Code used in a submitted research paper, entitled with "Effects of measurement errors on relationships between resting-state functional connectivity and psychological phenotypes: Structural equation modeling"
The analyses comprised of the three modules:
1. preprocess, where parcellation, spike regression, and global signal regression are applied. Intermediate files are created.
2. apply_model, where CFA or SEM is applied to RSFC and phenotype measures. Intermediate files are created.
3. postprocess, where graphs are drawn and statistical results are obtained

We used Slurm Workload Manager to conduct analyses on resting-state functional connectivity in 2. apply_model.
Analyses on measurement models with method factors representing order in days were conducted with fc_arg_slurm_lavaan_rel_msst_split_half_family_order_in_day.sh, called from sbatch_for_rel_with_filename.sh
If you have any question, please email to t-haitani@atr.jp.
