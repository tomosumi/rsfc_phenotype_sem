# rsfc_phenotype_sem
Code used in a submitted research paper, entitled with "Effects of measurement errors on relationships between resting-state functional connectivity and psychological phenotypes: Structural equation modeling"
The analyses comprised of the three modules:
1. preprocess, where parcellation, spike regression, and global signal regression are applied. Intermediate files are created.
2. apply_model, where CFA or SEM is applied to RSFC and phenotype measures. Intermediate files are created.
3. postprocess, where graphs are drawn and statistical results are obtained

We used Slurm Workload Manager to conduct analyses in 2. apply_model.
If you have any question, please email to t-haitani@atr.jp
