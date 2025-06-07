#!/bin/bash

# $1 is a filename of shell script calling .py file
# $2 is a .npy file storing FC data for analysis
# $3 is a number of edges
# $4 is a number of division
# $5 is a string of covariates

pyfile='/home/cezanne/t-haitani/hcp_data/code/Python/apply_model.py'

n_divide=$4
n_array=$(( n_divide - 1 ))
random_seed=$6

job_output=$(sbatch --array 0-$n_array \
    --partition=mm,lm \
    --mem-per-cpu=2G \
    $1 $2 $3 $5 $n_divide $random_seed)
echo "$job_output"
