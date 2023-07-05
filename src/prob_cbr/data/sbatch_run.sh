#!/usr/bin/env bash
#SBATCH --job-name=prior_map
#SBATCH --output=/mnt/nfs/scratch1/rajarshi/deep_case_based_reasoning/compute_prior_maps-%A_%a.out
#SBATCH -e /mnt/nfs/scratch1/rajarshi/deep_case_based_reasoning/res_%j.err
#SBATCH --partition=longq
#SBATCH --time=07-00:00:00
#SBATCH --mem=10G
#SBATCH --array=0-99

# Set to scratch/work since server syncing will occur from here
# Ensure sufficient space else runs crash without error message

# Flag --count specifies number of hyperparam settings (runs) tried per job
# If SBATCH --array flag is say 0-7 (8 jobs) then total (8 x count)
# hyperparam settings will be tried
wandb agent rajarshd/Open-BIo-Link-src_prob_cbr/y11zgpjw