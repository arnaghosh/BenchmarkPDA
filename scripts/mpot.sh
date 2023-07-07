#!/usr/bin/env bash
#SBATCH --array=0-239%60
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=10GB
#SBATCH --time=1:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/mpot.%A.%a.out
#SBATCH --error=sbatch_err/mpot.%A.%a.err
#SBATCH --job-name=mpot

. /etc/profile
module load anaconda/3
conda activate ffcv

python hp_search_train_val.py --method mpot --dset office-home \
                            --source_domain Art --target_domain Clipart \
                            --sweep_idx $SLURM_ARRAY_TASK_ID