#!/usr/bin/env bash
#SBATCH --array=0-24%25
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=10GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/ba3us.%A.%a.out
#SBATCH --error=sbatch_err/ba3us.%A.%a.err
#SBATCH --job-name=ba3us

. /etc/profile
module load anaconda/3
conda activate ffcv

python hp_search_train_val.py --method ba3us --dset office-home \
                            --source_domain Art --target_domain Clipart \
                            --sweep_idx $SLURM_ARRAY_TASK_ID