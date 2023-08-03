#!/usr/bin/env bash
#SBATCH --array=0-134%60
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=10GB
#SBATCH --time=2:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/jumbot.%A.%a.out
#SBATCH --error=sbatch_err/jumbot.%A.%a.err
#SBATCH --job-name=jumbot

. /etc/profile
module unload python
module load anaconda/3
conda activate ffcv_eg

python hp_search_train_val.py --method jumbot --dset office-home \
                            --source_domain Art --target_domain Real_World --data_folder $SCRATCH/DomainAdaptation/datasets \
                            --sweep_idx $SLURM_ARRAY_TASK_ID