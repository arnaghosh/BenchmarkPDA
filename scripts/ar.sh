#!/usr/bin/env bash
#SBATCH --array=0-47%48
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=10GB
#SBATCH --time=1:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/ar.%A.%a.out
#SBATCH --error=sbatch_err/ar.%A.%a.err
#SBATCH --job-name=ar

. /etc/profile
module unload python
module load anaconda/3
if [ $USER == 'roy.eyono' ]
then
    conda activate ffcv_eg
    target=Real_World
elif [ $USER == 'ghosharn' ]
then
    conda activate ffcv_new
    target=Clipart
else
    conda activate ffcv
fi

python hp_search_train_val.py --method ar --dset office-home \
                            --source_domain Art --target_domain $target \
                            --data_folder $SCRATCH/DomainAdaptation/datasets \
                            --sweep_idx $SLURM_ARRAY_TASK_ID