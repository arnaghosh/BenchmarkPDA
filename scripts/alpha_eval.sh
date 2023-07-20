#!/usr/bin/env bash
#SBATCH --array=0-4%5
#SBATCH --partition=long
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=16GB
#SBATCH --time=5:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/alpha_eval.%A.%a.out
#SBATCH --error=sbatch_err/alpha_eval.%A.%a.err
#SBATCH --job-name=alpha_eval

. /etc/profile
module load anaconda/3
conda activate ffcv

if [[ "$CUDA_VISIBLE_DEVICES" == *"MIG"* ]] 
then 
    unset CUDA_VISIBLE_DEVICES
fi

methods=('ar' 'ba3us' 'jumbot' 'mpot' 'pada')

python hp_search_train_val.py --method ${methods[SLURM_ARRAY_TASK_ID]} \
                            --mode 'eval' --dset office-home \
                            --source_domain Art --target_domain Clipart