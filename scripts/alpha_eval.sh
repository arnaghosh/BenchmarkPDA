#!/usr/bin/env bash
#SBATCH --array=0-10%11
#SBATCH --partition=long
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/alpha_eval.%A.%a.out
#SBATCH --error=sbatch_err/alpha_eval.%A.%a.err
#SBATCH --job-name=alpha_eval

. /etc/profile
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

if [[ "$CUDA_VISIBLE_DEVICES" == *"MIG"* ]] 
then 
    unset CUDA_VISIBLE_DEVICES
fi

methods=('ar' 'ba3us' 'jumbot' 'jumbot' 'jumbot' 'mpot' 'mpot' 'mpot' 'mpot' 'mpot' 'pada')
idxs=(0 0 0 50 100 0 50 100 150 200 0)

start_idx=${idxs[SLURM_ARRAY_TASK_ID]}
end_idx=$((50+start_idx))

for ((i=$start_idx; i<=$end_idx; i++))
do
    python hp_search_train_val.py --method ${methods[SLURM_ARRAY_TASK_ID]} \
                                --mode 'eval' --dset office-home \
                                --source_domain Art --target_domain $target --sweep_idx $i
done
