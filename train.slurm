#!/bin/bash
#SBATCH -p hgx2q
#SBATCH --job-name=OronaTSE
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time 04-0:00:00    # time (D-HH:MM:SS)
module purge
module load slurm/20.02.7
source /home/qinghua/anaconda3/bin/activate orona_tse
export PYTHONPATH=root_path:$PYTHONPATH
srun python train.py ${@}
