#!/bin/bash
#SBATCH --job-name=RISEDisplay
#SBATCH --output=notebookjob%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

#Move to your working directory
cd /home/aveera

#Load modules
module load Python/3.8.6-GCCcore-10.2.0
module load git

#Activate your virtual environment
source /data/aveera/venv/bin/activate

#Add virtual environment library directory to search path
export LD_LIBRARY_PATH=/apps/foss2020b/software/Python/3.8.6-GCCcore-10.2.0/lib:$LD_LIBRARY_PATH
export TQDM_DISABLE=1   

#Run the notebook headlessly using the environment's python executable
/data/aveera/venv/bin/python -m jupyter nbconvert --to notebook --execute RISEDisplay.ipynb --output executed_RISEDisplay.ipynb
