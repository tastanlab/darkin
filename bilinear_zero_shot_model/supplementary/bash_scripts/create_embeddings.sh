#!/bin/bash
#
# -= Resources =-
#
#SBATCH --job-name=create_protein_embeddings
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --mem=50G
# SBATCH --ntasks-per-node=4
#SBATCH --qos=cuda
#SBATCH --partition=cuda
#SBATCH --time=7-0
#SBATCH --output=slurm_outputs/%j-job.out
#SBATCH --mail-type=ALL

### GPU request
#SBATCH --gres=gpu:1
### SBATCH -w gpu01

USER="mpekey"

#Module File
echo "Loading CUDA..."
module load cuda/11.8

# Setting the necessary environment variables
export PATH=/cta/capps/cuda/11.8/bin:$PATH
export CUDA_HOME=/cta/capps/cuda/11.8
export LD_LIBRARY_PATH=/cta/capps/cuda/11.8/lib64:$LD_LIBRARY_PATH

# Source your .bashrc (if necessary for other reasons)
source /cta/users/${USER}/.bashrc

# Activate your conda environment
source activate DKZ_pytorch
env

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Python script..."
echo "==============================================================================="

# Put Python script command below
COMMAND="python3 /scripts/data/embeddings.py --model_name esm2_t6_8M_UR50D --sequence_type kinase"

echo ${COMMAND}
echo "-------------------------------------------"
$COMMAND

RET=$?
echo
echo "Solver exited with return code: $RET"
exit $RET
