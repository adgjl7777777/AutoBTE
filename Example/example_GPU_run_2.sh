#!/bin/bash
#SBATCH -J name
#SBATCH -p gpupart
#SBATCH -o ./out/%x.o.%j.out
#SBATCH -e ./out/%x.e.%j.err
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --nodelist=nanode05
#SBATCH -t 10000:00:00
#SBATCH --gres=gpu:a100:2
##SBATCH --qos=short     

######################MPI base setting######################
NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS
############################################################

####################openmpi setting####################
export PATH=/home/Software/openmpi-4.1.0-10G/bin/:$PATH
export LD_LIBRARY_PATH=/home/Software/openmpi-4.1.0-10G/lib/:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/bin/:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/extras/qd/lib/:$LD_LIBRARY_PATH
############################################################

###################DFT(Vasp) path setting###################
export PATH=/home/Software/vasp.6.2.0.openacc/:$PATH
export LD_LIBRARY_PATH="/home/Software/BLAS-3.10.0/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/Software/lapack-3.10.0/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/Software/scalapack-2.1.0/:$LD_LIBRARY_PATH"
export ASE_VASP_COMMAND="mpirun -n 2 --bind-to socket vasp_std"
## export ASE_VASP_COMMAND="mpirun --bind-to socket vasp_std"
export VASP_PP_PATH="/home/transcendence/ppot/"
############################################################

srun python3 vasp.py