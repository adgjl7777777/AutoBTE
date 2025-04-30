#!/bin/bash
#SBATCH -J name
#SBATCH -p gpupart
#SBATCH -o ./out/%x.o.%j.out
#SBATCH -e ./out/%x.e.%j.err
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --nodelist=nanode04
#SBATCH -t 10000:00:00
#SBATCH --gres=gpu:2
##SBATCH --qos=short            

###################NVIDIA HPC SDK setting###################
NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS
############################################################

######################openmpi setting#######################
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/bin/:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/extras/qd/lib/:$LD_LIBRARY_PATH
############################################################

###################DFT(Vasp) path setting###################
export PATH=/home/Software/vasp.6.2.0.openacc.AMD/:$PATH
export LD_LIBRARY_PATH="/home/Software/BLAS-3.10.0/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/Software/lapack-3.10.0/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/Software/scalapack-2.1.0/:$LD_LIBRARY_PATH"
export ASE_VASP_COMMAND="mpirun -n 2 --bind-to socket /home/Software/vasp.6.2.0.openacc.AMD/vasp_std"
##export ASE_VASP_COMMAND="mpirun --bind-to socket /home/Software/vasp.6.2.0.openacc.AMD/vasp_std"
export VASP_PP_PATH="/home/transcendence/ppot/"
############################################################

python3 vasp.py