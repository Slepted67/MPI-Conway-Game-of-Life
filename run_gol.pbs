#!/bin/bash
#PBS -l nodes=4:ppn=1
#PBS -l walltime=00:01:00
#PBS -q batch
#PBS -N GOL_MPI
#PBS -j oe

cd $PBS_O_WORKDIR

mpirun -np 4 ./gol_mpi
