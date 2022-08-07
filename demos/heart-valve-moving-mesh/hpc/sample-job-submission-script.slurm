#!/bin/bash
#SBATCH -J 16			# job name
#SBATCH -o err.16.o%j	# error file name
#SBATCH -N 2			# number of nodes
#SBATCH -n 40			# total number of cores
#SBATCH -p skx-dev		# job submission queue
#SBATCH -t 02:00:00		# job duration
#SBATCH -A BHV-FSI-S	# job allocation

# this set of modules may change over time
module load intel/19.1.1
module load tacc-singularity
module load mvapich2

# use when you get python conflicts
mv ~/.local ~/.local-temp-rename

# helper variables to make filepaths shorter
SW=$STOCKYARD"/stampede2/fenics-container"
FENICS=$SW"/fenics.sif"


# use to disable hyperthreading (default is usually 1 already)
export OMP_NUM_THREADS=1

# add the dependencies to the PYTHONPATH environment variable
export PYTHONPATH=$PYTHONPATH":"$SW"/COFFEE"
export PYTHONPATH=$PYTHONPATH":"$SW"/FInAT"
export PYTHONPATH=$PYTHONPATH":"$SW"/networkx"
export PYTHONPATH=$PYTHONPATH":"$SW"/pulp"
export PYTHONPATH=$PYTHONPATH":"$SW"/tsfc"
export PYTHONPATH=$PYTHONPATH":"$SW"/singledispatch"
export PYTHONPATH=$PYTHONPATH":"$SW"/tIGAr"
export PYTHONPATH=$PYTHONPATH":"$SW"/ShNAPr"

# actually run a script
ibrun singularity exec $FENICS python3 heart-valve-moving-mesh.py \
--mesh-folder           ./mesh/                         \
--results-folder        ./results/                      \
--restarts-folder       ./restarts/	                    \
--valve-folder          ./valve/fine/                   \
--num-patches           10                              \
--stent-patches         4 5 6 7 8 9 10                  \
--valve-smesh-suffix    .txt                            \
--delta-t               1e-4                            \
--num-steps             20000                           \
--block-max-iters       3                               \
--block-tol             1e-3                            \
--ksp-max-iters         300                             \
--leaflet-material      isotropic-lee-sacks             \
--leaflet-c0            676080                          \
--leaflet-c1            132848                          \
--leaflet-c2            38.1878                         \
--leaflet-thickness     0.0386                          \
--r-self                0.0308                          \
--r-max                 0.0237                          \
> log