#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com), Jonathan Gustafsson Frennert (jonathan.frennert@gmail.com)
#
# Run an sbatch arrayjob with a file containing a list of
# commands to run with dataset on offline node.
# 
# Assuming this file has been edited and renamed slurm_arrayjob.sh, here's an
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```

# constants
CONDA_ENV_NAME=pdiot-ml

MAIN_HOME=/home
MAIN_USER=
MAIN_PROJECT=pdiot-ml
MAIN_PATH=${MAIN_HOME}/${MAIN_USER}
MAIN_PROJECT_PATH=${MAIN_PATH}/${MAIN_PROJECT}

SCRATCH_HOME=/disk/scratch
SCRATCH_USER=
SCRATCH_PROJECT=pdiot-ml
SCRATCH_PATH=${SCRATCH_HOME}/${SCRATCH_USER}
SCRATCH_PROJECT_PATH=${SCRATCH_PATH}/${SCRATCH_PROJECT}

DATA_DN=data
OUTPUT_DN=ckpt
INPUT_PATH=${DATA_DN}/sets

DATA_SCRIPT_FN=pd_odgt.py


# ====================
# Options for sbatch
# ====================
# FMI about options, see https://slurm.schedmd.com/sbatch.html
# N.B. options supplied on the command line will overwrite these set here

# *** To set any of these options, remove the first comment hash '# ' ***
# i.e. `# # SBATCH ...` -> `#SBATCH ...`

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
# #SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
# #SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
# #SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
# #SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
# #SBATCH --cpus-per-task=4

# Maximum time for the job to run, format: days-hours:minutes:seconds
# #SBATCH --time=1-04:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
# #SBATCH --partition=PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]


# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
mkdir -p ${SCRATCH_PATH}

# Activate your conda environment
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_HOME"

# input data directory path on the DFS
src_path=${MAIN_PROJECT_PATH}/${INPUT_PATH}


# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_PROJECT_PATH}/${INPUT_PATH}
mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# ======================
# Pre-processing data
# ======================
# pre-processing the data on the scratch disk with the data script

echo "Pre-processing data in scratch space"

python ${MAIN_PROJECT_PATH}/${DATA_DN}/${DATA_SCRIPT_FN} --dir ${SCRATCH_PROJECT_PATH}/${INPUT_PATH}


# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_file=$1
ID=$[SLURM_ARRAY_TASK_ID + 1]
EXP="`sed \"${ID}q;d\" ${experiment_file}`"
IFS=$',' read -ra VALS <<< "$EXP"

LEN=${#VALS[@]}
COMMAND=${VALS[$LEN - 1]}
echo "Running provided command: ${COMMAND}"
if eval "${COMMAND}"; then
    echo "Command ran successfully!"
else
    echo "Command failed!"
    exit 1
fi


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"


EXP_NAME=${VALS[$LEN - 2]}
src_path=${SCRATCH_PROJECT_PATH}/${OUTPUT_DN}/${EXP_NAME}
dest_path=${MAIN_PROJECT_PATH}/${OUTPUT_DN}/${EXP_NAME}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# ======================================
# Cleaning up scratch space
# ======================================
# to make sure that we do not have issues with new batch experiments and to free up space, we delete our experiment after transfer

echo "Deleting output files in scratch space"
rm -r ${src_path}/*


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "Job finished successfully!"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
