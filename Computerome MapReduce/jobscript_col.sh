#!/bin/sh

# This is the jobscript file to submit the collector.py to the queueing system of Computerome

#PBS -W group_list=pr_course -A pr_course
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l mem=10GB
#PBS -l walltime=1:00:00
#PBS -e /home/projects/pr_course/people/konlyr/tools/collector.err
#PBS -o /home/projects/pr_course/people/konlyr/tools/collector.out
#PBS -N collector
#PBS -M s232994@dtu.dk
#PBS -m abe

# Load required modules
module load python36

# Set up and activate virtual environment
VENV_DIR="$HOME/myenv"

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Set the NLTK_DATA environment variable to use the downloaded data
export NLTK_DATA="$VENV_DIR/nltk_data"

# Run the collector script
python /home/projects/pr_course/people/konlyr/tools/collector.py