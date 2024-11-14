#!/bin/sh
#PBS -W group_list=pr_course -A pr_course
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l mem=10GB
#PBS -l walltime=3:00:00
#PBS -e /home/projects/pr_course/people/konlyr/tools/tools.err
#PBS -o /home/projects/pr_course/people/konlyr/tools/tools.out
#PBS -N tools
#PBS -M s232994@dtu.dk
#PBS -m abe

# Load required modules
module load python36

# Set up and activate virtual environment
VENV_DIR="$HOME/myenv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip and install necessary packages
pip install --upgrade pip
pip install joblib pandas nltk

# Download necessary NLTK data
python -c "
import nltk
nltk.download('stopwords', download_dir='$VENV_DIR/nltk_data')
nltk.download('wordnet', download_dir='$VENV_DIR/nltk_data')
nltk.download('averaged_perceptron_tagger', download_dir='$VENV_DIR/nltk_data')
nltk.download('punkt', download_dir='$VENV_DIR/nltk_data')
nltk.download('omw-1.4', download_dir='$VENV_DIR/nltk_data')
"

# Set the NLTK_DATA environment variable to use the downloaded data
export NLTK_DATA="$VENV_DIR/nltk_data"

# Run the administrator script using the Python from the virtual environment
python /home/projects/pr_course/people/konlyr/tools/administrator.py

