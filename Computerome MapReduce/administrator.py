#!/bin/python3

import os
import subprocess
import pandas as pd

# This is the administrator for MapReduce
# Reads the initial file, splits it into chunks and assigns each chunk to a worker

# Specify environment directory path
VENV_DIR = "/home/people/konlyr/myenv"

# Function to submit a job to the Queueing System
def submit2(command, runtime, cores, ram, directory='', modules='', group='pr_course',
            output='/dev/null', error='/dev/null'):
    runtime = int(runtime)
    cores = int(cores)
    ram = int(ram)
    if cores > 10:
        print("Can't use more than 10 cores on a node")
        sys.exit(1)
    if ram > 120:
        print("Can't use more than 120 GB on a node")
        sys.exit(1)
    if runtime < 1:
        print("Must allocate at least 1 minute runtime")
        sys.exit(1)
    minutes = runtime % 60
    hours = int(runtime / 60)
    walltime = "{:d}:{:02d}:00".format(hours, minutes)
    if directory == '':
        directory = os.getcwd()

    # Creating the job script as a string
    script = f'''#!/bin/sh
#PBS -A {group} -W group_list={group}
#PBS -e {error} -o {output}
#PBS -d {directory}
#PBS -l nodes=1:ppn={cores},mem={ram}GB
#PBS -l walltime={walltime}
'''
    if modules:
        script += f'module load {modules}\n'
    script += f'{command}\n'

    # Submit the job
    job = subprocess.run(['qsub'], input=script, stdout=subprocess.PIPE, universal_newlines=True)
    jobid = job.stdout.split('.')[0]
    return jobid

# Function to split the dataset into chunks and save them
def split_dataframe(input_tsv_gz, output_dir, num_parts=100):
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the entire dataset without assuming any header (since the original file does not have one)
    df = pd.read_csv(input_tsv_gz, sep='\t', compression='gzip', header=None)

    # Calculate the chunk size to split the DataFrame into the specified number of parts
    chunk_size = len(df) // num_parts + (1 if len(df) % num_parts else 0)

    # Split the DataFrame into smaller chunks without headers
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]

        # Write each chunk to a separate file without including headers
        chunk.to_csv(f"{output_dir}/chunk_{i}.tsv.gz", sep='\t', index=False, header=False, compression='gzip')

# Administrator function to submit chunks to workers in the queueing system
def administrator(input_tsv_gz, output_dir, runtime, cores, ram):
    worker_path = "/home/projects/pr_course/people/konlyr/tools/worker.py"

    # Split the dataset into chunks
    split_dataframe(input_tsv_gz, output_dir)

    # Submit each chunk for preprocessing
    chunk_files = os.listdir(output_dir)
    for chunk_file in chunk_files:
        if not chunk_file.endswith('.tsv.gz'):
            continue
        input_file_path = os.path.join(output_dir, chunk_file)
        output_file_path = os.path.join(output_dir, f"{chunk_file}_out.tsv")
        error_file_path = os.path.join(output_dir, f"{chunk_file}_err")

        job_id = submit2(
            command=f'export NLTK_DATA={VENV_DIR}/nltk_data && {VENV_DIR}/bin/python3 {worker_path} {input_file_path} {output_file_path}',
            directory=output_dir,
            modules='tools python36',
            runtime=runtime,
            cores=cores,
            ram=ram,
            output=output_file_path,
            error=error_file_path
        )
        print(f'Submitted job {job_id} for {chunk_file}')

if __name__ == "__main__":
    administrator("/home/projects/pr_course/people/konlyr/tools/preprocessed_depression_dataset_full.tsv.gz",
                  "/home/projects/pr_course/people/konlyr/tools/split_chunks",
                  runtime=180, cores=1, ram=4)