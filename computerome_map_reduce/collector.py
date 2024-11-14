#!/bin/python3

import os
import pandas as pd

# Collector function to merge processed chunks
def collector(output_dir, final_output_file):
    # List all output files that match the new pattern
    output_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith('.tsv_out.tsv')],
        key=lambda x: int(x.split('_')[1].split('.')[0])  # Sort by sequence number
    )

    # Read all files and add them to a list of DataFrames
    df_list = []
    for filename in output_files:
        file_path = os.path.join(output_dir, filename)
        df = pd.read_csv(file_path, sep='\t', header=None)
        df_list.append(df)

    # Concatenate all DataFrames
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(final_output_file, sep='\t', index=False, header=False)

if __name__ == "__main__":
    collector("/home/projects/pr_course/people/konlyr/tools/split_chunks",
              "/home/projects/pr_course/people/konlyr/tools/final_cleaned_dataset.tsv")
