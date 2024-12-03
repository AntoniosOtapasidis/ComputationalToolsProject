#!/bin/python3

import os
import sys
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import re
import logging
import time

# This is the worker for MapReduce. Perform lemmatization of the text of each row of the chunk assigned by the administrator

# Setup logging with UTF-8 encoding to handle Unicode characters
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler for logging to 'worker.log', with UTF-8 encoding
file_handler = logging.FileHandler('worker.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Create a logging format and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Set the NLTK data path to the location where omw-1.4 and other resources are downloaded
os.environ['NLTK_DATA'] = '/home/people/konlyr/nltk_data'

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Preprocess text function with POS tagging
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]

        pos_tags = pos_tag(words)
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        return ' '.join(lemmatized_words)
    except Exception as e:
        logger.error(f"Error in preprocess_text with text: {text} - {str(e)}")
        return ''

# Function to lemmatize or return the original if lemmatization fails
def lemmatize_or_original(text):
    lemmatized = preprocess_text(text)
    if lemmatized.strip():
        logger.info(f"Lemmatized row for text: {text[:30].encode('utf-8', errors='ignore').decode('utf-8')}...")
        return lemmatized
    else:
        logger.warning(f"Lemmatization returned empty for text: {text[:30].encode('utf-8', errors='ignore').decode('utf-8')}...")
        return text

# Worker function to process a chunk
def worker(input_file, output_file):
    try:
        # Log file paths
        logger.info(f'Reading from: {input_file}')
        logger.info(f'Writing to: {output_file}')
        
        # Read the chunk without headers
        df = pd.read_csv(input_file, sep='\t', compression='gzip', header=None)

        # Assign column names explicitly
        df.columns = ['body', 'unique_id']

        # Log the shape and the column names
        logger.info(f"Dataframe shape after reading: {df.shape}")
        logger.info(f"Dataframe columns: {df.columns}")

        # Preprocess the 'body' column
        logger.info('Starting text preprocessing...')
        df['cleaned_body'] = df['body'].apply(lambda x: lemmatize_or_original(x) if isinstance(x, str) else x)

        # Log after preprocessing to confirm non-empty rows
        non_empty_cleaned = df['cleaned_body'].str.strip().replace('', pd.NA).dropna().shape[0]
        logger.info(f"Number of non-empty cleaned_body rows: {non_empty_cleaned}")

        # Print first few rows to ensure data integrity
        logger.info(f"First few rows of cleaned_body DataFrame:\n{df[['body', 'unique_id', 'cleaned_body']].head()}")

        # Check if DataFrame is empty before writing
        if df.empty:
            logger.warning("Warning: Dataframe to be saved is empty.")
        else:
            logger.info(f"Dataframe shape before saving: {df.shape}")
            # Save the processed DataFrame without compression for debugging purposes
            df[['body', 'unique_id', 'cleaned_body']].to_csv(
                output_file.replace(".gz", ""), sep='\t', index=False, header=False
            )
            logger.info(f'Uncompressed output written to: {output_file.replace(".gz", "")}')
            
            # Attempt writing with compression as well
            df[['body', 'unique_id', 'cleaned_body']].to_csv(
                output_file, sep='\t', index=False, compression='gzip', header=False
            )
            logger.info(f'Compressed output written to: {output_file}')
            time.sleep(5)  # Wait to ensure write completes

    except pd.errors.EmptyDataError:
        logger.warning(f"Warning: {input_file} is empty and will be skipped.")
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    worker(input_file, output_file)