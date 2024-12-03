import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from torch.multiprocessing import Pool, set_start_method
import pickle

# Load the TSV file into a DataFrame
file_path = '../share/use_this_one.tsv.gz'
df = pd.read_csv(file_path, sep='\t', compression='gzip')

# Display the first few rows of the DataFrame
print(df.head())

# Load pre-trained model tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load pre-trained model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def distilbert_vectorize_batch(texts, device):
    # Tokenize input texts
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Move inputs to the GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get the hidden states from DistilBERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings for the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embeddings

def process_batch(batch, device):
    return distilbert_vectorize_batch(batch, device)

def worker_init(gpu_id):
    torch.cuda.set_device(gpu_id)
    global model
    model.to(f'cuda:{gpu_id}')

if __name__ == '__main__':
    # Set the start method to 'spawn'
    set_start_method('spawn')

    # Define batch size
    batch_size = 100000

    # Split the data into batches
    batches = [df['body'][i:i + batch_size].tolist() for i in range(0, len(df), batch_size)]

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Create a pool of workers, one for each GPU
    with Pool(processes=num_gpus, initializer=worker_init, initargs=(torch.cuda.current_device(),)) as pool:
        # Distribute the batches across the GPUs
        results = pool.starmap(process_batch, [(batch, f'cuda:{i % num_gpus}') for i, batch in enumerate(batches)])

    print('Flattening the results')
    # Flatten the list of results

    # Save the results to a pickle file
    with open('data/results_bert.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    flattened_results = [item for sublist in results for item in sublist]

    flattened_results = np.vstack(flattened_results)
    
    # Convert the flattened results to a DataFrame
    results_df = pd.DataFrame(flattened_results)

    # Define the output file path
    output_file_path = 'data/bert_embeddings.csv.gz'

    # Save the DataFrame to a compressed CSV file
    results_df.to_csv(output_file_path, index=False, compression='gzip')

    print(f"Results saved to {output_file_path}")