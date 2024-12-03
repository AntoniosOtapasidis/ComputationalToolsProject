import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
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

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move the model to the GPU
model.to(device)

def distilbert_vectorize(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Move inputs to the GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get the hidden states from DistilBERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings for the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embeddings

# Apply DistilBERT vectorization to each row in the 'body' column
results = df['body'].apply(distilbert_vectorize)

del model, tokenizer, device, df

# Save the results to a pickle file using with open
pickle_file_path = 'data/bert_embeddings.pkl'
with open(pickle_file_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Results saved to {pickle_file_path}")

# Convert the 'distilbert_vector' column to a DataFrame
results_df = pd.DataFrame(results.tolist().squeeze())

del results

# Define the output file path
output_file_path = 'data/bert_embeddings.csv.gz'

# Save the DataFrame to a compressed CSV file
results_df.to_csv(output_file_path, index=False, compression='gzip')

print(f"Results saved to {output_file_path}")