#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import networkx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import scipy.sparse as sp
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import scipy
import os


# In[2]:


#compress the dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

# Load the dataset

#compress the dataset
depression = pd.read_csv("use_this_one.tsv.gz", sep='\t' , compression='gzip')
depression.head()

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(depression["cleaned_body"])
feature_names = tfidf_vectorizer.get_feature_names_out()



# Define chunk size and output directory
chunk_size = 10000  # Number of rows per batch
output_dir = "tfidf_chunks"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Check if chunks already exist
existing_chunks = {int(f.split('_')[-1].split('.')[0]) for f in os.listdir(output_dir) if f.startswith("tfidf_chunk")}
print(f"Existing chunks: {existing_chunks}")

# Fit the vectorizer on the full dataset
if not existing_chunks:
    print("No existing chunks found. Fitting the vectorizer on the full dataset...")
    tfidf_vectorizer.fit(depression["cleaned_body"])
    print("Vectorizer fitted.")
else:
    print("Skipping vectorizer fitting as chunks already exist.")

# Process the data in chunks
for i in range(0, len(depression), chunk_size):
    chunk_index = i // chunk_size
    chunk_filename = f"{output_dir}/tfidf_chunk_{chunk_index}.npz"
    
    if chunk_index in existing_chunks:
        print(f"Skipping chunk {chunk_index}: already exists.")
        continue
    
    # Process current chunk
    print(f"Processing chunk {chunk_index}...")
    batch = depression["cleaned_body"][i:i+chunk_size]
    tfidf_matrix = tfidf_vectorizer.transform(batch)  # Transform the current chunk
    
    # Save each chunk as a sparse matrix
    scipy.sparse.save_npz(chunk_filename, tfidf_matrix)
    print(f"Saved chunk {chunk_index} to {chunk_filename}")

print("Processing complete. All chunks saved.")


# In[4]:



# Directory where chunks are saved
output_dir = "tfidf_chunks"

# Combined output file
combined_filename = "combined_tfidf_matrix.npz"

# Check if the combined file already exists
if os.path.exists(combined_filename):
    print(f"The combined matrix already exists at {combined_filename}. Skipping reassembly.")
else:
    # List all saved chunk files
    chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".npz")])
    
    if not chunk_files:
        print("No chunk files found in the output directory. Exiting.")
    else:
        print(f"Found {len(chunk_files)} chunk files. Reassembling the matrix...")

        # Load and combine all chunks
        sparse_matrix_list = [scipy.sparse.load_npz(file) for file in chunk_files]
        combined_matrix = scipy.sparse.vstack(sparse_matrix_list)

        # Save the reassembled matrix
        scipy.sparse.save_npz(combined_filename, combined_matrix)
        print(f"Reassembled matrix saved to {combined_filename}")





# Load the sparse matrix
combined_matrix = scipy.sparse.load_npz("combined_tfidf_matrix.npz")

# Check the shape
print("Shape of the combined matrix:", combined_matrix.shape)


# In[6]:


# Check basic properties
print(f"Shape: {combined_matrix.shape}")
print(f"Number of non-zero elements: {combined_matrix.nnz}")


# In[7]:

"""
# Accessing elements (convert a portion to dense if needed)
row_index = 0  # Example row
dense_row = combined_matrix[row_index].toarray()
print(f"First row: {dense_row}")


import numpy as np

# Assuming `combined_matrix` is the TF-IDF sparse matrix
# and `feature_names` is the list of terms from the TF-IDF vectorizer

# Step 1: Calculate term frequencies
term_frequencies = np.array(combined_matrix.sum(axis=0)).flatten()

# Step 2: Sort the term frequencies
sorted_indices = np.argsort(term_frequencies)[::-1]  # Indices of terms sorted by frequency
sorted_frequencies = term_frequencies[sorted_indices]

# Step 3: Normalize term frequencies
max_frequency = sorted_frequencies[0]
normalized_frequencies = sorted_frequencies / max_frequency

# Step 4: Store results in a dictionary
tfs = [
    {"Term": feature_names[term_index], "Frequency": freq, "Normalized_Frequency": norm_freq}
    for term_index, freq, norm_freq in zip(sorted_indices, sorted_frequencies, normalized_frequencies)
]

# Display the first few results
for entry in tfs[:10]:
    print(entry)


import pandas as pd
import matplotlib.pyplot as plt

# Convert the list of dictionaries to a Pandas DataFrame
tfs_df = pd.DataFrame(tfs)

# Sort by "Normalized_Frequency" in ascending order and select the top 10
top_tf = tfs_df.sort_values(by="Normalized_Frequency", ascending=False).head(10)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_tf['Term'], top_tf["Normalized_Frequency"])
plt.xlabel("Terms")
plt.ylabel("Normalized TF Score")
plt.title("Top 10 Terms with Highest TF Scores")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


import csv

# Save to a CSV file
with open("term_frequencies.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Term", "Frequency", "Normalized_Frequency"])
    writer.writeheader()
    writer.writerows(tfs)

print("Term frequencies saved to 'term_frequencies.csv'")

import pandas as pd
import numpy as np


# In[8]:



from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import scipy
import os

# Example large dataset: Subset with label == 1.0
subset = depression[depression["label"] == 1.0]

# Chunk size for processing
chunk_size = 10000  # Number of rows per batch

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the full subset
tfidf_vectorizer.fit(subset["cleaned_body"])

# Directory to save chunks
output_dir = "tfidf_subset_chunks_label1"
os.makedirs(output_dir, exist_ok=True)

# Process the subset in chunks
for i in range(0, len(subset), chunk_size):
    batch = subset["cleaned_body"][i:i+chunk_size]
    tfidf_matrix = tfidf_vectorizer.transform(batch)  # Transform the current chunk

    # Save each chunk as a sparse matrix
    chunk_filename = f"{output_dir}/tfidf_chunk_{i//chunk_size}.npz"
    scipy.sparse.save_npz(chunk_filename, tfidf_matrix)
    print(f"Saved chunk {i//chunk_size} to {chunk_filename}")

print("Subset processing complete. All chunks saved.")

# Reassemble the sparse matrix from saved chunks
chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".npz")])
sparse_matrix_list = [scipy.sparse.load_npz(file) for file in chunk_files]
combined_matrix = scipy.sparse.vstack(sparse_matrix_list)

# Save the reassembled matrix
combined_filename = "combined_subset_tfidf_matrix_label1.npz"
scipy.sparse.save_npz(combined_filename, combined_matrix)
print(f"Reassembled matrix saved to {combined_filename}")

# Load the reassembled sparse matrix
combined_matrix = scipy.sparse.load_npz("combined_subset_tfidf_matrix_label1.npz")

# Analyze the reassembled matrix
print(f"Shape: {combined_matrix.shape}")
print(f"Number of non-zero elements: {combined_matrix.nnz}")

# Accessing elements (convert a portion to dense if needed)
row_index = 0  # Example row
dense_row = combined_matrix[row_index].toarray()
print(f"First row: {dense_row}")



# In[9]:


"""
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import scipy
import os

# Example large dataset: Subset with label == 0.0
subset = depression[depression["label"] == 0.0]

# Chunk size for processing
chunk_size = 10000  # Number of rows per batch

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the full subset
tfidf_vectorizer.fit(subset["cleaned_body"])

# Directory to save chunks
output_dir = "tfidf_subset_chunks"
os.makedirs(output_dir, exist_ok=True)

# Process the subset in chunks
for i in range(0, len(subset), chunk_size):
    batch = subset["cleaned_body"][i:i+chunk_size]
    tfidf_matrix = tfidf_vectorizer.transform(batch)  # Transform the current chunk

    # Save each chunk as a sparse matrix
    chunk_filename = f"{output_dir}/tfidf_chunk_{i//chunk_size}.npz"
    scipy.sparse.save_npz(chunk_filename, tfidf_matrix)
    print(f"Saved chunk {i//chunk_size} to {chunk_filename}")

print("Subset processing complete. All chunks saved.")

# Reassemble the sparse matrix from saved chunks
chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".npz")])
sparse_matrix_list = [scipy.sparse.load_npz(file) for file in chunk_files]
combined_matrix = scipy.sparse.vstack(sparse_matrix_list)

# Save the reassembled matrix
combined_filename = "combined_subset_tfidf_matrix.npz"
scipy.sparse.save_npz(combined_filename, combined_matrix)
print(f"Reassembled matrix saved to {combined_filename}")

# Load the reassembled sparse matrix
combined_matrix = scipy.sparse.load_npz("combined_subset_tfidf_matrix.npz")

# Analyze the reassembled matrix
print(f"Shape: {combined_matrix.shape}")
print(f"Number of non-zero elements: {combined_matrix.nnz}")

# Accessing elements (convert a portion to dense if needed)
row_index = 0  # Example row
dense_row = combined_matrix[row_index].toarray()
print(f"First row: {dense_row}")

"""


# In[ ]:

"""
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import scipy.sparse
import pandas as pd
import numpy as np

# Load your dataset (replace this with your actual data loading code)
# Assuming `depression` is your DataFrame with columns 'cleaned_body' and 'label'

# Separate subsets based on label
label_0_subset = depression[depression["label"] == 0.0]["cleaned_body"]
label_1_subset = depression[depression["label"] == 1.0]["cleaned_body"]

# Initialize TF-IDF Vectorizer and fit on the combined data for consistent vocabulary
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(pd.concat([label_0_subset, label_1_subset]))

# Transform the subsets
label_0_matrix = tfidf_vectorizer.transform(label_0_subset)
label_1_matrix = tfidf_vectorizer.transform(label_1_subset)

# Compute term frequencies
term_frequencies_label_0 = label_0_matrix.sum(axis=0).A1  # Sum across rows for label 0
term_frequencies_label_1 = label_1_matrix.sum(axis=0).A1  # Sum across rows for label 1

# Get feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create dictionaries for word clouds
word_freq_label_0 = {
    feature_names[i]: term_frequencies_label_0[i]
    for i in range(len(term_frequencies_label_0))
    if term_frequencies_label_0[i] > 0
}

word_freq_label_1 = {
    feature_names[i]: term_frequencies_label_1[i]
    for i in range(len(term_frequencies_label_1))
    if term_frequencies_label_1[i] > 0
}

# Generate word clouds
wordcloud_label_0 = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq_label_0)
wordcloud_label_1 = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq_label_1)

# Plot the word clouds
plt.figure(figsize=(16, 8))

# Label 0 Word Cloud
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_label_0, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Label 0", fontsize=16)

# Label 1 Word Cloud
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_label_1, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Label 1", fontsize=16)
plt.savefig("wordcloud.png", dpi=300)  # Save the plot as a PNG file
plt.tight_layout()
plt.show()
"""


# In[ ]:
"""


import pandas as pd
from collections import Counter
from math import log2

# Initialize variables
doc_freq = Counter()  # Counter to accumulate document frequencies
unique_id = 0  # Total number of documents

# Process dataset in chunks
chunk_size = 10000  # Adjust based on your system's memory
chunks = pd.read_csv("use_this_one.tsv.gz", sep='\t' , compression='gzip',chunksize=chunk_size)  # Load your dataset in chunks

for chunk in chunks:
    # Tokenize the 'cleaned_body' column into sets of unique tokens per document
    chunk['tokens'] = chunk['cleaned_body'].apply(lambda x: set(x.split()))  # Tokenize into sets
    unique_id += len(chunk)  # Update total document count
    
    # Update document frequencies
    for tokens in chunk['tokens']:
        doc_freq.update(tokens)

# Calculate IDF for each term
idf = {token: log2(unique_id / (1 + count)) for token, count in doc_freq.items()}

# Convert IDF to DataFrame
idf_df = pd.DataFrame(list(idf.items()), columns=['Term', 'IDF'])

# Save IDF values to CSV
idf_df.to_csv("idf_values.csv", index=False)
print("IDF values saved to 'idf_values.csv'")

# Sort and visualize terms with the lowest IDF scores
top_idf_df = idf_df.sort_values(by="IDF", ascending=True).head(10)

# Plot the bar chart
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_idf_df['Term'], top_idf_df['IDF'], color="orange")
plt.xlabel("Terms")
plt.ylabel("IDF Score")
plt.title("Top 10 Terms with Lowest IDF Scores (Most Common Terms)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("IDF_scores.png", dpi=300)  # Save the plot as a PNG file
plt.show()


"""

# In[13]:

"""
from joblib import Parallel, delayed
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Define the chunk size
chunk_size = 10000  # Adjust based on your system's memory capacity

# Split the sparse matrix into chunks
def split_sparse_matrix(sparse_matrix, chunk_size):
    n_rows = sparse_matrix.shape[0]
    return [sparse_matrix[i:min(i + chunk_size, n_rows)] for i in range(0, n_rows, chunk_size)]

# Function to process a chunk
def process_chunk(chunk, n_components):
    chunk = chunk.astype('float32')
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(chunk)

# Split the matrix into chunks
chunks = split_sparse_matrix(combined_matrix, chunk_size)

# Parallelize the SVD process
n_components = 300  # Number of components for Truncated SVD
processed_chunks = Parallel(n_jobs=-1)(delayed(process_chunk)(chunk, n_components) for chunk in chunks)

# Combine the processed chunks into a single matrix
tfidf_reduced = np.vstack(processed_chunks)

print(f"TF-IDF dimensionality reduced: {tfidf_reduced.shape}")

"""
#from sklearn.metrics import silhouette_score
#import matplotlib.pyplot as plt

"""
"""
"""
# Determine the optimal number of clusters using KMeans with k-means++ initialization
#inertia = []
silhouette_scores = []
range_n_clusters = range(2, 8)

for n_clusters in range_n_clusters:
    # Use KMeans with k-means++ initialization
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10, max_iter=300)
    kmeans.fit(combined_matrix)
 #   inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(combined_matrix, kmeans.labels_))

# Print the silhouette scores for each number of clusters
for n_clusters, score in zip(range_n_clusters, silhouette_scores):
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.4f}")


# kMEANS Visualization
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reduce to 2 dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
reduced_data = tsne.fit_transform(combined_matrix)

# Select the optimal number of clusters based on silhouette scores or domain knowledge
optimal_n_clusters = 4  # Replace this with the best value
kmeans = KMeans(n_clusters=optimal_n_clusters, init='k-means++', random_state=42, n_init=10, max_iter=300)
kmeans.fit(combined_matrix)
labels = kmeans.labels_

# Plot the t-SNE visualization of KMeans clustering
plt.figure(figsize=(10, 8))
unique_labels = set(labels)
colors = [plt.cm.tab10(i / float(len(unique_labels) - 1)) for i in unique_labels]

for label, color in zip(unique_labels, colors):
    label_name = f"Cluster {label}"
    plt.scatter(
        reduced_data[labels == label, 0],
        reduced_data[labels == label, 1],
        c=[color],
        label=label_name,
        s=15,
    )

plt.title(f't-SNE Visualization of KMeans Clustering (k={optimal_n_clusters})')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("kmeans_tsne_visualization.png", dpi=300)
plt.show()

"""




"""
# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertia, marker='o', label='Inertia')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.legend()
plt.tight_layout()
plt.savefig("elbow_curve_kmeans_pp.png", dpi=300)  # Save the plot as a PNG file
plt.show()

# Plot the Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', color='green', label='Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal Clusters')
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_scores_kmeans_pp.png", dpi=300)  # Save the plot as a PNG file
plt.show()

"""
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Step 1: Dimensionality Reduction using Truncated SVD
n_components = 300  # Adjust based on dataset size and variance retained
svd = TruncatedSVD(n_components=n_components, random_state=42)
reduced_matrix = svd.fit_transform(combined_matrix)

print(f"Original dimensions: {combined_matrix.shape}")
print(f"Reduced dimensions: {reduced_matrix.shape}")

# Step 2: Clustering using Mini-Batch K-Means
n_clusters = 2  # Set the number of clusters based on the domain knowledge or Elbow method
batch_size = 500  # Adjust based on memory and dataset size

kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
kmeans.fit(reduced_matrix)

# Step 3: Analyze the clustering results
labels = kmeans.labels_

# Calculate the silhouette score to evaluate clustering
silhouette_avg = silhouette_score(reduced_matrix, labels)
print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

# Step 4: Optional - Save the labels to the dataset

# Assuming you have a dataframe called 'data' that matches combined_matrix
# Add labels to the dataframe for further analysis
data_with_labels = pd.DataFrame(reduced_matrix, columns=[f"component_{i}" for i in range(n_components)])
data_with_labels['Cluster'] = labels

# Save to CSV
data_with_labels.to_csv("clustered_data.csv", index=False)
print("Clustered data saved to 'clustered_data.csv'")

# Step 5: Visualize Clustering Results
# Use the first two components for 2D visualization
x = reduced_matrix[:, 0]
y = reduced_matrix[:, 1]

# Create a scatter plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(x, y, c=labels, cmap='viridis', s=10)
plt.title("Clustering Results (2D Scatter Plot)", fontsize=16)
plt.xlabel("Component 1", fontsize=12)
plt.ylabel("Component 2", fontsize=12)
plt.colorbar(scatter, label="Cluster")
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("clustering_results.png", dpi=300)
plt.show()

print("Clustering results plot saved as 'clustering_results.png'")


# In[ ]:

"""
# Plot the Silhouette Scores
# Plot the Silhouette Scores and save it
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', label='Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Clusters')
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=300)  # Save the plot as a PNG file
plt.show()
"""
"""

"""


"""

"""
"""
# In[47]:

from sklearn.model_selection import train_test_split

# Split the data into train, temp (validation + test), and test sets
text_data = depression['cleaned_body']
labels = depression['label']

# First split the dataset into training and temp (validation + test) sets (50% for temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    text_data, labels, test_size=0.5, random_state=42
)

# Then split the temp set into validation and test sets (50% for validation and 50% for test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
"""

"""
# Baseline model
from sklearn.metrics import accuracy_score, classification_report

# Compute the majority class in the training set
majority_class = y_train.value_counts().idxmax()
print(f"Majority class (baseline prediction): {majority_class}")

# Predict the majority class for all test samples
y_baseline_pred = np.full_like(y_test, majority_class)

# Evaluate the baseline model
print("Baseline Model Accuracy (Majority Class):", accuracy_score(y_test, y_baseline_pred))
print("Baseline Model Classification Report:")
print(classification_report(y_test, y_baseline_pred))
"""

