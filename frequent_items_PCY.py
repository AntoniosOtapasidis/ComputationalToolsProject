import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gc
from joblib import Parallel, delayed
import swifter
import time
import os
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import ast
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import heapq
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA

# This script works with the output of the "data_preprocessing" script
# It was submitted to DTU's HPC using a jobscript.sh file

# Define function to balance dataset based on "label" variable
def balance_classes(X, y): 
    """
    Balances the dataset by undersampling the majority class.

    Args:
        X (pd.Series): Feature data.
        y (pd.Series): Labels corresponding to the features.

    Returns:
        tuple: Balanced X and y.
    """
    # Combine X and y into a DataFrame for easier handling
    data = pd.DataFrame({'text': X, 'label': y})
    
    # Separate the classes
    class_0 = data[data['label'] == 0]
    class_1 = data[data['label'] == 1]
    
    # Undersample the majority class to match the size of the minority class
    if len(class_0) < len(class_1):
        class_1 = class_1.sample(n=len(class_0), random_state=42)
    else:
        class_0 = class_0.sample(n=len(class_1), random_state=42)
    
    # Combine the balanced classes and shuffle
    balanced_data = pd.concat([class_0, class_1]).sample(frac=1, random_state=42)
    
    return balanced_data['text'], balanced_data['label']

# Define the preprocessing function to further clean the "cleaned_body" variable
def preprocess_text(text):
    words = text.split()
    # Remove stopwords, numbers, and meaningless fragments
    filtered_words = [
        word for word in words 
        if word not in ENGLISH_STOP_WORDS       
        and word.isalpha()                      
        and len(word) > 2           
    ]
    return set(filtered_words)

# Check if the preprocessed dataset already exists
preprocessed_file = 'depression_preprocessed.csv'

if os.path.exists(preprocessed_file):
    # If preprocessed data exists, load it directly
    depression = pd.read_csv(preprocessed_file)
    print("Loaded preprocessed dataset.")
    depression['item_sets'] = depression['item_sets'].apply(ast.literal_eval)
else:
    # If not, read the "original" dataset and preprocess it
    # Here, be original we mean the output of the "data_preprocessing" script
    depression = pd.read_csv('use_this_one.tsv.gz', sep='\t')
    print("Loaded original dataset.")
    
    # Convert 'subreddit' to category dtype and 'label' to int8 dtype
    depression['subreddit'] = depression['subreddit'].astype('category')
    depression['label'] = pd.to_numeric(depression['label'], errors='coerce')
    depression['label'] = depression['label'].astype('int8')
    
    # Drop the 'Unnamed: 0' column
    if 'Unnamed: 0' in depression.columns:
        depression = depression.drop(columns=['Unnamed: 0'])

    # Convert 'label' to integer
    depression['label'] = depression['label'].astype(int)

    # Verify that some subreddits are only labeled with 1 and some only with 0
    label_counts = depression.groupby('subreddit')['label'].value_counts().unstack()
    print(label_counts)

    # Split the dataset into 10 smaller chunks to process them in parallel
    num_chunks = 10
    batch_size = len(depression) // num_chunks

    # Loop over each chunk and process it
    item_sets_batches = []
    for i in range(num_chunks):
        start_index = i * batch_size
        if i == num_chunks - 1:  
            end_index = len(depression)
        else:
            end_index = (i + 1) * batch_size

        # Extract the current chunk
        chunk = depression.iloc[start_index:end_index]

        # Apply preprocessing in parallel using swifter
        chunk['item_sets'] = chunk['cleaned_body'].swifter.apply(preprocess_text)
        
        # Append processed chunk
        item_sets_batches.append(chunk['item_sets'])
        
        # Delete chunk to free memory space
        del chunk

    # Combine all the item sets together
    depression['item_sets'] = pd.concat(item_sets_batches).reset_index(drop=True)

    # Save the preprocessed dataset to CSV
    depression.to_csv(preprocessed_file, index=False)
    print("Preprocessed dataset saved.")


# Extract text and labels
X = depression['cleaned_body']
y = depression['label']

# Use the balancing function
balanced_X, balanced_y = balance_classes(X, y)

# Update the 'depression' data frame with the balanced data
depression = pd.DataFrame({'cleaned_body': balanced_X, 'label': balanced_y})
print("Balanced dataset created.")
print("Dimensions of the balanced dataset:", depression.shape)
print("Number of each label in the balanced dataset:")
print(depression['label'].value_counts())

# Recreate the 'item_sets' column using the preprocessing function
depression['item_sets'] = depression['cleaned_body'].swifter.apply(preprocess_text)
print("Recreated 'item_sets' column after balancing the classes.")

# Take a quick look at the processed item sets
print(depression['item_sets'].head())

# Convert the 'item_sets' column into a list of transactions
transactions = depression['item_sets'].tolist()

# Take a look at the first few transactions to understand the data structure
print(transactions[:5])


# Here we implement the frequent items mining using the PCY algorithm

# Define support threshold
min_support_items = 20000  # Adjustable, decided after multiple runs for more meaningful results
min_support_bucket = 10000  # Adjustable, decided after multiple runs for more meaningful results
min_support_pairs = 20000  # Adjustable, decided after multiple runs for more meaningful results
min_support_triplets = 30000  # Adjustable, decided after multiple runs for more meaningful results

# First Pass: Item Count and Bucket Count
# Check if the outputs of the PCY's first pass exist, if yes then load them and don't run the chunk
if os.path.exists('item_count2.pkl') and os.path.exists('bucket_count2.pkl'):
    with open('item_count2.pkl', 'rb') as f:
        item_count = pickle.load(f)

    with open('bucket_count2.pkl', 'rb') as f:
        bucket_count = pickle.load(f)

    # Just a verification that pkl files were loaded properly
    print("Sample of loaded item counts:")
    print(list(item_count.items())[:10])

    print("Sample of loaded bucket counts:")
    print(list(bucket_count.items())[:10])

    print("Loaded item_count and bucket_count from files.")
else:
    # Initialize counters for single items and hash buckets
    item_count = defaultdict(int)
    bucket_count = defaultdict(int)

    # Define a hash function for pairs
    def hash_pair(pair, num_buckets):
        hash_object = hashlib.md5(str(pair).encode()) 
        return int(hash_object.hexdigest(), 16) % num_buckets

    num_buckets = 1000  # Adjustable

    # Split the transactions into smaller chunks for parallel processing
    num_chunks = 16  # Adjustable
    chunk_size = len(transactions) // num_chunks

    # Helper function to process a chunk of transactions with print progress
    def process_chunk(transactions_chunk, chunk_index):
        local_item_count = defaultdict(int)
        local_bucket_count = defaultdict(int)

        # Print to indicate processing start
        print(f"Start processing chunk {chunk_index + 1}/{num_chunks}")
        
        # Iterate over each transaction in the chunk
        for idx, transaction in enumerate(transactions_chunk):
            # Count individual items
            for item in transaction:
                local_item_count[item] += 1

            # Hash pairs into buckets
            transaction_list = list(transaction)
            for i in range(len(transaction_list)):
                for j in range(i + 1, len(transaction_list)):
                    pair = frozenset([transaction_list[i], transaction_list[j]])
                    bucket = hash_pair(pair, num_buckets)
                    local_bucket_count[bucket] += 1

            # Provide progress within the chunk
            if idx % 50000 == 0:  # Print progress every 50,000 transactions
                print(f"Chunk {chunk_index + 1}: Processed {idx} transactions out of {len(transactions_chunk)}")

        return local_item_count, local_bucket_count

    # Start timing the first pass
    start_time = time.time()

    # Use Joblib to process the chunks in parallel while printing progress
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_chunk)(transactions[i*chunk_size:(i+1)*chunk_size], i)
        for i in range(num_chunks)
    )

    # Initialize final counters for single items and hash buckets
    for local_item_count, local_bucket_count in results:
        # Combine item counts
        for item, count in local_item_count.items():
            item_count[item] += count
        
        # Combine bucket counts
        for bucket, count in local_bucket_count.items():
            bucket_count[bucket] += count

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"First pass completed in {total_time:.2f} seconds")

    # Save item count and bucket count to files
    with open('item_count2.pkl', 'wb') as f:
        pickle.dump(item_count, f)

    with open('bucket_count2.pkl', 'wb') as f:
        pickle.dump(bucket_count, f)

# Apply threshold to identify frequent items and buckets after first pass
frequent_items = {item for item, count in item_count.items() if count >= min_support_items}
frequent_buckets = {bucket for bucket, count in bucket_count.items() if count >= min_support_bucket}
print(f"Number of frequent buckets with threshold at {min_support_bucket}: {len(frequent_buckets)}")
print(f"Number of frequent items with threshold at {min_support_items}: {len(frequent_items)}")

print("Finished with PCY first pass!")


# Second Pass: Candidate Pairs Count
# Check if the output of the PCY's second pass exists, if yes then load it and don't run the chunk
if os.path.exists('candidate_pairs_count2.pkl'):
    with open('candidate_pairs_count2.pkl', 'rb') as f:
        candidate_pairs_count = pickle.load(f)

    # Just a verification that pkl files were loaded properly
    print("Sample of loaded item counts:")
    print(list(candidate_pairs_count.items())[:10])

    print("Loaded candidate_pairs_count from file.")
else:
    # Convert transactions to a simple list type if necessary
    transactions = [list(transaction) for transaction in transactions]

    # Split the transactions into smaller chunks for parallel processing
    num_chunks = 16  # Adjustable
    chunk_size = len(transactions) // num_chunks

    # Helper function to process a chunk of transactions for the second pass
    def process_second_pass_chunk(transactions_chunk, chunk_index):
        local_candidate_pairs_count = defaultdict(int)

        # Print to indicate the start of chunk processing
        print(f"Start processing chunk {chunk_index + 1}/{num_chunks} for second pass")

        # Iterate over each transaction in the chunk
        for idx, transaction in enumerate(transactions_chunk):
            # Consider only frequent items in the transactions
            frequent_transaction_items = [item for item in transaction if item in frequent_items]

            # Count candidate pairs that fall into frequent buckets
            for i in range(len(frequent_transaction_items)):
                for j in range(i + 1, len(frequent_transaction_items)):
                    pair = frozenset([frequent_transaction_items[i], frequent_transaction_items[j]])
                    bucket = hash_pair(pair, num_buckets)

                    if bucket in frequent_buckets:
                        local_candidate_pairs_count[pair] += 1

            # Provide progress within the chunk
            if idx % 50000 == 0:  # Print progress every 50,000 transactions
                print(f"Chunk {chunk_index + 1}: Processed {idx} transactions out of {len(transactions_chunk)}")

        return local_candidate_pairs_count

    # Start timing the second pass
    start_time = time.time()

    # Use Joblib to process the chunks in parallel while printing progress
    try:
        second_pass_results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(process_second_pass_chunk)(transactions[i*chunk_size:(i+1)*chunk_size], i)
            for i in range(num_chunks)
        )
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")

    # Initialize a dictionary to store counts of candidate pairs
    candidate_pairs_count = defaultdict(int)

    # Combine the results from all chunks with tqdm for progress tracking
    for local_candidate_pairs_count in tqdm(second_pass_results, desc="Combining Second Pass Results", unit="chunk"):
        # Combine candidate pairs counts
        for pair, count in local_candidate_pairs_count.items():
            candidate_pairs_count[pair] += count

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Second pass completed in {total_time:.2f} seconds")

    # Save candidate pairs count to file
    with open('candidate_pairs_count2.pkl', 'wb') as f:
        pickle.dump(candidate_pairs_count, f)

# Apply threshold to identify frequent pairs
frequent_pairs = {pair for pair, count in candidate_pairs_count.items() if count >= min_support_pairs}
print(f"Number of frequent pairs with threshold at {min_support_pairs}: {len(frequent_pairs)}")

print('finished with PCY second pass!')


# Third Pass: Candidate Triplets Count
# Check if the output of the PCY's third pass exists, if yes then load it and don't run the chunk
if os.path.exists('candidate_triplets_count2.pkl'):
    with open('candidate_triplets_count2.pkl', 'rb') as f:
        candidate_triplets_count = pickle.load(f)

    # Just a verification that pkl files were loaded properly
    print("Sample of loaded triplet counts:")
    print(list(candidate_triplets_count.items())[:10])

    print("Loaded candidate_triplets_count from file.")
else:
    # Split the transactions into smaller chunks for parallel processing
    num_chunks = 16  # Adjustable
    chunk_size = len(transactions) // num_chunks

    # Helper function to process a chunk of transactions for the third pass
    def process_third_pass_chunk(transactions_chunk, chunk_index):
        local_candidate_triplets_count = defaultdict(int)

        # Print to indicate the start of chunk processing
        print(f"Start processing chunk {chunk_index + 1}/{num_chunks} for third pass")

        # Iterate over each transaction in the chunk
        for idx, transaction in enumerate(transactions_chunk):
            # Consider only frequent items in the transactions
            frequent_transaction_items = [item for item in transaction if item in frequent_items]

            # Generate candidate triplets from the frequent items
            triplets = combinations(frequent_transaction_items, 3)

            # Count candidate triplets that involve frequent pairs
            for triplet in triplets:
                triplet_pairs = [frozenset([triplet[0], triplet[1]]),
                                 frozenset([triplet[0], triplet[2]]),
                                 frozenset([triplet[1], triplet[2]])]

                # Only consider triplets where all pairs are frequent
                if all(pair in frequent_pairs for pair in triplet_pairs):
                    sorted_triplet = frozenset(triplet)
                    local_candidate_triplets_count[sorted_triplet] += 1

            # Provide progress within the chunk
            if idx % 50000 == 0:  # Print progress every 50,000 transactions
                print(f"Chunk {chunk_index + 1}: Processed {idx} transactions out of {len(transactions_chunk)}")

        return local_candidate_triplets_count

    # Start timing the third pass
    start_time = time.time()

    # Use Joblib to process the chunks in parallel while printing progress
    try:
        third_pass_results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(process_third_pass_chunk)(transactions[i * chunk_size:(i + 1) * chunk_size], i)
            for i in range(num_chunks)
        )
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")

    # Initialize a dictionary to store counts of candidate triplets
    candidate_triplets_count = defaultdict(int)

    # Combine the results from all chunks with tqdm for progress tracking
    for local_candidate_triplets_count in tqdm(third_pass_results, desc="Combining Third Pass Results", unit="chunk"):
        # Combine candidate triplets counts
        for triplet, count in local_candidate_triplets_count.items():
            candidate_triplets_count[triplet] += count

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Third pass completed in {total_time:.2f} seconds")

    # Save candidate triplets count to file
    with open('candidate_triplets_count2.pkl', 'wb') as f:
        pickle.dump(candidate_triplets_count, f)


# Apply threshold to identify frequent triplets
frequent_triplets = {triplet for triplet, count in candidate_triplets_count.items() if count >= min_support_triplets}
print(f"Number of frequent triplets with threshold at {min_support_triplets}: {len(frequent_triplets)}")

print('Finished with PCY Third Pass!')


### EDA - WORD TRIPLET DISTRIBUTION ANALYSIS ###
# Here we try to identify triplets that are mostly present in text labeled as depressive (1) and less present in text
# labeled as non-depressive (0). We can create a barplot to visualize this

# Check if the .png file already exists
if not os.path.exists("test_image_updated_triplets_optimized.png"):

    # Step 1: Split the dataset
    num_chunks = 16  # Adjustable
    depressive_posts = depression[depression['label'] == 1]
    non_depressive_posts = depression[depression['label'] == 0]
    depressive_chunks = np.array_split(depressive_posts, num_chunks)
    non_depressive_chunks = np.array_split(non_depressive_posts, num_chunks)

    # Initialize dictionaries to store counts for each frequent triplet
    depressive_triplet_counts = Counter()
    non_depressive_triplet_counts = Counter()

    # Function to count frequent triplets in a dataset
    def count_frequent_triplets_chunk(posts_chunk, frequent_triplets):
        local_triplet_counts = Counter()
        for item_set in posts_chunk['item_sets']:
            # Generate all triplets from the current item set
            triplets = combinations(item_set, 3)
            for triplet in triplets:
                sorted_triplet = frozenset(sorted(triplet))
                if sorted_triplet in frequent_triplets:
                    local_triplet_counts[sorted_triplet] += 1
        return local_triplet_counts

    # Step 2: Count frequent triplets in depressive posts
    depressive_triplet_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(count_frequent_triplets_chunk)(chunk, frequent_triplets) for chunk in depressive_chunks
    )

    # Combine the counts from all chunks for depressive posts
    for local_count in depressive_triplet_results:
        depressive_triplet_counts.update(local_count)

    # Step 3: Count frequent triplets in non-depressive posts
    non_depressive_triplet_results = Parallel(n_jobs=-1, backend='loky')(
        delayed(count_frequent_triplets_chunk)(chunk, frequent_triplets) for chunk in non_depressive_chunks
    )

    # Combine the counts from all chunks for non-depressive posts
    for local_count in non_depressive_triplet_results:
        non_depressive_triplet_counts.update(local_count)

    print("Finished with counting frequent triplets in the posts.")

    # Step 4: Convert frozenset triplets to strings in a readable format
    def triplet_to_string(triplet):
        return " - ".join(sorted(triplet)) 

    # Step 5: Merge both counters to combine similar triplets
    final_triplet_counts = defaultdict(lambda: {'Depressive_Count': 0, 'Non_Depressive_Count': 0})

    # Update with depressive counts
    for triplet, count in depressive_triplet_counts.items():
        triplet_str = triplet_to_string(triplet)
        final_triplet_counts[triplet_str]['Depressive_Count'] += count

    # Update with non-depressive counts
    for triplet, count in non_depressive_triplet_counts.items():
        triplet_str = triplet_to_string(triplet)
        final_triplet_counts[triplet_str]['Non_Depressive_Count'] += count

    # Step 6: Convert the final counts to a data frame for easier visualization
    df_triplet_counts = pd.DataFrame.from_dict(final_triplet_counts, orient='index').reset_index().rename(columns={'index': 'Triplet'})

    # Step 7: Sort and get top triplets
    top_10_triplets = heapq.nlargest(10, df_triplet_counts.to_dict('records'), key=lambda x: x['Depressive_Count'] - x['Non_Depressive_Count'])

    # Convert top 10 triplets back to a data frame for plotting
    top_10_pairs_df = pd.DataFrame(top_10_triplets)

    # Step 8: Plot the top 10 frequent triplets more prevalent in depressive posts
    plt.figure(figsize=(12, 6))
    plt.barh(top_10_pairs_df['Triplet'], top_10_pairs_df['Depressive_Count'], color='red', alpha=0.7, label='Depressive')
    plt.barh(top_10_pairs_df['Triplet'], top_10_pairs_df['Non_Depressive_Count'], color='blue', alpha=0.7, label='Non-Depressive')
    plt.xlabel('Count')
    plt.title('Top 10 Word Triplets in Depressive vs Non-Depressive Posts')
    plt.legend()
    plt.savefig("test_image_updated_triplets_optimized.png")

    print("Finished with updated plots")
else:
    print("Plot already exists, skipping the analysis.")



### CLASSIFICATION PART ###
# Here we make use of the information acquired by the frequent triplets to try and predict the label of a post by
# the existence or not of a frequent triplet in it

# Split the original dataset into training and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(
    depression['item_sets'], depression['label'], test_size=0.3, random_state=42)

# Balance the training set
X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

# Split the temp set into validation and test sets (no balancing applied)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Verify alignment of shapes
print(f"Balanced Train shape: {X_train_balanced.shape}, Train labels: {y_train_balanced.value_counts()}")
print(f"Validation shape: {X_val.shape}, Validation labels: {y_val.value_counts()}")
print(f"Test shape: {X_test.shape}, Test labels: {y_test.value_counts()}")

# Convert item_sets to a binary matrix representation for model training using frequent triplets
# Creating a list of all frequent triplets
frequent_triplets_list = list(frequent_triplets) 
triplet_to_index = {triplet: idx for idx, triplet in enumerate(frequent_triplets_list)}

# Helper function to convert item_sets to a binary vector based on frequent triplets
def convert_to_triplet_vector(item_set, triplet_to_index):
    vector = [0] * len(triplet_to_index)
    for triplet in triplet_to_index:
        if triplet.issubset(item_set): 
            vector[triplet_to_index[triplet]] = 1
    return vector

# Apply the conversion to the dataset using frequent triplets as features
X_train_matrix = [convert_to_triplet_vector(item_set, triplet_to_index) for item_set in X_train_balanced]
X_val_matrix = [convert_to_triplet_vector(item_set, triplet_to_index) for item_set in X_val]
X_test_matrix = [convert_to_triplet_vector(item_set, triplet_to_index) for item_set in X_test]

print("Preprocessing for modeling complete.")


### K-means ###
# Hyperparameter tuning for K-means clustering
def evaluate_k_means(k, X_train_matrix, X_val_matrix, y_val):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_matrix)
    
    # Assign clusters to classes based on majority voting
    cluster_labels = kmeans.predict(X_train_matrix)
    label_mapping = {}
    for cluster in set(cluster_labels):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        majority_label = y_train_balanced.iloc[indices].mode()[0]
        label_mapping[cluster] = majority_label
    
    # Predict labels for validation set
    y_val_pred = [label_mapping[cluster] for cluster in kmeans.predict(X_val_matrix)]
    
    # Calculate accuracy for validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    return k, accuracy

possible_ks = range(2, 11)  # Testing different values for k from 2 to 10
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(evaluate_k_means)(k, X_train_matrix, X_val_matrix, y_val) for k in possible_ks
)

# Print accuracy results for all tested values of k
for k, accuracy in results:
    print(f"k={k}, Validation Accuracy: {accuracy}")

# Find the best k based on validation accuracy
best_k, best_accuracy = max(results, key=lambda x: x[1])
print(f"Best k determined: {best_k} with Validation Accuracy: {best_accuracy}")

# Train final K-means model with best k
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_train_matrix)

# Update cluster_labels for subsequent usage in the plots (make it available globally)
cluster_labels = kmeans.predict(X_train_matrix)

# Predict clusters for the test set
test_cluster_labels = kmeans.predict(X_test_matrix)

# Assign clusters to classes based on majority voting
label_mapping = {}
for cluster in set(test_cluster_labels):
    indices = [i for i, label in enumerate(test_cluster_labels) if label == cluster]
    majority_label = pd.Series([y_test.iloc[i] for i in indices]).mode()[0]
    label_mapping[cluster] = majority_label

# Map predicted clusters to class labels
y_test_pred = [label_mapping[cluster] for cluster in test_cluster_labels]

# Evaluate K-means model on the test set
print("\nK-Means Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
print(f"Precision: {precision_score(y_test, y_test_pred)}")
print(f"Recall: {recall_score(y_test, y_test_pred)}")

# Create a baseline model to compare our model's performance against - assign everything ot the majority class
y_baseline_pred = [y_train_balanced.mode()[0]] * len(y_test)

print("\nBaseline Model Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_baseline_pred)}")
print(f"Precision: {precision_score(y_test, y_baseline_pred)}")
print(f"Recall: {recall_score(y_test, y_baseline_pred)}")


# Create some plots for the k-means clustering, see if we get anything of value
# Elbow plot for K-means clustering
inertia_values = []
for k in possible_ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_matrix)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(possible_ks, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Plot for Determining Optimal k')
plt.grid(True)
plt.savefig('kmeans_elbow_plot_2.png')
print('Elbow plot saved as kmeans_elbow_plot_2.png')

# PCA for 2D visualization of clusters
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_matrix)

plt.figure(figsize=(10, 7))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'K-Means Clustering with k={best_k}')
plt.legend()
plt.savefig('kmeans_pca_clusters_plot_2.png')
print('PCA clusters plot saved as kmeans_pca_clusters_plot_2.png')

# Distribution of depressive and non-eepressive posts in clusters
cluster_distribution = pd.DataFrame({'Cluster': cluster_labels, 'Label': y_train_balanced})
cluster_counts = cluster_distribution.groupby(['Cluster', 'Label']).size().unstack(fill_value=0)

cluster_counts.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.xlabel('Cluster')
plt.ylabel('Number of Posts')
plt.title(f'Distribution of Depressive vs. Non-Depressive Posts Across {best_k} Clusters')
plt.legend(['Non-Depressive', 'Depressive'])
plt.savefig('kmeans_cluster_distribution_plot_2.png')
print('Cluster distribution plot saved as kmeans_cluster_distribution_plot_2.png')


### RandomForest ###
# Define hyperparameter grid - it's not very "rich" since due to the big data that we deal with, it needed a 
# very long time to run
param_grid = {
    'n_estimators': [20, 50], 
    'max_depth': [None, 5, 10],  
    'min_samples_split': [10], 
}

# Create a Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use RandomizedSearchCV to find the best hyperparameters
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,  # 3-fold cross-validation on the training set
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

# Fit random search on the training set
random_search.fit(X_train_matrix, y_train_balanced)

# Get the best parameters from the random search
best_params = random_search.best_params_
print(f"Best Hyperparameters from Randomized Search: {best_params}")

# Train final Random Forest on combined training and validation Set
# Combine training and validation sets for final training
X_combined_train_val = X_train_matrix + X_val_matrix
y_combined_train_val = list(y_train_balanced) + list(y_val)

# Train the Random Forest with the best hyperparameters found
final_random_forest = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
final_random_forest.fit(X_combined_train_val, y_combined_train_val)

# Evaluate the Random Forest on the test set
# Predict the labels for the test set
y_test_pred = final_random_forest.predict(X_test_matrix)

# Calculate accuracy, precision, and recall on the test set
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("\nRandom Forest Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")