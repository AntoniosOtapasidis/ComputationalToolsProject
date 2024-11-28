import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

# Path to the pickle file
file_path = 'data/bert_embeddings.pkl'

# Load the data from the pickle file
with open(file_path, 'rb') as file:
    bert_embeddings = pickle.load(file)

# Convert bert_embeddings to a NumPy array
bert_embeddings = np.array(bert_embeddings).squeeze()

# Load the TSV file
tsv_file_path = '../share/use_this_one.tsv.gz'
data = pd.read_csv(tsv_file_path, sep='\t', compression='gzip')

print(f"TSV data size: {data.shape}")

# Ensure the length of bert_embeddings matches the length of data
assert len(bert_embeddings) == len(data), "Length of bert_embeddings and data must match"

# Create a new DataFrame with bert_embeddings and label column
new_df = pd.DataFrame({
    'bert_embeddings': list(bert_embeddings),
    'label': data['label']
})

# Delete the loaded data to free up memory
del data

print(f"New DataFrame size: {new_df.shape}")

# Set the seed
seed = 42

# Split the new DataFrame into train, test, and validation sets
train_data, temp_data = train_test_split(new_df, test_size=0.5, random_state=seed)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)  # 0.5 x 0.5 = 0.25

print(f"Train data size: {train_data.shape}")
print(f"Validation data size: {val_data.shape}")
print(f"Test data size: {test_data.shape}")

# Extract the embeddings and labels for each set
train_embeddings = np.vstack(train_data['bert_embeddings']).squeeze()
print(f"Train embeddings shape: {train_embeddings.shape}")
val_embeddings = np.vstack(val_data['bert_embeddings']).squeeze()
test_embeddings = np.vstack(test_data['bert_embeddings']).squeeze()

train_labels = train_data['label'].values
val_labels = val_data['label'].values
test_labels = test_data['label'].values

# Perform KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=seed)
kmeans.fit(train_embeddings)

# Get the cluster labels
train_data['cluster_labels'] = kmeans.labels_
# Determine the most common label in each cluster
cluster_to_label = {}
for cluster in range(2):
    cluster_labels = train_data[train_data['cluster_labels'] == cluster]['label']
    most_common_label = cluster_labels.mode()[0]
    cluster_to_label[cluster] = most_common_label

# Predict the labels for the validation set
val_cluster_labels = kmeans.predict(val_embeddings)
val_predicted_labels = [cluster_to_label[cluster] for cluster in val_cluster_labels]

# Calculate the accuracy score
accuracy = np.mean(val_predicted_labels == val_labels)
print(f"Validation accuracy: {accuracy}")
# train accuracy
train_cluster_labels = kmeans.predict(train_embeddings)
train_predicted_labels = [cluster_to_label[cluster] for cluster in train_cluster_labels]
train_accuracy = np.mean(train_predicted_labels == train_labels)
print(f"Train accuracy: {train_accuracy}")

# Baseline model: predict the most common label in the training data
most_common_label = train_data['label'].mode()[0]
baseline_val_predicted_labels = [most_common_label] * len(val_labels)
baseline_val_accuracy = np.mean(baseline_val_predicted_labels == val_labels)
print(f"Baseline validation accuracy: {baseline_val_accuracy}")

# Decision Tree model
decision_tree = DecisionTreeClassifier(max_depth=10, random_state=seed)
decision_tree.fit(train_embeddings, train_labels)

# Training accuracy for Decision Tree
train_predicted_labels_dt = decision_tree.predict(train_embeddings)
train_accuracy_dt = np.mean(train_predicted_labels_dt == train_labels)
print(f"Decision Tree Train accuracy: {train_accuracy_dt}")

# Validation accuracy for Decision Tree
val_predicted_labels_dt = decision_tree.predict(val_embeddings)
val_accuracy_dt = np.mean(val_predicted_labels_dt == val_labels)
print(f"Decision Tree Validation accuracy: {val_accuracy_dt}")

# AdaBoost model
adaboost = AdaBoostClassifier(n_estimators=25, random_state=seed)
adaboost.fit(train_embeddings, train_labels)

# Training accuracy for AdaBoost
train_predicted_labels_ab = adaboost.predict(train_embeddings)
train_accuracy_ab = np.mean(train_predicted_labels_ab == train_labels)
print(f"AdaBoost Train accuracy: {train_accuracy_ab}")

# Validation accuracy for AdaBoost
val_predicted_labels_ab = adaboost.predict(val_embeddings)
val_accuracy_ab = np.mean(val_predicted_labels_ab == val_labels)
print(f"AdaBoost Validation accuracy: {val_accuracy_ab}")

cluster_dict = {
    'kmeans': kmeans,
    'cluster_to_label': cluster_to_label,
    'val_accuracy': accuracy,
    'train_accuracy': train_accuracy,
    'baseline_val_accuracy': baseline_val_accuracy,
    'decision_tree': decision_tree,
    'decision_tree_train_accuracy': train_accuracy_dt,
    'decision_tree_val_accuracy': val_accuracy_dt,
    'adaboost': adaboost,
    'adaboost_train_accuracy': train_accuracy_ab,
    'adaboost_val_accuracy': val_accuracy_ab,
}
# Save the cluster_dict to a pickle file
output_file_path = 'data/bert_cluster_results_v2.pkl'
with open(output_file_path, 'wb') as output_file:
    pickle.dump(cluster_dict, output_file)

print(f"Cluster results saved to {output_file_path}")