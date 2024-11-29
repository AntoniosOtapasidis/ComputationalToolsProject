import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances_argmin_min
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier


file_path = "data/bert_embeddings.pkl"
tsv_file_path = "../share/use_this_one.tsv.gz"


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
    data = pd.DataFrame({"text": X, "label": y})

    # Separate the classes
    class_0 = data[data["label"] == 0]
    class_1 = data[data["label"] == 1]

    # Undersample the majority class to match the size of the minority class
    if len(class_0) < len(class_1):
        class_1 = class_1.sample(n=len(class_0), random_state=42)
    else:
        class_0 = class_0.sample(n=len(class_1), random_state=42)

    # Combine the balanced classes and shuffle
    balanced_data = pd.concat([class_0, class_1]).sample(frac=1, random_state=42)

    return balanced_data["text"], balanced_data["label"]


def preprocess_data():
    # Path to the pickle file

    # Load the data from the pickle file
    with open(file_path, "rb") as file:
        bert_embeddings = pickle.load(file)

    # Convert bert_embeddings to a NumPy array
    bert_embeddings = np.array(bert_embeddings).squeeze()

    # Load the TSV file
    data = pd.read_csv(tsv_file_path, sep="\t", compression="gzip")

    print(f"TSV data size: {data.shape}")

    # Ensure the length of bert_embeddings matches the length of data
    assert len(bert_embeddings) == len(
        data
    ), "Length of bert_embeddings and data must match"

    # Create a new DataFrame with bert_embeddings and label column
    new_df = pd.DataFrame(
        {"bert_embeddings": list(bert_embeddings), "label": data["label"]}
    )

    # Step 2: Split the original dataset into training and temp sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        new_df["bert_emebddings"], new_df["label"], test_size=0.7, random_state=42
    )

    # Step 3: Balance the training set
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

    # Step 4: Split the temp set into validation and test sets (no balancing applied)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Verify alignment of shapes
    print(
        f"Balanced Train shape: {X_train_balanced.shape}, Train labels: {y_train_balanced.value_counts()}"
    )
    print(f"Validation shape: {X_val.shape}, Validation labels: {y_val.value_counts()}")
    print(f"Test shape: {X_test.shape}, Test labels: {y_test.value_counts()}")

    return X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test

def kmeans_clustering(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test):
    best_k = 0
    best_accuracy = 0
    results = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train_balanced.tolist())
        
        # Get the cluster labels
        train_data = pd.DataFrame({"embeddings": X_train_balanced.tolist(), "label": y_train_balanced})
        train_data['cluster_labels'] = kmeans.labels_
        
        # Determine the most common label in each cluster
        cluster_to_label = {}
        for cluster in range(k):
            cluster_labels = train_data[train_data['cluster_labels'] == cluster]['label']
            most_common_label = cluster_labels.mode()[0]
            cluster_to_label[cluster] = most_common_label
        
        # Predict the labels for the validation set
        val_cluster_labels = kmeans.predict(X_val.tolist())
        val_predicted_labels = [cluster_to_label[cluster] for cluster in val_cluster_labels]
        
        # Calculate the accuracy
        accuracy = np.mean(val_predicted_labels == y_val)
        
        print(f"K={k}, Validation Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

        results.append([k, accuracy])

    print(f"Best K: {best_k}, Best Validation Accuracy: {best_accuracy}")

    # Train the final model with the best K
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_kmeans.fit(X_train_balanced.tolist())

    # Predict the closest cluster each sample in X_test belongs to
    test_clusters = final_kmeans.predict(X_test.tolist())

    # Find the closest cluster for each sample in X_test
    closest, _ = pairwise_distances_argmin_min(final_kmeans.cluster_centers_, X_test.tolist())

    # Map the clusters to the labels
    cluster_labels = {i: y_train_balanced[closest[i]] for i in range(best_k)}

    # Predict the labels for the test set
    y_test_pred = [cluster_labels[cluster] for cluster in test_clusters]

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Test Accuracy: {test_accuracy}")

    results.append(["Test", test_accuracy])

    # Output the results as a table
    print(tabulate(results, headers=["K", "Accuracy"], tablefmt="grid"))

def random_forest_classifier(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test):
    best_n_estimators = 0
    best_accuracy = 0
    results = []

    for n in range(10, 110, 10):
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train_balanced.tolist(), y_train_balanced)
        
        # Predict the labels for the validation set
        y_val_pred = rf.predict(X_val.tolist())
        
        # Calculate the accuracy
        accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"n_estimators={n}, Validation Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_estimators = n

        results.append([n, accuracy])

    print(f"Best n_estimators: {best_n_estimators}, Best Validation Accuracy: {best_accuracy}")

    # Train the final model with the best n_estimators
    final_rf = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
    final_rf.fit(X_train_balanced.tolist(), y_train_balanced)

    # Predict the labels for the test set
    y_test_pred = final_rf.predict(X_test.tolist())

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Test Accuracy: {test_accuracy}")

    results.append(["Test", test_accuracy])

    # Output the results as a table
    print(tabulate(results, headers=["n_estimators", "Accuracy"], tablefmt="grid"))

def main():
    X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test = preprocess_data()
    kmeans_clustering(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test)
    random_forest_classifier(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test)
    


if __name__ == "__main__":
    main()
