import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from cuml.cluster import KMeans
#from sklearn.cluster import KMeans
from cuml.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances_argmin_min
from tabulate import tabulate

file_path = "data/bert_embeddings.pkl"
tsv_file_path = "../share/use_this_one.tsv.gz"

def balance_classes(X, y):
    """
    Balances the dataset by undersampling the majority class.

    Args:
        X (np.ndarray): Feature data.
        y (pd.Series): Labels corresponding to the features.

    Returns:
        tuple: Balanced X and y.
    """
    # Combine X and y into a DataFrame for easier handling
    data = pd.DataFrame({"text": list(X), "label": y})
    print(f"Original data size: {data.shape}")
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

    print(f"Balanced data size: {balanced_data.shape}")

    # Convert the 'text' column back to a NumPy array
    X_balanced = np.array(balanced_data["text"].tolist())
    y_balanced = balanced_data["label"].values

    return X_balanced, y_balanced

def preprocess_data():
    # Path to the pickle file

    # Load the data from the pickle file
    with open(file_path, "rb") as file:
        bert_embeddings = pickle.load(file)

    # Convert bert_embeddings to a NumPy array
    bert_embeddings = np.array(bert_embeddings).squeeze()

    # Ensure the embeddings are 2D arrays
    bert_embeddings = bert_embeddings.reshape((bert_embeddings.shape[0], -1))

    # Load the TSV file
    data = pd.read_csv(tsv_file_path, sep="\t", compression="gzip")

    print(f"TSV data size: {data.shape}")

    # Ensure the length of bert_embeddings matches the length of data
    assert len(bert_embeddings) == len(
        data
    ), "Length of bert_embeddings and data must match"

    # # Randomly sample 1000 entries
    # sampled_indices = np.random.choice(len(data), 1000, replace=False)
    # bert_embeddings = bert_embeddings[sampled_indices]
    # data = data.iloc[sampled_indices]

    # Create a new DataFrame with bert_embeddings and label column
    new_df = pd.DataFrame(
        {"bert_embeddings": list(bert_embeddings), "label": data["label"]}
    )
    del data

    # Balance the entire dataset
    X_balanced, y_balanced = balance_classes(new_df["bert_embeddings"], new_df["label"])

    # Split the balanced dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    del X_temp, y_temp
    del new_df

    # Reshape the dfs to 2D numpy arrays
    X_train = np.vstack(X_train.tolist()).squeeze()
    X_val = np.vstack([np.array(x[0]).squeeze() for x in X_val])
    X_test = np.vstack([np.array(x[0]).squeeze() for x in X_test])
    print(f"Balanced Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(X_val[0].shape)
    y_train = y_train
    y_val = y_val
    y_test = y_test

    # Verify alignment of shapes
    print(
        f"Balanced Train shape: {X_train.shape}, Train labels: {y_train.shape}"
    )
    print(f"Validation shape: {X_val.shape}, Validation labels: {y_val.shape}")
    print(f"Test shape: {X_test.shape}, Test labels: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def kmeans_clustering(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test):
    best_k = 0
    best_accuracy = 0
    results = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train_balanced)  # No reshaping here
        
        # Predict the closest cluster each sample in X_val belongs to
        val_clusters = kmeans.predict(X_val)  # No reshaping here
        
        # Find the closest cluster for each sample in X_train_balanced
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_train_balanced)  # No reshaping here
        
        # Map the clusters to the labels using training labels
        cluster_labels = {i: y_train_balanced[closest[i]] for i in range(k)}
        
        # Predict the labels for the validation set
        y_val_pred = [cluster_labels[cluster] for cluster in val_clusters]
        
        # Calculate the accuracy
        accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"K={k}, Validation Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

        results.append([k, accuracy])

    print(f"Best K: {best_k}, Best Validation Accuracy: {best_accuracy}")

    # Train the final model with the best K
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_kmeans.fit(X_train_balanced)  # No reshaping here

    # Predict the closest cluster each sample in X_test belongs to
    test_clusters = final_kmeans.predict(X_test)  # No reshaping here

    # Find the closest cluster for each sample in X_train_balanced
    closest, _ = pairwise_distances_argmin_min(final_kmeans.cluster_centers_, X_train_balanced)  # No reshaping here

    # Map the clusters to the labels using training labels
    cluster_labels = {i: y_train_balanced[closest[i]] for i in range(best_k)}

    # Predict the labels for the test set
    y_test_pred = [cluster_labels[cluster] for cluster in test_clusters]

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Test Accuracy: {test_accuracy}")

    results.append(["Test", test_accuracy])

    # Output the results as a table
    print(tabulate(results, headers=["K", "Accuracy"], tablefmt="grid"))
    return final_kmeans
    

def random_forest_classifier(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test):
    best_n_estimators = 0
    best_accuracy = 0
    results = []

    for n in range(100, 1000, 100):
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train_balanced, y_train_balanced)  # No reshaping here
        
        # Predict the labels for the validation set
        y_val_pred = rf.predict(X_val)  # No reshaping here
        
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
    final_rf.fit(X_train_balanced, y_train_balanced)

    # Predict the labels for the test set
    y_test_pred = final_rf.predict(X_test)  # No reshaping here

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Test Accuracy: {test_accuracy}")

    results.append(["Test", test_accuracy])

    # Output the results as a table
    print(tabulate(results, headers=["n_estimators", "Accuracy"], tablefmt="grid"))
    return final_rf

def baseline_classifier(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test):
    # Convert y_train_balanced to a pandas Series
    y_train_balanced = pd.Series(y_train_balanced)
    most_common_class = y_train_balanced.mode()[0]
    
    # Predict the most common class for validation and test sets
    y_val_pred = [most_common_class] * len(y_val)
    y_test_pred = [most_common_class] * len(y_test)
    
    # Calculate the accuracy
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Baseline Validation Accuracy: {val_accuracy}")
    print(f"Baseline Test Accuracy: {test_accuracy}")
    
    return most_common_class

def main():
    X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test = preprocess_data()
    # final_kmeans = kmeans_clustering(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test)
    final_rf = random_forest_classifier(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test)
    most_common_class = baseline_classifier(X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test)
    
    # Save the final models to disk
    with open("final_kmeans.pkl", "wb") as f:
        pickle.dump(final_kmeans, f)

    with open("final_rf.pkl", "wb") as f:
        pickle.dump(final_rf, f)
        
    with open("most_common_class.pkl", "wb") as f:
        pickle.dump(most_common_class, f)

if __name__ == "__main__":
    main()