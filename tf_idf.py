import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
#from cuml.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score
#from sklearn.cluster import KMeans
from cuml.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.sparse import vstack
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tabulate import tabulate

# Load and preprocess dataset
depression = pd.read_csv("../share/use_this_one.tsv.gz", sep="\t", compression="gzip")

# Keep relevant columns
columns_to_keep = ['label', 'cleaned_body', 'unique_id']
depression = depression[columns_to_keep]



# Helper Function to Balance Classes
def balance_classes(X, y):
    """
    Balances the dataset by undersampling the majority class.
    """
    data = pd.DataFrame({'text': X, 'label': y})

    # Separate the classes
    class_0 = data[data['label'] == 0]
    class_1 = data[data['label'] == 1]

    # Undersample the majority class to match the size of the minority class
    if len(class_0) < len(class_1):
        class_1 = class_1.sample(n=len(class_0), random_state=42)
    else:
        class_0 = class_0.sample(n=len(class_1), random_state=42)

    balanced_data = pd.concat([class_0, class_1]).sample(frac=1, random_state=42)
    return balanced_data['text'], balanced_data['label']

print("Random Forest Starts Now")

# Define Chunked TF-IDF Vectorizer
class ChunkedTfidfVectorizer:
    def __init__(self, chunk_size=10000, ngram_range=(1, 3), min_df=0.01):
        self.chunk_size = chunk_size
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=ngram_range, min_df=min_df, stop_words="english"
        )

    def fit(self, text_data):
        self.vectorizer.fit(text_data)

    def transform(self, text_data):
        n_chunks = len(text_data) // self.chunk_size + int(len(text_data) % self.chunk_size > 0)
        tfidf_chunks = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self._process_chunk)(text_data, i) for i in range(n_chunks)
        )
        return vstack(tfidf_chunks)

    def _process_chunk(self, text_data, chunk_index):
        start = chunk_index * self.chunk_size
        end = start + self.chunk_size
        chunk = text_data[start:end]
        return self.vectorizer.transform(chunk)

    def fit_transform(self, text_data):
        self.fit(text_data)
        return self.transform(text_data)

# Define TF-IDF Classifier with Hold-Out Validation
class TfidfClassifierHoldOut:
    """TF-IDF Classifier with Hold-Out Validation for Hyperparameter Tuning."""
    
    def __init__(self, n_estimators_list):
        """
        Initialize with a list of estimators to test.
        """
        self.n_estimators_list = n_estimators_list
        self.results = []

    def hold_out_validate(self, tfidf_matrix_train, labels_train, tfidf_matrix_val, labels_val):
        """
        Perform hold-out validation for each number of estimators.
        """
        print(f"Performing hold-out validation for n_estimators: {self.n_estimators_list}")
        
        for n_estimators in self.n_estimators_list:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(tfidf_matrix_train.toarray(), labels_train)
            predictions = model.predict(tfidf_matrix_val.toarray())
            accuracy = accuracy_score(labels_val, predictions)
            self.results.append((n_estimators, accuracy))
            print(f"n_estimators: {n_estimators}, Accuracy: {accuracy:.4f}")
        
        print(tabulate(self.results, headers=["n_estimators", "Accuracy"], tablefmt="grid"))
    
    def plot_results(self):
        """
        Plot the hold-out validation results.
        """
        n_estimators = [r[0] for r in self.results]
        accuracies = [r[1] for r in self.results]

        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators, accuracies, 'o-', label="Accuracy")
        plt.title("Hold-Out Validation Accuracy vs Number of Trees")
        plt.xlabel("Number of Trees (n_estimators)")
        plt.ylabel("Accuracy")
        plt.xticks(n_estimators)
        plt.grid()
        plt.legend()
        plt.savefig("hold_out_validation_results.png")
        plt.show()

# Define K-means Clustering with Hold-Out Validation
class KMeansHoldOut:
    """K-means Clustering with Hold-Out Validation for Hyperparameter Tuning."""
    
    def __init__(self, k_values):
        """
        Initialize with a list of K values to test.
        """
        self.k_values = k_values
        self.results = []

    def hold_out_validate(self, tfidf_matrix_train, labels_train, tfidf_matrix_val, labels_val):
        """
        Perform hold-out validation for each K value.
        """
        print(f"Performing hold-out validation for K values: {self.k_values}")
        
        for k in self.k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(tfidf_matrix_train)
            cluster_labels = kmeans.predict(tfidf_matrix_val)
            accuracy = accuracy_score(labels_val, cluster_labels)
            self.results.append((k, accuracy))
            print(f"K={k}, Accuracy={accuracy:.4f}")
        
        print(tabulate(self.results, headers=["K", "Accuracy"], tablefmt="grid"))
    
    def plot_results(self):
        """
        Plot the hold-out validation results.
        """
        k_values = [r[0] for r in self.results]
        accuracies = [r[1] for r in self.results]

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, 'o-', label="Accuracy")
        plt.title("Hold-Out Validation Accuracy vs Number of Clusters (K)")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Accuracy")
        plt.xticks(k_values)
        plt.grid()
        plt.legend()
        plt.savefig("hold_out_validation_kmeans_results.png")
        plt.show()

# Main Workflow
def main():
    # Step 1: Split the dataset into training (70%), validation (15%), and test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        depression['cleaned_body'], depression['label'], test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Step 2: Balance the training set
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

    # Step 3: Initialize Chunked TF-IDF Vectorizer
    chunked_vectorizer = ChunkedTfidfVectorizer(chunk_size=10000, ngram_range=(1, 3), min_df=0.01)
    print("Fitting the vectorizer...")
    # Step 4: Fit and transform the training and validation datasets
    tfidf_matrix_train = chunked_vectorizer.fit_transform(X_train_balanced)
    tfidf_matrix_val = chunked_vectorizer.transform(X_val)

    # Step 5: Perform Hold-Out Validation for Random Forest
    n_estimators_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    classifier_hold_out = TfidfClassifierHoldOut(n_estimators_list=n_estimators_list)
    classifier_hold_out.hold_out_validate(tfidf_matrix_train, y_train_balanced, tfidf_matrix_val, y_val)

    # Step 6: Plot Hold-Out Validation Results for Random Forest
    classifier_hold_out.plot_results()

    # Step 7: Perform Hold-Out Validation for K-means
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    kmeans_hold_out = KMeansHoldOut(k_values=k_values)
    kmeans_hold_out.hold_out_validate(tfidf_matrix_train, y_train_balanced, tfidf_matrix_val, y_val)

    # Step 8: Plot Hold-Out Validation Results for K-means
    kmeans_hold_out.plot_results()

    # Step 9: Evaluate the Best Random Forest Model on Test Set
    best_n_estimators = max(classifier_hold_out.results, key=lambda x: x[1])[0]
    print(f"\nBest n_estimators based on hold-out validation: {best_n_estimators}")
    best_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
    best_model.fit(tfidf_matrix_train.toarray(), y_train_balanced)

    # Transform the test set
    tfidf_matrix_test = chunked_vectorizer.transform(X_test).toarray()

    # Test set evaluation
    predictions = best_model.predict(tfidf_matrix_test)
    print("\nTest Set Evaluation:")
    print(classification_report(y_test, predictions))
    test_accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot the ROC Curve for the Test Set
    y_test_proba = best_model.predict_proba(tfidf_matrix_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("roc_curve_best_model.png")
    plt.show()

    print(f"Test ROC AUC Score: {roc_auc:.4f}")

    # Step 10: Evaluate the Best K-means Model on Test Set
    best_k = max(kmeans_hold_out.results, key=lambda x: x[1])[0]
    print(f"\nBest K based on hold-out validation: {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(tfidf_matrix_train)
    cluster_labels_test = kmeans.predict(tfidf_matrix_test)
    test_accuracy_kmeans = accuracy_score(y_test, cluster_labels_test)
    print(f"Test Accuracy for K-means: {test_accuracy_kmeans:.4f}")

    # Perform PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    reduced_data_test = pca.fit_transform(tfidf_matrix_test)

    # Plot clusters in 2D space
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_data_test[:, 0], reduced_data_test[:, 1], c=cluster_labels_test, cmap="viridis", s=10)
    plt.colorbar()
    plt.title(f"K-means Clustering with K={best_k} on Test Set")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.savefig(f"kmeans_clustering_test_k_{best_k}.png")
    plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Function to Perform K-means Clustering and Evaluate for Different K
def kmeans_clustering_analysis(tfidf_matrix, k_values, pca_components=2):
    """
    Perform K-means clustering for a range of K values and evaluate using silhouette scores.
    Also performs dimensionality reduction using PCA for visualization.
    """
    silhouette_scores = []

    # Perform K-means for each K and compute silhouette scores
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"K={k}, Silhouette Score={silhouette_avg:.4f}")
    
    # Identify the optimal K based on maximum silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"\nBest K based on silhouette score: {best_k}")

    print(tabulate(zip(k_values, silhouette_scores), headers=["K", "Silhouette Score"], tablefmt="grid"))

    # Perform PCA for visualization
    pca = PCA(n_components=pca_components, random_state=42)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    # Apply K-means with the best K
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Plot clusters in 2D space
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap="viridis", s=10)
    plt.colorbar()
    plt.title(f"K-means Clustering with K={best_k}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.savefig(f"kmeans_clustering_k_{best_k}.png")
    plt.show()

    return best_k, silhouette_scores

# Run the main function
if __name__ == "__main__":
    main()
