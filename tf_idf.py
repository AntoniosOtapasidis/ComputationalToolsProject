import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.sparse import vstack
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Load and preprocess dataset
depression = pd.read_csv("use_this_one.tsv.gz", sep="\t", compression="gzip")

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

# Define TF-IDF Classifier with Cross-Validation
class TfidfClassifierCV:
    """TF-IDF Classifier with Cross-Validation for Hyperparameter Tuning."""
    
    def __init__(self, n_estimators_list):
        """
        Initialize with a list of estimators to test.
        """
        self.n_estimators_list = n_estimators_list
        self.results = []

    def cross_validate(self, tfidf_matrix, labels, k=10):
        """
        Perform k-fold cross-validation for each number of estimators.
        """
        print(f"Performing {k}-fold cross-validation for n_estimators: {self.n_estimators_list}")
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for n_estimators in self.n_estimators_list:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            scores = cross_val_score(model, tfidf_matrix, labels, cv=kf, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            self.results.append((n_estimators, mean_score, std_score))
            print(f"n_estimators: {n_estimators}, Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    
    def plot_results(self):
        """
        Plot the cross-validation results.
        """
        n_estimators = [r[0] for r in self.results]
        mean_scores = [r[1] for r in self.results]
        std_scores = [r[2] for r in self.results]

        plt.figure(figsize=(10, 6))
        plt.errorbar(n_estimators, mean_scores, yerr=std_scores, fmt='o-', capsize=5, label="Accuracy ± Std Dev")
        plt.title("Cross-Validation Accuracy vs Number of Trees")
        plt.xlabel("Number of Trees (n_estimators)")
        plt.ylabel("Accuracy")
        plt.xticks(n_estimators)
        plt.grid()
        plt.legend()
        plt.savefig("cross_validation_results.png")
        plt.show()

# Main Workflow
def main():
    # Step 1: Split the dataset into training (70%) and test (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        depression['cleaned_body'], depression['label'], test_size=0.30, random_state=42
    )

    # Step 2: Balance the training set
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

    # Step 3: Combine training and validation for cross-validation
    combined_data = pd.concat([X_train_balanced, X_temp])
    combined_labels = pd.concat([y_train_balanced, y_temp])

    # Step 4: Initialize Chunked TF-IDF Vectorizer
    chunked_vectorizer = ChunkedTfidfVectorizer(chunk_size=10000, ngram_range=(1, 3), min_df=0.01)
    
    # Step 5: Fit and transform the combined dataset
    tfidf_matrix = chunked_vectorizer.fit_transform(combined_data)

    # Step 6: Perform Cross-Validation
    n_estimators_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    classifier_cv = TfidfClassifierCV(n_estimators_list=n_estimators_list)
    classifier_cv.cross_validate(tfidf_matrix, combined_labels, k=10)

    # Step 7: Plot Cross-Validation Results
    classifier_cv.plot_results()

    # Step 8: Evaluate the Best Model on Test Set
    best_n_estimators = max(classifier_cv.results, key=lambda x: x[1])[0]
    print(f"\nBest n_estimators based on cross-validation: {best_n_estimators}")
    best_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42, n_jobs=-1)
    best_model.fit(tfidf_matrix, combined_labels)

    # Transform the test set
    X_test_tfidf = chunked_vectorizer.transform(X_temp)

    # Test set evaluation
    predictions = best_model.predict(X_test_tfidf)
    print("\nTest Set Evaluation:")
    print(classification_report(y_temp, predictions))
    test_accuracy = accuracy_score(y_temp, predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot the ROC Curve for the Test Set
    y_test_proba = best_model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_temp, y_test_proba)
    roc_auc = roc_auc_score(y_temp, y_test_proba)

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

# Main Workflow
def main():
    # Step 1: Split the dataset into training (70%) and test (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        depression['cleaned_body'], depression['label'], test_size=0.30, random_state=42
    )

    # Step 2: Balance the training set
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

    # Step 3: Combine training and validation for cross-validation
    combined_data = pd.concat([X_train_balanced, X_temp])
    combined_labels = pd.concat([y_train_balanced, y_temp])

    # Step 4: Initialize Chunked TF-IDF Vectorizer
    chunked_vectorizer = ChunkedTfidfVectorizer(chunk_size=10000, ngram_range=(1, 3), min_df=0.01)
    print("Fitting the vectorizer...")
    # Step 5: Fit and transform the combined dataset
    tfidf_matrix = chunked_vectorizer.fit_transform(combined_data)

    # Step 6: Perform Cross-Validation for Random Forest
    n_estimators_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    classifier_cv = TfidfClassifierCV(n_estimators_list=n_estimators_list)
    classifier_cv.cross_validate(tfidf_matrix, combined_labels, k=10)

    # Step 7: Plot Cross-Validation Results
    classifier_cv.plot_results()

    # Step 8: Perform K-means Clustering
    k_values = range(2, 11)  # Test K from 2 to 10
    best_k, silhouette_scores = kmeans_clustering_analysis(tfidf_matrix, k_values)

    # Step 9: Evaluate the Best Model on Test Set
    best_n_estimators = max(classifier_cv.results, key=lambda x: x[1])[0]
    print(f"\nBest n_estimators based on cross-validation: {best_n_estimators}")
    best_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42, n_jobs=-1)
    best_model.fit(tfidf_matrix, combined_labels)

    # Transform the test set
    X_test_tfidf = chunked_vectorizer.transform(X_temp)

    # Test set evaluation
    predictions = best_model.predict(X_test_tfidf)
    print("\nTest Set Evaluation:")
    print(classification_report(y_temp, predictions))
    test_accuracy = accuracy_score(y_temp, predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot the ROC Curve for the Test Set
    y_test_proba = best_model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_temp, y_test_proba)
    roc_auc = roc_auc_score(y_temp, y_test_proba)

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

# Run the main function
if __name__ == "__main__":
    main()
