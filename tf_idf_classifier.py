import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.sparse import vstack
import matplotlib.pyplot as plt
#compress the dataset
depression = pd.read_csv("use_this_one.tsv.gz", sep='\t' , compression='gzip')
depression.head()

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(depression["cleaned_body"])
feature_names = tfidf_vectorizer.get_feature_names_out()



columns_to_keep = ['label', 'cleaned_body', 'unique_id']
depression = depression[columns_to_keep]  # Keep only the relevant columns

class ChunkedTfidfVectorizer:
    """TF-IDF Vectorizer for processing large datasets in chunks."""

    def __init__(self, chunk_size=10000, ngram_range=(1, 2), min_df=0.01):
        """
        Args:
            chunk_size (int): Number of rows to process at a time.
            ngram_range (tuple): Range of n-grams for TF-IDF.
            min_df (float): Minimum document frequency for tokens.
        """
        self.chunk_size = chunk_size
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=ngram_range, min_df=min_df, stop_words="english"
        )

    def fit(self, text_data):
        """Fit the vectorizer on the entire dataset."""
        self.vectorizer.fit(text_data)

    def transform(self, text_data):
        """Transform the dataset into TF-IDF features in chunks."""
        tfidf_chunks = []
        for start in range(0, len(text_data), self.chunk_size):
            chunk = text_data[start : start + self.chunk_size]
            tfidf_chunk = self.vectorizer.transform(chunk)
            tfidf_chunks.append(tfidf_chunk)
        return vstack(tfidf_chunks)

    def fit_transform(self, text_data):
        """Fit and transform the dataset."""
        self.fit(text_data)
        return self.transform(text_data)


class TfidfClassifier:
    """TF-IDF Classifier for Text Classification."""

    def __init__(self):
        """Initialize RandomForest classifier."""
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, tfidf_matrix, labels):
        """Train the classifier.

        Args:
            tfidf_matrix (sparse matrix): TF-IDF feature matrix.
            labels (pandas.Series or list): Target labels for training.
        """
        self.classifier.fit(tfidf_matrix, labels)

    def predict(self, tfidf_matrix):
        """Predict labels for new data.

        Args:
            tfidf_matrix (sparse matrix): TF-IDF feature matrix for prediction.

        Returns:
            numpy array: Predicted labels.
        """
        return self.classifier.predict(tfidf_matrix)

    def evaluate(self, tfidf_matrix, labels):
        """Evaluate the model.

        Args:
            tfidf_matrix (sparse matrix): TF-IDF feature matrix for evaluation.
            labels (pandas.Series or list): True labels.

        Returns:
            None: Prints classification report.
        """
        predictions = self.predict(tfidf_matrix)
        print(classification_report(labels, predictions))


# Assuming 'cleaned_body' contains the tokenized, lemmatized, and standardized text, and 'label' is the target column
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

# Verify alignment of shapes
print(f"Train shape: {X_train.shape}, Train labels: {y_train.shape}")
print(f"Validation shape: {X_val.shape}, Validation labels: {y_val.shape}")
print(f"Test shape: {X_test.shape}, Test labels: {y_test.shape}")

# Initialize the Chunked TF-IDF Vectorizer
chunked_vectorizer = ChunkedTfidfVectorizer(chunk_size=10000, ngram_range=(1, 3), min_df=0.01)

# Fit and transform the training data
X_train_tfidf = chunked_vectorizer.fit_transform(X_train)

# Transform the validation and test data
X_val_tfidf = chunked_vectorizer.transform(X_val)
X_test_tfidf = chunked_vectorizer.transform(X_test)

# Initialize the classifier (Random Forest)
classifier = TfidfClassifier()

# Train the model on the TF-IDF-transformed training data
classifier.train(X_train_tfidf, y_train)

# Evaluate the model on the TF-IDF-transformed validation data
print("Evaluation on Validation Set:")
classifier.evaluate(X_val_tfidf, y_val)

# Evaluate the model on the TF-IDF-transformed test data
print("Evaluation on Test Set:")
classifier.evaluate(X_test_tfidf, y_test)



from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

    # Generate predictions and probabilities for ROC and confusion matrix
y_test_pred = classifier.predict(X_test_tfidf)
y_test_proba = classifier.classifier.predict_proba(X_test_tfidf)[:, 1]

    # ROC Curve
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
plt.savefig("roc_curve.png")  # Save the ROC curve
plt.show()

    # Feature Importance
feature_importances = classifier.classifier.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]  # Sort by importance descending

    # Select top 20 most important features
top_n = 20
top_indices = sorted_indices[:top_n]

    # Map indices to feature names
top_features = [chunked_vectorizer.vectorizer.get_feature_names_out()[i] for i in top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), feature_importances[top_indices], align="center")
plt.yticks(range(top_n), top_features)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Important Features")
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top
plt.savefig("feature_importance.png")  # Save the feature importance plot
plt.show()


