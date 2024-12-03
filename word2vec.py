import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from joblib import dump

# Path to your dataset
file_path = '/zhome/27/f/203294/ComputationalToolsProject/use_this_one.tsv'

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(file_path, sep='\t')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Balance the dataset
print("Balancing the dataset...")
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]
balanced_df = pd.concat([
    df_0.sample(len(df_1), replace=False, random_state=42),
    df_1
]).sample(frac=1, random_state=42)
print(f"Balanced dataset size: {len(balanced_df)}")

# Tokenize the cleaned_body column of balanced dataset
print("Tokenizing text...")
tokenized_text = balanced_df['cleaned_body'].apply(lambda x: x.split())
print("Text tokenized.")

# Train Word2Vec model on the balanced dataset
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    sentences=tokenized_text,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
word2vec_model.save("word2vec_model.model")
print("Word2Vec model saved.")

# Function to compute sentence embeddings
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Compute embeddings for balanced data
print("Computing embeddings...")
balanced_embeddings = np.array([get_sentence_embedding(sentence, word2vec_model) for sentence in balanced_df['cleaned_body']])
balanced_labels = balanced_df['label'].values

# Save embeddings and labels
print("Saving embeddings and labels...")
np.save("/zhome/27/f/203294/ComputationalToolsProject/balanced_embeddings.npy", balanced_embeddings)
np.save("/zhome/27/f/203294/ComputationalToolsProject/balanced_labels.npy", balanced_labels)
print("Embeddings and labels saved.")

# Dimensionality Reduction for visualization
print("Reducing dimensions for visualization...")
pca = PCA(n_components=2)  # Reduce to 2 dimensions for plotting
reduced_embeddings = pca.fit_transform(balanced_embeddings)

# Cluster reduced embeddings using k-means
print("Clustering reduced embeddings with k-means...")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(reduced_embeddings)
print("Reduced embeddings clustered.")

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(balanced_embeddings, balanced_labels, test_size=0.3, random_state=42)
print("Data split done.")

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(random_state=42, n_jobs=4)
clf.fit(X_train, y_train)
print("Random Forest model trained.")

# Evaluation
print("Evaluating model...")
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Prediction probabilities required for ROC curve
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('/zhome/27/f/203294/ComputationalToolsProject/roc_curve.png')
plt.close()

# Plotting k-means clustering
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
plt.title('K-means Clustering of Sentence Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.savefig('/zhome/27/f/203294/ComputationalToolsProject/kmeans_clustering.png')
plt.show()

# Save the trained model and data
print("Saving models and data...")
dump(clf, "balanced_random_forest_model.joblib")
np.save("X_train_balanced.npy", X_train)
np.save("X_test_balanced.npy", X_test)
np.save("y_train_balanced.npy", y_train)
np.save("y_test_balanced.npy", y_test)
print("Models and data saved.")
