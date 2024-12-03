import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load precomputed embeddings and labels
print("Loading embeddings and labels...")
embeddings = np.load("/zhome/27/f/203294/ComputationalToolsProject/balanced_embeddings.npy")
labels = np.load("/zhome/27/f/203294/ComputationalToolsProject/balanced_labels.npy")
print(f"Loaded embeddings of shape: {embeddings.shape}")
print(f"Loaded labels of shape: {labels.shape}")

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# To store metrics
accuracies = []
conf_matrices = []

print("Starting 10-fold cross-validation...")
fold = 1
for train_index, test_index in kf.split(embeddings, labels):
    print(f"\nFold {fold}")
    
    # Split data into train and test sets
    X_train, X_test = embeddings[train_index], embeddings[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    print(f"Accuracy for Fold {fold}: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    fold += 1

# Calculate average accuracy
average_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy across 10 folds: {average_accuracy:.2f}")

# Save the cross-validation results for future analysis (optional)
np.save("/zhome/27/f/203294/ComputationalToolsProject/cross_validation_accuracies.npy", accuracies)
print("Cross-validation accuracies saved.")
