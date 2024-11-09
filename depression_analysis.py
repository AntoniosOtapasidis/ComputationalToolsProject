import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Read dataset with pandas
#depression = pd.read_csv('reddit_depression_dataset.csv')
#depression = pd.read_csv('./data/reddit_depression_dataset.csv')

#compress the dataset
depression.to_csv("reddit_depression_dataset.tsv.gz", sep='\t', index=False, compression='gzip')


depression = pd.read_csv('reddit_depression_dataset.tsv.gz', sep='\t')
print(depression.head())

# Display descriptive statistics for numerical columns
# Get an overview of the DataFrame, including counts of non-null entries per column
print(depression.info())

# Count of NaNs in each column
print(depression.isna().sum())

#Preprocessing
#1) Exclude the empty/NA body

# Drop rows where 'body' column is NaN
depression = depression.dropna(subset=['body'])
depression = depression.dropna(subset=['Unnamed: 0'])
# Drop rows where all columns except 'subreddit' and 'Unnamed: 0' are NaN
depression = depression.dropna(how='all', subset=[col for col in depression.columns if col not in ['subreddit', 'Unnamed: 0']])

# Replace NaN values in 'num_comments' column with 0.0
depression['num_comments'] = depression['num_comments'].fillna(0.0)
print(depression.head())

# Check if there is a 0 in the 'upvotes' column
has_zero = (depression['upvotes'] == 0).any()

print("Is there a 0 in the upvotes column?", has_zero)

#I don t thing we need the created_UTC since it is just the number the reddit was created
depression= depression.drop(columns=['created_utc'])  # Feature matrix without 'label' column

y = depression['label']
X = depression.drop(columns=['label'])  # Feature matrix without 'label' column


#Data Preprocessing
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Download for better lemmatization support

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the custom preprocessing function with lemmatization
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back to a single string
    return ' '.join(lemmatized_words)

# Apply preprocessing and skip empty results
X['cleaned_body'] = X['body'].iloc[:10000].apply(lambda x: preprocess_text(x) if preprocess_text(x).strip() else x)

# Display the result
print(X[['body', 'cleaned_body']].head())

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Convert non-string values in 'cleaned_body' to empty strings
X['cleaned_body'] = X['cleaned_body'].fillna('').astype(str)

# Combine all preprocessed text into a single string for word cloud generation
text_for_wordcloud = ' '.join(X['cleaned_body'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_wordcloud)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


from sklearn.model_selection import train_test_split

# Make a copy to avoid SettingWithCopyWarning and limit to the first 999 rows for consistency
X = X.iloc[:999].copy()
y = y.iloc[:999].copy()

# Train-validation-test split with an 80-10-10 ratio
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verify the split sizes
print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])

from datasketch import MinHash
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from collections import Counter

# Define MinHash parameters
num_perm = 1000  # Number of hash functions

# Function to apply MinHashing
def minhash_vectorize(text, num_perm=1000):
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    return np.array(minhash.hashvalues)

# Apply MinHash vectorization to the cleaned_body in X_train and X_val
X_train['minhash_vector'] = X_train['cleaned_body'].apply(lambda x: minhash_vectorize(x, num_perm))
X_val['minhash_vector'] = X_val['cleaned_body'].apply(lambda x: minhash_vectorize(x, num_perm))

# Stack the MinHash vectors into a 2D array for KMeans input
minhash_vectors_train = np.vstack(X_train['minhash_vector'].values)
minhash_vectors_val = np.vstack(X_val['minhash_vector'].values)

# Define and fit the KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(minhash_vectors_train)

# Assign each training review to a cluster
X_train['cluster'] = kmeans.labels_

# Map each cluster to the most frequent sentiment in that cluster
cluster_to_sentiment = {}
for cluster in range(3):
    cluster_labels = y_train[X_train['cluster'] == cluster]
    most_common_sentiment = Counter(cluster_labels).most_common(1)[0][0]
    cluster_to_sentiment[cluster] = most_common_sentiment

# Predict the clusters for the validation set
val_clusters = kmeans.predict(minhash_vectors_val)

# Map the clusters to sentiments based on training data cluster assignments
y_val_pred = [cluster_to_sentiment[cluster] for cluster in val_clusters]

# Calculate and print accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation accuracy:", accuracy)

# Print cluster composition to examine the distribution of sentiments in clusters
for cluster in range(3):
    print(f"Cluster {cluster} composition:", Counter(y_train[X_train['cluster'] == cluster]))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce MinHash vectors to 2D for visualization
pca = PCA(n_components=2)
train_vectors_2d = pca.fit_transform(minhash_vectors_train)

# Plot the clusters with different colors
plt.figure(figsize=(10, 7))
for cluster in range(3):
    cluster_points = train_vectors_2d[X_train['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.6)

plt.title('KMeans Clusters of MinHash Vectors (PCA-reduced to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initialize CountVectorizer with binary=True for one-hot encoding
vectorizer = CountVectorizer(binary=True)

# Fit on training data and transform both training and validation data
X_train_oh = vectorizer.fit_transform(X_train['cleaned_body'])
X_val_oh = vectorizer.transform(X_val['cleaned_body'])

# Define and fit the KMeans model with 3 clusters on the one-hot encoded training data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_oh)

# Assign each training review to a cluster
X_train['cluster'] = kmeans.labels_

# Map each cluster to the most frequent sentiment in that cluster
cluster_to_sentiment = {}
for cluster in range(3):
    cluster_labels = y_train[X_train['cluster'] == cluster]
    most_common_sentiment = Counter(cluster_labels).most_common(1)[0][0]
    cluster_to_sentiment[cluster] = most_common_sentiment

# Predict the clusters for the validation set
val_clusters = kmeans.predict(X_val_oh)

# Map the clusters to sentiments based on training data cluster assignments
y_val_pred = [cluster_to_sentiment[cluster] for cluster in val_clusters]

# Calculate and print accuracy
accuracy_train = accuracy_score(y_train, [cluster_to_sentiment[cluster] for cluster in kmeans.labels_])
accuracy_val = accuracy_score(y_val, y_val_pred)
print("Training accuracy:", accuracy_train)
print("Validation accuracy:", accuracy_val)

# Plotting the clusters for training data
# Reduce the dimensionality for visualization using PCA
pca = PCA(n_components=2)
train_vectors_2d = pca.fit_transform(X_train_oh.toarray())  # Convert sparse matrix to dense for PCA

plt.figure(figsize=(10, 7))
for cluster in range(3):
    cluster_points = train_vectors_2d[X_train['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.6)

plt.title('KMeans Clusters of One-Hot Encoded Vectors (PCA-reduced to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
