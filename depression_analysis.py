import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from datasketch import MinHash
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from multiprocessing import Pool
import pickle


DATA_PATH = "data/"
DATA_FILE = "reddit_depression_dataset.csv"
NUM_CORES = 16
SAMPLE_SIZE = 1000


# Define the custom preprocessing function with lemmatization for pool.map
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]

    # Lemmatize each word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back to a single string
    return " ".join(lemmatized_words)


def preprocess(data_path):

    df = pd.read_csv(data_path)
    print(df.head())

    # Count the number of rows where 'label' is NaN
    num_nan_labels = df["label"].isna().sum()
    print(f"Number of rows with NaN 'label': {num_nan_labels}")

    # Drop rows where 'label' column is NaN
    df = df.dropna(subset=["label"])

    # Display descriptive statistics for numerical columns
    # Get an overview of the DataFrame, including counts of non-null entries per column
    print(df.info())

    # Count of NaNs in each column
    print(df.isna().sum())

    # Preprocessing
    # 1) Exclude the empty/NA body

    # Drop rows where 'body' column is NaN
    df = df.dropna(subset=["body"])
    df = df.dropna(subset=["Unnamed: 0"])
    # Drop rows where all columns except 'subreddit' and 'Unnamed: 0' are NaN
    df = df.dropna(
        how="all",
        subset=[col for col in df.columns if col not in ["subreddit", "Unnamed: 0"]],
    )

    # Replace NaN values in 'num_comments' column with 0.0
    df["num_comments"] = df["num_comments"].fillna(0.0)
    print(df.head())

    # Check if there is a 0 in the 'upvotes' column
    has_zero = (df["upvotes"] == 0).any()

    print("Is there a 0 in the upvotes column?", has_zero)

    # I don t thing we need the created_UTC since it is just the number the reddit was created
    df = df.drop(columns=["created_utc"])  # Feature matrix without 'label' column

    # Download NLTK resources if not already downloaded
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")  # Download for better lemmatization support

    # Initialize the WordNet lemmatizer
    global lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Apply preprocessing using multiprocessing
    with Pool(processes=NUM_CORES) as pool:
        cleaned_bodies = pool.map(preprocess_text, df["body"].tolist())

    # Assign the cleaned bodies back to the DataFrame
    df["cleaned_body"] = cleaned_bodies

    # Display the result
    print(df[["body", "cleaned_body"]].head())

    # Convert non-string values in 'cleaned_body' to empty strings
    df["cleaned_body"] = df["cleaned_body"].fillna("").astype(str)

    return df


def word_cloud_function(X, path="figures/wordcloud.png"):
    # Combine all preprocessed text into a single string for word cloud generation
    text_for_wordcloud = " ".join(X["cleaned_body"])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text_for_wordcloud
    )

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(path)


def minhash_vectorize(text, num_perm=500):
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode("utf8"))
    return np.array(minhash.hashvalues)


def minhashing(X):
    # Define MinHash parameters
    num_perm = 500 # Number of hash functions

    # Apply MinHash vectorization using multiprocessing
    with Pool(processes=NUM_CORES) as pool:
        minhash_vectors = pool.map(minhash_vectorize, X["cleaned_body"].tolist())

    # Assign the MinHash vectors back to the DataFrame
    X["minhash_vector"] = minhash_vectors
    return X


def kmeans_clustering(X_train, y_train, X_val, y_val, num_clusters=2):
    # Convert MinHash vectors to 2D array
    X_train_minhash = np.vstack(X_train)
    X_val_minhash = np.vstack(X_val)

    # Define and fit the KMeans model with specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_train_minhash)

    # Assign each training review to a cluster
    train_clusters = kmeans.labels_

    # Map each cluster to the most frequent sentiment in that cluster
    cluster_to_sentiment = {}
    for cluster in range(num_clusters):
        cluster_labels = y_train[train_clusters == cluster]
        most_common_sentiment = Counter(cluster_labels).most_common(1)[0][0]
        cluster_to_sentiment[cluster] = most_common_sentiment

    # Predict the clusters for the validation set
    val_clusters = kmeans.predict(X_val_minhash)

    # Map the clusters to sentiments based on training data cluster assignments
    y_train_pred = [cluster_to_sentiment[cluster] for cluster in train_clusters]
    y_val_pred = [cluster_to_sentiment[cluster] for cluster in val_clusters]

    # Calculate and print accuracy
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    print("Training accuracy:", accuracy_train)
    print("Validation accuracy:", accuracy_val)

    return kmeans, cluster_to_sentiment, accuracy_train, accuracy_val


if __name__ == "__main__":
    # df = preprocess(DATA_PATH + DATA_FILE)
    # with open(DATA_PATH + "df_clean.pkl", "wb") as f:
    #     pickle.dump(df, f)
    # word_cloud_function(df)
    # # Save the cleaned DataFrame to a pickle file
    ### FOR READING THE FILE
    # with open(DATA_PATH + "df_clean.pkl", "rb") as f:
    #     df = pickle.load(f)
        
    # df = minhashing(df)
    # with open(DATA_PATH + "df_minhash.pkl", "wb") as f:
    #     pickle.dump(df, f)
    ### FOR READING THE FILE
    with open(DATA_PATH + "df_minhash.pkl", "rb") as f:
        df = pickle.load(f)
    # Drop rows where 'label' column is NaN
    df = df.dropna(subset=["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["minhash_vector"], df["label"], test_size=0.2, random_state=42
    )

    kmeans, cluster_to_sentiment, accuracy_train, accuracy_val = kmeans_clustering(
        X_train, y_train, X_test, y_test, num_clusters=3
    )
    k_means_output = {
        "kmeans": kmeans,
        "cluster_to_sentiment": cluster_to_sentiment,
        "accuracy_train": accuracy_train,
        "accuracy_val": accuracy_val,
    }
    with open(DATA_PATH + "k_means_output.pkl", "wb") as f:
        pickle.dump(k_means_output, f)

    # If statement just to not run the code right now
    # have to fix the code prfore running
    if False:
        X = X.iloc[:999].copy()
        y = y.iloc[:999].copy()

        # Train-validation-test split with an 80-10-10 ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Verify the split sizes
        print("Training set size:", X_train.shape[0])
        print("Validation set size:", X_val.shape[0])
        print("Test set size:", X_test.shape[0])
        # Initialize CountVectorizer with binary=True for one-hot encoding
        vectorizer = CountVectorizer(binary=True)

        # Fit on training data and transform both training and validation data
        X_train_oh = vectorizer.fit_transform(X_train["cleaned_body"])
        X_val_oh = vectorizer.transform(X_val["cleaned_body"])

        # Perform KMeans clustering
        kmeans, cluster_to_sentiment, accuracy_train, accuracy_val = kmeans_clustering(
            X_train_oh, y_train, X_val_oh
        )

        # Plotting the clusters for training data
        # Reduce the dimensionality for visualization using PCA
        pca = PCA(n_components=2)
        train_vectors_2d = pca.fit_transform(
            X_train_oh.toarray()
        )  # Convert sparse matrix to dense for PCA

        plt.figure(figsize=(10, 7))
        for cluster in range(3):
            cluster_points = train_vectors_2d[X_train["cluster"] == cluster]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f"Cluster {cluster}",
                alpha=0.6,
            )

        plt.title("KMeans Clusters of One-Hot Encoded Vectors (PCA-reduced to 2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()
