import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

depression = pd.read_csv('./data/reddit_depression_dataset.csv')

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

# Combine all preprocessed text into a single string for word cloud generation
text_for_wordcloud = ' '.join(X['cleaned_body'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_wordcloud)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Train test split
X = X.iloc[:999].copy()  # Make a copy to avoid SettingWithCopyWarning

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


# Apply minhashing
from datasketch import MinHash, MinHashLSH
import numpy as np

# Define MinHash parameters
num_perm = 100  # Number of hash functions

# Function to apply MinHashing
def minhash_vectorize(text, num_perm=100):
    # Initialize MinHash object
    minhash = MinHash(num_perm=num_perm)
    
    # Apply shingles of size 1 (i.e., individual words)
    for word in text.split():
        minhash.update(word.encode('utf8'))  # Convert each word to bytes for hashing
        
    # Return the hash values as a numpy array
    return np.array(minhash.hashvalues)

# Apply MinHash vectorization to the cleaned_body in X_train
X_train['minhash_vector'] = X_train['cleaned_body'].apply(lambda x: minhash_vectorize(x, num_perm))

# Display a sample of the vectorized result
print(X_train[['cleaned_body', 'minhash_vector']].head())
