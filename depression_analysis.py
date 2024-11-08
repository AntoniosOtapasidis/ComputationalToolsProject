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


X = X.iloc[:999].copy()  # Make a copy to avoid SettingWithCopyWarning
#In case you have not noticed the nltk removed the very common english words from the text!!
