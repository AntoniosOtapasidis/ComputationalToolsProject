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
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Preprocess function without lemmatization
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply preprocessing only to the first 10,000 rows
X.loc[:999, 'cleaned_body'] = X['body'].iloc[:1000].apply(preprocess_text)

X = X.iloc[:999].copy()  # Make a copy to avoid SettingWithCopyWarning
#In case you have not noticed the nltk removed the very common english words from the text!!
