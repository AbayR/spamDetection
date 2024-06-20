import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = 'spam.csv'
spam_data = pd.read_csv(file_path)


# Preprocess text data
def preprocess_text_simple(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    common_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                        'itself',
                        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                        'as',
                        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                        'through',
                        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                        'off',
                        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                        'how',
                        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                        'not',
                        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                        'should', 'now'}
    tokens = [word for word in text.split() if word not in common_stopwords]
    return ' '.join(tokens)


spam_data['Message'] = spam_data['Message'].apply(preprocess_text_simple)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=20)  # Adjust max_features to limit the number of features
tfidf_matrix = tfidf_vectorizer.fit_transform(spam_data['Message'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Add the Category column to the DataFrame
tfidf_df['Category'] = spam_data['Category']

# Calculate the mean TF-IDF score for each feature for spam and ham messages
tfidf_means = tfidf_df.groupby('Category').mean().T

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(tfidf_means, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('TF-IDF Feature Scores for Spam and Ham Messages')
plt.xlabel('Category')
plt.ylabel('TF-IDF Feature')
plt.show()
