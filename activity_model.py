import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
file_path = 'data/activities.csv'  
activities_df = pd.read_csv(file_path)
def preprocess_text(text):
    # Tokenization (convert from string to list)
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization (convert words to their root form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove punctuation and numbers
    words = [word for word in words if word.isalpha()]

    # Re-join words to form the cleaned text
    cleaned_text = ' '.join(words)
    return cleaned_text
activities_df['processed_description'] = activities_df['Description'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer with Bi-grams and Tri-grams
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

# Fit and transform the processed descriptions
tfidf_matrix = vectorizer.fit_transform(activities_df['processed_description'])

# Function to extract top N keywords from each description
def extract_keywords(row, top_n=10):
    row_id = row.name
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix[row_id].toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return ', '.join(top_keywords)
# Extract keywords for each activity
activities_df['keywords'] = activities_df.apply(extract_keywords, axis=1)

# Outputting the first row keywords as a sample
print(activities_df['keywords'][0])
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_input_words(input_words):
    # Assuming input_words is a list of words
    cleaned_input = ' '.join([preprocess_text(word) for word in input_words])
    return cleaned_input

def find_similar_activities(input_words, activities_df, vectorizer, tfidf_matrix, top_n=3):
    # Preprocess and vectorize input words
    cleaned_input = preprocess_input_words(input_words)
    input_vector = vectorizer.transform([cleaned_input])

    # Ensure the input vector has the same number of features as the tfidf_matrix
    if input_vector.shape[1] != tfidf_matrix.shape[1]:
        raise ValueError("Dimension mismatch between input vector and TF-IDF matrix")

    # Calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)

    # Get top N similar activities
    top_activities_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    top_activities = activities_df.iloc[top_activities_indices]

    return top_activities[['Activity']]



def initialize_activities_model(file_path):
    # Read CSV file
    activities_df = pd.read_csv(file_path)

    # Preprocess descriptions
    activities_df['processed_description'] = activities_df['Description'].apply(preprocess_text)

    # Initialize TF-IDF Vectorizer with Bi-grams and Tri-grams
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

    # Fit and transform the processed descriptions
    tfidf_matrix = vectorizer.fit_transform(activities_df['processed_description'])

    return activities_df, vectorizer, tfidf_matrix
