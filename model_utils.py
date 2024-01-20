import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression

# Download necessary NLTK data
nltk.download('stopwords')  
nltk.download('wordnet')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


def bert_tokenize(text):
    return tokenizer.encode(text, add_special_tokens=True)

def get_bert_embeddings(text, model, tokenizer):
    # Tokenize the text
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)

    # Convert token to tensor
    input_ids = torch.tensor([tokenized_text])

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs[2]

    # Use the average of the last 4 layers' hidden states as the text embedding
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    # Calculate the average for each token across the last 4 layers
    token_vecs_cat = []
    for token in token_embeddings:
        cat_vec = torch.mean(token[-4:], dim=0)
        token_vecs_cat.append(cat_vec)

    # Use the average of all token vectors as the final embedding
    text_embedding = torch.mean(torch.stack(token_vecs_cat), dim=0)

    return text_embedding.numpy()


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove punctuation and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)  # Remove tags
    text = re.sub("(\\d|\\W)+", " ", text)  # Remove special characters and digits
    text = text.split()  # Convert to list from string
    lem = WordNetLemmatizer()  # Lemmatization
    text = [lem.lemmatize(word) for word in text if not word in stopwords.words('english')] 
    return " ".join(text)

def load_data(bert_model, tokenizer):
    nace_data = pd.read_csv('data/NACE Codes.csv')
    nace_data['Cleaned_Description'] = nace_data['Description'].apply(lambda x: preprocess_text(x))
    nace_data['BERT_Embeddings'] = nace_data['Description'].apply(lambda x: get_bert_embeddings(x, bert_model, tokenizer))
    return nace_data

def train_model(nace_data):
    X = list(nace_data['BERT_Embeddings'])
    y = nace_data['Code']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using Logistic Regression instead of MultinomialNB
    clf = LogisticRegression(max_iter=1000)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf


def predict_nace_code(description, logistic_model, bert_model, tokenizer, top_n=5):
    bert_embedding = get_bert_embeddings(description, bert_model, tokenizer)
    probabilities = logistic_model.predict_proba([bert_embedding])[0]

    # Get the indices of the top N probabilities
    top_n_indices = probabilities.argsort()[-top_n:][::-1]

    # Retrieve the corresponding NACE codes and probabilities
    predicted_codes = [(logistic_model.classes_[i], probabilities[i]) for i in top_n_indices]
    return predicted_codes


