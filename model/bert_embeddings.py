from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
nace_data = pd.read_csv('Nace Codes.csv')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_tokenize(text):
    return tokenizer.encode(text, add_special_tokens=True)

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True, # Whether the model returns all hidden-states.
                                 )

def get_bert_embeddings(text):
    # Tokenize the text
    tokenized_text = bert_tokenize(text)

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

# Example: Transforming your dataset
nace_data['BERT_Embeddings'] = nace_data['Description'].apply(lambda x: get_bert_embeddings(x))

# Converting embeddings to a suitable format
X = list(nace_data['BERT_Embeddings'])
y = nace_data['Code']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Model Initialization
clf = LogisticRegression(max_iter=1000)

# Train the model
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Evaluating the model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

def predict_with_bert(description, model, tokenizer):
    # Get BERT embeddings for the description
    bert_embedding = get_bert_embeddings(description)

    # Predict using the trained classifier
    predicted_nace_code = model.predict([bert_embedding])
    return predicted_nace_code[0]

# Example Usage
new_description = " metal products"
predicted_code = predict_with_bert(new_description, clf, tokenizer)
print("Predicted NACE Code:", predicted_code)
