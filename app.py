from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Load precomputed embeddings and metadata
name_embeddings_np = np.load("product_name_embeddings.npy")
data = pd.read_csv("product_metadata.csv")


index = faiss.IndexFlatL2(name_embeddings_np.shape[1])
index.add(name_embeddings_np)

# Normalize sentiment columns
data['Positive Reviews'] = data['Positive Reviews'] / data['Positive Reviews'].max()
data['Neutral Reviews'] = data['Neutral Reviews'] / data['Neutral Reviews'].max()
data['Negative Reviews'] = data['Negative Reviews'] / data['Negative Reviews'].max()

def get_bert_embeddings_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

from fuzzywuzzy import process

def match_input_to_main_category(user_input):
    main_categories = ['Refrigerators', 'Mobile Phones', 'Televisions', 'Air Conditioners', 'Earphones',
                       'Headsets', 'Smart Watches', 'Digital Cameras', 'speakers', 'printers', 'scanners',
                       'tablets', 'laptops', 'desktop computers', 'power banks']
    
    matched_category, confidence = process.extractOne(user_input, main_categories)
    if confidence >= 80:  # Adjust the confidence threshold as needed
        return matched_category
    return None


def process_user_input(user_input):
    price_limit = None
    num_products = 1  # Default to 1

    match = re.search(r'under (\d+)', user_input.lower())
    if match:
        try:
            price_limit = int(match.group(1))
        except (IndexError, ValueError):
            price_limit = None

    match = re.search(r'top (\d+)', user_input.lower())
    if match:
        num_products = int(match.group(1))

    if re.search(r'\b(a|an|one)\b', user_input.lower()):
        num_products = 1

    return price_limit, num_products


def get_recommendations(user_input, data, embeddings, num_products=5):
    price_limit, num_products = process_user_input(user_input)
    category = match_input_to_main_category(user_input)

    if not category or category not in data['category'].unique():
        return None, f"No products found for your query."

    filtered_data = data[data['category'] == category]
    if price_limit:
        filtered_data = filtered_data[filtered_data['Price'] <= price_limit]

    if filtered_data.empty:
        return None, f"No products found under ₹{price_limit} for the category '{category}'. You may want to try a different price range or category."

    filtered_embeddings = get_bert_embeddings_batch(filtered_data['Product Name'].tolist()).numpy()
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)
    user_embedding = get_bert_embeddings_batch([user_input]).numpy()
    distances, indices = temp_index.search(user_embedding, min(num_products, len(filtered_data)))
    recommended_products = filtered_data.iloc[indices[0]]

    # Change sorting from 'Positive Reviews' to 'Rating'
    recommended_products = recommended_products.sort_values(by='Rating', ascending=False)

    recommendations = f"<div>Here are the recommendations for the category '{category}' based on the sentiment analysis of reviews😊:</div><br>"
    for idx, row in recommended_products.iterrows():
        recommendations += f"""
        <div>
            <strong>Product Name:</strong> {row['Product Name']}<br>
            <strong>Category:</strong> {row['category']}<br>
            <strong>Price:</strong> ₹{row['Price']}<br>
            <strong>Rating:</strong> {row['Rating']} ⭐<br>
            <strong>Product URL:</strong> <a href="{row['Product URL']}" target="_blank">Link 🔗</a><br>
            It has {row['Positive Reviews'] * 100:.2f}% positive reviews 👍 and {row['Negative Reviews'] * 100:.2f}% negative reviews 👎.
        </div>
        <hr>
        """

    recommendations += "<div>Do you have any other queries? 🤔</div>" 
    
    return recommended_products, recommendations


@app.route('/')
def index():
    return render_template('html.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({'response': 'Please enter a message. 🙏'})
    
    user_input = user_input.lower()
    
    # Casual conversation handling
    casual_responses = {
        'hi': ["Hi there! What can I help you with? 👋", "Need any assistance? 😄"],
        'hello': ["Hello! How can I assist you today? 😊", "Need any assistance? 😄"],
        'how are you': ["I'm doing great! How can I assist you today? 😊"],
        'thank you': ["You're welcome! If you have any more questions, feel free to ask. 🙏"],
        'thanks': ["No problem! If you need more help, just let me know. 😊"],
        'bye': ["Goodbye! Have a great day! 👋", "See you later! Take care! 😊"],
        'r u fine':["Yah! I am fine 😁, and what about You?"]
    }

    for key in casual_responses:
        if key in user_input:
            return jsonify({'response': random.choice(casual_responses[key])})
    
    recommended_products, response = get_recommendations(user_input, data, name_embeddings_np)
    
    if recommended_products is None or response is None:
        response = f"No products found for the given input '{user_input}'. Please ask about a product like a Electronics or provide more details. 🤔"

    return jsonify({'response':  response})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Chatbot is running"}), 200
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Render automatically assigns a port
    app.run(host="0.0.0.0", port=port)
