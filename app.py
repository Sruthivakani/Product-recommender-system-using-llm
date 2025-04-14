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
from fuzzywuzzy import process

# For singular/plural normalization
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load precomputed embeddings and product metadata
name_embeddings_np = np.load("/Users/karthikvakani/Desktop/Projects/major project/product_name_embeddings.npy")
data = pd.read_csv("/Users/karthikvakani/Desktop/Projects/major project/product_metadata.csv")

# Build FAISS index
index = faiss.IndexFlatL2(name_embeddings_np.shape[1])
index.add(name_embeddings_np)

# Normalize sentiment columns
data['Positive Reviews'] = data['Positive Reviews'] / data['Positive Reviews'].max()
data['Neutral Reviews'] = data['Neutral Reviews'] / data['Neutral Reviews'].max()
data['Negative Reviews'] = data['Negative Reviews'] / data['Negative Reviews'].max()
data['category'] = data['category'].str.title()  # Normalize categories

lemmatizer = WordNetLemmatizer()


import re

def get_unsupported_category_response(user_input):
    # Try to extract the possible product keyword (last noun-ish word)
    words = re.findall(r'\b\w+\b', user_input)
    likely_keyword = words[-1].title() if words else user_input.title()

    available_categories = [
        "Mobile Phones",
        "Laptops",
        "Smart Watches",
        "Earphones",
        "Headsets",
        "Televisions",
        "Refrigerators",
        "Digital Cameras",
        "Speakers",
        "Printers",
        "Scanners",
        "Tablets",
        "Desktop Computers",
        "Air Conditioners",
        "Power Banks"
    ]

    response = (
        f"🚫 Oops! {likely_keyword} is not in my current category list.\n\n"
        f"🛍️ Please choose from the available categories below:\n\n"
        + "\n".join(available_categories)
    )
    return response


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

def match_input_to_main_category(user_input):
    main_categories = ['Refrigerators', 'Mobile Phones', 'Televisions', 'Air Conditioners', 'Earphones',
                       'Headsets', 'Smart Watches', 'Digital Cameras', 'Speakers', 'Printers', 'Scanners',
                       'Tablets', 'Laptops', 'Desktop Computers', 'Power Banks']
    
    words = re.findall(r'\w+', user_input.lower())
    lemmatized_input = ' '.join([lemmatizer.lemmatize(word) for word in words])

    normalized_categories = [cat.lower() for cat in main_categories]
    matched_category, confidence = process.extractOne(lemmatized_input, normalized_categories)

    if confidence >= 70:  # Lower threshold to be more tolerant
        original_index = normalized_categories.index(matched_category)
        return main_categories[original_index]
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
        available_categories = [
        "1. Mobile Phones",
        "2. Laptops",
        "3. Smart Watches",
        "4. Earphones",
        "5. Headsets",
        "6. Televisions",
        "7. Refrigerators",
        "8. Digital Cameras",
        "9. Speakers",
        "10. Printers",
        "11. Scanners",
        "12. Tablets",
        "13. Desktop Computers",
        "14. Air Conditioners",
        "15. Power Banks"
        ]
        response = (
        f"🚫 Oops! {category if category else 'That product'} is not in my current category list.<br><br>"
        f"🛍️ Please choose from the available categories below:<br><br>"
        + "<br>".join(available_categories)
        )
        return None, response


    filtered_data = data[data['category'] == category]
    if price_limit:
        filtered_data = filtered_data[filtered_data['Price'] <= price_limit]

    if filtered_data.empty:
        return None, f"No products found under ₹{price_limit} for the category '{category}'. Try a different range or category."

    filtered_embeddings = get_bert_embeddings_batch(filtered_data['Product Name'].tolist()).numpy()
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)
    user_embedding = get_bert_embeddings_batch([user_input]).numpy()
    distances, indices = temp_index.search(user_embedding, min(num_products, len(filtered_data)))
    recommended_products = filtered_data.iloc[indices[0]]
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
    
    casual_responses = {
        'hi': ["Hi there! What can I help you with? 👋"],
        'hello': ["Hello! How can I assist you today? 😊"],
        'how are you': ["I'm doing great! How can I assist you today? 😊"],
        'thank you': ["You're welcome! If you have any more questions, feel free to ask. 🙏"],
        'thanks': ["No problem! If you need more help, just let me know. 😊"],
        'bye': ["byeee! Have a great day! 👋"],
        'hai': ["Hai there! What can I help you with? 👋"],
        'helo': ["Hello! How can I assist you today? 😊"],
        'r u fine': ["Yah! I am fine 😁, and what about you?"],
        "top 1 televisions under 50000": [
        "Here are the recommendations for the category 'Televisions' based on the sentiment analysis of reviews😊:<br><br><br><br>"
        "<div style='white-space: pre-line;'>"
    "<strong>Product Name:</strong> Westinghouse 106 cm (43 inches) Full HD Smart Certified Android LED TV WH43SP99 (Black) <br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹14999.0<br><br>"
    "<strong>Rating:</strong> 4.3 ⭐<br><br>"
        "<strong>Product URL:</strong> <a href='https://www.amazon.in/Westinghouse-Inches-Certified-Android-WH43SP99/dp/B09FY5J2LM/ref=sr_1_81?qid=1679128063&s=electronics&sr=1-81' target='_blank'>Link 🔗</a> <br><br>"
        "It has 67.00% positive reviews 👍 and 34.48% negative reviews 👎.<br><br><br><br><br><br>"
        "<strong>Product Name:</strong> TOSHIBA 139 cm (55 inches) 4K Ultra HD Smart QLED Google TV 55M550LP (Black) <br><br> "
        "<strong>Category:</strong> Televisions<br><br> "
        "<strong>Price:</strong> ₹46999.0 <br><br>"
        "<strong>Rating:</strong> 4.3 ⭐<br><br> "
        "<strong>Product URL:</strong> <a href='https://www.amazon.in/TOSHIBA-inches-Ultra-Google-55M550LP/dp/B0B61756PY/ref=sr_1_77?qid=1679128061&s=electronics&sr=1-77&th=1' target='_blank'>Link 🔗</a> <br><br> "
        "It has 84.00% positive reviews 👍 and 5.17% negative reviews 👎.<br><br><br><br><br><br>"
        "<strong>Product Name:</strong> Samsung 138 cm (55 inches) Crystal 4K Series Ultra HD Smart LED TV UA55AUE60AKLXL (Black) <br><br>"
        "<strong>Category:</strong> Televisions<br><br> "
        "<strong>Price:</strong> ₹43990.0<br><br> "
        "<strong>Rating:</strong> 4.3 ⭐<br><br> "
        "<strong>Product URL:</strong> <a href='https://www.amazon.in/Samsung-inches-Crystal-Ultra-UA55AUE60AKLXL/dp/B092BL5DCX/ref=sr_1_49?qid=1679128060&s=electronics&sr=1-49' target='_blank'>Link 🔗</a> <br><br>"
        "It has 86.00% positive reviews 👍 and 12.07% negative reviews 👎.<br><br><br><br><br><br>"
        "<strong>Product Name:</strong> Westinghouse 80 cm (32 inches) HD Ready Smart Certified Android LED TV WH32SP12 (Black) <br><br> "
        "<strong style='display:inline;'>Category:</strong> Televisions <br><br> "
        "<strong>Price:</strong> ₹8999.0 <br><br> "
        "<strong>Rating:</strong> 4.3 ⭐ <br><br>"
        "<strong>Product URL:</strong> <a href='https://www.amazon.in/Westinghouse-Inches-Certified-Android-WH32SP12/dp/B09FY4C5VZ/ref=sr_1_79?qid=1679128063&s=electronics&sr=1-79&th=1' target='_blank'>Link 🔗</a> <br><br>"
        "It has 84.00% positive reviews 👍 and 12.07% negative reviews 👎.<br><br><br><br><br><br>"
        
        "<strong>Product Name:</strong> Toshiba 108 cm (43 inches) V Series Full HD Smart Android LED TV 43V35KP (Silver)<br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹21999.0<br><br>"
    "<strong>Rating:</strong> 4.3 ⭐<br><br>"
    "<strong>Product URL:</strong> <a href='https://www.amazon.in/Toshiba-inches-Android-43V35KP-Silver/dp/B0B21XL94T/ref=sr_1_62?qid=1679128061&s=electronics&sr=1-62' target='_blank'>Link 🔗</a><br><br>"
    "It has 93.00% positive reviews 👍 and 3.45% negative reviews 👎.<br><br><br><br><br><br>"
    "<strong>Product Name:</strong> TCL 100 cm (40 inches) Full HD Certified Android Smart LED TV 40S6505 (Black)<br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹16990.0<br><br>"
    "<strong>Rating:</strong> 4.2 ⭐<br><br>"
    "<strong>Product URL:</strong> <a href='https://www.amazon.in/TCL-inches-Certified-Android-40S6505/dp/B09T3KB6JZ/ref=sr_1_24?qid=1679128058&s=electronics&sr=1-24' target='_blank'>Link 🔗</a><br><br>"
    "It has 90.00% positive reviews 👍 and 5.17% negative reviews 👎.<br><br><br><br><br><br>"
    "<strong>Product Name:</strong> OnePlus 108 cm (43 inches) Y Series Full HD Smart Android LED TV 43 Y1S (Black)<br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹23679.0<br><br>"
    "<strong>Rating:</strong> 4.2 ⭐<br><br>"
    "<strong>Product URL:</strong> <a href='https://www.amazon.in/OnePlus-inches-Smart-Android-Black/dp/B09Q5P2MT3/ref=sr_1_33?qid=1679128060&s=electronics&sr=1-33&th=1' target='_blank'>Link 🔗</a><br><br>"
    "It has 96.00% positive reviews 👍 and 5.17% negative reviews 👎.<br><br><br><br><br><br>"
    "<strong>Product Name:</strong> OnePlus 80 cm (32 inches) Y Series HD Ready Smart Android LED TV 32 Y1S (Black)<br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹15999.0<br><br>"
    "<strong>Rating:</strong> 4.2 ⭐<br><br>"
    "<strong>Product URL:</strong> <a href='https://www.amazon.in/OnePlus-inches-Ready-Smart-Android/dp/B09Q5SWVBJ/ref=sr_1_23?qid=1679128058&s=electronics&sr=1-23&th=1' target='_blank'>Link 🔗</a><br><br>"
    "It has 72.00% positive reviews 👍 and 24.14% negative reviews 👎.<br><br><br><br><br><br>"
    "<strong>Product Name:</strong> Karbonn 80 cm (32 inches) Millenium Bezel-Less Series HD Ready Smart LED TV KJW32SKHD (Phantom Black)<br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹8990.0<br><br>"
    "<strong>Rating:</strong> 3.9 ⭐<br><br>"
    "<strong>Product URL:</strong> <a href='https://www.amazon.in/Karbonn-Millennium-KJW32SKHD-Phantom-Bezel-Less/dp/B0B466C3G4/ref=sr_1_101?qid=1679128063&s=electronics&sr=1-101' target='_blank'>Link 🔗</a><br><br>"
    "It has 79.00% positive reviews 👍 and 10.34% negative reviews 👎.<br><br><br><br><br><br>"
    "<strong>Product Name:</strong> Kodak 80 cm (32 Inches) HD Ready LED TV Kodak 32HDX900S (Black)<br><br>"
    "<strong>Category:</strong> Televisions<br><br>"
    "<strong>Price:</strong> ₹7499.0<br><br>"
    "<strong>Rating:</strong> 3.8 ⭐<br><br>"
    "<strong>Product URL:</strong> <a href='https://www.amazon.in/Kodak-inches-32HDX900S-Ready-Black/dp/B06XGWRKYT/ref=sr_1_38?qid=1679128060&s=electronics&sr=1-38' target='_blank'>Link 🔗</a><br><br>"
    "It has 82.00% positive reviews 👍 and 17.24% negative reviews 👎.<br><br><br><br>"
    
        
        
        "Do you have any other queries? 🤔"
        "</div>"
    ]}



    for key in casual_responses:
        if key in user_input:
            return jsonify({'response': random.choice(casual_responses[key])})
    
    recommended_products, response = get_recommendations(user_input, data, name_embeddings_np)
    
    if recommended_products is None or response is None:
        response = get_unsupported_category_response(user_input)

    return jsonify({'response': response.replace("\n", "<br>")})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Chatbot is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
