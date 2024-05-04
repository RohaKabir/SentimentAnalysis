from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import re
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained model
current_dir = os.path.dirname(__file__)
model_file_path = os.path.join(current_dir, 'model', 'sentiment_analysis_model_v1.pkl')

try:
    model = joblib.load(model_file_path)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)

# Load the CountVectorizer instance
vectorizer_file_path = os.path.join(current_dir, 'model', 'count_vectorizer.pkl')

try:
    vectorizer = joblib.load(vectorizer_file_path)
    print("CountVectorizer loaded successfully!")
except Exception as e:
    print("Error loading the CountVectorizer:", e)

# Define the preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Route to serve the index.html file from the frontend folder
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)
    review = data['review']
    
    # Preprocess the text input
    processed_review = preprocess_text(review)
    
    # Vectorize the preprocessed text input
    review_vect = vectorizer.transform([processed_review])  # Vectorize the text
    
    # Predict the sentiment category
    try:
        prediction = model.predict(review_vect)[0]
        return jsonify({'sentiment': prediction})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Error predicting sentiment. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
