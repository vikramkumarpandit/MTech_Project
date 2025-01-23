from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting {"text": "your news text here"}
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess and predict
    text = [data['text']]
    text_vectorized = vectorizer.transform(text)
    prediction = model.predict(text_vectorized)[0]  # 0: real, 1: fake

    return jsonify({'prediction': 'Fake' if prediction == 1 else 'Real'})

if __name__ == '__main__':
    app.run(debug=True)
