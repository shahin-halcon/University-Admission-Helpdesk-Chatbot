from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random

app = Flask(__name__)

# --- 1. LOAD SAVED ASSETS ---
with open('intents.json') as file:
    data = json.load(file)

model = load_model('helpdesk_lstm_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20  

# --- 2. DEFINE ROUTES ---

# Route 1: Serve the HTML webpage
@app.route("/")
def home():
    return render_template("index.html")

# Route 2: API endpoint to handle the chat logic
@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    
    # Process text
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)
    
    # Predict
    prediction = model.predict(padded_sequence, verbose=0)
    tag_index = np.argmax(prediction)
    tag = lbl_encoder.inverse_transform([tag_index])[0]
    confidence = np.max(prediction)
    
    # Generate Response
    response_text = "I'm not entirely sure about that. Could you contact admin@iuk.ac.in?"
    if confidence > 0.6:
        for intent in data['intents']:
            if intent['tag'] == tag:
                response_text = random.choice(intent['responses'])
                break
                
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)