import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# --- 1. LOAD THE DATA ---
with open('intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []

# Loop through the JSON to extract patterns and tags
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

# --- 2. PREPROCESSING: LABEL ENCODING ---
# Neural networks need numbers, not text tags like "course_fees"
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

# --- 3. PREPROCESSING: TOKENIZATION ---
# Convert the words in your sentences into numerical tokens
vocab_size = 1000
embedding_dim = 16
max_len = 20  # Max length of a user's question
oov_token = "<OOV>" # Out Of Vocabulary token for words it hasn't seen

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)

# --- 4. PREPROCESSING: PADDING ---
# Ensure every sequence is exactly 20 numbers long (pads with 0s if shorter)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# --- 5. BUILD THE LSTM ARCHITECTURE ---
model = Sequential()
# Embedding layer turns your word tokens into dense vectors
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
# The core LSTM layer to understand sequence and context
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
# Output layer with softmax to give probabilities for each intent tag
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

print(model.summary())

# --- 6. TRAIN THE MODEL ---
# Since the dataset is small, we use a higher number of epochs
epochs = 500
print("Training the model...")
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, verbose=1)

# --- 7. SAVE EVERYTHING FOR DEPLOYMENT ---
# Save the model
model.save("helpdesk_lstm_model.h5")

# Save the tokenizer so the bot can process new user input the exact same way
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Save the label encoder so the bot knows which number corresponds to which tag
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

print("Training Complete. Model and preprocessing tools saved!")