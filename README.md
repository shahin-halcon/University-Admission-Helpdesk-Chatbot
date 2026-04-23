```markdown
# 🎓 University Admission Helpdesk Chatbot

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Lightweight-black.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An LSTM-based university helpdesk chatbot featuring a Flask web interface. It is designed to answer common admission and campus-related questions quickly and efficiently.

---

## 🌟 Overview

This project provides a complete question-and-answer chatbot pipeline, from training a neural network on custom intents to serving it via a clean, interactive browser chat interface. 

**Highlights:**
* **Custom LSTM Classifier:** Trained on specific university helpdesk intents.
* **Interactive UI:** Real-time responses using a lightweight Flask backend.
* **Smart Fallbacks:** Graceful fallback responses for low-confidence predictions.
* **Lightweight & Accessible:** Perfect for learning, portfolio demos, and small deployments.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** TensorFlow / Keras, NumPy, scikit-learn
* **Frontend:** HTML, CSS, JavaScript (Fetch API)

---

## 🚀 Quick Start

### Prerequisites
* Python 3.9+ 
* `pip` installed

### 1. Install Dependencies

```bash
pip install flask numpy tensorflow scikit-learn
```

### 2. Train the Model
Run the training script to generate the model and preprocessing artifacts.

```bash
python train.py
```

*Expected generated files: `helpdesk_lstm_model.h5`, `tokenizer.pickle`, `label_encoder.pickle`*

### 3. Run the Web App
Start the Flask server to launch the chat interface.

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000/**

---

## 🧠 How It Works

1. **Data Preparation:** Patterns and tags are loaded from `intents.json`. Text is converted into token sequences, and tag labels are encoded into numeric classes.
2. **Model Training:** An embedding layer learns word vectors, an LSTM layer captures sequence context, and dense layers classify the intents via softmax.
3. **Inference Flow:** User text is tokenized and padded. The model predicts intent probabilities, and the highest-probability intent is selected.
4. **Response Generation:** If confidence exceeds the threshold (0.6), a response from that intent is returned. Otherwise, a fallback message guides the user to a human administrator.

---

## 📊 Model Details

| Parameter | Value |
| :--- | :--- |
| **Vocabulary Size** | 1000 |
| **Embedding Dimension** | 16 |
| **Max Sequence Length** | 20 |
| **LSTM Units** | 64 |
| **Hidden Dense Units** | 64 |
| **Output Activation** | Softmax |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Optimizer** | Adam |
| **Epochs** | 500 |

---

## 🎯 Supported Intent Categories

| Tag | Purpose |
| :--- | :--- |
| `courses_provided` | Available courses/departments |
| `hod_info` | Heads of department |
| `course_fees` | Fee structure |
| `contact_emails` | Official contact emails |
| `reach_by_bus` | Travel guidance by bus |
| `reach_by_air` | Travel guidance by air |
| `reach_by_train` | Travel guidance by train |

---

## 🔌 API Reference

**Endpoint:** `POST /get`

**Form Field:**
* `msg`: The user's text message.

**Example Request:**
```bash
curl -X POST [http://127.0.0.1:5000/get](http://127.0.0.1:5000/get) -d "msg=What courses do you offer?"
```

**Example Response:**
```json
{
  "response": "We provide the following courses: B.Tech, M.Tech, MBA, and Ph.D. programs."
}
```

---

## 📁 Project Structure

```text
.
├── app.py
├── train.py
├── intents.json
├── tokenizer.pickle
├── label_encoder.pickle
├── helpdesk_lstm_model.h5          # Generated after training
├── templates/
│   └── index.html
├── LICENSE
└── README.md
```

---

## 🚧 Known Limitations & Roadmap

**Current Limitations:**
* A small intent dataset may limit generalization.
* Lacks context memory across multiple conversational turns.
* Single-language support (English).

**Suggested Improvements (Contributions Welcome!):**
* Add more patterns per intent to improve robustness.
* Introduce text normalization and data validation.
* Add logging for model confidence and errors.
* Move hyperparameters to a dedicated config file.
* Include a `requirements.txt` and `Dockerfile` for easier deployment.

---

## 📄 License & Author

* **Author:** Shahin Sharaf
* **License:** This project is licensed under the MIT License. See the `LICENSE` file for details.
```
