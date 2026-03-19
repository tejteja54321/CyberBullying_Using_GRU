from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
MODEL_PATH = "cnn_gru_model.h5"
model = load_model(MODEL_PATH)

with open('tokenizer2.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        return render_template("preview.html", df_view=df)

@app.route('/home')
def home():
    return render_template('index.html')

# -----------------------------
# Label Mapping
# -----------------------------
labels = {
    0: 'Age_Cyberbullying',
    1: 'Ethnicity_Cyberbullying',
    2: 'Gender_Cyberbullying',
    3: 'Not_Cyberbullying',
    4: 'Other_Cyberbullying',
    5: 'Religion_Cyberbullying'
}

# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        msg = request.form['message']
        text_input = msg.lower().strip()

        # -----------------------------
        # Neutral Words List
        # -----------------------------
        neutral_words = [
            "hi", "hello", "hey", "good", "morning", "evening",
            "thanks", "thank you", "ok", "okay", "fine",
            "nice", "cool", "great", "awesome", "welcome",
            "bye", "goodbye", "see you",
            "how are you", "what are you doing",
            "have a nice day", "good afternoon"
        ]

        # -----------------------------
        # Rule 1: Neutral Text Check
        # -----------------------------
        if any(word in text_input for word in neutral_words):
            return render_template('index.html', prediction_value="Not_Cyberbullying")

        # -----------------------------
        # Preprocessing
        # -----------------------------
        msg_df = pd.DataFrame(index=[0], data=msg, columns=['data'])

        sequences = tokenizer.texts_to_sequences(msg_df['data'].astype('U'))
        new_text = pad_sequences(sequences, maxlen=28)

        # -----------------------------
        # Model Prediction
        # -----------------------------
        pred = model.predict(new_text, batch_size=1, verbose=0)[0]

        confidence = np.max(pred)
        predicted_class = np.argmax(pred)

        print("Confidence:", confidence)
        print("Predicted index:", predicted_class)

        # -----------------------------
        # Rule 2: Low Confidence Check
        # -----------------------------
        if confidence < 0.5:
            return render_template('index.html', prediction_value="Not_Cyberbullying")

        # -----------------------------
        # Final Prediction
        # -----------------------------
        result = labels.get(predicted_class, "Unknown")

        return render_template('index.html', prediction_value=result)

    else:
        return render_template('index.html', error="Invalid message")

# -----------------------------
# Chart Route
# -----------------------------
@app.route('/chart')
def chart():
    return render_template('chart.html')

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)