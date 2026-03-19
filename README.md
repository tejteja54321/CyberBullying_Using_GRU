# 🚀 Cyberbullying Detection using Deep Learning (CNN + GRU)

## 📌 Project Overview

This project is a **Deep Learning-based Cyberbullying Detection System** that classifies user input text into different types of cyberbullying categories.

It uses a **Hybrid CNN + GRU model** to capture both:

* Local text patterns (via CNN)
* Contextual sequence information (via GRU)

The system is deployed using **Flask**, allowing real-time predictions through a web interface.

---

## 🎯 Objective

To build an intelligent system that can:

* Detect cyberbullying in text
* Classify it into specific categories
* Avoid false predictions on normal/neutral text

---

## 🧠 Model Architecture

### 🔥 Hybrid Model: CNN + GRU

* **Embedding Layer** → Converts words into vectors
* **Conv1D (CNN)** → Extracts important word patterns
* **MaxPooling** → Reduces dimensionality
* **GRU Layer** → Understands context and sequence
* **Dense Layers** → Final classification

---

## 📊 Classes

The model predicts the following categories:

* Age Cyberbullying
* Ethnicity Cyberbullying
* Gender Cyberbullying
* Religion Cyberbullying
* Other Cyberbullying
* Not Cyberbullying

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Flask (for deployment)

---

## 📁 Project Structure

```
├── app.py
├── cnn_gru_model.h5
├── tokenizer.pickle
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── upload.html
│   ├── preview.html
│   └── chart.html
├── static/
├── dataset/
│   └── cyberbullying_tweets.csv
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/cyberbullying-detection.git
cd cyberbullying-detection
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Flask App

```
python app.py
```

### 4️⃣ Open Browser

```
http://127.0.0.1:5000/
```

---

## ⚠️ Challenges Faced (IMPORTANT)

### 1. ❌ Wrong Predictions for Normal Text

Example:

```
Input: "hi"
Output: Other_Cyberbullying
```

### Reason:

* Model was trained only on bullying categories
* No understanding of **neutral/non-bullying text**

---

### 2. ❌ Overconfidence Problem

* Model forced to choose a class even when input is unrelated
* Leads to incorrect predictions

---

### 3. ❌ Small Dataset Limitation

* Limited generalization
* Poor handling of unseen text

---

### 4. ❌ Hardware Constraints

* BERT and Transformer models caused:

  * Kernel crashes
  * High memory usage
* Needed a **lightweight CPU-friendly solution**

---

## ✅ Solutions Implemented

### 🔥 1. Neutral Text Filtering

A predefined list of general words was added:

```
hi, hello, hey, good, thanks, etc.
```

If detected → directly classified as:

```
Not_Cyberbullying
```

---

### 🔥 2. Confidence Thresholding

```
if confidence < 0.5:
    Not_Cyberbullying
```

This prevents the model from making forced incorrect predictions.

---

### 🔥 3. Hybrid Approach (DL + Rule-Based)

Final system uses:

* Deep Learning Model (CNN + GRU)
* Rule-Based Filtering
* Confidence Scoring

---

### 🔥 4. Lightweight Model Design

Instead of heavy models like BERT:

* Used GRU-based architecture
* Reduced sequence length
* Optimized for CPU

---

## 📈 Model Performance

| Model     | Accuracy |
| --------- | -------- |
| LSTM      | ~85%     |
| GRU       | ~88%     |
| CNN + GRU | ~90–94%  |

---

## 💡 Key Learnings

* Deep Learning alone is not enough
* Real-world systems require:

  * Rule-based logic
  * Confidence thresholds
* Dataset quality matters more than model complexity
* Lightweight models are essential for deployment

---

## 🔮 Future Improvements

* Use Attention Mechanism (GRU + Attention)
* Increase dataset size
* Add multilingual support
* Deploy on cloud (AWS / Render)
* Replace rule-based filtering with smarter NLP logic

---

## 👨‍💻 Author

**Teja**
Full Stack Python Developer | AI Engineer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
