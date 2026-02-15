# 📧 Email / Message Tone Classification (NLP Project)

## 📌 Overview

This project classifies short emails or messages into different tones using:

- Text preprocessing  
- TF-IDF vectorization (Unigrams + Bigrams)  
- Multinomial Naive Bayes  

The model predicts one of the following tones:

- Formal  
- Informal  
- Promotional  
- Urgent  

It also visualizes:

- Label distribution  
- Confusion matrix  
- Top important words per tone  

---

## 🚀 Key Features

✔ Text cleaning with regex  
✔ Multi-class text classification  
✔ TF-IDF feature extraction  
✔ Naive Bayes classifier  
✔ Confusion matrix visualization  
✔ Top keyword analysis per tone  
✔ Interactive tone prediction  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Regex  

---

## 📂 Dataset

The dataset is manually created inside the script and contains:

- text → Email or message content  
- label → Tone category  

Example labels:

- Formal  
- Informal  
- Promotional  
- Urgent  

---

## 🔎 Project Workflow

### 1️⃣ Text Cleaning

- Convert text to lowercase  
- Remove numbers and special characters  
- Keep only alphabetic characters  

```python
clean()
```

---

### 2️⃣ Data Visualization

A count plot shows the distribution of tone categories.

---

### 3️⃣ Train-Test Split

- 60% training  
- 40% testing  
- Stratified split to maintain class balance  

---

### 4️⃣ TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=1000,
    ngram_range=(1,2),
    stop_words='english'
)
```

- Uses unigrams and bigrams  
- Removes English stopwords  
- Limits vocabulary size  

---

### 5️⃣ Model Training

Model used:

```
Multinomial Naive Bayes
```

Suitable for text classification problems.

---

### 6️⃣ Evaluation

Metrics used:

- Accuracy  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

Confusion matrix is visualized using a heatmap.

---

## 📊 Word Importance Per Tone

For each class, the model identifies:

- Top 10 words with highest importance  

Separate bar charts are generated for:

- Formal  
- Informal  
- Promotional  
- Urgent  

This helps interpret how the model differentiates tone.

---

## 🔮 Interactive Prediction

You can type your own message:

```
Type your email/message (or 'quit' to exit):
```

The model predicts the tone in real-time.

Example:

- "Please send the updated report." → Formal  
- "Limited time offer, buy now!" → Promotional  

---

## 📦 Installation

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Then type your message to test predictions.

---

## 🎯 Use Cases

- Email tone detection  
- Customer communication analysis  
- Chat moderation  
- NLP beginner project  
- Multi-class text classification practice  

---

## 📈 What This Project Demonstrates

- NLP preprocessing  
- TF-IDF feature engineering  
- Multi-class Naive Bayes classification  
- Model interpretability  
- Interactive prediction system  

---

## 👨‍💻 Author

Built as part of Natural Language Processing practice and text classification experiments.

If you found this useful, consider starring the repository ⭐
