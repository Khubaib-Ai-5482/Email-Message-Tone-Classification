import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'text': [
        "Dear Sir, I am writing to apply for the position of Data Analyst.",
        "Hey buddy, wanna catch up tonight?",
        "Huge discounts on our new products!",
        "Please review the report ASAP!",
        "Looking forward to meeting you next week.",
        "Don't miss our limited time offer!",
        "Hi, can we discuss the project tomorrow?",
        "Urgent: submit your assignment before 5 PM!",
        "Congratulations on your promotion!",
        "Hey, let's grab coffee later."
    ],
    'label': [
        "Formal",
        "Informal",
        "Promotional",
        "Urgent",
        "Formal",
        "Promotional",
        "Formal",
        "Urgent",
        "Formal",
        "Informal"
    ]
}

df = pd.DataFrame(data)

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean)

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df, palette='viridis')
plt.show()

X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.show()

features = tfidf.get_feature_names_out()
coef = model.feature_log_prob_

for i, l in enumerate(model.classes_):
    top_idx = coef[i].argsort()[-10:]
    plt.figure(figsize=(7,4))
    plt.barh([features[j] for j in top_idx], coef[i][top_idx], color='green' if l!="Urgent" else 'red')
    plt.show()

def predict_tone(text):
    text = clean(text)
    vec = tfidf.transform([text])
    return model.predict(vec)[0]

while True:
    t = input("Type your email/message (or 'quit' to exit): ")
    if t.lower() == 'quit':
        break
    print("Predicted Tone:", predict_tone(t))
