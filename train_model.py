
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

data = pd.read_csv("scam_dataset.csv")

print("Dataset Loaded Successfully!")
print("\nDataset Overview:")
print(data.head())

# -----------------------------------------
# 2. TEXT PREPROCESSING
# -----------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r'http\S+|www\S+',
        ' URL ',
        text
    )

    text = re.sub(
        r'[^a-z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


data["message"] = data["message"].apply(clean_text)

# -----------------------------------------
# 3. SEPARATE FEATURES AND LABELS
# -----------------------------------------

X = data["message"]
y = data["label"]

# -----------------------------------------
# 4. SPLIT DATASET
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))

# -----------------------------------------
# 5. TF-IDF VECTORIZATION
# -----------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

# -----------------------------------------
# 6. TRAIN MACHINE LEARNING MODEL
# -----------------------------------------

model = MultinomialNB(alpha=0.5)

model.fit(
    X_train_vectorized,
    y_train
)

print("\nMachine Learning Model Trained Successfully!")

# -----------------------------------------
# 7. MODEL EVALUATION
# -----------------------------------------

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL EVALUATION ==========")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["legitimate", "scam"]
    )
)

# -----------------------------------------
# 8. SAVE TRAINED MODEL
# -----------------------------------------

joblib.dump(
    model,
    "scam_detector.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print("\n======================================")
print("CYBERGUARD AI MODEL SAVED SUCCESSFULLY!")
print("======================================")

