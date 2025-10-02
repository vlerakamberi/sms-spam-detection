import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# paths
DATA_FILE = "data/archive/spam.csv"
MODEL_FOLDER = "models"
OUTPUT_FOLDER = "outputs"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# load csv
def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")
    if "v1" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
    else:
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
        df = df[["label", "text"]]
    return df

# text cleaning
def clean_msg(msg):
    msg = str(msg).lower()
    msg = re.sub(r"http\S+", " ", msg)
    msg = re.sub(r"[^a-z0-9\s]", " ", msg)
    msg = " ".join([w for w in msg.split() if len(w) > 2])
    msg = " ".join([w for w in msg.split() if not w.isdigit()])
    return msg

if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_FILE)

    print("Cleaning text...")
    df["text_clean"] = df["text"].apply(clean_msg)

    print("Messages total:", len(df))
    print("Spam:", len(df[df['label']=="spam"]))
    print("Ham:", len(df[df['label']=="ham"]))

    # top spam words
    spam_words = " ".join(df[df['label']=="spam"]["text_clean"]).split()
    print("Top 5 spam words:", Counter(spam_words).most_common(5))

    X = df["text_clean"]
    y = df["label"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # vectorize
    vec = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1,2), max_features=4000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    print("Vectorized shape:", X_train_vec.shape)

    # train Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_pred = nb.predict(X_test_vec)
    print("NB accuracy:", accuracy_score(y_test, nb_pred))
    print(classification_report(y_test, nb_pred))

    # train Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)
    print("LR accuracy:", accuracy_score(y_test, lr_pred))
    print(classification_report(y_test, lr_pred))

    # choose best
    if accuracy_score(y_test, lr_pred) >= accuracy_score(y_test, nb_pred):
        model = lr
        model_name = "logistic_regression"
        pred = lr_pred
    else:
        model = nb
        model_name = "naive_bayes"
        pred = nb_pred

    print("Best model:", model_name)

    # bar chart of messages
    df['label'].value_counts().plot(kind='bar', color=['skyblue','salmon'])
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "message_count.png"))

    # save model + vectorizer
    joblib.dump(model, os.path.join(MODEL_FOLDER, f"{model_name}.joblib"))
    joblib.dump(vec, os.path.join(MODEL_FOLDER, "tfidf_vectorizer.joblib"))
    print("Model saved")

    # confusion matrix
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"))
    plt.show()
