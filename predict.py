import os
import re
import joblib

MODEL_DIR = "models"
VECT_NAME = "tfidf_vectorizer.joblib"

# text cleaning
def clean_msg(msg):
    msg = str(msg).lower()
    msg = re.sub(r"http\S+", " ", msg)
    msg = re.sub(r"[^a-z0-9\s]", " ", msg)
    msg = " ".join(msg.split())
    return msg

# load the first model we find in models folder
def load_model():
    files = os.listdir(MODEL_DIR)
    model_file = [f for f in files if f.endswith(".joblib") and "tfidf_vectorizer" not in f]
    if not model_file:
        raise FileNotFoundError("No model found in 'models/'")
    model_file = model_file[0]

    clf = joblib.load(os.path.join(MODEL_DIR, model_file))
    vec = joblib.load(os.path.join(MODEL_DIR, VECT_NAME))
    return clf, vec, model_file

if __name__ == "__main__":
    clf, vec, model_file = load_model()
    print(f"Loaded model: {model_file}")

    while True:
        text = input("\nType a message (or 'quit' to exit):\n> ")
        if text.strip().lower() in ("quit", "exit"):
            print("Bye!")
            break

        msg = clean_msg(text)
        msg_vec = vec.transform([msg])
        pred = clf.predict(msg_vec)[0]

        print("Prediction:", pred)
