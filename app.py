from flask import Flask, request, jsonify
import joblib
import re
import os

app = Flask(__name__)

# load model and vectorizer
model_file = [f for f in os.listdir("models") if f.endswith(".joblib") and "tfidf_vectorizer" not in f][0]
model = joblib.load(os.path.join("models", model_file))
vec = joblib.load(os.path.join("models", "tfidf_vectorizer.joblib"))

# text cleaning
def clean_msg(msg):
    msg = msg.lower()
    msg = re.sub(r"http\S+", " ", msg)
    msg = re.sub(r"[^a-z0-9\s]", " ", msg)
    msg = " ".join(msg.split())
    return msg

@app.route("/predict", methods=["POST"])
def predict_label():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    msg = clean_msg(data["text"])
    msg_vec = vec.transform([msg])
    pred = model.predict(msg_vec)[0]

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(port=5000)
