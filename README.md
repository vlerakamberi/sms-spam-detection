Spam Detection AI Project

Project Description
A simple SMS spam detection system using machine learning.
Classifies messages as spam or ham.
Includes data cleaning, TF-IDF feature extraction, model training, evaluation, and a Flask API for predictions.

Dataset
Source: SMS Spam Collection Dataset
URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
Categories: 2 (spam, ham)
Total Samples: 5,572

Project Structure
data/archive/spam.csv       - dataset
models/                     - saved models and vectorizer
outputs/                    - charts, confusion matrix
train.py                    - train models
predict.py                  - interactive prediction script
app.py                      - Flask API
test_request.py             - test API requests
requirements.txt            - dependencies
README.md                   - this file

Requirements
Python 3.x
pandas
scikit-learn
matplotlib
seaborn
Flask
joblib

Install:
pip install -r requirements.txt

Usage

Train the Models
python train.py
- Trains Naive Bayes and Logistic Regression.
- Saves best model and TF-IDF vectorizer in models/.
- Generates charts in outputs/.

Predict Messages
python predict.py
- Enter a message.
- Outputs spam or ham.
- Type quit to exit.

Flask API
python app.py
- Runs on http://127.0.0.1:5000
- POST to /predict with JSON:
  {"text": "Congratulations! You won a free iPhone!"}
- Returns:
  {"prediction": "spam"}

Results
Best Model: Naive Bayes
Accuracy: ~96–97%
Confusion matrix and charts saved in outputs/.

Notes
Demonstrates end-to-end ML workflow on text data.
TF-IDF vectorization used for features.
Two models trained and compared.
