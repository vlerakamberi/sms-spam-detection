import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Congratulations! You won a free iPhone!"}

response = requests.post(url, json=data)
print(response.json())
