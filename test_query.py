import requests
import json

url = "http://127.0.0.1:8000/query"
payload = {"query": "What is the key topic?", "mode": "qa"}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e)
