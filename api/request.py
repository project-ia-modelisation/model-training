import requests

data = {
    "model_path": "./data/generated_model_{last_number}.obj",
    "ground_truth_path": "./data/ground_truth.obj"
}
response = requests.post("http://localhost:5001/evaluate", json=data)
print(response.json())
