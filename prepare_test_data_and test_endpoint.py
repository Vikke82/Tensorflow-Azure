import tensorflow as tf
import json
import requests

# Step 1: Load the MNIST test data
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Step 2: Normalize the test data (scale pixel values to 0–1)
x_test = x_test / 255.0

# Step 3: Select a few test samples (e.g., first 5 images)
test_samples = x_test[:5].tolist()  # Convert to list format for JSON serialization

# Step 4: Create a JSON payload
test_data_json = json.dumps({"data": test_samples})

# Print the JSON payload (for debugging or sending in a POST request)
#print(test_data_json)

# Define the local Flask app URL (or your Azure deployed scoring URI)
#url = "http://127.0.0.1:5000/predict"  # Local URL, or replace with your Azure URI
url = "https://my-mnist-endpoint-ville.northeurope.inference.ml.azure.com/score"

# Set the headers
headers = {'Content-Type': 'application/json'}

# Send the POST request with the test data
response = requests.post(url, data=test_data_json, headers=headers)

# Print the response (predicted labels)
print(response.json())
