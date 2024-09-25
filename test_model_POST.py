import requests
import json
import tensorflow as tf
from tensorflow import keras


# Load the MNIST dataset and select a subset for testing
(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalize the test data

# Select the first 5 test images for testing
test_data = x_test[:5].tolist()  # Convert to list format for JSON serialization

# Create a JSON payload
input_data = json.dumps({"data": test_data})

# Define the scoring URI
scoring_uri = "http://<ACI_SERVICE_URI>"  # Replace with actual scoring URI

# Send the request to the web service
headers = {"Content-Type": "application/json"}
response = requests.post(scoring_uri, data=input_data, headers=headers)

# Print the predictions from the model
print(response.json())
