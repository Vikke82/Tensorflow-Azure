import tensorflow as tf
import json
import requests
import matplotlib.pyplot as plt
# Step 1: Load the MNIST test data
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()



# Step 3: Select a few test samples (e.g., first 5 images)
test_samples = x_test[401:405].tolist()  # Convert to list format for JSON serialization

# Visualize sample images
plt.figure(figsize=(10, 5))
for i in range(len(test_samples)):
    plt.subplot(2, 5, i+1)
    image = test_samples[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Step 4: Create a JSON payload
test_data_json = json.dumps({"data": test_samples})

# Print the JSON payload (for debugging or sending in a POST request)
#print(test_data_json)

# Define the local Flask app URL (or your Azure deployed scoring URI)
#url = "http://127.0.0.1:5000/predict"  # Local URL, or replace with your Azure URI
#url = "https://my-mnist-endpoint-ville.northeurope.inference.ml.azure.com/score"
url = "http://127.0.0.1:31311/score"
#url = "https://myworkspace-ville.northeurope.inference.ml.azure.com/score"

# Set the headers
headers = {'Content-Type': 'application/json'}



# Send the POST request with the test data
response = requests.post(url, data=test_data_json, headers=headers)

# Print the response (predicted labels)
print(response.json())
