import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Simulate the init() function from score.py
def init():
    global model
    # Load your locally saved model (ensure this path is correct)
    model_path = "my_model.keras"  # Adjust the model path
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")

# Simulate the preprocess_data() function from score.py
def preprocess_data(data):
    data = np.array(data)  # Convert to numpy array
    data = data / 255.0    # Normalize pixel values (as done during training)
    return data

# Simulate the run() function from score.py
def run(raw_data):
    try:
        # Simulate incoming JSON data and parse it
        input_data = json.loads(raw_data)['data']
        
        # Preprocess the data
        input_data = preprocess_data(input_data)

        # Make predictions
        predictions = model.predict(input_data)

        # Convert predictions to class labels (0-9)
        predicted_labels = np.argmax(predictions, axis=1)

        # Return predictions as JSON
        return json.dumps({"predictions": predicted_labels.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Initialize the model
init()

# Example test data (use actual test data here, e.g., from MNIST)

# Load MNIST test data
(_, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalize the test data (as done during training)
x_test = x_test / 255.0

# Select a few test samples (e.g., first 50 samples)
test_samples = x_test[:50].tolist()  # Convert to list format for JSON serialization

# Simulate incoming request data
test_data = json.dumps({"data": test_samples})

# Run the test on the local model
output = run(test_data)
print(f"Predicted labels: {output}")


# Test the run function locally
output = run(test_data)
print(f"Model predictions: {output}")
