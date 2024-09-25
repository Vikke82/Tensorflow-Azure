from flask import Flask, request, jsonify
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Initialize the model
def init():
    global model
    model_path = "my_model.keras"  # Adjust the model path
    model = keras.models.load_model(model_path)

# Preprocess input data
def preprocess_data(data):
    data = np.array(data)
    data = data / 255.0  # Normalize pixel values
    return data

# Flask route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from request
        raw_data = request.get_json()

        # Preprocess and predict
        input_data = preprocess_data(raw_data['data'])
        predictions = model.predict(input_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # Return predictions as JSON response
        return jsonify({"predictions": predicted_labels.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

# Start the Flask app
if __name__ == '__main__':
    init()  # Initialize the model
    app.run(debug=True)
