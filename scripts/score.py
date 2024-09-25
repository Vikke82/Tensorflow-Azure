import json
import numpy as np
from azureml.core.model import Model
from tensorflow import keras

# Initialize the model
def init():
    global model
    # Load the model from Azure ML model registry
    model_path = Model.get_model_path('my_tensorflow_model')  # Replace 'my_mnist_model' with the actual model name
    model = keras.models.load_model(model_path)

# Preprocess incoming data (normalize it)
def preprocess_data(data):
    # Convert to numpy array and normalize (as done during training)
    data = np.array(data)
    data = data / 255.0  # Normalization step (0-255 to 0-1) grayscale 8-bit images, grayscale 0-255
    return data # 

# Run the model on the input data
def run(raw_data):
    try:
        # Parse the input data from the request
        input_data = json.loads(raw_data)#['data']  # Expecting 'data' key in the input
        
        # Preprocess the input data
        input_data = preprocess_data(input_data)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Convert the model output (probabilities) to predicted class labels (0-9)
        predicted_labels = np.argmax(predictions, axis=1)

        # Return the predicted labels as JSON response
        return json.dumps({"predictions": predicted_labels.tolist()})
    
    except Exception as e:
        # Return error message in case of exception
        return json.dumps({"error": str(e)})
