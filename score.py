import json
import numpy as np
import tensorflow as tf
import os
import logging

logging.basicConfig(level=logging.DEBUG)
os.environ["AZUREML_MODEL_DIR"] = "/var/azureml-app/azureml-models/my_tensorflow_model/3"

print("TensorFlow Version in score.py: ", tf.__version__)
# Initialize the model
def init():
    print("Init on score.py")
    logging.debug("Initializing model...")
    global model
    # Load the model from Azure ML model registry
    #model_path = Model.get_model_path('my_tensorflow_model', version=3)  # Replace 'my_mnist_model' with the actual model name
    #model_path = "my_model.keras"
    # model_dir = os.getenv('AZUREML_MODEL_DIR', default='/var/azureml-app/')
    # model_path = os.path.join(model_dir, "my_model.keras")  # Replace with actual model file name
    model_path = "/var/azureml-app/azureml-models/my_tensorflow_model/3/my_model.keras"

    #model_path = "my_model.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    logging.debug("Model initialized successfully.")
    print("End of init")

# Preprocess incoming data (normalize it)
def preprocess_data(data):
    # Convert to numpy array and normalize (as done during training)
    data = np.array(data)
    data = data / 255.0  # Normalization step (0-255 to 0-1) grayscale 8-bit images, grayscale 0-255
    return data # 

# Run the model on the input data
def run(raw_data):
    print("Run on score.py")
    logging.debug(f"Received input data")
    try:
        # Parse the input data from the request
        input_data = json.loads(raw_data)['data']  # Expecting 'data' key in the input
        
        # Preprocess the input data
        input_data = preprocess_data(input_data)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Convert the model output (probabilities) to predicted class labels (0-9)
        predicted_labels = np.argmax(predictions, axis=1)

        logging.debug("Returning predictions.")

        print("Run done "+str(predicted_labels.tolist()))
        # Return the predicted labels as JSON response
        return json.dumps({"predictions": predicted_labels.tolist()})
    
    except Exception as e:
        print("error "+str(e))
        # Return error message in case of exception
        return json.dumps({"error": str(e)})
