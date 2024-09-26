import tensorflow as tf
from tensorflow import keras #tensorflow=2.16.1
from keras import layers, utils
import matplotlib.pyplot as plt


# Load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Visualize sample images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    image = x_train[i]
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Preprocess the data, normalize to 0..1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#2D image to 1D vector
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))




#evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc)

#predict the model
predictions = model.predict(x_test)
predicted_labels = [tf.argmax(prediction).numpy() for prediction in predictions]
print(predicted_labels[:10])


print("Tensorflow version in training "+tf.__version__)
# Save the model in TensorFlow's SavedModel format
model.save('my_model.keras', save_format='keras')
