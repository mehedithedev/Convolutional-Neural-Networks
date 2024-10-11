import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical #Utility to convert integer labels to categorical (One-hot encoded format)
import matplotlib.pyplot as plt

# Load the MNIST dataset. Training images comprise of 60,00 images, whereas testing is 10,000
(train_images , train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the images
# Reshape the images to add a channel dimension (1 for grayscale) and normalize the pixel to range [0,1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
# Normalization helps improve the model performance by scaling pixel values to a range [0,1]

train_labels = to_categorical(train_labels) # Transform training labels to one-hot format
test_labels = to_categorical(test_labels) 
# This converts integer labels(0-9) to one-hot encoded (e.g.- 3 becomes [0,0,0,1,0,0,0,0,0,0])


# Create a data augmentation generator
data_gen = tf.karas.preprocessing.image.ImageDataGenerator(
    rotation_range = 10, # Randomly rotate images by up to 10 degrees
    width_shift_range = 0.1, # Randomly shift images horizontally by 10% of their width
    height_shift_range = 0.1, #Randomly shift images vertically by 10% of their height
    zoom_range = 0.1, # Randomly zoom in or out by 10% 
    shear_range = 0.1 # Randomly shear the images (distort them)

    )

# Define the CNN model
# We use a Sequential model, which allows us to stack layers on top of each other sequentially
model = models.Sequential()

#  Add layers to the model
# First convolutional layer: extract features from the images using 32 filters of size 3x3
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model 

model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

# Traind the model
model.fit(train_images, train_labels, epochs = 5, batch_size = 64, validation_split = 0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")  