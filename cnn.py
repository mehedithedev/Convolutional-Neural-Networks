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
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
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
# 32 is the number of filters/kernals that will learn to recognize different patterns in teh images

# Max pooling layer: reuce the spatial dimensions (height & width) of these feature maps to retain essential features
model.add(layers.MaxPooling2D((2,2))) # The halves the height and width of the feature maps

# Second convolutional layer
model.add(layers.Conv2D(64, (3,3), activation='relu')) # Extract more complex features
model.add(layers.MaxPooling2D((2,2))) # Further reduces dimensions

# Third convolutional layer
model.add(layers.Conv2D(64, (3,3), activation='relu')) # Continuation of feature extraction

# Flatten the output to feed into the fully connected layer
# Flattening converts the 2D matrices from the convolutional layers into 1D vectors for the dense layer
model.add(layers.Flatten()) # Converting to 1D

# Fully connected layer: learn the complex features from the flattened 1D vector
model.add(layers.Dense(64, activation='relu'))

# Output layer: predicts the class probabilities for digits 0 - 9
model.add(layers.Dense(10, activation='softmax')) #Softmax converts scores to probabilities for each class 

# Compile the model 
# This sets up the model for training with specified optimizer, loss funciton, and metrics to track 
model.compile(
        optimizer = 'adam', # Adam optimizer is po;ular for training neural networks
        loss = 'categorical_crossentropy', #Loss funciton for multi-class classification
        metrics = ['accuracy'] #Tracking accuracy during training
    )

# Fit the model using data generator
# We train the model on the augmented training data, and 'history' stores training metrics
history = model.fit(
    data_gen.flow(train_images, train_labels, batch_size = 64), # Using the data augmentation for training
    epochs = 10, # Numbers of times to go through the eitire dataset
    validation_data = (test_images, test_labels))


# Evaluate the model on the test set
# This gives us the final loss and accuracy on unseen data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.4f}") # print the test accuracy

# Plot traininig & validation accuracy and loss
plt.figure(figsize=(12,5)) # Define the of the figure

# Plot training and accuracy
# history: this object is returned by model.fit(), which contains information about the training process
# history.history: dictionary that holds lists of metrics recorded during training
plt.subplot(1,2,1) # subplot for accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Plot training & validation loss
plt.subplot(1,2,2) # subplot for loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout() # Adjusts subplots to fit into the figure area
plt.show()