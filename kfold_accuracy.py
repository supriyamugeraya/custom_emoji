import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the trained model
model_name = input("Enter the model name (with extension) to check accuracy: ")
model = load_model(model_name)

# Load the test data
with open("test_images", "rb") as f:
    test_images = np.array(pickle.load(f))
with open("test_labels", "rb") as f:
    test_labels = np.array(pickle.load(f), dtype=np.uint8)

# Reshape and one-hot encode labels
image_x, image_y = test_images.shape[1], test_images.shape[2]
test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
test_labels = to_categorical(test_labels)

# Evaluate the model
scores = model.evaluate(test_images, test_labels, verbose=1)
accuracy = scores[1] * 100

# Save the accuracy to a file
output_file = "model_accuracy.txt"
with open(output_file, "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}%\n")

print(f"Model Accuracy: {accuracy:.2f}% saved to {output_file}")
