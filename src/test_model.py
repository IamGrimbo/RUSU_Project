import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'saved_models/fruit_classifier_model.h5'
model = load_model(model_path)

# Define the categories
data_dir = '../Data/train'
categories = sorted(os.listdir(data_dir))

def prepare_image(image_path, img_size=(100, 100)):
    """Prepare an image for classification."""
    img_array = cv2.imread(image_path)
    if img_array is None:
        print(f"Failed to read image from {image_path}")
        return None
    img_resized = cv2.resize(img_array, img_size)
    img_resized = img_resized / 255.0
    return np.expand_dims(img_resized, axis=0)

def predict_image(image_path):
    """Predict the category of an image."""
    prepared_image = prepare_image(image_path)
    if prepared_image is None:
        return "Invalid image"
    predictions = model.predict(prepared_image)
    print(f"Predictions: {predictions}")
    class_idx = np.argmax(predictions, axis=1)[0]
    print(f"Predicted class index: {class_idx}")
    if class_idx < 0 or class_idx >= len(categories):
        return "Invalid prediction index"
    return categories[class_idx]

# Test the model on new images
test_image_path = 'data/test/0060.jpg'  # replace with your test image path
predicted_category = predict_image(test_image_path)
print(f'The predicted category for the test image is: {predicted_category}')