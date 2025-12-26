import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Load the trained model (assuming it's saved as 'best_model_fixed.h5')
model_path = os.path.join(os.path.dirname(__file__), 'best_model_fixed.h5')
model = tf.keras.models.load_model(model_path)

# Define class names (based on the tomato leaf disease training data)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict():
    """
    Predict tomato leaf disease from the uploaded image.

    Returns:
        str: Predicted disease class name
    """
    try:
        # Path to the uploaded image
        img_path = 'media/input/test/test.jpg'

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # EfficientNet preprocessing

        # Make prediction
        predictions = model.predict(img_array, verbose=0)

        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]

        return predicted_class

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error in prediction"

# Function for general use (can be used outside Django)
def predict_from_path(img_path):
    """
    Predict tomato leaf disease from a given image path.

    Args:
        img_path (str): Path to the input image file

    Returns:
        str: Predicted disease class name
    """
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # EfficientNet preprocessing

        # Make prediction
        predictions = model.predict(img_array, verbose=0)

        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]

        return predicted_class

    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error in prediction"