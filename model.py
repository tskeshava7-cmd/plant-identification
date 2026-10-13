from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Complete list of 30 classes sorted alphabetically to match folder names
CLASS_NAMES = [
    "aloevera",
    "banana",
    "bilimbi",
    "cantaloupe",
    "cassava",
    "coconut",
    "corn",
    "cucumber",
    "curcuma",
    "eggplant",
    "galangal",
    "ginger",
    "guava",
    "kale",
    "longbeans",
    "mango",
    "melon",
    "orange",
    "paddy",
    "papaya",
    "peper chili",
    "pineapple",
    "pomelo",
    "shallot",
    "soybeans",
    "spinach",
    "sweet potatoes",
    "tobacco",
    "waterapple",
    "watermelon"
]

def load_custom_plant_model():
    """
    Loads the custom-trained 30-class plant/crop identification model (.keras format).
    """
    if os.path.exists("plant_model.keras"):
        model = tf.keras.models.load_model("plant_model.keras")
        return model
    else:
        return None

def preprocess_image(image: Image.Image):
    """
    Resizes and prepares an uploaded PIL image for the custom model (224x224).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((224, 224))  # Changed from 150 to 224 to match the model input shape
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    return img_array