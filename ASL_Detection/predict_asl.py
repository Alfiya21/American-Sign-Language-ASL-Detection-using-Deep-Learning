import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. LOAD TRAINED MODEL
MODEL_PATH = os.path.join("model", "asl_model.h5")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# 2. CLASS LABELS (AUTO FROM DATASET)
TRAIN_DIR = os.path.join(
    "dataset",
    "ASL_Dataset",
    "asl_alphabet_train",
    "asl_alphabet_train"
)

class_labels = sorted(os.listdir(TRAIN_DIR))
print("Class labels:", class_labels)
# 3. IMAGE PREPROCESS FUNCTION
IMG_SIZE = 64

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or path incorrect")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 4. PREDICT FUNCTION
def predict_asl(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = np.max(prediction)

    return predicted_label, confidence

# 5. TEST WITH ONE IMAGE

# PUT A REAL IMAGE PATH HERE
TEST_IMAGE_PATH = os.path.join(
    "dataset",
    "ASL_Dataset",
    "asl_alphabet_train",
    "asl_alphabet_train",
    "A",        # change letter
    "A1.jpg"    # change image name
)

label, conf = predict_asl(TEST_IMAGE_PATH)
print(f"Predicted ASL Sign: {label} (Confidence: {conf:.2f})")
