
# 1. IMPORT LIBRARIES

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 2. DATASET PATH (FINAL & CORRECT)

TRAIN_DIR = os.path.join(
    "dataset",
    "ASL_Dataset",
    "asl_alphabet_train",
    "asl_alphabet_train"
)

print("TRAIN_DIR:", TRAIN_DIR)
print("Detected class folders:", os.listdir(TRAIN_DIR))

# 3. PARAMETERS

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15

# 4. DATA PREPROCESSING

train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

NUM_CLASSES = train_data.num_classes
print("Number of classes detected:", NUM_CLASSES)
print("Class labels:", train_data.class_indices)

# 5. BUILD CNN MODEL

model = Sequential()

model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

#  FINAL OUTPUT LAYER (DYNAMIC)
model.add(Dense(NUM_CLASSES, activation="softmax"))


# 6. COMPILE MODEL

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. TRAIN MODEL

history = model.fit(
    train_data,
    epochs=EPOCHS,
    verbose=1
)

# 8. SAVE MODEL

os.makedirs("model", exist_ok=True)
model.save("model/asl_model.h5")

print("Model saved to model/asl_model.h5")

# 9. VISUALIZE TRAINING

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
