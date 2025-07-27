# train_model.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = "C:/Users/madhu/OneDrive/Documents/ElevateLabs/signlanguagerecognition/archive (7)"
MODEL_PATH = "model/sign_model.h5"

# Load dataset
print("Loading dataset...")
train_df = pd.read_csv(os.path.join(DATASET_PATH, "sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(DATASET_PATH, "sign_mnist_test.csv"))

# Separate features and labels
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y_test = test_df.iloc[:, 0].values

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Save label map for later inference use
label_map = {i: chr(c + 65) for i, c in enumerate(lb.classes_)}
np.save("model/label_map.npy", label_map)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

# Build CNN model
print("Building model...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(24, activation='softmax')  # 24 letters (excluding J & Z)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save the model
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")


