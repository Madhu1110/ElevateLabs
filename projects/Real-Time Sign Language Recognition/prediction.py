# predict_from_webcam.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load trained model and label map
model = load_model("model/sign_model.h5")
label_map = np.load("model/label_map.npy", allow_pickle=True).item()

# Start video capture
cap = cv2.VideoCapture(0)

# ROI coordinates (you can adjust this)
roi_top, roi_right, roi_bottom, roi_left = 100, 350, 300, 550

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame to avoid mirror image
    frame = cv2.flip(frame, 1)
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Draw the ROI box on the screen
    cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (255, 0, 0), 2)

    # Preprocess the ROI for prediction
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Predict the sign
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)
    predicted_letter = label_map[predicted_class]

    # Show predicted letter on screen
    cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show the frames
    cv2.imshow("Sign Language Recognition", frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
