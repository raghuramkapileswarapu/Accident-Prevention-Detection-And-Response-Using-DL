import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import time

model = AccidentDetectionModel("C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\model.json", 
                               "C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = 90  # Adjust this threshold as needed

def preprocess_frame(frame):
    # Additional preprocessing steps if needed
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))
    roi = roi / 255.0  # Normalize the image
    return roi

def startapplication():
    video = cv2.VideoCapture(0)
    history = []  # To keep track of past predictions
    history_length = 5  # Number of frames to consider for consistency
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        roi = preprocess_frame(frame)
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        prob = round(prob[0][0] * 100, 2)

        # Keep track of the predictions
        if len(history) >= history_length:
            history.pop(0)  # Remove the oldest prediction

        history.append((pred, prob))

        # Check if majority of the past predictions are "Accident" and above the threshold
        accident_count = sum(1 for p, prob in history if p == "Accident" and prob > THRESHOLD)
        if accident_count > history_length // 2:
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, "Accident " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
