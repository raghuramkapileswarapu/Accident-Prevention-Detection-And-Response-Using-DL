import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
from twilio.rest import Client
import time

# Initialize the model
model = AccidentDetectionModel("C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\model.json", 
                               "C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

# Twilio configuration for WhatsApp
twilio_account_sid = 'ACa5517bab7aa1e805024d18dca81fd4f2'
twilio_auth_token = 'de959f8f4867a40eb019b4b67b57b569'
twilio_whatsapp_number = 'whatsapp:+14155238886'
recipient_whatsapp_number = 'whatsapp:+919848699624'

def send_whatsapp_message(message):
    client = Client(twilio_account_sid, twilio_auth_token)
    try:
        message = client.messages.create(
            body=message,
            from_=twilio_whatsapp_number,
            to=recipient_whatsapp_number
        )
        print(f"Message sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")

def startapplication():
    video = cv2.VideoCapture('C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\CAM 1.mp4')  # for camera use video = cv2.VideoCapture(0)
    last_alert_time = 0  # Initialize last alert time
    alert_display_start_time = 0  # Initialize alert display start time
    display_alert = False  # Flag to control alert display

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to read frame or end of video.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = (round(prob[0][0] * 100, 2))
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)
            
            current_time = time.time()
            if current_time - last_alert_time > 5:
                message = f"Accident detected with {prob}% confidence."
                send_whatsapp_message(message)
                last_alert_time = current_time
                
                # Start displaying "Alert Sent" for 2 seconds
                alert_display_start_time = current_time
                display_alert = True
        
        # Display "Alert Sent" if within 2 seconds of sending the alert
        if display_alert:
            if time.time() - alert_display_start_time < 2:
                cv2.putText(frame, "Alert Sent", (frame.shape[1] - 200, 30), font, 1, (0, 255, 0), 2)
            else:
                display_alert = False

        cv2.imshow('Video', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
