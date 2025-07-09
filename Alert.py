import cv2
from detection import AccidentDetectionModel
import numpy as np
from twilio.rest import Client
import time

# Load the model
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
    video = cv2.VideoCapture('C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\CAM 1.mp4')  # Use 0 for webcam

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    last_alert_time = 0  # Initialize last alert time

    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video or failed to read frame.")
            break  # Exit if frame couldn't be read (end of video)

        # Check if the frame is valid
        if frame is None:
            print("Empty frame received.")
            continue  # Skip to the next iteration of the loop

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))
        roi = roi / 255.0  # Normalize if necessary
        roi = np.expand_dims(roi, axis=0)

        # Debug: Print ROI to check
        print("ROI shape:", roi.shape)

        pred, prob = model.predict_accident(roi)
        accident_value = prob[0][0]
        # Debug: Print predictions and accident value
        print(f"Prediction: {pred}, Accident Value: {accident_value * 100:.2f}%")

        if pred == "Accident" or accident_value > 0.4:
            accident_value_percent = round(accident_value * 100, 2)
            print(f"Accident detected with {accident_value_percent}% confidence.")
            
            current_time = time.time()
            print(f"Current Time: {current_time}, Last Alert Time: {last_alert_time}")

            # Check if it's time to send the alert
            if current_time - last_alert_time > 8:
                message = f"Accident detected with {accident_value_percent}% confidence."
                print("Condition met for sending message.")
                send_whatsapp_message(message)
                last_alert_time = current_time
                
                # Draw background for text
                cv2.rectangle(frame, (8, 80), (550, 140), (0, 0, 0), -1)  # Black rectangle for accident values and alert
                cv2.putText(frame, f"Accident Value: {accident_value_percent}%", (10, 100), font, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Alert sent", (10, 130), font, 0.7, (0, 255, 0), 2)
                
                # Show the frame with labels for 3 seconds
                cv2.imshow('Video', frame)
                cv2.waitKey(3000)  # Wait for 3000 ms (3 seconds)
            else:
                print("Condition not met for sending message.")
                cv2.putText(frame, "No Alert sent", (10, 130), font, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Accident Value: {accident_value * 100:.2f}%", (10, 40), font, 0.7, (0, 0, 255), 2)  # Display accident value if no accident detected

        cv2.imshow('Video', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
