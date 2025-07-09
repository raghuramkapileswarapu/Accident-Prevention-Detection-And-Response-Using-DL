import cv2
import time
import playsound

# Load the pre-trained Haar cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(r'C:\Users\chand\Downloads\haarcascade_eye.xml')

# Capture video feed from the webcam or external camera
cap = cv2.VideoCapture(0)

# Alert sound file
# Alert sound file
alert_sound = r'C:\Users\chand\OneDrive\Desktop\Project\Accident-Detection-System-main\Sleep Detection\Alert.wav'


closed_eye_start_time = None
closed_eye_duration = 0

while True:
    # Read the current frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes in the grayscale frame
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(eyes) == 0:  # No eyes detected
        if closed_eye_start_time is None:
            closed_eye_start_time = time.time()
        closed_eye_duration = time.time() - closed_eye_start_time
        
        if closed_eye_duration > 2:  # Eyes closed for more than 2 seconds
            playsound.playsound(alert_sound)
            closed_eye_start_time = None
            closed_eye_duration = 0
    else:
        closed_eye_start_time = None
        closed_eye_duration = 0  # Reset if eyes are detected
    
    for (x, y, w, h) in eyes:
        # Draw rectangles around the detected eyes
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the frame with eye rectangles and closed eye duration
    cv2.putText(frame, f"Closed Eye Duration: {closed_eye_duration:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Drowsiness Detection', frame)
    
    # If 'q' is pushed, the loop will end
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video recording, then shut the window
cap.release()
cv2.destroyAllWindows()
