import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\model.json", 
                               "C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))
    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)
        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)
    return frame

def startapplication():
    video_files = ["C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\CAM 1.mp4", 
                   "C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\CAM 2.mp4", 
                   "C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\CAM 3.mp4", 
                   "C:\\Users\\chand\\OneDrive\\Desktop\\Project\\Accident-Detection-System-main\\Accident-Detection-System-main\\CAM 4.mp4"]
    videos = [cv2.VideoCapture(video) for video in video_files]

    # Get screen resolution
    screen_width = 1920
    screen_height = 1080
    
    # Adjust window size to be slightly smaller than screen size
    window_width = int(screen_width * 0.95)
    window_height = int(screen_height * 0.95)
    
    target_width = window_width // 2
    target_height = window_height // 2

    while True:
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Restart video if it ends
                ret, frame = video.read()
            frame = process_frame(frame)
            frame = cv2.resize(frame, (target_width, target_height))
            frames.append(frame)

        if len(frames) == 4:
            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            final_frame = np.vstack((top_row, bottom_row))

            cv2.namedWindow("Accident Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Accident Detection", window_width, window_height)
            cv2.imshow("Accident Detection", final_frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    for video in videos:
        video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    startapplication()


