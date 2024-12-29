import cv2
import tensorflow as tf
from serial import Serial
import numpy as np
import time

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model("face_model.keras")

# Connect to Arduino
arduino = Serial('/dev/ttyUSB0', baudrate=9600, timeout=2)  # Replace 'COM9' with your Arduino port
print("Connected to Arduino on COM9")

# Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Loaded Haar Cascade for face detection")



import time

def fingerprint_verification():
    """Ask Arduino to verify fingerprint with a timeout."""
    print("Waiting for fingerprint...")
    time.sleep(5) 
    arduino.write(b'F')  # Command Arduino to verify fingerprint
    
    start_time = time.time()  # Record the start time
    timeout_duration = 5  # Timeout after 2 seconds

    while True:
        response = arduino.readline().decode().strip()
        if response:  # Only print non-empty responses
            print(f"Arduino response: {response}")  # Debugging print
        
        if response == "Fingerprint matched":
            print("Fingerprint matched! Unlocking solenoid lock...")
            arduino.write(b'U')  # Unlock command
            return True
        elif response == "Fingerprint not matched":
            print("Fingerprint not matched. Access denied.")
            return False
        
        # Check if timeout has elapsed
        if time.time() - start_time > timeout_duration:
            print("Fingerprint verification timed out. Returning to motion detection.")
            return False

def real_time_recognition():
    """Perform real-time face recognition with a 3-second camera display."""
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return False

    start_time = time.time()  # Record the start time
    timeout_duration = 10  # Timeout after 10 seconds
    camera_display_time = 3  # Keep camera open for at least 3 seconds

    face_recognized = False
    recognition_start_time = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            cropped_face = cv2.resize(cropped_face, (50, 50)).reshape(1, 50, 50, 1)

            model_out = model.predict(cropped_face)[0]
            confidence = model_out[np.argmax(model_out)] * 100

            print(model_out, confidence)

            if np.argmax(model_out) == 0:
                label = 'Vaibhav'
            elif np.argmax(model_out) == 1:
                label = 'Sruthi'
            elif np.argmax(model_out) == 2:
                label = "Intruder"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x+4, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(frame, str(confidence), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            if label == "Intruder":
                print("Intruder detected! Access denied.")
                time.sleep(5)  # Wait for 5 seconds before returning to motion detection
                cap.release()
                cv2.destroyAllWindows()
                return False  # Return to motion detection

            if confidence >= 70 and label != "Intruder":
                face_recognized = True
                if recognition_start_time is None:
                    recognition_start_time = time.time()

                # Check if the camera has displayed the face for at least 3 seconds
                if time.time() - recognition_start_time >= camera_display_time:
                    print(f"Recognized {label}. Initiating fingerprint verification...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return fingerprint_verification()

        cv2.imshow('Real-Time Face Recognition', frame)

        # Timeout condition
        if time.time() - start_time > timeout_duration:
            print("Face recognition timed out. Returning to motion detection.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False




def check_motion_and_recognize_face():
    """Main function to detect motion, recognize face, and unlock lock."""
    print("Starting motion detection and face recognition...")
    try:
        while True:
            arduino.write(b'M')  # Send 'M' to Arduino to check motion
            response = arduino.readline().decode().strip()
            print(f"Arduino response: {response}")  # Debugging print
            if response == "Motion detected":
                print("Motion detected! Starting face recognition...")
                if real_time_recognition():
                    print("Access granted.")
                else:
                    print("Access denied.")
                    time.sleep(5)  # Add a 5-second delay before restarting the loop
            elif response == "No motion detected":
                print("No motion detected.")
            time.sleep(1)  # Small delay to prevent excessive polling
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
    finally:
        arduino.close()

# Start the main function
check_motion_and_recognize_face()