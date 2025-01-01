import os

# Force OpenCV to use X11 instead of Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import tensorflow as tf
from serial import Serial
import numpy as np
import time
import requests

# Load Haar Cascade and Face Model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("face_model.keras")

# Connect to Arduino
arduino = Serial('/dev/ttyUSB0', baudrate=9600, timeout=2)  # Replace '/dev/ttyUSB0' with your Arduino port
print("Connected to Arduino on /dev/ttyUSB0")

# Map recognized face labels to fingerprint IDs
face_to_fingerprint = {
    'Vaibhav': 1,
    'Sruthi': 2,
    'Kamran': 3,
    'Karthik': 4
}

# Telegram Bot Configuration
bot_token = "7620011385:AAHC3ip1Ha-NeuiTpsvMydRroxIlYJtblro"  # Replace with your bot token
chat_id = "7396267168"  # Replace with your chat ID

def send_telegram_message(message):
    """Send a text message using the Telegram bot."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Telegram alert sent successfully!")
        else:
            print(f"Failed to send Telegram alert. Response: {response.text}")
    except Exception as e:
        print(f"An error occurred while sending Telegram alert: {e}")

def send_telegram_photo(photo_path):
    """Send a photo using the Telegram bot."""
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        files = {"photo": photo}
        data = {"chat_id": chat_id}
        try:
            response = requests.post(url, data=data, files=files)
            if response.status_code == 200:
                print("Photo sent successfully!")
            else:
                print(f"Failed to send photo. Response: {response.text}")
        except Exception as e:
            print(f"An error occurred while sending photo: {e}")

def fingerprint_verification(expected_id):
    """Ask Arduino to verify fingerprint with the expected ID."""
    global cap  # Ensure access to the camera object
    print(f"Waiting for fingerprint ID {expected_id}...")
    time.sleep(5)  # Allow time for fingerprint placement
    arduino.write(f'F{expected_id}'.encode())  # Send fingerprint ID to Arduino

    start_time = time.time()
    timeout_duration = 5  # Timeout after 5 seconds

    while True:
        response = arduino.readline().decode().strip()
        if response:
            print(f"Arduino response: {response}")

        if response == f"Fingerprint {expected_id} matched":
            print("Fingerprint matched! Unlocking solenoid lock...")
            arduino.write(b'U')  # Unlock command

            # Capture and send photo on success
            success_photo_path = "access_granted.jpg"
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)  # Reinitialize camera if needed
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(success_photo_path, frame)
                send_telegram_message("✅ Access Granted: Both face and fingerprint verified. Solenoid lock unlocked.")
                send_telegram_photo(success_photo_path)
            else:
                print("Failed to capture photo during successful access.")
            return True

        elif response == f"Fingerprint {expected_id} not matched":
            print("Fingerprint not matched. Access denied.")

            # Capture and send photo on failure
            failure_photo_path = "fingerprint_failed.jpg"
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)  # Reinitialize camera if needed
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(failure_photo_path, frame)
                send_telegram_message("⚠️ Fingerprint verification failed. Access denied.")
                send_telegram_photo(failure_photo_path)
            else:
                print("Failed to capture photo during fingerprint failure.")
            return False

        if time.time() - start_time > timeout_duration:
            print("Fingerprint verification timed out. Returning to motion detection.")
            return False

def real_time_recognition():
    """Perform real-time face recognition."""
    global cap  # To use the camera object in fingerprint verification
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return False

    start_time = time.time()
    timeout_duration = 10  # Timeout after 10 seconds
    camera_display_time = 5  # Keep camera open for at least 5 seconds

    recognition_start_time = None
    face_recognized = False

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

            # Map model predictions to labels
            if np.argmax(model_out) == 0:
                label = 'Vaibhav'
            elif np.argmax(model_out) == 1:
                label = 'Sruthi'
            elif np.argmax(model_out) == 2:
                label = 'Kamran'
            elif np.argmax(model_out) == 3:
                label = 'Karthik'
            elif np.argmax(model_out) == 4:
                label = 'Intruder'

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}%", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if label == "Intruder":
                print("Intruder detected! Access denied.")
                intruder_photo_path = "intruder_detected.jpg"
                success = cv2.imwrite(intruder_photo_path, frame)  # Save the intruder's image
                if success:
                    send_telegram_message("⚠️ Face ID Failed - Intruder detected! Access denied.")
                    send_telegram_photo(intruder_photo_path)
                else:
                    print("Failed to save the intruder photo.")
                time.sleep(5)
                cap.release()
                cv2.destroyAllWindows()
                return False

            if confidence >= 70 and label != "Intruder":
                face_recognized = True
                if recognition_start_time is None:
                    recognition_start_time = time.time()

                if time.time() - recognition_start_time >= camera_display_time:
                    print(f"Recognized {label}. Initiating fingerprint verification...")
                    expected_fingerprint_id = face_to_fingerprint.get(label)
                    if expected_fingerprint_id is not None:
                        cap.release()
                        cv2.destroyAllWindows()
                        return fingerprint_verification(expected_fingerprint_id)
                    else:
                        print("No fingerprint ID assigned for this label. Access denied.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

        cv2.imshow('Real-Time Face Recognition', frame)

        if time.time() - start_time > timeout_duration:
            print("Face recognition timed out. Returning to motion detection.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

def check_motion_and_recognize_face():
    """Detect motion, recognize face, and unlock the lock."""
    print("Starting motion detection and face recognition...")
    try:
        while True:
            arduino.write(b'M')  # Send 'M' to Arduino to check motion
            response = arduino.readline().decode().strip()
            if response:
                print(f"Arduino response: {response}")

            if response == "Motion detected":
                print("Motion detected! Starting face recognition...")
                if real_time_recognition():
                    print("Access granted.")
                else:
                    print("Access denied. Returning to motion detection.")
                    time.sleep(5)
            elif response == "No motion detected":
                print("No motion detected.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
    finally:
        arduino.close()

check_motion_and_recognize_face()
