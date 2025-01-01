import cv2
import tensorflow as tf
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model("face_model.keras")

def real_time_recognition():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5) 
        # confidence = model.predict([data])[0]
    
        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            cropped_face = cv2.resize(cropped_face, (50, 50)).reshape(1, 50, 50, 1)

            model_out = model.predict(cropped_face)[0]

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
            cv2.putText(frame, label, (x+4, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            cv2.putText(frame, str(model_out[np.argmax(model_out)]*100), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            
        cv2.imshow('Real-Time Face Recognition', frame)

        # if confidence >= 70 and label != "Unknown":
        #         print(f"Recognized {label}. Initiating fingerprint verification...")
        #         cap.release()
        #         cv2.destroyAllWindows()
        #         time.sleep(10)
        #         return fingerprint_verification()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

real_time_recognition()