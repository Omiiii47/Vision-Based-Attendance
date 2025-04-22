import cv2
import pickle
import numpy as np
import os

# Load the Haar Cascade Classifier for face detection
facedetect = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

# Initialize an empty list to store face data
faces_data = []

# Counter to keep track of the number of frames processed
i = 0

# Get user input for their name
name = input("Enter your name: ")

# Open a video capture object using the default camera (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) <= 5 and i % 5 == 0:
            faces_data.append(resized_img)

        i += 1

        cv2.putText(frame, str(len(faces_data)), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 5:
        break

cap.release()
cv2.destroyAllWindows()

# Save face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(5, -1)

if 'names.pkl' not in os.listdir('Data/'):
    names = [name] * 5
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 5
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('Data/'):
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
