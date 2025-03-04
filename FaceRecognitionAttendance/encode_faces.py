import face_recognition
import pickle
import cv2
import os

# Path to student images
STUDENT_DIR = "students"
ENCODINGS_FILE = "data/encodings.pickle"

known_encodings = []
known_names = []

for file in os.listdir(STUDENT_DIR):
    if file.endswith(("png", "jpg", "jpeg")):
        name = os.path.splitext(file)[0]  # Student name
        image_path = os.path.join(STUDENT_DIR, file)

        # Load image and encode face
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:  # Ensure at least one face is found
            known_encodings.append(encodings[0])
            known_names.append(name)

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("Face encodings saved successfully!")
