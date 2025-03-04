import face_recognition
import cv2

def detect_faces(image):
    """Detect faces in an image."""
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    return face_locations, face_encodings
