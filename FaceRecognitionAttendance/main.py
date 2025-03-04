import cv2
import face_recognition
import pickle
import pandas as pd
import os
from datetime import datetime
from utils.video_processing import get_frames

# Load encodings
ENCODINGS_FILE = "data/encodings.pickle"
ATTENDANCE_FILE = "data/attendance.xlsx"
VIDEO_PATH = "videos/cctv_footage.mp4"

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Load attendance file or create a new one
if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
    df.columns = df.columns.str.strip()  # Remove spaces from column names
else:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])

print("Current DataFrame columns:", df.columns.tolist())  # Debugging

# Process video frames
for frame in get_frames(VIDEO_PATH):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Mark attendance
            now = datetime.now()
            if not ((df["Name"] == name) & (df["Date"] == now.strftime("%Y-%m-%d"))).any():
                df.loc[len(df)] = {"Name": name, "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}

    # Display frame (Optional)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save updated attendance
df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
print("Attendance updated successfully!")

cv2.destroyAllWindows()
