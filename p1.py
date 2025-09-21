import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

# ================================
# STEP 1: Load & Encode Known Faces
# ================================
known_face_encodings = []
known_face_names = []

path = "known_faces"  # Folder where you store known people's images
if not os.path.exists(path):
    print(f"⚠️ Folder '{path}' not found! Create it and add images (e.g., jubedh.jpg)")
    exit()

for filename in os.listdir(path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if len(encoding) > 0:  # Ensure face was detected
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])
            print(f"✅ Loaded face for: {filename}")
        else:
            print(f"⚠️ No face found in {filename}, skipping...")

# ================================
# STEP 2: Start Webcam
# ================================
video_capture = cv2.VideoCapture(0)
attendance = []

print("\n📷 Starting webcam... Press 'Q' to quit.\n")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Error: Could not read from webcam.")
        break

    # Resize frame for faster processing (1/4 size)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Mark attendance only once per person
            if name not in attendance:
                attendance.append(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"📝 Attendance: {name} marked present at {current_time}")

        # Draw rectangle & name on face
        top, right, bottom, left = [v * 4 for v in face_location]  # scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

    # Show video feed
    cv2.imshow('Face Recognition Attendance', frame)

    # Exit when 'Q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or ESC key
        print("Exiting program...")
        break


video_capture.release()
cv2.destroyAllWindows()

# ================================
# STEP 3: Save Attendance to CSV
# ================================
if attendance:
    df = pd.DataFrame(attendance, columns=["Name"])
    df["Date"] = datetime.now().strftime("%Y-%m-%d")
    df["Time"] = datetime.now().strftime("%H:%M:%S")
    df.to_csv("attendance.csv", mode='a', header=not os.path.exists("attendance.csv"), index=False)
    print("\n✅ Attendance saved to attendance.csv")
else:
    print("\n⚠️ No attendance recorded.")
 