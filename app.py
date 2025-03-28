import streamlit as st
import cv2
import numpy as np
import os
import easyocr
from deepface import DeepFace
from PIL import Image
import mediapipe as mp
import time

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ===================== 📌 CNIC VALIDATION FUNCTION =====================
def is_valid_cnic(image_path):
    """Checks if the uploaded CNIC image contains government logos and key text patterns."""
    try:
        result = reader.readtext(image_path)
        keywords = ["Government", "Pakistan", "Identity", "Card"]  # Key CNIC-related words
        
        detected_texts = [text.lower() for _, text, _ in result]
        if any(keyword.lower() in detected_texts for keyword in keywords):
            return True
        return False
    except Exception as e:
        return False

# ===================== 📌 FACE EXTRACTION FUNCTION =====================
def extract_face(image_path, output_name):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Error: Image not found!"

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, "⚠️ No face detected! Please upload a clear image with a visible face."
    elif len(faces) > 1:
        return None, "⚠️ Multiple faces detected! Please upload an image with only your face."

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    cv2.imwrite(output_name, face)
    return output_name, None

# ===================== 📌 LIVE FACE CAPTURE FUNCTION =====================
def capture_live_face():
    cap = cv2.VideoCapture(0)
    instructions = [
        "Move closer to the camera",
        "Look up",
        "Look down",
        "Look left",
        "Look right",
        "Smile! 😃"
    ]
    instruction_index = 0
    detected_movements = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                mouth = face_landmarks.landmark[13]
                
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                left_x, right_x = int(left_eye.x * w), int(right_eye.x * w)

                cv2.putText(frame, instructions[instruction_index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if instruction_index == 0 and nose_y < h // 4:
                    detected_movements.add("closer")
                elif instruction_index == 1 and nose_y < h // 5:
                    detected_movements.add("up")
                elif instruction_index == 2 and nose_y > h // 2:
                    detected_movements.add("down")
                elif instruction_index == 3 and left_x > w // 3:
                    detected_movements.add("left")
                elif instruction_index == 4 and right_x < 2 * w // 3:
                    detected_movements.add("right")
                elif instruction_index == 5 and mouth.y > 0.45:
                    detected_movements.add("smile")

                if instructions[instruction_index].split()[0].lower() in detected_movements:
                    instruction_index += 1
                    time.sleep(1)

                if instruction_index == len(instructions):
                    cv2.putText(frame, "✅ Face Verified!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imwrite("live_captured_face.jpg", frame)
                    cap.release()
                    return "live_captured_face.jpg"

        cv2.imshow("Live Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 CNIC Face Verification System")
st.write("Upload your **CNIC image** and use the live camera for real-time verification!")

cnic_file = st.file_uploader("📤 Upload CNIC Image", type=["jpg", "png", "jpeg"])

if cnic_file:
    cnic_path = "uploaded_cnic.jpg"
    with open(cnic_path, "wb") as f:
        f.write(cnic_file.getbuffer())

    if not is_valid_cnic(cnic_path):
        st.error("❌ Invalid CNIC image! Please upload a valid CNIC.")
    else:
        st.write("✅ CNIC validated! Now capture your live photo.")
        if st.button("📸 Capture Live Photo"):
            profile_path = capture_live_face()
            if profile_path:
                cnic_face_path, cnic_error = extract_face(cnic_path, "cnic_face.jpg")
                profile_face_path, profile_error = extract_face(profile_path, "profile_face.jpg")
                if cnic_error:
                    st.error(cnic_error)
                elif profile_error:
                    st.error(profile_error)
                else:
                    result, verify_error = verify_faces(profile_face_path, cnic_face_path)
                    if verify_error:
                        st.error(verify_error)
                    else:
                        match_status = "✔️ Match!" if result["verified"] else "❌ No Match!"
                        st.write(f"**Distance:** {result['distance']:.4f}")
                        st.write(f"**Threshold:** {result['threshold']:.4f}")
                        st.markdown(f"### {match_status}")
            else:
                st.error("❌ Live capture failed. Please try again.")
