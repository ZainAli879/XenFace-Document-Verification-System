import streamlit as st
import cv2
import numpy as np
import os
import easyocr
from deepface import DeepFace
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

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

# ===================== 📌 IMAGE RESIZING FUNCTION =====================
def resize_image(image_path, output_name):
    img = Image.open(image_path)
    img = img.resize((250, 250))  # Resize for DeepFace processing
    img.save(output_name)
    return output_name

# ===================== 📌 FACE VERIFICATION FUNCTION =====================
def verify_faces(img1_path, img2_path, model_name="Facenet512", detector_backend="retinaface", threshold=0.75):
    try:
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            return None, "⚠️ One or both processed images are missing. Verification cannot proceed."

        # Resize images before verification
        img1_path = resize_image(img1_path, "resized_profile.jpg")
        img2_path = resize_image(img2_path, "resized_cnic.jpg")

        # Perform DeepFace verification
        result = DeepFace.verify(img1_path, img2_path, model_name=model_name, detector_backend=detector_backend)

        # Apply custom threshold
        result["verified"] = result["distance"] <= threshold
        result["threshold"] = threshold

        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 CNIC Face Verification System")
st.write("Upload your **CNIC image** and take a live photo for verification!")

col1, col2 = st.columns(2)

with col1:
    cnic_file = st.file_uploader("📤 Upload CNIC Image", type=["jpg", "png", "jpeg"])
with col2:
    st.write("📸 Capture Live Profile Picture")
    live_photo = st.camera_input("Take a photo")

if cnic_file and live_photo:
    cnic_path = "uploaded_cnic.jpg"
    profile_path = "captured_profile.jpg"

    with open(cnic_path, "wb") as f:
        f.write(cnic_file.getbuffer())
    with open(profile_path, "wb") as f:
        f.write(live_photo.getbuffer())

    # Validate CNIC Image
    if not is_valid_cnic(cnic_path):
        st.error("❌ Invalid CNIC image! Please upload a valid CNIC with recognizable government logos and text.")
    else:
        # Extract Faces with unique names
        cnic_face_path, cnic_error = extract_face(cnic_path, "cnic_face.jpg")
        profile_face_path, profile_error = extract_face(profile_path, "profile_face.jpg")

        if cnic_error:
            st.error(cnic_error)
        elif profile_error:
            st.error(profile_error)
        else:
            # Display Processed Faces
            st.subheader("📷 Processed Face Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(profile_face_path), caption="Captured Profile Face", use_container_width=True)
            with col2:
                st.image(Image.open(cnic_face_path), caption="Extracted CNIC Face", use_container_width=True)
            
            # Perform Face Verification
            result, verify_error = verify_faces(profile_face_path, cnic_face_path)
            
            if verify_error:
                st.error(verify_error)
            else:
                # Display Verification Result
                st.subheader("✅ Verification Result")
                distance = result["distance"]
                threshold = result["threshold"]
                match_status = "✔️ Match!" if result["verified"] else "❌ No Match!"
                
                st.write(f"**Distance:** {distance:.4f}")
                st.write(f"**Threshold:** {threshold:.4f}")
                st.markdown(f"### {match_status}")
else:
    st.warning("⚠️ Please upload CNIC and capture a live photo to proceed!")
