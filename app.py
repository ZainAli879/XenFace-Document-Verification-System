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
    """Checks if the uploaded CNIC image contains a valid CNIC number."""
    try:
        result = reader.readtext(image_path)
        for _, text, _ in result:
            clean_text = text.replace(" ", "").replace("-", "")
            if clean_text.isdigit() and len(clean_text) == 13:  # CNIC format
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
def verify_faces(img1_path, img2_path, model_name="ArcFace", detector_backend="mtcnn", threshold=0.65):
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
st.write("Upload your **CNIC image** and **profile picture**, and we'll check if they match!")

col1, col2 = st.columns(2)

with col1:
    cnic_file = st.file_uploader("📤 Upload CNIC Image", type=["jpg", "png", "jpeg"])
with col2:
    profile_file = st.file_uploader("📤 Upload Profile Image", type=["jpg", "png", "jpeg"])

if cnic_file and profile_file:
    cnic_path = "uploaded_cnic.jpg"
    profile_path = "uploaded_profile.jpg"

    with open(cnic_path, "wb") as f:
        f.write(cnic_file.getbuffer())
    with open(profile_path, "wb") as f:
        f.write(profile_file.getbuffer())

    # Validate CNIC Image
    if not is_valid_cnic(cnic_path):
        st.error("❌ Invalid CNIC image! Please upload a valid CNIC with a readable ID number.")
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
                st.image(Image.open(profile_face_path), caption="Extracted Profile Face", use_column_width=True)
            with col2:
                st.image(Image.open(cnic_face_path), caption="Extracted CNIC Face", use_column_width=True)
            
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
    st.warning("⚠️ Please upload both images to proceed!")
