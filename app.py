import streamlit as st
import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from PIL import Image, ImageEnhance

# ===================== 📌 CONFIGURE STREAMLIT THEME =====================
st.set_page_config(page_title="XenFace - Document Verification", page_icon="🔍", layout="wide")

# ===================== 📌 FUNCTION: Extract & Enhance Face =====================
def extract_and_enhance_face(image_path, output_name="enhanced_face.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Error: Image not found!"

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, "⚠️ No face detected! Upload a clear image."
    elif len(faces) > 1:
        return None, "⚠️ Multiple faces detected! Please upload an image with only one face."

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    enhanced_face = enhance_image_quality(face)
    cv2.imwrite(output_name, enhanced_face)
    return output_name, None

# ===================== 📌 FUNCTION: Enhance Image Quality =====================
def enhance_image_quality(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(pil_img)
    enhanced_img = enhancer.enhance(2.0)  # Enhance sharpness
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(1.5)  # Enhance contrast
    return cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)

# ===================== 📌 FUNCTION: Verify Faces =====================
def verify_faces(img1_path, img2_path, threshold=0.66):
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name="ArcFace", detector_backend="mtcnn")
        result["verified"] = result["distance"] <= threshold
        result["threshold"] = threshold
        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 XenFace - Document Verification System")
st.write("Upload your **CNIC image** and **profile picture** to verify identity.")

# 📌 Sidebar
st.sidebar.header("Settings")
enable_cnic_crop = st.sidebar.checkbox("Enable CNIC Face Cropping", value=True)

# 📌 File Uploaders
col1, col2 = st.columns(2)
with col1:
    cnic_file = st.file_uploader("📤 Upload CNIC Image", type=["jpg", "png", "jpeg"])
with col2:
    profile_file = st.file_uploader("📤 Upload Profile Image", type=["jpg", "png", "jpeg"])

if cnic_file and profile_file:
    cnic_path, profile_path = "uploaded_cnic.jpg", "uploaded_profile.jpg"
    with open(cnic_path, "wb") as f: f.write(cnic_file.getbuffer())
    with open(profile_path, "wb") as f: f.write(profile_file.getbuffer())

    with st.spinner("Processing images..."):
        profile_path, profile_error = extract_and_enhance_face(profile_path, "profile_face.jpg")
        if profile_error:
            st.error(profile_error)

        if enable_cnic_crop:
            cnic_path, cnic_error = extract_and_enhance_face(cnic_path, "cnic_face.jpg")
            if cnic_error:
                st.error(cnic_error)
        
    st.subheader("📷 Processed Face Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(profile_path, caption="Enhanced Profile Picture", use_container_width=True)
    with col2:
        st.image(cnic_path, caption="Processed CNIC Image", use_container_width=True)

    if st.button("🔍 Start Verification"):
        with st.spinner("Verifying faces..."):
            result, verify_error = verify_faces(profile_path, cnic_path)
            time.sleep(2)

        if verify_error:
            st.error(verify_error)
        else:
            st.subheader("✅ Verification Result")
            st.markdown(f"### {'✅ Identity Verified!' if result['verified'] else '⚠️ Identity Mismatch!'}")
else:
    st.warning("⚠️ Please upload both images to proceed!")
