import streamlit as st
import cv2
import numpy as np
import os
import time
import easyocr
import re
from deepface import DeepFace
from PIL import Image

# ===================== 📌 CONFIGURE STREAMLIT THEME =====================
st.set_page_config(page_title="XenFace - Document Verification", page_icon="🔍", layout="wide")

# ===================== 📌 FUNCTION: Validate CNIC Image =====================
def detect_cnic_template(image_path, template_path):
    """
    Detects CNIC by matching a predefined CNIC template using OpenCV.
    """
    img = cv2.imread(image_path, 0)  # Load in grayscale
    template = cv2.imread(template_path, 0)  # Load CNIC template

    if img is None or template is None:
        return False, "❌ Error: Unable to read image or template!"

    # Match template
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > 0.8:  # Threshold for match confidence
        return True, None
    else:
        return False, "❌ No valid CNIC detected!"

# ===================== 📌 FUNCTION: Blur CNIC Text =====================
def blur_cnic_text(image_path, output_name="blurred_cnic.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Error: CNIC Image not found!"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        x_max = int(max(top_right[0], bottom_right[0]))
        y_max = int(max(bottom_left[1], bottom_right[1]))

        img[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(img[y_min:y_max, x_min:x_max], (15, 15), 10)

    cv2.imwrite(output_name, img)
    return output_name, None

# ===================== 📌 FUNCTION: Extract Face =====================
def extract_face(image_path, output_name="face.jpg"):
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
    cv2.imwrite(output_name, face)
    return output_name, None

# ===================== 📌 FUNCTION: Verify Faces =====================
def verify_faces(img1_path, img2_path, threshold=0.66):
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name="ArcFace", detector_backend="opencv")
        result["verified"] = result["distance"] <= threshold
        result["threshold"] = threshold
        result["similarity_score"] = 1 - result["distance"]
        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 XenFace - Document Verification System")
st.write("Upload your **CNIC image** and **profile picture** to verify identity.")

# 📌 Sidebar
st.sidebar.header("Settings")
enable_cnic_crop = st.sidebar.checkbox("Enable CNIC Face Cropping", value=True)
enable_cnic_blur = st.sidebar.checkbox("Blur CNIC Text Information", value=True)

st.sidebar.header("How to Use XenFace")
st.sidebar.markdown("""
### 1️⃣ Upload CNIC Image
- Select a valid CNIC image containing a clear number.

### 2️⃣ Upload Profile Image
- Choose a clear profile picture for comparison.

### 3️⃣ Enable/Disable Options
- Toggle CNIC face cropping and text blurring.

### 4️⃣ View Processed Images
- Processed images will be displayed before verification.

### 5️⃣ Start Verification
- Click the button to verify identity.
""")

# 📌 File Uploaders
col1, col2 = st.columns(2)
with col1:
    cnic_file = st.file_uploader("📄 Upload CNIC Image Only", type=["jpg", "png", "jpeg"])
with col2:
    profile_file = st.file_uploader("📄 Upload Profile Image", type=["jpg", "png", "jpeg"])

if cnic_file and profile_file:
    cnic_path, profile_path = "uploaded_cnic.jpg", "uploaded_profile.jpg"
    with open(cnic_path, "wb") as f: f.write(cnic_file.getbuffer())
    with open(profile_path, "wb") as f: f.write(profile_file.getbuffer())

    with st.spinner("Processing images..."):
        is_valid_cnic, cnic_error = is_cnic_image(cnic_path)
        is_valid_profile, profile_error = is_cnic_image(profile_path)

        if not is_valid_cnic:
            st.error(cnic_error)
        elif is_valid_profile:
            st.error("❌ Profile picture cannot be a CNIC image! Please upload a real profile photo.")
        else:
            if enable_cnic_crop:
                cnic_path, cnic_error = extract_face(cnic_path, "cnic_face.jpg")
                if cnic_error:
                    st.error(cnic_error)

            if enable_cnic_blur:
                cnic_path, _ = blur_cnic_text(cnic_path, "blurred_cnic.jpg")

            profile_path, profile_error = extract_face(profile_path, "profile_face.jpg")
            if profile_error:
                st.error(profile_error)
            else:
                st.subheader("📷 Processed Face Images")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(profile_path, caption="Profile Picture", use_container_width=True)
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
                        st.markdown(f"### {'✅ Identity Verified! Congrats Your Documents are successfully verified' if result['verified'] else '⚠️ Identity Mismatch!Please Upload Your Original Documents'}")
                        st.write(f"**Distance Score:** {result['distance']:.4f}")
