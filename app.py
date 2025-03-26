import streamlit as st
import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

# ===================== 📌 CONFIGURE STREAMLIT THEME =====================
st.set_page_config(page_title="XenFace - Document Verification", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
        .big-font {font-size:20px !important; font-weight: bold; color: #4CAF50;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; border: none; padding: 12px; width: 100%;}
        .stSpinner {color: #4CAF50 !important;}
        .sidebar .sidebar-content {position: fixed; width: 300px;}
        .stFileUploader>label {font-size: 16px !important; font-weight: bold;}
        .stFileUploader div {max-width: 90%;} /* Reduce upload field size */
    </style>
""", unsafe_allow_html=True)

# ===================== 📌 FUNCTION: Extract & Validate Single Face =====================
def extract_single_face(image_path, output_name="cropped_face.jpg"):
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

# ===================== 📌 FUNCTION: Blur CNIC Text Details =====================
def blur_cnic_text(image_path, output_name="blurred_cnic.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Error: CNIC Image not found!"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 10)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 255  

    img = np.where(mask[:, :, None] == 255, img, cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(output_name, img)
    return output_name, None

# ===================== 📌 FUNCTION: Add Watermark =====================
def add_watermark(image_path, output_name="watermarked_cnic.jpg", text="XenFace Secure"):
    img = Image.open(image_path).convert("RGBA")
    watermark = Image.new("RGBA", img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(watermark)
    font = ImageFont.load_default()

    text_position = (10, img.size[1] - 30)
    draw.text(text_position, text, fill=(255, 255, 255, 128), font=font)

    img = Image.alpha_composite(img, watermark)
    img.convert("RGB").save(output_name)
    return output_name

# ===================== 📌 FUNCTION: Resize Image =====================
def resize_image(image_path, output_name="resized.jpg"):
    img = Image.open(image_path).resize((250, 250))
    img.save(output_name)
    return output_name

# ===================== 📌 FUNCTION: Verify Faces =====================
def verify_faces(img1_path, img2_path, model_name="ArcFace", detector_backend="mtcnn", threshold=0.66):
    try:
        img1_path = resize_image(img1_path, "resized_profile.jpg")
        img2_path = resize_image(img2_path, "resized_cnic.jpg")
        result = DeepFace.verify(img1_path, img2_path, model_name=model_name, detector_backend=detector_backend)
        result["verified"] = result["distance"] <= threshold
        result["threshold"] = threshold
        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 XenFace - Document Verification System")
st.write("Upload your **CNIC image** and **profile picture** to verify identity.")

# 📌 Sidebar (Fixed and Fully Visible)
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Face Recognition Model", ["ArcFace", "VGG-Face", "Facenet"])
enable_watermark = st.sidebar.checkbox("Enable Watermarking", value=True)
enable_blur = st.sidebar.checkbox("Blur CNIC Text Details", value=True)
enable_face_crop = st.sidebar.checkbox("Enable Face Cropping", value=True)

st.sidebar.subheader("ℹ️ How It Works?")
st.sidebar.write("""
1️⃣ Upload your **CNIC Image** & **Profile Picture**  
2️⃣ The system extracts & enhances your face  
3️⃣ CNIC text can be blurred, and watermark added  
4️⃣ Your identity is verified with AI-powered face matching  
""")

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
        if enable_face_crop:
            profile_path, profile_error = extract_single_face(profile_path, "profile_face.jpg")
            cnic_path, cnic_error = extract_single_face(cnic_path, "cnic_face.jpg")

            if profile_error:
                st.error(profile_error)
            if cnic_error:
                st.error(cnic_error)

        if enable_blur:
            cnic_path, _ = blur_cnic_text(cnic_path, "blurred_cnic.jpg")

        if enable_watermark:
            cnic_path = add_watermark(cnic_path, "watermarked_cnic.jpg")

        resized_cnic_path = resize_image(cnic_path, "resized_cnic.jpg")
        resized_profile_path = resize_image(profile_path, "resized_profile.jpg")
        time.sleep(1)

    st.subheader("📷 Processed Face Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_profile_path, caption="Profile Picture", use_container_width=True)
    with col2:
        st.image(resized_cnic_path, caption="Processed CNIC Image", use_container_width=True)

    if st.button("🔍 Start Verification"):
        with st.spinner("Verifying faces..."):
            result, verify_error = verify_faces(resized_profile_path, resized_cnic_path, model_choice)
            time.sleep(2)

        if verify_error:
            st.error(verify_error)
        else:
            st.subheader("✅ Verification Result")
            st.markdown(f"### {'✅ Identity Verified!' if result['verified'] else '⚠️ Identity Mismatch!'}")
else:
    st.warning("⚠️ Please upload both images to proceed!")
