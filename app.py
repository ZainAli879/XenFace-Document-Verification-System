import streamlit as st
import cv2
import numpy as np
import time
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

# ===================== 📌 CONFIGURE STREAMLIT =====================
st.set_page_config(page_title="XenFace - Document Verification", page_icon="🔍", layout="wide")

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
        return None, "⚠️ Multiple faces detected! Upload an image with only one face."

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    cv2.imwrite(output_name, face)
    return output_name, None

# ===================== 📌 FUNCTION: Add Watermark =====================
def add_watermark(image_path, output_name="watermarked_cnic.jpg", text="XenFace Secure"):
    try:
        img = Image.open(image_path).convert("RGBA")
        watermark = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)
        font = ImageFont.load_default()
        draw.text((10, img.size[1] - 30), text, fill=(255, 255, 255, 128), font=font)
        img = Image.alpha_composite(img, watermark)
        img.convert("RGB").save(output_name)
        return output_name
    except Exception:
        return None  # Prevent crashing

# ===================== 📌 FUNCTION: Resize Image =====================
def resize_image(image_path, output_name="resized.jpg"):
    try:
        img = Image.open(image_path).resize((250, 250))
        img.save(output_name)
        return output_name
    except Exception:
        return None  # Prevent crashing

# ===================== 📌 FUNCTION: Verify Faces =====================
def verify_faces(img1_path, img2_path, model_name="ArcFace", detector_backend="mtcnn", threshold=0.66):
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name=model_name, detector_backend=detector_backend)
        result["verified"] = result["distance"] <= threshold
        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 XenFace - Document Verification System")
st.write("Upload your **CNIC image** and **profile picture** to verify identity.")

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
        # Face extraction
        profile_path, profile_error = extract_single_face(profile_path, "profile_face.jpg")
        cnic_path, cnic_error = extract_single_face(cnic_path, "cnic_face.jpg")

        # Display errors & prevent processing invalid images
        if profile_error:
            st.error(profile_error)
        if cnic_error:
            st.error(cnic_error)

        if not profile_path or not cnic_path:
            st.warning("⚠️ Cannot proceed due to face detection errors.")
        else:
            # Watermark & Resize (Only for valid images)
            cnic_path = add_watermark(cnic_path, "watermarked_cnic.jpg") if cnic_path else None
            resized_profile_path = resize_image(profile_path, "resized_profile.jpg")
            resized_cnic_path = resize_image(cnic_path, "resized_cnic.jpg")

            # Display Processed Images
            if resized_profile_path and resized_cnic_path:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_profile_path, caption="Profile Picture", use_container_width=True)
                with col2:
                    st.image(resized_cnic_path, caption="Processed CNIC Image", use_container_width=True)

                if st.button("🔍 Start Verification"):
                    with st.spinner("Verifying faces..."):
                        result, verify_error = verify_faces(resized_profile_path, resized_cnic_path)
                        time.sleep(2)

                    if verify_error:
                        st.error(verify_error)
                    else:
                        st.subheader("✅ Verification Result")
                        st.markdown(f"### {'✅ Identity Verified!' if result['verified'] else '⚠️ Identity Mismatch!'}")
            else:
                st.warning("⚠️ Verification skipped due to image errors.")
else:
    st.warning("⚠️ Please upload both images to proceed!")
