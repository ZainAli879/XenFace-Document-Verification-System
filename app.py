import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import time
from deepface import DeepFace
from PIL import Image
from mtcnn import MTCNN
import threading

# ===================== 📌 CONFIGURE STREAMLIT THEME =====================
st.set_page_config(page_title="XenFace - Document Verification", page_icon="🔍", layout="wide")

# ===================== 📌 CACHE EASYOCR READER =====================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])
reader = get_ocr_reader()

# ===================== 📌 FUNCTION: Resize Image =====================
def resize_image(image_path, max_size=800):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(image_path, img)
    return image_path

# ===================== 📌 FUNCTION: Validate CNIC Image =====================
def is_cnic_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False, "❌ Error: Unable to read image!"
    
    results = reader.readtext(img)
    extracted_text = " ".join([res[1] for res in results])
    cnic_pattern = r"\b\d{5}-\d{7}-\d\b"
    return bool(re.search(cnic_pattern, extracted_text)), "❌ No valid CNIC detected!" if not re.search(cnic_pattern, extracted_text) else None

# ===================== 📌 FUNCTION: Blur CNIC Text (Parallel) =====================
def blur_cnic_text(image_path, output_name="blurred_cnic.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Error: CNIC Image not found!"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    
    for bbox, _, _ in results:
        x_min, y_min = map(int, bbox[0])
        x_max, y_max = map(int, bbox[2])
        img[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(img[y_min:y_max, x_min:x_max], (15, 15), 10)
    
    cv2.imwrite(output_name, img)
    return output_name, None

# ===================== 📌 FUNCTION: Extract Face (Using MTCNN) =====================
def extract_face(image_path, output_name="face.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return None, "❌ Error: Image not found!"
    
    detector = MTCNN()
    faces = detector.detect_faces(img)
    
    if len(faces) == 0:
        return None, "⚠️ No face detected! Upload a clear image."
    elif len(faces) > 1:
        return None, "⚠️ Multiple faces detected! Please upload an image with only one face."
    
    x, y, w, h = faces[0]['box']
    face = img[y:y+h, x:x+w]
    cv2.imwrite(output_name, face)
    return output_name, None

# ===================== 📌 FUNCTION: Verify Faces =====================
def verify_faces(img1_path, img2_path, threshold=0.66):
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name="ArcFace", detector_backend="opencv")
        result["verified"] = result["distance"] <= threshold
        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 XenFace - Document Verification System")
st.write("Upload your **CNIC image** and **profile picture** to verify identity.")

st.sidebar.header("Settings")
enable_cnic_crop = st.sidebar.checkbox("Enable CNIC Face Cropping", value=True)
enable_cnic_blur = st.sidebar.checkbox("Blur CNIC Text Information", value=True)

col1, col2 = st.columns(2)
with col1:
    cnic_file = st.file_uploader("📄 Upload CNIC Image", type=["jpg", "png", "jpeg"])
with col2:
    profile_file = st.file_uploader("📄 Upload Profile Image", type=["jpg", "png", "jpeg"])

if cnic_file and profile_file:
    cnic_path, profile_path = "uploaded_cnic.jpg", "uploaded_profile.jpg"
    with open(cnic_path, "wb") as f: f.write(cnic_file.getbuffer())
    with open(profile_path, "wb") as f: f.write(profile_file.getbuffer())
    
    resize_image(cnic_path)
    resize_image(profile_path)

    with st.spinner("Processing images..."):
        is_valid_cnic, cnic_error = is_cnic_image(cnic_path)
        is_valid_profile, profile_error = is_cnic_image(profile_path)
        
        if not is_valid_cnic:
            st.error(cnic_error)
        elif is_valid_profile:
            st.error("❌ Profile picture cannot be a CNIC image!")
        else:
            cnic_face_path, cnic_blur_path = "cnic_face.jpg", "blurred_cnic.jpg"
            profile_face_path = "profile_face.jpg"
            
            threads = []
            if enable_cnic_crop:
                t1 = threading.Thread(target=extract_face, args=(cnic_path, cnic_face_path))
                threads.append(t1)
                t1.start()
            if enable_cnic_blur:
                t2 = threading.Thread(target=blur_cnic_text, args=(cnic_path, cnic_blur_path))
                threads.append(t2)
                t2.start()
            
            for t in threads:
                t.join()

            profile_face_path, profile_error = extract_face(profile_path, "profile_face.jpg")
            if profile_error:
                st.error(profile_error)
            else:
                st.subheader("📷 Processed Face Images")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(profile_face_path, caption="Profile Picture")
                with col2:
                    st.image(cnic_face_path, caption="Processed CNIC Image")
                
                if st.button("🔍 Start Verification"):
                    with st.spinner("Verifying faces..."):
                        result, verify_error = verify_faces(profile_face_path, cnic_face_path)
                        time.sleep(2)
                    
                    if verify_error:
                        st.error(verify_error)
                    else:
                        st.subheader("✅ Verification Result")
                        st.markdown(f"### {'✅ Identity Verified!' if result['verified'] else '⚠️ Identity Mismatch!'}")
                        st.write(f"**Distance Score:** {result['distance']:.4f}")
