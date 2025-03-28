import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import tempfile

# ===================== 📌 CONFIGURE STREAMLIT =====================
st.set_page_config(page_title="XenFace - Document Verification", page_icon="🔍", layout="wide")

# ===================== 📌 CACHED MODELS =====================
@st.cache_resource
def load_deepface():
    return DeepFace.build_model("ArcFace")

deepface_model = load_deepface()

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'])

easyocr_reader = load_easyocr()

# ===================== 📌 FUNCTION: Validate CNIC Image =====================
def is_cnic_image(image):
    img = np.array(image)
    results = easyocr_reader.readtext(img)

    extracted_text = " ".join([res[1] for res in results])
    cnic_pattern = r"\b\d{5}-\d{7}-\d\b"

    return (True, None) if re.search(cnic_pattern, extracted_text) else (False, "❌ No valid CNIC image detected!")

# ===================== 📌 FUNCTION: Extract Face =====================
def extract_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, "⚠️ No face detected!"
    elif len(faces) > 1:
        return None, "⚠️ Multiple faces detected! Please upload an image with only one face."
    
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return Image.fromarray(face), None

# ===================== 📌 FUNCTION: Verify Faces =====================
def verify_faces(img1, img2, threshold=0.66):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp1:
            img1.save(temp1.name)
            img1_path = temp1.name
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp2:
            img2.save(temp2.name)
            img2_path = temp2.name

        result = DeepFace.verify(img1_path, img2_path, model_name="ArcFace", detector_backend="opencv")
        distance = result["distance"]
        result["verified"] = distance <= threshold
        result["threshold"] = threshold
        result["similarity_score"] = round((1 - distance) * 100, 2)
        return result, None
    except Exception as e:
        return None, f"❌ Error during verification: {str(e)}"

# ===================== 📌 STREAMLIT UI =====================
st.title("🔍 XenFace - Document Verification System")
st.write("Upload your **CNIC image** and **profile picture** to verify identity.")

# 📌 Sidebar Settings
st.sidebar.header("Settings")
enable_cnic_crop = st.sidebar.checkbox("Enable CNIC Face Cropping", value=True)

# 📌 File Uploaders
col1, col2 = st.columns(2)
with col1:
    cnic_file = st.file_uploader("📄 Upload CNIC Image", type=["jpg", "png", "jpeg"])
with col2:
    profile_file = st.file_uploader("📄 Upload Profile Image", type=["jpg", "png", "jpeg"])

if cnic_file and profile_file:
    cnic_image = Image.open(cnic_file).convert("RGB")
    profile_image = Image.open(profile_file).convert("RGB")

    with st.spinner("Processing images..."):
        is_valid_cnic, cnic_error = is_cnic_image(cnic_image)
        is_valid_profile, profile_error = is_cnic_image(profile_image)

        if not is_valid_cnic:
            st.error(cnic_error)
        elif is_valid_profile:
            st.error("❌ Profile picture cannot be a CNIC image! Upload a real profile photo.")
        else:
            if enable_cnic_crop:
                cnic_face, cnic_error = extract_face(cnic_image)
                if cnic_error:
                    st.error(cnic_error)
                else:
                    cnic_image = cnic_face
            
            profile_face, profile_error = extract_face(profile_image)
            if profile_error:
                st.error(profile_error)
            else:
                st.subheader("📷 Processed Face Images")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(profile_face, caption="Profile Picture", use_container_width=True)
                with col2:
                    st.image(cnic_image, caption="Processed CNIC Image", use_container_width=True)

                with st.form(key="verify_form"):
                    submit = st.form_submit_button("🔍 Start Verification")
                    if submit:
                        with st.spinner("Verifying faces..."):
                            result, verify_error = verify_faces(profile_face, cnic_image)
                        
                        if verify_error:
                            st.error(verify_error)
                        else:
                            st.subheader("✅ Verification Result")
                            st.markdown(f"### {'✅ Identity Verified! Your documents are successfully verified.' if result['verified'] else '⚠️ Identity Mismatch! Please upload your original documents.'}")
                            st.write(f"**Similarity Score:** {result['similarity_score']}%")
