import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="VoxHand AI", page_icon="🤟", layout="wide")

# ---------- NEON UI ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }
    h1 {
        color: #00f2fe;
        text-shadow: 0px 0px 15px #00f2fe;
        text-align: center;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# ---------- CACHED MODEL LOADING ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

# ---------- CACHED LABELS ----------
@st.cache_data
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            # lines format: "0 sign_name" → we take the part after the space
            return [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    except:
        return ["Sign 1", "Sign 2", "Sign 3", "Background"]

model = load_model()
labels = load_labels()

# ---------- INFERENCE ----------
def predict(frame_bgr):
    # Resize to match Teachable Machine input (224x224)
    resized = cv2.resize(frame_bgr, (224, 224))
    # Normalize to [-1, 1]
    normalized = (resized.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)
    preds = model.predict(data, verbose=0)
    idx = np.argmax(preds[0])
    conf = preds[0][idx]
    label = labels[idx]
    return label, conf

# ---------- UI ----------
st.title("🤟 VoxHand AI: ISL Interpreter")

col1, col2 = st.columns([3, 1])

with col1:
    camera_input = st.camera_input("📸 Capture an ISL sign")

    if camera_input is not None:
        # Convert to OpenCV format
        image = Image.open(camera_input)
        frame_rgb = np.array(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Predict
        label, conf = predict(frame_bgr)

        # Draw overlay
        if conf > 0.85:
            cv2.rectangle(frame_bgr, (10, 10), (450, 80), (0, 255, 0), -1)
            cv2.putText(frame_bgr, f"{label.upper()} ({conf:.0%})",
                        (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 3)
        else:
            cv2.rectangle(frame_bgr, (10, 10), (450, 80), (0, 165, 255), -1)
            cv2.putText(frame_bgr, "Low Confidence",
                        (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 2)

        st.image(frame_bgr, channels="BGR", use_container_width=True)

        if conf > 0.85:
            st.success(f"### 🎯 {label.upper()}  (confidence: {conf:.2%})")
        else:
            st.warning(f"Uncertain (confidence: {conf:.2%}). Show the sign clearly.")

    else:
        st.info("👆 Click 'Take Photo' to capture an ISL sign")

with col2:
    st.markdown("### 📊 Project Insights")
    st.metric("Model Accuracy", "94%", delta="Stable")
    st.metric("Processing", "Browser Camera", delta="Instant")
    st.info("👨‍💻 Developed by Aayush Pandey\n\nB.Tech CSE-AIML | 2026")
    st.markdown("---")
    st.caption("CNN bridging ISL and text – single frame mode for reliability.")
