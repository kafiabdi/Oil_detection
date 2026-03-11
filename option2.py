import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

# ----------------------------
# CONFIGURATION
# ----------------------------

st.set_page_config(
    page_title="AI Oil Spill Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (256, 256)
MODEL_PATH = "unet_final (1).h5"

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource(show_spinner=True)
def load_model():
    # No custom_objects needed for plain Keras U-Net
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ----------------------------
# SIDEBAR UI
# ----------------------------

sidebar = st.sidebar
sidebar.title("🔧 Settings")

with sidebar.expander("⚙️ Inference Options"):
    threshold = st.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.01)
    overlay_alpha = st.slider("Overlay Transparency", 0.1, 1.0, 0.65, 0.05)

with sidebar.expander("🎨 Theme"):
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] {background-color: #001f3f;}
        </style>
        """,
        unsafe_allow_html=True,
    )
st.markdown("---")

sidebar.markdown("Need help? 📖")
sidebar.info("Upload an image → adjust threshold → explore heatmap/overlay")

# ----------------------------
# FILE UPLOADER
# ----------------------------

st.title("🌊 AI Oil Spill Detection")
st.markdown(
    "Upload a marine or aerial image and let the model highlight possible oil spill regions."
)

uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# ----------------------------
# UTILS
# ----------------------------

def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_mask(image):
    tensor = preprocess(image)
    pred = model.predict(tensor)[0, :, :, 0]
    return pred

# ----------------------------
# PROCESS AND SHOW
# ----------------------------

if uploaded_files:
    tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_files))])

    for tab, file in zip(tabs, uploaded_files):
        with tab:
            image = Image.open(file).convert("RGB")
            st.subheader("Original Input")

            pred = predict_mask(image)
            mask_binary = (pred > threshold).astype(np.uint8)

            mask_rs = cv2.resize(mask_binary, image.size)

            overlay = np.array(image).copy()
            overlay[mask_rs == 1] = [255, 0, 0]
            blended = cv2.addWeighted(np.array(image), 1 - overlay_alpha, overlay, overlay_alpha, 0)

            st.markdown("#### 🔁 Before/After Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                st.image(blended, caption="Overlay", use_column_width=True)

            st.markdown("---")

            if st.checkbox("🔥 Show Confidence Heatmap", key=f"heat_{file.name}"):
                fig, ax = plt.subplots()
                ax.imshow(pred, cmap="inferno")
                ax.axis("off")
                st.pyplot(fig)

            st.markdown("### 📊 Prediction Stats")
            oil_pct = (np.sum(mask_binary) / mask_binary.size) * 100
            st.metric("Oil Coverage (%)", f"{oil_pct:.2f}")
            st.metric("Mean Confidence", f"{np.mean(pred):.3f}")
            st.metric("Max Confidence", f"{np.max(pred):.3f}")

            buffer = io.BytesIO()
            Image.fromarray(blended).save(buffer, format="PNG")
            buffer.seek(0)

            st.download_button(
                label="⬇️ Download Overlay Result",
                data=buffer,
                file_name=f"segmented_{file.name}",
                mime="image/png",
            )

else:
    st.info("📂 Upload images to begin oil spill detection.")

st.markdown(
    """<div style="text-align: center; font-size: 14px;">
       ⚡ Built with Streamlit & TensorFlow | AI Oil Spill Detection Demo
       </div>
    """,
    unsafe_allow_html=True,
)
