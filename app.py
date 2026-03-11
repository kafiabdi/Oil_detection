import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ----------------------------
# SIDEBAR
# ----------------------------

sidebar = st.sidebar
sidebar.title("🔧 Settings")

with sidebar.expander("⚙️ Inference Options"):
    threshold = st.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.01)
    overlay_alpha = st.slider("Overlay Transparency", 0.1, 1.0, 0.65, 0.05)

sidebar.info("Upload image → Adjust threshold → View metrics")

# ----------------------------
# MAIN UI
# ----------------------------

st.title("🌊 AI Oil Spill Detection System")
st.markdown(
    "Upload a marine or aerial image and detect possible oil spill regions using U-Net segmentation."
)

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

gt_file = st.file_uploader(
    "Upload Ground Truth Mask (Optional - for Performance Metrics)",
    type=["jpg", "jpeg", "png"],
)

# ----------------------------
# UTILITIES
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
# PROCESSING
# ----------------------------

if uploaded_files:

    tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_files))])

    for tab, file in zip(tabs, uploaded_files):
        with tab:

            image = Image.open(file).convert("RGB")
            pred = predict_mask(image)

            mask_binary = (pred > threshold).astype(np.uint8)
            mask_resized = cv2.resize(mask_binary, image.size)

            # Create overlay
            overlay = np.array(image).copy()
            overlay[mask_resized == 1] = [255, 0, 0]
            blended = cv2.addWeighted(
                np.array(image), 1 - overlay_alpha,
                overlay, overlay_alpha, 0
            )

            # ----------------------------
            # BEFORE / AFTER
            # ----------------------------

            st.subheader("🔁 Before / After Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original", use_column_width=True)

            with col2:
                st.image(blended, caption="Detected Oil Spill Overlay", use_column_width=True)

            # ----------------------------
            # HEATMAP
            # ----------------------------

            if st.checkbox("🔥 Show Confidence Heatmap", key=file.name):
                fig_heat, ax_heat = plt.subplots(figsize=(4, 4))
                ax_heat.imshow(pred, cmap="inferno")
                ax_heat.axis("off")
                st.pyplot(fig_heat)

            # ----------------------------
            # BASIC STATS
            # ----------------------------

            st.markdown("### 📊 Prediction Statistics")

            oil_pct = (np.sum(mask_binary) / mask_binary.size) * 100
            st.metric("Oil Coverage (%)", f"{oil_pct:.2f}")
            st.metric("Mean Confidence", f"{np.mean(pred):.3f}")
            st.metric("Max Confidence", f"{np.max(pred):.3f}")

            # ----------------------------
            # PERFORMANCE METRICS
            # ----------------------------

            if gt_file is not None:

                gt_image = Image.open(gt_file).convert("L")
                gt_image = gt_image.resize(image.size)
                gt_array = np.array(gt_image)
                gt_binary = (gt_array > 127).astype(np.uint8)

                y_true = gt_binary.flatten()
                y_pred = mask_resized.flatten()

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                st.markdown("## 📈 Model Performance")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.4f}")
                c2.metric("Precision", f"{prec:.4f}")
                c3.metric("Recall", f"{rec:.4f}")
                c4.metric("F1 Score", f"{f1:.4f}")

                # ----------------------------
                # Confusion Matrix
                # ----------------------------

                cm = confusion_matrix(y_true, y_pred)

                st.markdown("### 🔢 Confusion Matrix")

                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    ax=ax_cm,
                )

                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title("Confusion Matrix")

                st.pyplot(fig_cm)

                # ----------------------------
                # Accuracy Pie Chart
                # ----------------------------

                correct = np.sum(y_true == y_pred)
                incorrect = np.sum(y_true != y_pred)

                st.markdown("### 🥧 Accuracy Distribution")

                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                ax_pie.pie(
                    [correct, incorrect],
                    labels=["Correct", "Incorrect"],
                    autopct="%1.1f%%",
                )
                ax_pie.set_title("Prediction Accuracy")

                st.pyplot(fig_pie)

            # ----------------------------
            # DOWNLOAD RESULT
            # ----------------------------

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
    st.info("📂 Upload image(s) to begin detection.")

# ----------------------------
# FOOTER
# ----------------------------

st.markdown(
    """
    <div style="text-align: center; font-size: 14px;">
    ⚡ Built with Streamlit & TensorFlow | U-Net Oil Spill Detection
    </div>
    """,
    unsafe_allow_html=True,
)