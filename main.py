import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="🌽 Corn Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-text {
        color: black;
        font-weight: bold;
    }
    .frame-card {
        background-color: #f9f9f9;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Class names and descriptions
CLASS_NAMES = ['Grey Leaf Spot', 'Corn Rust', 'Leaf Blight']
CLASS_DESCRIPTIONS = {
    'Grey Leaf Spot': {
        'description': 'Penyakit yang disebabkan oleh jamur Cercospora zeae-maydis',
        'symptoms': 'Bercak abu-abu persegi panjang pada daun',
        'treatment': 'Fungisida berbahan aktif strobilurin atau triazole'
    },
    'Corn Rust': {
        'description': 'Penyakit yang disebabkan oleh jamur Puccinia sorghi',
        'symptoms': 'Pustula karat berwarna coklat kemerahan pada daun',
        'treatment': 'Fungisida berbahan aktif triazole atau strobilurin'
    },
    'Leaf Blight': {
        'description': 'Penyakit yang disebabkan oleh jamur Exserohilum turcicum',
        'symptoms': 'Bercak coklat memanjang dengan bentuk elips pada daun',
        'treatment': 'Fungisida berbahan aktif mancozeb atau chlorothalonil'
    }
}

@st.cache_resource
def load_onnx_model():
    """Load ONNX model with caching"""
    model_path = os.path.join('../models', 'corn_disease_model.onnx')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers():
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image_array = np.array(image)
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    image_resized = cv2.resize(image_array, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def predict_disease(image, session):
    """Predict disease from image using ONNX model"""
    try:
        processed_image = preprocess_image(image)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        predictions = session.run([output_name], {input_name: processed_image})[0]
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        all_probabilities = predictions[0]
        
        return predicted_class_idx, confidence, all_probabilities
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# ── NEW ──────────────────────────────────────────────────────────────────────
def extract_frames_from_video(video_path, target_fps=20):
    """
    Extract 1 frame per second from a video.
    Assumes the video runs at `target_fps` frames per second;
    falls back to the actual FPS reported by OpenCV when available.

    Returns a list of tuples: (second_index, rgb_frame_as_numpy_array)
    """
    cap = cv2.VideoCapture(video_path)

    # Prefer the real FPS from the file; use target_fps as the fallback
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(actual_fps)) if actual_fps > 0 else target_fps

    frames = []
    frame_idx = 0
    second_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Keep only the first frame of every second-long window
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((second_idx, frame_rgb))
            second_idx += 1

        frame_idx += 1

    cap.release()
    return frames
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<h1 class="main-header">🌽 Corn Disease Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Information")
        st.info("""
        **How to use:**
        1. Upload a video of corn leaf
        2. Wait for the analysis
        3. View the prediction results per second
        
        **Supported formats:**
        - MP4, AVI, MOV, MKV
        - Max size: 200MB
        """)
        
        st.header("🔬 Model Info")
        st.write("**Model Type:** Custom CNN")
        st.write("**Framework:** ONNX Runtime")
        st.write("**Classes:** 3 diseases")
        st.write("**Accuracy:** 98.60%")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Video")

        # ── CHANGED: video uploader instead of image uploader ────────────────
        uploaded_file = st.file_uploader(
            "Choose a video of corn leaf",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a clear video of corn leaf for disease detection"
        )

        if uploaded_file is not None:
            # Show the video player
            st.video(uploaded_file)
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
        # ─────────────────────────────────────────────────────────────────────

    with col2:
        st.header("🔍 Prediction Results")

        if uploaded_file is not None:
            # Load model
            with st.spinner("Loading model..."):
                session = load_onnx_model()

            # ── CHANGED: save video to temp file, extract frames, predict ────
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Extracting frames from video…"):
                    frames = extract_frames_from_video(tmp_path, target_fps=20)

                if not frames:
                    st.error("❌ Could not extract any frames from the video.")
                else:
                    st.write(f"**Total seconds analysed:** {len(frames)}")

                    all_predictions = []   # collect class names for summary

                    progress = st.progress(0, text="Analysing frames…")
                    for i, (second_idx, frame_rgb) in enumerate(frames):
                        pil_image = Image.fromarray(frame_rgb)
                        pred_idx, confidence, probabilities = predict_disease(
                            pil_image, session
                        )

                        if pred_idx is not None:
                            predicted_class = CLASS_NAMES[pred_idx]
                            all_predictions.append(predicted_class)
                            conf_color = get_confidence_color(confidence)

                            # Per-second result card
                            with st.expander(
                                f"⏱ Second {second_idx + 1} — {predicted_class} "
                                f"({confidence:.2%})",
                                expanded=False
                            ):
                                img_col, info_col = st.columns([1, 1])
                                with img_col:
                                    st.image(
                                        pil_image,
                                        caption=f"Frame at second {second_idx + 1}",
                                        use_container_width=True
                                    )
                                with info_col:
                                    st.markdown(
                                        f'<p class="{conf_color}">'
                                        f"Confidence: {confidence:.2%}</p>",
                                        unsafe_allow_html=True
                                    )
                                    st.bar_chart(
                                        {"Disease": CLASS_NAMES,
                                         "Probability": list(probabilities)},
                                        x="Disease", y="Probability"
                                    )

                        progress.progress(
                            (i + 1) / len(frames),
                            text=f"Analysing second {second_idx + 1} of {len(frames)}…"
                        )

                    progress.empty()

                    # ── Summary: most common prediction ──────────────────────
                    if all_predictions:
                        most_common_class, count = Counter(
                            all_predictions
                        ).most_common(1)[0]
                        st.markdown("---")
                        st.subheader("📊 Overall Summary")
                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <h3 class="prediction-text">
                                    🎯 Dominant Disease: {most_common_class}
                                </h3>
                                <p>Detected in <b>{count}</b> out of
                                <b>{len(all_predictions)}</b> seconds
                                ({count / len(all_predictions):.2%})</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        disease_info = CLASS_DESCRIPTIONS[most_common_class]
                        st.write(
                            f"**Description:** {disease_info['description']}"
                        )
                        st.write(
                            f"**Symptoms:** {disease_info['symptoms']}"
                        )
                        st.write(
                            f"**Treatment:** {disease_info['treatment']}"
                        )
            finally:
                os.unlink(tmp_path)   # clean up temp file
            # ─────────────────────────────────────────────────────────────────

        else:
            st.info("👆 Please upload a video to start analysis")

    # Additional information
    st.markdown("---")
    st.header("📚 About the Diseases")
    
    cols = st.columns(3)
    for i, (disease, info) in enumerate(CLASS_DESCRIPTIONS.items()):
        with cols[i]:
            st.subheader(f"🦠 {disease}")
            st.write(f"**Cause:** {info['description']}")
            st.write(f"**Symptoms:** {info['symptoms']}")
            st.write(f"**Treatment:** {info['treatment']}")

if __name__ == "__main__":
    main()
