import os
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from streamlit_mic_recorder import mic_recorder
import firebase_admin
from firebase_admin import db as firebase_db
import io

# 1. Page layout configuration
st.set_page_config(page_title="Acoustic Diagnostic Lab", page_icon="🫀", layout="centered")

# Visual framework styling matching your dark theme
st.markdown("""
    <style>
    .report-box {
        background-color: #000000;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-title { color: #888; font-size: 14px; font-weight: bold; letter-spacing: 1px; }
    .metric-value { font-size: 36px; font-weight: bold; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# Relative model location for Cloud Docker/Linux containers
MODEL_PATH = "83_desktop.h5"

# 2. Cached model loading optimization
@st.cache_resource
def load_acoustic_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error initializing neural network framework: {str(e)}")
            return None
    else:
        st.error(f"Error: Model file not found at root directory: {MODEL_PATH}")
        return None

# 3. Suppressed Firebase background initialization
try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app(options={
            'databaseURL': 'https://firebaseio.com'
        })
except Exception:
    pass

model = load_acoustic_model()

# 4. Corrected feature extraction pipeline math
def extract_3d_spectrogram(audio_data, sample_rate=4000, max_pad_len=128):
    try:
        rms = np.sqrt(np.mean(audio_data**2)) + 1e-6
        audio_data = audio_data / rms
        spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=sample_rate, n_mels=128, fmin=20, fmax=400, hop_length=64
        )
        static = librosa.power_to_db(spectrogram, ref=np.max)
        static_norm = (static - np.mean(static)) / (np.std(static) + 1e-4)
        delta = librosa.feature.delta(static_norm)
        delta_delta = librosa.feature.delta(static_norm, order=2)
        
        channels = [static_norm, delta, delta_delta]
        processed_channels = []
        for ch in channels:
            # FIX: explicit index added to shape query to prevent tuple vs int type exceptions
            if ch.shape[1] < max_pad_len:
                pad_width = max_pad_len - ch.shape[1]
                ch = np.pad(ch, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                ch = ch[:, :max_pad_len]
            processed_channels.append(ch)
        return np.stack(processed_channels, axis=-1)
    except Exception:
        return None

def compute_prediction(audio_arr, sr):
    features = extract_3d_spectrogram(audio_arr, sample_rate=sr)
    if features is None:
        return None
    test_batch = np.expand_dims(features, axis=0).astype(np.float32)
    prediction_raw = model.predict(test_batch, verbose=0)
    prediction = float(prediction_raw.item() if hasattr(prediction_raw, 'item') else prediction_raw)
    return prediction * 100

# --- USER INTERFACE FRONTEND ---
st.title("🔬 Acoustic Diagnostic Lab")
st.caption("Cloud Synced Real-Time Diagnostics Pipeline")

col1, col2 = st.columns(2)
audio_data_buffer = None
chosen_sr = 4000

# Left Input Block: Native Local PC Files
with col1:
    st.subheader("Process Files")
    uploaded_file = st.file_uploader("Upload .WAV File Natively from PC", type=["wav"])
    if uploaded_file is not None:
        audio_data_buffer, chosen_sr = librosa.load(uploaded_file, sr=4000)

# Right Input Block: Secure Browser Mic Capture
with col2:
    st.subheader("Live Signal Input")
    st.write("Capture Real-Time Beat via Browser Mic")
    audio_record = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="🛑 Stop & Process",
        just_once=False,
        key='recorder'
    )
    if audio_record is not None:
        raw_audio_bytes = audio_record['bytes']
        audio_data_buffer, chosen_sr = librosa.load(io.BytesIO(raw_audio_bytes), sr=4000)

# --- ENGINE OUTPUT VIEWPORT ---
st.markdown("<br>", unsafe_allow_html=True)

if model is None:
    st.warning("Awaiting Model Path Verification...")
elif audio_data_buffer is not None:
    with st.spinner("Computing sound maps and evaluating acoustic matrices..."):
        prob_percent = compute_prediction(audio_data_buffer, chosen_sr)
        
    if prob_percent is not None:
        if prob_percent > 50.0:
            verdict_msg = "Abnormal Murmur Signatures Detected. Review diagnostic data charts."
            verdict_color = "#FF3333" # Red text
        else:
            verdict_msg = "Normal acoustic envelope pacing profile registered."
            verdict_color = "#00FFFF" # Cyan text
            
        st.markdown(f"""
            <div class="report-box">
                <div class="metric-title">CLOUD SYNCED DIAGNOSTIC REPORT</div>
                <div class="metric-value">Probability: {prob_percent:.2f}%</div>
                <div style="color: {verdict_color}; font-weight: bold; font-size: 15px;">{verdict_msg}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Feature processing failure across channels.")
else:
    st.markdown("""
        <div class="report-box">
            <div class="metric-title">CLOUD SYNCED DIAGNOSTIC REPORT</div>
            <div class="metric-value" style="color: #666;">Probability: --</div>
            <div style="color: #00FFFF; font-weight: bold; font-size: 15px;">System Idle: Awaiting acoustic sample injection.</div>
        </div>
    """, unsafe_allow_html=True)
