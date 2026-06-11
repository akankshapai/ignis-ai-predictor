import sys
# sys.path.insert(0, r"C:\tflib")  # Local dev only — disabled for cloud deployment
# On Streamlit Community Cloud, TensorFlow is installed via requirements.txt

import streamlit as st
import numpy as np
from PIL import Image
import time

# TFLite interpreter — uses the lightweight tflite-runtime package on the cloud
# and falls back to the full tf.lite.Interpreter when running locally.
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    import sys
    sys.path.insert(0, r"C:\tflib")   # local dev path
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="IGNIS-AI: Satellite Wildfire Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — DARK COMMAND CENTER THEME
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global / Background ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0d14;
        color: #e8eaf0;
        font-family: 'Courier New', monospace;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d111e 0%, #10172a 100%);
        border-right: 1px solid #1f2d45;
    }
    [data-testid="stSidebar"] * { color: #a8b8d0 !important; }

    /* ── Main Title ── */
    .ignis-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #ff6b00, #ff2d55, #ff6b00);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer {
        to { background-position: 200% center; }
    }

    /* ── Subtitle / Tag line ── */
    .ignis-sub {
        color: #4a7fa5;
        font-size: 0.85rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        margin-top: -0.6rem;
        margin-bottom: 1.4rem;
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #1f2d45;
        margin: 1.2rem 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #0e1726 0%, #141f35 100%);
        border: 1px solid #1f3050;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        color: #5a7a9a;
        text-transform: uppercase;
    }
    .metric-value-green {
        font-size: 1.15rem;
        font-weight: 700;
        color: #00e676;
    }
    .metric-value-amber {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffa726;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #0e1726;
        border: 2px dashed #1f3a58;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #ff6b00;
    }

    /* ── FIRE ALERT banner ── */
    .alert-fire {
        background: linear-gradient(135deg, #2d0a00 0%, #1a0000 100%);
        border: 2px solid #ff2d55;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        color: #ff4466;
        font-size: 1.4rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-align: center;
        animation: pulse-red 1.2s ease-in-out infinite;
        text-shadow: 0 0 18px #ff2d55aa;
        box-shadow: 0 0 30px #ff2d5533;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px #ff2d5555; }
        50%       { box-shadow: 0 0 45px #ff2d55cc; }
    }

    /* ── CLEAR banner ── */
    .alert-clear {
        background: linear-gradient(135deg, #001a0d 0%, #00100a 100%);
        border: 2px solid #00e676;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        color: #00e676;
        font-size: 1.4rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-align: center;
        text-shadow: 0 0 14px #00e67688;
        box-shadow: 0 0 28px #00e67622;
    }

    /* ── Confidence bar wrapper ── */
    .conf-bar-bg {
        background: #0e1726;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    /* ── Image container ── */
    .img-frame {
        border: 1px solid #1f3050;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Spinner override ── */
    [data-testid="stSpinner"] { color: #ff6b00 !important; }

    /* ── Footer ── */
    .ignis-footer {
        color: #2a3a50;
        font-size: 0.7rem;
        text-align: center;
        letter-spacing: 0.15em;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# SIDEBAR — SYSTEM METRICS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ IGNIS-AI")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("**SYSTEM STATUS**")

    st.markdown(
        """
        <div class='metric-card'>
            <div class='metric-label'>Model Status</div>
            <div class='metric-value-green'>● ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='metric-card'>
            <div class='metric-label'>Detection Threshold</div>
            <div class='metric-value-amber'>Dynamic (Slider)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**🎚️ SENSITIVITY CONTROLS**")
    threshold = st.slider(
        "System Sensitivity Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        format="%.2f",
        help="Lower = more sensitive (catches more fires, more false positives). "
             "Higher = more conservative (fewer false positives, may miss weak signals).",
    )
    # Dynamic threshold indicator
    if threshold < 0.35:
        thresh_color = "#ff4466"
        thresh_label = "HIGH SENSITIVITY"
    elif threshold > 0.65:
        thresh_color = "#ffa726"
        thresh_label = "CONSERVATIVE"
    else:
        thresh_color = "#00e676"
        thresh_label = "BALANCED"
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>Active Threshold</div>
            <div style='color:{thresh_color}; font-size:1.4rem; font-weight:900;
                        letter-spacing:0.08em; margin-top:0.2rem;'>
                {threshold:.0%}
            </div>
            <div style='color:{thresh_color}; font-size:0.65rem;
                        letter-spacing:0.2em; margin-top:0.2rem;'>
                {thresh_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='metric-card'>
            <div class='metric-label'>Neural Architecture</div>
            <div class='metric-value-amber'>CNN · TensorFlow</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='metric-card'>
            <div class='metric-label'>Input Resolution</div>
            <div class='metric-value-amber'>150 × 150 px</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#2a3a50;font-size:0.7rem;letter-spacing:0.15em'>"
        "IGNIS-AI v1.0 · CLASSIFIED</span>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(
    "<div class='ignis-title'>🛰️ IGNIS-AI: Satellite Wildfire Predictor</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='ignis-sub'>Orbital · Deep Learning · Threat Classification System</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TFLITE MODEL LOADER (cached for performance)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the TFLite interpreter and allocate tensors once, then cache."""
    interpreter = TFLiteInterpreter(model_path="wildfire_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# ──────────────────────────────────────────────
# FILE UPLOADER
# ──────────────────────────────────────────────
st.markdown("### 📡 Upload Satellite Image")
uploaded_file = st.file_uploader(
    "Drag & drop a satellite image for analysis",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

# ──────────────────────────────────────────────
# ANALYSIS PIPELINE
# ──────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_img, col_result = st.columns([1, 1], gap="large")

    # ── Left: Image Display ──
    with col_img:
        st.markdown("#### 🌐 Satellite Feed")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)

    # ── Right: Prediction ──
    with col_result:
        st.markdown("#### 🔬 Threat Analysis")

        with st.spinner("⏳ Running orbital pixel analysis matrix..."):
            time.sleep(1.2)

            # Load TFLite interpreter
            interpreter = load_model()
            input_details  = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Pre-process
            img_resized = image.resize((150, 150))
            img_array   = np.array(img_resized, dtype=np.float32) / 255.0
            img_array   = np.expand_dims(img_array, axis=0)   # (1, 150, 150, 3)

            # Run inference via TFLite
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]["index"])
            fire_prob  = float(prediction[0][0])

        confidence_pct = fire_prob * 100
        raw_prob_pct   = fire_prob * 100

        # ── Result Banner ──
        if fire_prob >= threshold:
            st.markdown(
                "<div class='alert-fire'>🚨 CRITICAL THREAT: ACTIVE WILDFIRE DETECTED!</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='alert-clear'>✅ SYSTEM CLEAR: NO FIRE DETECTED</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Raw Probability Gauge ────────────────────────────────────────
        gauge_color = "#ff2d55" if fire_prob >= threshold else "#00e676"
        marker_pos  = max(5, min(95, threshold * 100))
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>🔍 Raw Model Probability Score</div>
                <div style='display:flex; justify-content:space-between;
                            align-items:baseline; margin-top:0.3rem;'>
                    <div style='color:{gauge_color}; font-size:2rem;
                                font-weight:900; letter-spacing:0.04em;'>
                        {raw_prob_pct:.2f}%
                    </div>
                    <div style='color:#3a5a7a; font-size:0.7rem;
                                letter-spacing:0.15em;'>
                        FIRE PROBABILITY
                    </div>
                </div>
                <div style='position:relative; background:#0a1020;
                            border-radius:8px; height:18px;
                            overflow:visible; margin-top:0.5rem;
                            border:1px solid #1f3050;'>
                    <div style='width:{raw_prob_pct:.1f}%; height:100%;
                                background:linear-gradient(90deg,
                                    #1a6aff 0%, {gauge_color} 100%);
                                border-radius:8px;
                                box-shadow: 0 0 10px {gauge_color}55;'>
                    </div>
                    <div style='position:absolute; top:-5px;
                                left:{marker_pos:.1f}%;
                                width:3px; height:28px;
                                background:#ffa726;
                                border-radius:2px;
                                box-shadow: 0 0 8px #ffa72699;'>
                    </div>
                </div>
                <div style='display:flex; justify-content:space-between;
                            margin-top:0.4rem;'>
                    <span style='color:#1a3a5a; font-size:0.65rem;'>0%</span>
                    <span style='color:#ffa726; font-size:0.65rem;
                                 letter-spacing:0.1em;'>
                        ▲ THRESHOLD {threshold:.0%}
                    </span>
                    <span style='color:#1a3a5a; font-size:0.65rem;'>100%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence Metrics ──
        if fire_prob >= threshold:
            bar_color   = "#ff2d55"
            label_color = "#ff4466"
            label_text  = "🔥 Fire Confidence"
        else:
            bar_color   = "#00e676"
            label_color = "#00e676"
            label_text  = "🌿 Safe Confidence"
            confidence_pct = (1 - fire_prob) * 100

        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>{label_text}</div>
                <div style='color:{label_color}; font-size:2rem; font-weight:900;
                            letter-spacing:0.05em; margin-top:0.2rem;'>
                    {confidence_pct:.2f} %
                </div>
                <div class='conf-bar-bg'>
                    <div style='width:{confidence_pct:.1f}%; height:100%;
                                background:{bar_color};
                                border-radius:6px;
                                transition: width 0.6s ease;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Raw score (for transparency) ──
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Raw Model Output</div>
                <div style='color:#5a7a9a; font-size:1rem; font-weight:700;
                            letter-spacing:0.05em; margin-top:0.2rem;'>
                    σ(x) = {fire_prob:.6f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Classification label ──
        label = "WILDFIRE" if fire_prob >= threshold else "NO FIRE"
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Classification</div>
                <div style='color:{"#ff4466" if fire_prob >= threshold else "#00e676"};
                            font-size:1rem; font-weight:900;
                            letter-spacing:0.2em; margin-top:0.2rem;'>
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # ── Placeholder when no image is uploaded ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center; color:#1f3050; padding: 3rem;
                    border: 1px dashed #1f3050; border-radius:12px;'>
            <div style='font-size:3rem;'>🛰️</div>
            <div style='font-size:1rem; letter-spacing:0.2em; margin-top:0.5rem;'>
                AWAITING SATELLITE IMAGE FEED
            </div>
            <div style='font-size:0.75rem; margin-top:0.3rem; color:#0e1f30;'>
                Upload a PNG / JPG to begin orbital threat analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown(
    "<div class='ignis-footer'>IGNIS-AI · ORBITAL WILDFIRE INTELLIGENCE · CLASSIFIED SYSTEM</div>",
    unsafe_allow_html=True,
)
