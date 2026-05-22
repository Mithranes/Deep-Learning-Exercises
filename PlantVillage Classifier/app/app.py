import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #f7f5f0; }

h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #1a1a1a;
    line-height: 1.1;
}

.subtitle {
    color: #6b6b6b;
    font-size: 1rem;
    font-weight: 300;
    margin-top: -12px;
    margin-bottom: 32px;
}

.result-box {
    background: #ffffff;
    border: 1px solid #e5e0d8;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 24px;
}

.disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #1a1a1a;
    margin-bottom: 4px;
}

.plant-name {
    font-size: 0.9rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 20px;
}

.confidence-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.healthy-badge {
    display: inline-block;
    background: #d4edda;
    color: #1a6b35;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}

.disease-badge {
    display: inline-block;
    background: #fdecea;
    color: #8b2a1e;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}

.bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.bar-label {
    margin-top: 0.5rem;
    font-size: 0.78rem;
    color: #ffffff;
    width: 220px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
}

.bar-track {
    flex: 1;
    background: #ede9e1;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 6px;
    background: #3a6b45;
    transition: width 0.4s ease;
}

.bar-fill.dim { background: #9ab5a0; }

.bar-pct {
    font-size: 0.78rem;
    color: #777;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

.divider {
    border: none;
    border-top: 1px solid #e5e0d8;
    margin: 20px 0;
}

.upload-hint {
    font-size: 0.82rem;
    color: #aaa;
    text-align: center;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Class names (38 PlantVillage classes) ──────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def format_class(raw):
    parts = raw.split('___')
    plant   = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else ''
    return plant, disease

# ── Model loader ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_model.h5')

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("<h1>Plant Disease<br>Detector</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf photo — get an instant diagnosis.</p>', unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown('<p class="upload-hint">JPG or PNG · any resolution</p>', unsafe_allow_html=True)

if uploaded:
    img = Image.open(uploaded).convert('RGB')

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(img, use_container_width=True)

    with col2:
        with st.spinner("Analyzing leaf..."):
            # Preprocess
            img_resized = img.resize((224, 224))
            arr = np.array(img_resized, dtype=np.float32)
            arr = np.expand_dims(arr, axis=0)   # (1, 224, 224, 3) — no /255, EfficientNet handles it

            # Predict
            model = load_model()
            preds = model.predict(arr, verbose=0)[0]   # shape (38,)

        top5_idx = np.argsort(preds)[::-1][:5]
        top_idx  = top5_idx[0]

        plant, disease = format_class(CLASS_NAMES[top_idx])
        confidence     = preds[top_idx] * 100
        is_healthy     = 'healthy' in CLASS_NAMES[top_idx].lower()

        # ── Result card ────────────────────────────────────────────────────────
        badge = 'healthy-badge' if is_healthy else 'disease-badge'
        badge_text = '✓ Healthy' if is_healthy else '⚠ Disease detected'

        st.markdown(f"""
        <div class="result-box">
            <div class="plant-name">{plant}</div>
            <div class="disease-name">{disease}</div>
            <span class="{badge}">{badge_text}</span>
            <hr class="divider">
            <div class="confidence-label">Top 5 predictions</div>
        """, unsafe_allow_html=True)

        for i, idx in enumerate(top5_idx):
            p, d    = format_class(CLASS_NAMES[idx])
            pct     = preds[idx] * 100
            fill    = '' if i == 0 else ' dim'
            bar_w   = f"{pct:.1f}%"
            label   = f"{p} — {d}"
            st.markdown(f"""
            <div class="bar-row">
                <span class="bar-label" title="{label}">{label}</span>
                <div class="bar-track">
                    <div class="bar-fill{fill}" style="width:{bar_w}"></div>
                </div>
                <span class="bar-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)