"""Local image-upload interface. Run: python -m streamlit run app/demo.py"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

try:
    from app.inference import (
        CHECKPOINT_PATH, CNN_CHECKPOINT_PATH, PROJECT_DIR,
        load_cnn_model, load_model, predict_image,
    )
except ModuleNotFoundError:
    from inference import (
        CHECKPOINT_PATH, CNN_CHECKPOINT_PATH, PROJECT_DIR,
        load_cnn_model, load_model, predict_image,
    )


st.set_page_config(
    page_title="Deepfake Detector | NNDL Project",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

styles = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_cnn_model():
    return load_cnn_model()


def result_card(result, model_name):
    fake_pct = result["fake_probability"] * 100
    real_pct = result["real_probability"] * 100
    fake_class = "fake" if result["prediction"] == "FAKE" else ""
    st.markdown(
        f"""
        <div class="result-card {fake_class}">
          <div class="result-label">{model_name} prediction</div>
          <div class="result-value">{result['prediction']}</div>
          <div class="confidence"><strong>{result['confidence']:.1%}</strong> predicted-class confidence</div>
          <div class="prob-row"><span>P(FAKE)</span><strong>{fake_pct:.1f}%</strong></div>
          <div class="track"><div class="fill-fake" style="width:{fake_pct:.1f}%"></div></div>
          <div class="prob-row"><span>P(REAL)</span><strong>{real_pct:.1f}%</strong></div>
          <div class="track"><div class="fill-real" style="width:{real_pct:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="topline"><div class="brand"><span class="brand-mark"></span>DEEPFAKE DETECTOR</div>'
    '<div class="top-tag">NNDL PROJECT · TWO MODEL COMPARISON</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="eyebrow">IMAGE AUTHENTICITY RESEARCH</div>'
    '<div class="hero-title">Look closer at <span>what you see.</span></div>'
    '<div class="hero-copy">Upload a face image to see predictions from EfficientNet-B0 '
    'and our custom CNN. Compare each label and the probabilities behind it.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.02, .98], gap="large")
with left:
    st.markdown(
        '<div class="section-head"><div class="section-kicker">01 / INPUT</div>'
        '<div class="section-title">Upload an image</div>'
        '<div class="section-subtitle">JPG, PNG or WebP · a clear face image works best</div></div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    image = None
    image_digest = None
    if uploaded_file is not None:
        try:
            data = uploaded_file.getvalue()
            image = Image.open(BytesIO(data))
            image.load()
            image_digest = sha256(data).hexdigest()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f"Could not read this image: {exc}")
    if image is not None:
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        st.markdown(
            '<div class="panel panel-muted">Your image preview will appear here after upload. '
            'Both models run locally from the saved project checkpoints.</div>',
            unsafe_allow_html=True,
        )
    checkpoint_ready = CHECKPOINT_PATH.is_file() and CNN_CHECKPOINT_PATH.is_file()
    if not checkpoint_ready:
        st.error("A model checkpoint is missing. Check the checkpoints and checkpoints_cnn_baseline folders.")
    analyze = st.button(
        "Analyze image  →", type="primary", disabled=image is None or not checkpoint_ready
    )

with right:
    st.markdown(
        '<div class="section-head"><div class="section-kicker">02 / ANALYSIS</div>'
        '<div class="section-title">Model result</div>'
        '<div class="section-subtitle">Two models · decision threshold 0.50</div></div>',
        unsafe_allow_html=True,
    )
    if analyze:
        try:
            st.session_state["last_results"] = None
            with st.spinner("Loading models and analyzing image…"):
                model, transform, device = get_model()
                cnn_model, cnn_transform, cnn_device = get_cnn_model()
                st.session_state["last_results"] = {
                    "efficientnet": predict_image(image, model, transform, device),
                    "cnn": predict_image(image, cnn_model, cnn_transform, cnn_device),
                }
                st.session_state["last_image_digest"] = image_digest
        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
            st.error(f"Analysis failed: {exc}")
    results = st.session_state.get("last_results")
    if results is not None and st.session_state.get("last_image_digest") == image_digest:
        result_card(results["efficientnet"], "EfficientNet-B0")
        result_card(results["cnn"], "Custom CNN")
        st.markdown(
            '<p class="fine-print">The probabilities are raw model outputs. They are not '
            'verified authenticity or calibrated certainty.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel empty-result"><strong>Ready when you are.</strong>'
            '<p>Upload an image and select Analyze image to compare both predictions and class probabilities.</p></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
metrics = json.loads((PROJECT_DIR / "results" / "test_metrics.json").read_text(encoding="utf-8"))
stats = st.columns(3, gap="large")
for column, label, value in zip(
    stats,
    ["EFFICIENTNET TEST ACCURACY", "EFFICIENTNET FAKE-CLASS F1", "HELD-OUT IMAGES"],
    [f"{metrics['accuracy']:.1%}", f"{metrics['f1']:.3f}", "1,000"],
):
    with column:
        st.markdown(
            f'<div class="stat-label">{label}</div><div class="stat-value">{value}</div>',
            unsafe_allow_html=True,
        )
st.markdown(
    '<p class="fine-print">Results are from this project’s held-out dataset. Performance on images '
    'from other generators, cameras or sources has not been established.</p>',
    unsafe_allow_html=True,
)
