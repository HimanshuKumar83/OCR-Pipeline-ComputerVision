"""Streamlit demo interface for the robust OCR pipeline."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from ocr_pipeline import recognize_image


PROJECT_ROOT = Path(__file__).parent
SAMPLE_IMAGE = PROJECT_ROOT / "sample_input" / "sample.png"

st.set_page_config(page_title="Robust OCR Pipeline", page_icon="OCR", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #17211b; --muted: #657168; --accent: #d65a31; --paper: #f7f3eb; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247, 243, 235, 0.86); }
    .block-container { max-width: 1180px; padding-top: 3rem; }
    h1, h2, h3 { font-family: Georgia, serif; letter-spacing: 0; }
    h1 { font-size: 3.6rem !important; line-height: 1; color: var(--ink); }
    .eyebrow { color: var(--accent); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1.08rem; max-width: 680px; }
    [data-testid="stFileUploader"] { border: 1px dashed #c8b9a4; border-radius: 8px; background: #fffaf2; }
    [data-testid="stDownloadButton"] button, .stButton button { border-radius: 5px; }
    [data-testid="stMetricLabel"] p { color: #405047 !important; font-weight: 700; }
    [data-testid="stMetricValue"] { color: #17211b !important; }
    [data-testid="stMetricValue"] div { color: #17211b !important; }
    [data-testid="stDownloadButton"] button { background: #fffaf2 !important; color: #17211b !important; border: 1px solid #c8b9a4 !important; }
    [data-testid="stDownloadButton"] button:hover { background: #f0e4d2 !important; color: #17211b !important; border-color: #d65a31 !important; }
    [data-testid="stTextArea"] textarea { background: #fffaf2; color: #17211b; border: 1px solid #c8b9a4; border-radius: 6px; font-size: 1rem; line-height: 1.55; }
    .status { padding: .75rem 1rem; border-left: 4px solid var(--accent); background: #fffaf2; color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode uploaded bytes into an OpenCV BGR image."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


st.markdown('<div class="eyebrow">Computer vision / OCR</div>', unsafe_allow_html=True)
st.title("Read the signal in every image.")
st.markdown(
    '<p class="lede">Upload a sign, document, or scene capture. The pipeline improves local contrast, removes light noise, and turns the result into editable text.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pipeline demo")
    st.caption("OpenCV preprocessing + Tesseract OCR")
    use_sample = st.checkbox("Use included sample image", value=True)
    uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg", "bmp"])
    run_ocr = st.button("Run OCR", type="primary", width="stretch")
    st.divider()
    st.caption("Tesseract must be installed separately and available on PATH.")

image_bytes = None
image_name = "sample.png"
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_name = uploaded_file.name
elif use_sample and SAMPLE_IMAGE.exists():
    image_bytes = SAMPLE_IMAGE.read_bytes()

if image_bytes is None:
    st.info("Choose the included sample or upload an image from the sidebar.")
    st.stop()

image = decode_image(image_bytes)
if image is None:
    st.error("This file is not a valid image.")
    st.stop()

preview_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Input image")
    st.image(preview_rgb, caption=image_name, width="stretch")
with right:
    st.subheader("Best preprocessed image")
    st.caption("The selected candidate appears after OCR comparison.")

st.divider()
st.subheader("Extracted text")
if run_ocr:
    try:
        best, attempts = recognize_image(image)
    except Exception as error:
        if "TesseractNotFoundError" in type(error).__name__:
            st.error("Tesseract is not installed or is not available on PATH.")
        else:
            st.error(f"OCR could not run: {error}")
    else:
        with right:
            st.image(best.image, caption=f"{best.method} / PSM {best.psm}", width="stretch")
        if best.text:
            metric_columns = st.columns(3)
            metric_columns[0].metric("Confidence", f"{best.mean_confidence:.1f}%")
            metric_columns[1].metric("Preprocessing", best.method)
            metric_columns[2].metric("PSM mode", best.psm)
            st.text_area("OCR result", value=best.text, height=190, label_visibility="collapsed")
            st.download_button("Download text", data=best.text + "\n", file_name="ocr_output.txt", mime="text/plain")
            with st.expander("Show all OCR attempts"):
                for attempt in sorted(attempts, key=lambda item: item.score, reverse=True):
                    st.markdown(
                        f"**{attempt.method} / PSM {attempt.psm}**  "
                        f"Confidence: `{attempt.mean_confidence:.1f}%` | "
                        f"Words: `{attempt.word_count}` | Lines: `{attempt.line_count}`"
                    )
                    st.code(attempt.text or "(no text detected)", language="text")
        else:
            st.warning("OCR completed, but no text was detected.")
else:
    st.markdown('<div class="status">Ready to process. Click <strong>Run OCR</strong> in the sidebar.</div>', unsafe_allow_html=True)