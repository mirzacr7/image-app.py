"""
Streamlit Image Tools
File: streamlit_image_tools.py
How to run locally:
  pip install -r requirements.txt
  streamlit run streamlit_image_tools.py

Requirements (put in requirements.txt):
streamlit
opencv-python-headless
Pillow
numpy

This single-file Streamlit app implements:
 - upload image
 - display original (PIL/OpenCV)
 - convert to grayscale (OpenCV / PIL)
 - show image shape (height, width, channels)
 - rotate by 90/180/270 (cv2.rotate)
 - mirror (cv2.flip)
 - threshold + contour detection (no deep learning)
 - vertical / horizontal slicing
 - grid split
 - download processed image

"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Image Tools", layout="wide")

st.title("Image Tools — OpenCV & PIL playground")
st.write("Upload an image and experiment with common image operations (grayscale, rotate, mirror, threshold + contour detection, slicing, grid).")

# Sidebar controls
st.sidebar.header("Controls")
upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"]) 

use_pil_preview = st.sidebar.checkbox("Show PIL preview (instead of OpenCV)", value=False)

op_grayscale = st.sidebar.checkbox("Convert to grayscale", value=False)
rotate_option = st.sidebar.selectbox("Rotate", options=["None", "90° CW", "180°", "270° CW"], index=0)
mirror = st.sidebar.checkbox("Mirror horizontally (flip)", value=False)

st.sidebar.markdown("---")
contour_detect = st.sidebar.checkbox("Detect objects (threshold + contours)", value=False)
thresh_val = st.sidebar.slider("Threshold value", min_value=0, max_value=255, value=120)
min_area = st.sidebar.number_input("Min contour area (px)", min_value=1, value=100)

st.sidebar.markdown("---")
vertical_cut = st.sidebar.checkbox("Show vertical cut (select columns)", value=False)
if vertical_cut:
    v_start = st.sidebar.slider("Vertical start (px)", min_value=0, max_value=1000, value=0)
    v_end   = st.sidebar.slider("Vertical end (px)", min_value=0, max_value=1000, value=100)

horizontal_cut = st.sidebar.checkbox("Show horizontal cut (select rows)", value=False)
if horizontal_cut:
    h_start = st.sidebar.slider("Horizontal start (px)", min_value=0, max_value=1000, value=0)
    h_end   = st.sidebar.slider("Horizontal end (px)", min_value=0, max_value=1000, value=100)

st.sidebar.markdown("---")
show_grid = st.sidebar.checkbox("Show grid blocks", value=False)
if show_grid:
    grid_h = st.sidebar.number_input("Grid rows", min_value=1, max_value=20, value=4)
    grid_w = st.sidebar.number_input("Grid cols", min_value=1, max_value=20, value=4)

st.sidebar.markdown("---")
export_png = st.sidebar.checkbox("Enable download of processed image", value=True)

# Helper functions

def read_image_to_cv2(file) -> np.ndarray:
    """Read uploaded file (BytesIO) into OpenCV BGR ndarray"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert('RGB'))
    # convert RGB -> BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    if len(img_cv2.shape) == 2:
        return Image.fromarray(img_cv2)
    rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bgr_for_st_image(img_cv2: np.ndarray):
    """Return an RGB array suitable for st.image from BGR cv2 array"""
    if img_cv2 is None:
        return None
    if len(img_cv2.shape) == 2:
        return img_cv2
    return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)


def process_image(img_cv):
    """Apply operations according to sidebar settings and return processed image and metadata"""
    if img_cv is None:
        return None, {}

    h, w = img_cv.shape[:2]
    c = 1 if len(img_cv.shape) == 2 else img_cv.shape[2]

    out = img_cv.copy()

    # Rotate
    if rotate_option == "90° CW":
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_option == "180°":
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rotate_option == "270° CW":
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Mirror
    if mirror:
        out = cv2.flip(out, 1)

    # Grayscale
    if op_grayscale:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    meta = {"height": h, "width": w, "channels": c}

    # Contour detection (works on grayscale)
    contours_img = None
    if contour_detect:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv.copy()
        _, thresh = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = img_cv.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < int(min_area):
                continue
            x,y,wc,hc = cv2.boundingRect(cnt)
            cv2.rectangle(detected, (x,y), (x+wc, y+hc), (0,255,0), 2)
        contours_img = detected

    # Vertical / Horizontal cuts (safely clamp indices)
    cuts = {}
    if vertical_cut:
        s = max(0, min(v_start, w-1))
        e = max(1, min(v_end, w))
        cuts['vertical'] = img_cv[:, s:e]
    if horizontal_cut:
        s = max(0, min(h_start, h-1))
        e = max(1, min(h_end, h))
        cuts['horizontal'] = img_cv[s:e, :]

    # Grid blocks
    grid_blocks = None
    if show_grid:
        gh = int(grid_h)
        gw = int(grid_w)
        bh = h // gh
        bw = w // gw
        blocks = []
        for i in range(gh):
            row = []
            for j in range(gw):
                y0 = i*bh
                x0 = j*bw
                y1 = h if i==gh-1 else (i+1)*bh
                x1 = w if j==gw-1 else (j+1)*bw
                row.append(img_cv[y0:y1, x0:x1])
            blocks.append(row)
        grid_blocks = blocks

    return out, {
        'meta': meta,
        'contours_img': contours_img,
        'cuts': cuts,
        'grid_blocks': grid_blocks
    }

# Main UI

if upload is None:
    st.info("Please upload an image using the uploader in the sidebar.")
    st.stop()

# Read image bytes
try:
    img_cv = read_image_to_cv2(upload)
except Exception as e:
    st.error(f"Failed to read image: {e}")
    st.stop()

# Show original
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    if use_pil_preview:
        img_pil = Image.open(upload)
        st.image(img_pil, use_column_width=True)
    else:
        st.image(bgr_for_st_image(img_cv), caption=f"shape: {img_cv.shape}", use_column_width=True)

# Process
processed_img, info = process_image(img_cv)

with col2:
    st.subheader("Processed")
    if processed_img is None:
        st.write("No processed image")
    else:
        st.image(bgr_for_st_image(processed_img), use_column_width=True)

# Show meta
st.sidebar.markdown("### Image info")
meta = info.get('meta', {})
st.sidebar.write(meta)

# Show contours result if requested
if contour_detect and info.get('contours_img') is not None:
    st.markdown("### Contour detection (visualized)")
    st.image(bgr_for_st_image(info['contours_img']), use_column_width=True)

# Show cuts
if info.get('cuts'):
    st.markdown("### Cuts")
    for k,v in info['cuts'].items():
        st.write(k)
        st.image(bgr_for_st_image(v), use_column_width=True)

# Show grid blocks
if info.get('grid_blocks') is not None:
    st.markdown("### Grid blocks")
    gh = len(info['grid_blocks'])
    gw = len(info['grid_blocks'][0]) if gh>0 else 0
    for i in range(gh):
        cols = st.columns(gw)
        for j in range(gw):
            with cols[j]:
                st.image(bgr_for_st_image(info['grid_blocks'][i][j]), caption=f"({i},{j})")

# Download processed image
if export_png and processed_img is not None:
    buf = None
    if len(processed_img.shape) == 2:
        pil = Image.fromarray(processed_img)
    else:
        pil = cv2_to_pil(processed_img)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    st.download_button("Download processed image (PNG)", data=byte_im, file_name="processed.png", mime="image/png")

st.markdown("---")
st.info("This app uses only basic OpenCV / PIL operations — no deep learning. To deploy: create a GitHub repo with this file (streamlit_image_tools.py) and a requirements.txt, then deploy on Streamlit Cloud (share.streamlit.io) or run locally with `streamlit run streamlit_image_tools.py`.")
