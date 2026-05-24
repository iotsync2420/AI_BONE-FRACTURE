import streamlit as st
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bone Fracture Detection", layout="wide")

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ── EigenCAM: finds WHERE the model's features fire most strongly ─────────────
def compute_eigencam(img_bgr, torch_model, target_layers=(6, 12, 15, 18)):
    """
    EigenCAM on multiple backbone layers.
    Returns a heatmap (H,W) float32 in [0,1] aligned with img_bgr.
    """
    oh, ow = img_bgr.shape[:2]
    lb      = LetterBox(new_shape=(1024, 1024))
    resized = lb(image=img_bgr)
    inp     = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Hook target layers
    hooks, handles = {}, []
    def make_hook(name):
        def fn(m, i, out): hooks[name] = out.detach()
        return fn
    for idx in target_layers:
        try:
            handles.append(torch_model.model[idx].register_forward_hook(make_hook(idx)))
        except Exception:
            pass

    with torch.no_grad():
        torch_model(inp)
    for hh in handles:
        hh.remove()

    # Determine letterbox padding offset
    scale  = 1024 / max(oh, ow)
    new_h  = int(oh * scale)
    new_w  = int(ow * scale)
    pad_x  = (1024 - new_w) // 2
    pad_y  = (1024 - new_h) // 2

    combined = np.zeros((1024, 1024), dtype=np.float32)
    for feat in hooks.values():
        f = feat[0].cpu().numpy()          # (C, H, W)
        C, H, W = f.shape
        flat    = f.reshape(C, -1)         # (C, H*W)
        centered= flat - flat.mean(axis=1, keepdims=True)
        # First principal component via power iteration
        v = np.ones(C) / np.sqrt(C)
        cov = centered @ centered.T / (H * W)
        for _ in range(30):
            v = cov @ v
            norm = np.linalg.norm(v)
            if norm < 1e-10: break
            v /= norm
        cam = (v @ flat).reshape(H, W)
        cam = np.maximum(cam, 0)
        if cam.max() > 1e-8:
            cam /= cam.max()
        cam_up   = cv2.resize(cam, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        combined += cam_up

    # Crop letterbox padding and resize back to original
    cam_crop = combined[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    cam_orig = cv2.resize(cam_crop, (ow, oh), interpolation=cv2.INTER_CUBIC)
    cam_orig = np.maximum(cam_orig, 0)
    if cam_orig.max() > 1e-8:
        cam_orig /= cam_orig.max()
    return cam_orig


def get_fracture_boxes(cam, img_bgr, thresh_pct=0.60, margin=20, min_area_frac=0.002):
    """
    Find fracture bounding boxes from EigenCAM heatmap.
    - Smooth the CAM
    - Threshold at thresh_pct of peak
    - Find connected components (each = one fracture region)
    - Filter out tiny noise components
    Returns list of (x1,y1,x2,y2,score) sorted by score desc.
    """
    oh, ow = cam.shape
    smooth = gaussian_filter(cam, sigma=max(oh, ow) * 0.025)
    smooth /= smooth.max() + 1e-8

    binary  = (smooth >= thresh_pct).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    min_area = oh * ow * min_area_frac
    regions  = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        score = float(smooth[labels == i].mean())
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(ow, x + cw + margin)
        y2 = min(oh, y + ch + margin)
        regions.append((x1, y1, x2, y2, score))

    # Sort by score descending
    regions.sort(key=lambda r: -r[4])
    return regions


def draw_fracture_boxes(img_bgr, regions, cam):
    """Draw heatmap overlay + clean bounding boxes."""
    out = img_bgr.copy()

    # Heatmap overlay (subtle, 25% alpha)
    cam_u8    = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    out       = cv2.addWeighted(out, 0.75, cam_color, 0.25, 0)

    for i, (x1, y1, x2, y2, score) in enumerate(regions):
        label    = f"Fracture {score:.0%}"
        color    = (0, 255, 80)
        thick    = 2
        font     = cv2.FONT_HERSHEY_SIMPLEX
        fscale   = 0.55
        fthick   = 2

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        # Label pill
        (tw, th), bl = cv2.getTextSize(label, font, fscale, fthick)
        ly = max(y1 - 6, th + 6)
        cv2.rectangle(out, (x1, ly - th - bl - 2), (x1 + tw + 6, ly + 2), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 3, ly - bl), font, fscale, (0, 0, 0), fthick, cv2.LINE_AA)

    return out


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🦴 Bone Fracture Detection")
st.markdown("Upload an X-ray. The app uses **EigenCAM** on the model's backbone features to pinpoint *where* the fracture is — not just whether one exists.")

with st.sidebar:
    st.header("⚙️ Settings")
    thresh_pct = st.slider(
        "Heatmap threshold",
        min_value=0.40, max_value=0.85, value=0.60, step=0.05,
        help="Higher = tighter boxes around the hottest region. Lower = wider coverage."
    )
    margin_px = st.slider(
        "Box margin (px)", min_value=0, max_value=40, value=15, step=5,
        help="Extra padding around each detected region."
    )
    show_heatmap = st.checkbox("Show heatmap overlay", value=True)

uploaded = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode image.")
        st.stop()

    with st.spinner("Computing EigenCAM and detecting fractures…"):
        torch_model = model.model
        torch_model.eval()

        cam     = compute_eigencam(img_bgr, torch_model)
        regions = get_fracture_boxes(
            cam, img_bgr,
            thresh_pct=thresh_pct,
            margin=margin_px,
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        if not regions:
            st.success("✅ No significant fracture hotspot detected.")
            if show_heatmap:
                cam_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay   = cv2.addWeighted(img_bgr, 0.75, cam_color, 0.25, 0)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.image(img_rgb, use_container_width=True)
        else:
            draw_cam = cam if show_heatmap else np.zeros_like(cam)
            annotated = draw_fracture_boxes(img_bgr, regions, draw_cam)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.divider()
    if regions:
        st.subheader(f"🔍 {len(regions)} Fracture Region(s) Detected")
        rows = []
        for i, (x1, y1, x2, y2, score) in enumerate(regions):
            rows.append({
                "#":             i + 1,
                "CAM Score":     f"{score:.1%}",
                "Region (px)":   f"({x1},{y1}) → ({x2},{y2})",
                "Size":          f"{x2-x1} × {y2-y1} px",
            })
        st.table(rows)
        st.info(
            "**How it works:** EigenCAM extracts the first principal component of each "
            "backbone feature map, producing a spatial attention map that shows *where* "
            "the model focuses — independent of the bounding-box head. "
            "Boxes are drawn around the highest-activation connected regions."
        )
    else:
        st.info("Try lowering the **Heatmap threshold** in the sidebar if a fracture is expected.")
