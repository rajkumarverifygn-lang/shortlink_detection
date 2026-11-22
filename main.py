# Copyright (c) VerifyGN. All Rights Reserved.
# V-Count — Shortlink Detector & Counter (Streamlit)
# Run:  streamlit run main.py

import os
import io
import json
import time
import base64
import threading
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from engine import VCountEngine, verdict

try:
    if os.environ.get("VCOUNT_DISABLE_WEBRTC"):
        raise ImportError("WebRTC disabled")
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    HAS_WEBRTC = True
except Exception:  # pragma: no cover
    HAS_WEBRTC = False

# ============================================================================
# PATHS / CONFIG
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("VCOUNT_MODEL", os.path.join(BASE_DIR, "models", "best.vf"))
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"D:\verifygn\demo\wago_counting\wago_count_v1_vfgn26n_640_e2e_results\best.vf"
LOGO_PATH = os.path.join(BASE_DIR, "assets", "VFN_logo.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = r"D:\verifygn\VFN_logo.png"
SAMPLES_DIR = os.path.join(BASE_DIR, "assets", "samples")
MODEL_KEY = os.environ.get("VCOUNT_MODEL_KEY") or None

# Detection thresholds — confidence is HARD-CODED (change here only; no UI control)
CONF_THRESHOLD = 0.50
DEFAULT_IOU = 0.60

st.set_page_config(
    page_title="V-Count | Shortlink Counter",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# STYLE
# ============================================================================
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.vc-header {display:flex; align-items:center; gap:18px; padding:14px 22px; border-radius:14px;
  background:linear-gradient(90deg,#0f172a 0%,#1e293b 100%); color:#fff; margin-bottom:14px;
  box-shadow:0 4px 18px rgba(0,0,0,.18);}
.vc-header img {height:52px; background:#fff; border-radius:10px; padding:4px 8px;}
.vc-header h1 {font-size:1.65rem; margin:0; color:#fff; line-height:1.2;}
.vc-header p {margin:0; color:#94a3b8; font-size:.9rem;}
.vc-badge {margin-left:auto; background:#16a34a; color:#fff; padding:6px 14px; border-radius:999px;
  font-weight:700; font-size:.8rem; letter-spacing:.5px;}
.vc-card {border-radius:14px; padding:14px 10px; text-align:center; background:#f8fafc;
  border:1px solid #e2e8f0;}
.vc-card .lbl {font-size:.8rem; color:#64748b; font-weight:700; letter-spacing:1px; text-transform:uppercase;}
.vc-card .val {font-size:3.1rem; font-weight:800; color:#0f172a; line-height:1.1;}
.vc-verdict {border-radius:16px; padding:14px; text-align:center; color:#fff; font-weight:900;
  font-size:4rem; letter-spacing:4px; line-height:1.15; margin-top:10px;}
.vc-verdict small {display:block; font-size:.95rem; letter-spacing:1px; font-weight:600; opacity:.92;}
.vc-ok  {background:linear-gradient(135deg,#16a34a,#15803d); box-shadow:0 0 24px rgba(22,163,74,.45);}
.vc-nok {background:linear-gradient(135deg,#dc2626,#b91c1c); box-shadow:0 0 24px rgba(220,38,38,.45);}
.vc-na  {background:#64748b; font-size:2rem;}
.vc-foot {text-align:center; color:#94a3b8; font-size:.8rem; margin-top:24px;}
div.stButton > button[kind="primary"] {font-size:1.15rem; font-weight:800; padding:.6rem 0;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# MODEL (cached once per server)
# ============================================================================
@st.cache_resource(show_spinner="Loading VFGN26 shortlink model ...")
def load_engine(path: str, key):
    return VCountEngine(path, key=key)


def logo_b64() -> str:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


try:
    engine = load_engine(MODEL_PATH, MODEL_KEY)
except Exception as e:
    st.error(f"❌ Could not load model `{MODEL_PATH}`\n\n{e}")
    st.stop()

# ============================================================================
# SESSION STATE
# ============================================================================
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("last_check", None)
ss.setdefault("cam_running", False)


def log_result(source: str, count: int, set_count: int, enabled: bool, image_bgr=None):
    v = verdict(count, set_count, enabled)
    ss.history.insert(0, {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Set Count": set_count if enabled else "-",
        "Detected": int(count),
        "Result": v,
    })
    ss.last_check = {"verdict": v, "count": int(count), "set": set_count, "source": source,
                     "time": ss.history[0]["Time"],
                     "image": None if image_bgr is None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)}
    return v


def stable_count(counts) -> int:
    return Counter(counts).most_common(1)[0][0] if counts else 0


# ============================================================================
# UI HELPERS
# ============================================================================
def result_panel_html(count: int, set_count: int, enabled: bool, extra: str = "") -> str:
    v = verdict(count, set_count, enabled)
    set_txt = str(set_count) if enabled else "—"
    if v == "OK":
        vb = f'<div class="vc-verdict vc-ok">OK<small>Count matched ({count} = {set_count})</small></div>'
    elif v == "NOK":
        diff = count - set_count
        vb = (f'<div class="vc-verdict vc-nok">NOK<small>{"+" if diff > 0 else ""}{diff} '
              f'({"extra" if diff > 0 else "missing"})</small></div>')
    else:
        vb = '<div class="vc-verdict vc-na">CHECK OFF<small>Enable OK/NOK check in sidebar</small></div>'
    return f"""
<div style="display:flex;gap:10px;">
  <div class="vc-card" style="flex:1"><div class="lbl">Detected</div><div class="val">{count}</div></div>
  <div class="vc-card" style="flex:1"><div class="lbl">Set Count</div><div class="val">{set_txt}</div></div>
</div>
{vb}
<div style="color:#64748b;font-size:.82rem;margin-top:8px;text-align:center">{extra}</div>
"""


def decode_image(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ============================================================================
# HEADER
# ============================================================================
lb = logo_b64()
st.markdown(
    f"""
<div class="vc-header">
  {'<img src="data:image/png;base64,' + lb + '"/>' if lb else ''}
  <div><h1>V-Count · Shortlink Detector &amp; Counter</h1>
  <p>AI-powered WAGO shortlink detection, live counting &amp; OK / NOK verification</p></div>
  <div class="vc-badge">{engine.meta.get('variant', 'VFGN26')} · {engine.device.upper()}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR — SET COUNT & SETTINGS
# ============================================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width="stretch")
    st.markdown("### 🎯 Set Count & Check")
    set_count = st.number_input("Set (expected) count", min_value=0, max_value=999, value=10, step=1,
                                help="Detected count is compared with this value → OK / NOK")
    check_enabled = st.toggle("Enable OK / NOK check", value=True)

    st.markdown("### ⚙️ Detection Settings")
    conf = CONF_THRESHOLD  # fixed in code (see CONFIG section) — not user-editable
    iou = st.slider("Overlap (NMS IoU)", 0.10, 0.90, DEFAULT_IOU, 0.01,
                    help="Higher IoU (e.g. 0.60) keeps closely touching/parallel shortlinks separate")
    show_numbers = st.checkbox("Show index numbers", value=True)
    show_conf = st.checkbox("Show confidence %", value=False)
    stab_window = st.slider("Live stability window (frames)", 1, 30, 8,
                            help="Live count = most frequent count over last N frames (removes flicker)")

    st.markdown("### 📊 Session")
    n_ok = sum(1 for h in ss.history if h["Result"] == "OK")
    n_nok = sum(1 for h in ss.history if h["Result"] == "NOK")
    c1, c2, c3 = st.columns(3)
    c1.metric("Checks", len(ss.history))
    c2.metric("OK", n_ok)
    c3.metric("NOK", n_nok)

    with st.expander("ℹ️ Model info"):
        st.json({k: engine.meta.get(k) for k in ("variant", "names", "imgsz", "epoch", "file_size_mb")})


# ============================================================================
# TABS
# ============================================================================
tab_img, tab_live, tab_hist = st.tabs(["🖼️ Image Inspection", "🎥 Live Camera Counting", "📋 Check History"])

# ---------------------------------------------------------------------------
# TAB 1 — IMAGE (upload / camera snapshot / sample)
# ---------------------------------------------------------------------------
with tab_img:
    src = st.radio("Input source", ["📁 Upload Image", "📷 Camera Snapshot", "🧪 Sample Image"],
                   horizontal=True, label_visibility="collapsed")
    img_bgr, src_name = None, ""
    if src.startswith("📁"):
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            img_bgr, src_name = decode_image(up), f"Upload: {up.name}"
    elif src.startswith("📷"):
        snap = st.camera_input("Capture image from camera")
        if snap is not None:
            img_bgr, src_name = decode_image(snap), "Camera snapshot"
    else:
        samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".png"))) \
            if os.path.isdir(SAMPLES_DIR) else []
        if samples:
            pick = st.selectbox("Sample", samples)
            img_bgr, src_name = cv2.imread(os.path.join(SAMPLES_DIR, pick)), f"Sample: {pick}"
        else:
            st.info("No sample images found in assets/samples.")

    if img_bgr is None:
        st.info("⬆️ Upload an image or capture from camera to detect & count shortlinks.")
    else:
        boxes, scores, ms = engine.detect(img_bgr, conf, iou)
        count = len(boxes)
        vis = engine.annotate(img_bgr, boxes, scores, set_count, check_enabled, show_numbers, show_conf)

        col_img, col_res = st.columns([2.2, 1])
        with col_img:
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch",
                     caption=f"{src_name} · {img_bgr.shape[1]}×{img_bgr.shape[0]} px · {ms:.0f} ms")
        with col_res:
            avg = f" · avg conf {np.mean(scores) * 100:.0f}%" if count else ""
            st.markdown(result_panel_html(count, set_count, check_enabled, f"Inference {ms:.0f} ms{avg}"),
                        unsafe_allow_html=True)
            st.write("")
            if st.button("✔ CHECK & SAVE RESULT", type="primary", width="stretch", key="chk_img"):
                v = log_result(src_name, count, set_count, check_enabled, vis)
                (st.success if v == "OK" else st.error if v == "NOK" else st.info)(
                    f"Saved → {v} (detected {count}, set {set_count})")
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
            st.download_button("⬇ Download annotated image", buf.tobytes(),
                               file_name=f"vcount_{datetime.now():%Y%m%d_%H%M%S}.jpg",
                               mime="image/jpeg", width="stretch")

# ---------------------------------------------------------------------------
# TAB 2 — LIVE CAMERA
# ---------------------------------------------------------------------------
if HAS_WEBRTC:
    class LiveProcessor(VideoProcessorBase):
        """Runs in the WebRTC worker thread: detects, counts, overlays OK/NOK on every frame."""

        def __init__(self, eng):
            self.eng = eng
            self.conf, self.iou = CONF_THRESHOLD, DEFAULT_IOU
            self.set_count, self.check_enabled = 10, True
            self.show_numbers, self.show_conf = True, False
            self.counts = deque(maxlen=8)
            self.lock = threading.Lock()
            self.latest = {"count": 0, "stable": 0, "fps": 0.0, "ms": 0.0}
            self.frame = None
            self._t = time.time()

        def set_window(self, n):
            if self.counts.maxlen != n:
                self.counts = deque(self.counts, maxlen=n)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            boxes, scores, ms = self.eng.detect(img, self.conf, self.iou)
            self.counts.append(len(boxes))
            st_cnt = stable_count(self.counts)
            now = time.time()
            fps = 1.0 / max(now - self._t, 1e-3)
            self._t = now
            vis = self.eng.annotate(img, boxes, scores, self.set_count, self.check_enabled,
                                    self.show_numbers, self.show_conf, display_count=st_cnt, fps=fps)
            with self.lock:
                self.latest = {"count": len(boxes), "stable": st_cnt, "fps": fps, "ms": ms}
                self.frame = vis
            return av.VideoFrame.from_ndarray(vis, format="bgr24")


def ice_servers():
    try:
        if "ICE_SERVERS" in st.secrets:
            return json.loads(st.secrets["ICE_SERVERS"])
    except Exception:
        pass
    if os.environ.get("ICE_SERVERS"):
        return json.loads(os.environ["ICE_SERVERS"])
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


with tab_live:
    modes = (["🌐 Browser Camera (phone / laptop webcam)"] if HAS_WEBRTC else []) + \
            ["🔌 Server Camera (USB index / IP / RTSP)"]
    live_mode = st.radio("Camera mode", modes, horizontal=True, label_visibility="collapsed")
    if not HAS_WEBRTC:
        st.caption("Install `streamlit-webrtc` to enable browser camera streaming.")

    col_v, col_p = st.columns([2.2, 1])

    # ---------- Browser camera via WebRTC ----------
    if live_mode.startswith("🌐"):
        with col_v:
            ctx = webrtc_streamer(
                key="vcount-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": ice_servers()},
                video_processor_factory=lambda: LiveProcessor(engine),
                media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720},
                                                    "facingMode": "environment"}, "audio": False},
                async_processing=True,
            )
            st.caption("Press **START** and allow camera access. On phones the rear camera is preferred. "
                       "Deployed apps need **HTTPS** for camera access.")
        proc = ctx.video_processor if ctx else None
        if proc is not None:
            proc.conf, proc.iou = conf, iou
            proc.set_count, proc.check_enabled = set_count, check_enabled
            proc.show_numbers, proc.show_conf = show_numbers, show_conf
            proc.set_window(stab_window)

        with col_p:
            check_live = st.button("✔ CHECK NOW (capture count)", type="primary",
                                   width="stretch", key="chk_live", disabled=proc is None)
            panel = st.empty()
            msg = st.empty()
            if check_live and proc is not None:
                with proc.lock:
                    d, fr = dict(proc.latest), None if proc.frame is None else proc.frame.copy()
                v = log_result("Live (browser)", d["stable"], set_count, check_enabled, fr)
                (msg.success if v == "OK" else msg.error if v == "NOK" else msg.info)(
                    f"Checked → **{v}** (detected {d['stable']}, set {set_count})")

            if ctx and ctx.state.playing and proc is not None:
                while ctx.state.playing:
                    with proc.lock:
                        d = dict(proc.latest)
                    panel.markdown(result_panel_html(
                        d["stable"], set_count, check_enabled,
                        f"🔴 LIVE · raw {d['count']} · {d['fps']:.0f} FPS · {d['ms']:.0f} ms"),
                        unsafe_allow_html=True)
                    time.sleep(0.25)
            else:
                panel.markdown(result_panel_html(0, set_count, check_enabled, "Camera stopped — press START"),
                               unsafe_allow_html=True)

    # ---------- Server-side camera via OpenCV ----------
    else:
        with col_p:
            cam_src = st.text_input("Camera source", value="0",
                                    help="USB index (0, 1, ...) or IP/RTSP URL, e.g. rtsp://user:pass@ip:554/stream")
            b1, b2 = st.columns(2)
            if b1.button("▶ Start", width="stretch"):
                ss.cam_running = True
            if b2.button("⏹ Stop", width="stretch"):
                ss.cam_running = False
            check_srv = st.button("✔ CHECK NOW (capture count)", type="primary",
                                  width="stretch", key="chk_srv")
            panel = st.empty()
            msg = st.empty()
            if check_srv:
                d = ss.get("srv_latest")
                if d:
                    v = log_result(f"Live (camera {cam_src})", d["stable"], set_count, check_enabled, d["frame"])
                    (msg.success if v == "OK" else msg.error if v == "NOK" else msg.info)(
                        f"Checked → **{v}** (detected {d['stable']}, set {set_count})")
                else:
                    msg.warning("Start the camera first.")
        with col_v:
            frame_ph = st.empty()
            st.caption("Server camera = camera attached to the machine running this app (USB / industrial / IP).")

        if ss.cam_running:
            source = int(cam_src) if cam_src.strip().isdigit() else cam_src.strip()
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) and os.name == "nt" \
                else cv2.VideoCapture(source)
            if not cap.isOpened():
                st.error(f"Cannot open camera source: {cam_src}")
                ss.cam_running = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                counts = deque(maxlen=stab_window)
                t_prev = time.time()
                try:
                    while ss.cam_running:
                        ok, frame = cap.read()
                        if not ok:
                            st.warning("Camera frame not received.")
                            break
                        boxes, scores, ms = engine.detect(frame, conf, iou)
                        counts.append(len(boxes))
                        sc = stable_count(counts)
                        now = time.time()
                        fps = 1.0 / max(now - t_prev, 1e-3)
                        t_prev = now
                        vis = engine.annotate(frame, boxes, scores, set_count, check_enabled,
                                              show_numbers, show_conf, display_count=sc, fps=fps)
                        frame_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")
                        panel.markdown(result_panel_html(
                            sc, set_count, check_enabled,
                            f"🔴 LIVE · raw {len(boxes)} · {fps:.0f} FPS · {ms:.0f} ms"), unsafe_allow_html=True)
                        ss.srv_latest = {"stable": sc, "frame": vis}
                finally:
                    cap.release()
        else:
            panel.markdown(result_panel_html(0, set_count, check_enabled, "Camera stopped — press ▶ Start"),
                           unsafe_allow_html=True)
            frame_ph.info("Camera is stopped.")

# ---------------------------------------------------------------------------
# TAB 3 — HISTORY
# ---------------------------------------------------------------------------
with tab_hist:
    total = len(ss.history)
    n_ok = sum(1 for h in ss.history if h["Result"] == "OK")
    n_nok = sum(1 for h in ss.history if h["Result"] == "NOK")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total checks", total)
    m2.metric("✅ OK", n_ok)
    m3.metric("❌ NOK", n_nok)
    m4.metric("Yield", f"{(n_ok / max(n_ok + n_nok, 1)) * 100:.1f}%")

    lc = ss.last_check
    if lc:
        st.markdown("#### Last check")
        a, b = st.columns([2, 1])
        if lc["image"] is not None:
            a.image(lc["image"], width="stretch", caption=f"{lc['source']} · {lc['time']}")
        b.markdown(result_panel_html(lc["count"], lc["set"], lc["verdict"] != "-", lc["time"]),
                   unsafe_allow_html=True)

    if total:
        df = pd.DataFrame(ss.history)

        def _color(v):
            return ("background-color:#dcfce7;color:#166534;font-weight:700" if v == "OK" else
                    "background-color:#fee2e2;color:#991b1b;font-weight:700" if v == "NOK" else "")

        st.dataframe(df.style.map(_color, subset=["Result"]), width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV report", df.to_csv(index=False).encode(),
                           file_name=f"vcount_report_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv", width="stretch")
        if d2.button("🗑 Clear history", width="stretch"):
            ss.history, ss.last_check = [], None
            st.rerun()
    else:
        st.info("No checks yet. Use **CHECK** in Image or Live tab to record OK / NOK results.")

st.markdown('<div class="vc-foot">© VerifyGN · V-Count Shortlink Detector &amp; Counter</div>',
            unsafe_allow_html=True)
