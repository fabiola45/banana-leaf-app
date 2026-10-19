import io
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------- config
st.set_page_config(
    page_title="Banana Leaf Doctor",
    page_icon="🍌",
    layout="centered",
    initial_sidebar_state="collapsed",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "banana_leaf_mobilenetv2.keras")
CLASSES_PATH = os.path.join(HERE, "class_names.json")
IMG_SIZE = (224, 224)
CONFIDENCE_FLOOR = 0.60  # below this we ask for a better photo

# Same order as the notebook: sorted(df["label"].unique())
DEFAULT_CLASSES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

INFO = {
    "healthy": {
        "title": "Healthy leaf",
        "tone": "ok",
        "summary": "No sign of disease. The leaf surface looks even and green.",
        "signs": [
            "Even green colour across the blade",
            "No spots, streaks or yellow halos",
        ],
        "actions": [
            "Keep up regular weeding and good drainage.",
            "Check leaves every week so problems are caught early.",
            "Remove old dry leaves to keep air moving through the plants.",
        ],
    },
    "cordana": {
        "title": "Cordana leaf spot",
        "tone": "warn",
        "summary": "A fungal disease that makes oval spots with a yellow edge.",
        "signs": [
            "Oval or spindle-shaped brown spots",
            "Yellow ring around each spot, grey centre when old",
            "Spots join together on badly affected leaves",
        ],
        "actions": [
            "Cut off and burn or bury badly spotted leaves.",
            "Space plants so air and sunlight reach the leaves.",
            "Avoid watering over the leaves; water at the base.",
            "If it keeps spreading, ask your local agronomist about a suitable fungicide.",
        ],
    },
    "sigatoka": {
        "title": "Sigatoka leaf disease",
        "tone": "bad",
        "summary": "A serious fungal disease that destroys leaf area and cuts yield.",
        "signs": [
            "Thin yellow-brown streaks running along the leaf",
            "Streaks grow into dark brown or black patches",
            "Leaves dry out and collapse from the edges",
        ],
        "actions": [
            "Remove and destroy infected leaves quickly; do not leave them in the field.",
            "Improve drainage and thin out crowded suckers.",
            "Talk to your local extension officer about a spray programme, because the right product depends on your area.",
            "Check the neighbouring plants today.",
        ],
    },
    "pestalotiopsis": {
        "title": "Pestalotiopsis leaf spot",
        "tone": "warn",
        "summary": "A fungal leaf spot that often starts on damaged or stressed leaves.",
        "signs": [
            "Brown or grey patches, often with rings inside",
            "Dark specks in the middle of older spots",
            "Usually starts at leaf edges or damaged areas",
        ],
        "actions": [
            "Prune off affected leaves and keep tools clean.",
            "Protect leaves from wind and tool damage.",
            "Avoid overhead watering and keep the field tidy.",
            "Ask a local agronomist if the problem spreads.",
        ],
    },
}

TONE_COLORS = {
    "ok": ("#1F6B3A", "#DDF1DF"),
    "warn": ("#8A5A00", "#FFEFC2"),
    "bad": ("#9B2C1F", "#FBDDD6"),
}

# ---------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --leaf-deep: #123524;
  --leaf: #1F6B3A;
  --leaf-soft: #DDF1DF;
  --banana: #F4C430;
  --paper: #F4F8EE;
  --ink: #17261C;
  --muted: #5B6B5F;
  --line: #D5E2CF;
}

html, body, [class*="css"], .stApp { font-family: 'Source Sans 3', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 760px; padding-top: 1.2rem; padding-bottom: 4rem; }

h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.02em; }

/* hero */
.hero {
  position: relative; overflow: hidden; border-radius: 28px;
  padding: 3rem 2.2rem 2.6rem; color: #F4F8EE;
  background: linear-gradient(155deg, #123524 0%, #1F6B3A 100%);
  margin-bottom: 1.6rem;
}
.hero svg { position: absolute; right: -40px; top: -30px; width: 360px; opacity: .22; }
.hero h1 { color: #fff; font-size: clamp(2rem, 6vw, 3.1rem); line-height: 1.02; margin: 0 0 .9rem; font-weight: 800; position: relative; }
.hero p { font-size: 1.12rem; line-height: 1.5; max-width: 30rem; margin: 0 0 1.4rem; color: #DCEBDD; position: relative; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; position: relative; }
.chip { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); border-radius: 999px; padding: .3rem .8rem; font-size: .9rem; }
.chip.y { background: var(--banana); border-color: var(--banana); color: #3A2D00; font-weight: 600; }

/* upload area */
[data-testid="stFileUploader"] section {
  background: #fff; border: 2px dashed #9CC3A4; border-radius: 22px; padding: 1.6rem;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--leaf); background: #FBFEF8; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; border-radius: 10px 10px 0 0; padding: .6rem 1rem; }
.stTabs [aria-selected="true"] { color: var(--leaf); }
.stButton > button, [data-testid="stBaseButton-secondary"] {
  border-radius: 999px; font-weight: 600;
}

/* result */
.result {
  border-radius: 24px; padding: 1.6rem 1.7rem; margin: 1.2rem 0 1rem;
  border: 1px solid rgba(0,0,0,.05);
}
.result .kicker { font-size: .95rem; font-weight: 600; margin-bottom: .2rem; opacity: .85; }
.result h2 { margin: 0 0 .3rem; font-size: 2.1rem; font-weight: 800; }
.result p { margin: 0; font-size: 1.05rem; }
.conf { display: flex; align-items: baseline; gap: .5rem; margin-top: 1rem; }
.conf b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.6rem; line-height: 1; }
.conf span { font-size: .95rem; opacity: .8; }

.card { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; }
.card h3 { margin: 0 0 .7rem; font-size: 1.2rem; }
.card ul { margin: 0; padding-left: 1.1rem; }
.card li { margin: .35rem 0; line-height: 1.45; }

.bar-row { display: grid; grid-template-columns: 8.5rem 1fr 3.6rem; gap: .7rem; align-items: center; margin: .5rem 0; font-size: .98rem; }
.bar-track { background: #E8F0E2; height: 12px; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; background: #9CC3A4; }
.bar-fill.top { background: var(--leaf); }
.bar-row .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.bar-row.top .pct { color: var(--ink); font-weight: 600; }

.note { color: var(--muted); font-size: .92rem; line-height: 1.5; margin-top: 1.4rem; }
.stImage img { border-radius: 18px; }

@media (max-width: 560px) {
  .hero { padding: 2.2rem 1.4rem 2rem; border-radius: 22px; }
  .bar-row { grid-template-columns: 6.8rem 1fr 3.2rem; }
}
[data-testid="stImage"] img,
.stImage img {
  max-height: 420px;
  width: 100%;
  object-fit: contain;
  border-radius: 18px;
}
[data-testid="stCameraInput"] img,
[data-testid="stCameraInput"] video {
  max-height: 420px;
  object-fit: contain;
}
</style>
"""

HERO = """
<div class="hero">
  <svg viewBox="0 0 400 400" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round">
    <path d="M30 380 C120 300 220 160 380 20"/>
    <path d="M90 330 L40 300 M120 300 L60 250 M150 270 L90 210 M180 240 L130 170 M210 205 L170 130 M245 170 L215 95 M285 130 L270 60"/>
    <path d="M90 330 L160 345 M120 300 L200 305 M150 270 L240 265 M180 240 L275 220 M210 205 L305 170 M245 170 L335 120"/>
  </svg>
  <h1>Is your banana leaf healthy?</h1>
  <p>Take a photo of one leaf. In a few seconds you get the likely disease and what to do next.</p>
  <div class="chips">
    <span class="chip y">Healthy</span><span class="chip">Cordana</span>
    <span class="chip">Sigatoka</span><span class="chip">Pestalotiopsis</span>
  </div>
</div>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            names = json.load(f)
    else:
        names = DEFAULT_CLASSES
    return model, names


def predict(image: Image.Image, model, names):
    import tensorflow as tf

    # Same steps as the notebook: RGB -> resize 224x224 -> MobileNetV2 preprocessing
    img = ImageOps.exif_transpose(image).convert("RGB")
    arr = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
    arr = tf.image.resize(arr, IMG_SIZE)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    probs = model.predict(tf.expand_dims(arr, 0), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return [(names[i], float(probs[i])) for i in order]


def render_result(ranked):
    top, conf = ranked[0]
    info = INFO.get(top, INFO["healthy"])
    fg, bg = TONE_COLORS[info["tone"]]

    if conf < CONFIDENCE_FLOOR:
        st.markdown(
            f"""<div class="result" style="background:#EEF1EA;color:#2B372E">
            <div class="kicker">Not sure</div>
            <h2>I can't tell from this photo</h2>
            <p>Best guess is <b>{info['title'].lower()}</b> at {conf*100:.0f}%. Try a clear, close photo of a single leaf in daylight, with the leaf filling most of the frame.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="result" style="background:{bg};color:{fg}">
            <div class="kicker">Result</div>
            <h2>{info['title']}</h2>
            <p>{info['summary']}</p>
            <div class="conf"><b>{conf*100:.0f}%</b><span>sure</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    bars = ""
    for i, (name, p) in enumerate(ranked):
        label = INFO.get(name, {}).get("title", name).replace(" leaf disease", "").replace(" leaf spot", "").replace(" leaf", "")
        cls = "top" if i == 0 else ""
        bars += (
            f'<div class="bar-row {cls}"><span>{label}</span>'
            f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{p*100:.1f}%"></div></div>'
            f'<span class="pct">{p*100:.1f}%</span></div>'
        )
    st.markdown(f'<div class="card"><h3>How the model scored each option</h3>{bars}</div>', unsafe_allow_html=True)

    if conf >= CONFIDENCE_FLOOR:
        signs = "".join(f"<li>{s}</li>" for s in info["signs"])
        acts = "".join(f"<li>{a}</li>" for a in info["actions"])
        st.markdown(
            f'<div class="card"><h3>What to look for</h3><ul>{signs}</ul></div>'
            f'<div class="card"><h3>What to do now</h3><ul>{acts}</ul></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- UI
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Put `banana_leaf_mobilenetv2.keras` in the same folder as `app.py`."
    )
    st.stop()

tab_up, tab_cam = st.tabs(["Upload a photo", "Use the camera"])
image = None

with tab_up:
    f = st.file_uploader(
        "Drop a leaf photo here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="JPG or PNG",
    )
    if f is not None:
        image = Image.open(io.BytesIO(f.read()))

with tab_cam:
    c = st.camera_input("Take a photo of one leaf", label_visibility="collapsed")
    if c is not None:
        image = Image.open(io.BytesIO(c.getvalue()))

if image is not None:
    st.image(image, use_container_width=True)
    model, names = load_model()
    with st.spinner("Checking the leaf..."):
        ranked = predict(image, model, names)
    render_result(ranked)

st.markdown(
    '<p class="note">This tool gives a quick first opinion from a photo. It was trained on four leaf types only, '
    "so it can be wrong on other diseases, pests or nutrient problems. For a final decision, check with a local agronomist or extension officer.</p>",
    unsafe_allow_html=True,
)
