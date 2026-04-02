import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# 🌍 LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

lang_option = st.selectbox(
    "🌍 Language",
    ["🇮🇹 Italiano", "🇺🇸 English (US)"],
)

lang = "IT" if "Italiano" in lang_option else "EN"

TEXTS = {
    "IT": {
        "title": "Avvolgimento",
        "hook": "Volte iniziali mandrino",
    },
    "EN": {
        "title": "Coiling",
        "hook": "Initial mandrel turns",
    }
}

t = TEXTS[lang]

st.title(t["title"])

# =========================
# INPUTS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    d_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with col2:
    d_rame = st.number_input("Ø Rame (mm)", value=9.52)
    spessore = st.number_input("Spessore (mm)", value=7.0)
    lunghezza = st.number_input("Lunghezza (m)", value=50.0)

with col3:
    passo_assiale = st.number_input("Passo assiale", value=20.0)
    passo_radiale = st.number_input("Passo radiale", value=20.0)
    rit_top = st.number_input("Ritardo top", value=180.0)
    rit_bot = st.number_input("Ritardo bottom", value=180.0)
    hook_turns = st.slider(t["hook"], 0.0, 1.5, 0.5)

# =========================
# UTILS
# =========================

def length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim(points, target):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    if cum[-1] <= target:
        return points

    i = np.searchsorted(cum, target) - 1
    p0, p1 = points[i], points[i+1]
    alpha = (target - cum[i]) / np.linalg.norm(p1 - p0)

    return np.vstack([points[:i+1], p0 + alpha*(p1-p0)])

# =========================
# BUILD COIL (FIXED)
# =========================

def build():

    lunghezza_mm = lunghezza * 1000
    d_tubo = d_rame + 2*spessore

    r = d_aspo/2 + d_tubo/2
    z = 0.0
    theta = 0.0
    direction = 1

    dz_dtheta = passo_assiale / (2*np.pi)
    step = np.deg2rad(4)

    pts = []

    def add():
        pts.append([r*np.cos(theta), r*np.sin(theta), z])

    # =========================
    # 🔥 TRAM INICIAL REAL
    # =========================

    hook_angle = hook_turns * 2*np.pi
    steps = max(10, int(hook_angle / step))

    for _ in range(steps):
        theta += hook_angle / steps
        add()

    # =========================
    # HELIX
    # =========================

    while True:

        if len(pts) > 2 and length(np.array(pts)) >= lunghezza_mm:
            break

        # HELIX RUN
        while True:
            theta += step
            z += direction * dz_dtheta * step

            if direction == 1 and z >= spalla:
                z = spalla
                add()
                break

            if direction == -1 and z <= 0:
                z = 0
                add()
                break

            add()

            if length(np.array(pts)) >= lunghezza_mm:
                break

        if length(np.array(pts)) >= lunghezza_mm:
            break

        # RADIAL STEP
        r += passo_radiale
        add()

        # DWELL
        rit = rit_top if direction == 1 else rit_bot

        if rit > 0:
            dwell_steps = int(max(1, rit/4))
            dtheta = np.deg2rad(rit/dwell_steps)

            for _ in range(dwell_steps):
                theta += dtheta
                add()

                if length(np.array(pts)) >= lunghezza_mm:
                    break

        direction *= -1

    pts = trim(np.array(pts), lunghezza_mm)

    return pts, d_tubo

# =========================
# VIEWER
# =========================

def viewer(points, d):

    pts = json.dumps(points.tolist())

    html = f"""
    <div id="v" style="width:100%;height:600px"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const pts = {pts}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(45,2,0.1,10000);
    const ren = new THREE.WebGLRenderer();
    ren.setSize(800,600);
    document.getElementById("v").appendChild(ren.domElement);

    const controls = new THREE.OrbitControls(cam, ren.domElement);

    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, 1200, {d/2}, 24, false);
    const mesh = new THREE.Mesh(geo, new THREE.MeshNormalMaterial());
    scene.add(mesh);

    cam.position.set(400,400,200);

    function a(){{requestAnimationFrame(a);controls.update();ren.render(scene,cam);}}
    a();
    </script>
    """

    return html

# =========================
# RUN
# =========================

pts, d_tubo = build()

components.html(viewer(pts, d_tubo), height=650)
