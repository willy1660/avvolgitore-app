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
    index=0 if st.session_state.lang == "IT" else 1
)

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
lang = st.session_state.lang

TEXTS = {
    "IT": {
        "title": "Avvolgimento",
        "bobina": "🟦 Bobina",
        "tubo": "🟩 Tubo",
        "avvolg": "🟧 Avvolgimento",
        "viewer": "⚙️ Viewer",
        "diam_aspo": "Ø Aspo (mm)",
        "spalla": "Spalla (mm)",
        "rame": "Ø Rame",
        "isolamento": "Spessore isolamento (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
        "warning": "⚠️ Diametro esterno superiore a 750 mm."
    },
    "EN": {
        "title": "Coiling",
        "bobina": "🟦 Coil",
        "tubo": "🟩 Tube",
        "avvolg": "🟧 Winding",
        "viewer": "⚙️ Viewer",
        "diam_aspo": "Spool diameter (mm)",
        "spalla": "Width (mm)",
        "rame": "Copper size",
        "isolamento": "Insulation thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm)",
        "incremento": "Layer increment (mm)",
        "rit_min": "Bottom delay (°)",
        "rit_max": "Top delay (°)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Outer diameter",
        "warning": "⚠️ Outer diameter exceeds 750 mm."
    }
}

t = TEXTS[lang]

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1, 7])

logo_path = os.path.join(os.path.dirname(__file__), "New Logo PDM - rame.png")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)

with col_title:
    st.markdown(f"# {t['title']}")

# =========================
# CONSTANTS
# =========================

COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

EPS = 1e-9

# =========================
# GEOMETRY
# =========================

def build_coil(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
    ritardo_min_deg,
    ritardo_max_deg,
):
    lunghezza_mm = lunghezza_m * 1000.0
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    r = d_aspo_mm / 2 + d_tubo / 2
    z = 0.0
    theta = 0.0
    direction = 1

    dz = passo_assiale / (2 * np.pi)
    theta_step = np.deg2rad(4)

    points = []

    def add():
        points.append([r*np.cos(theta), r*np.sin(theta), z])

    # =========================
    # PRE-ENGAGEMENT (FIXED)
    # =========================

    theta_pre = np.deg2rad(180)
    steps = 20

    for i in range(steps):
        t = (i+1)/steps
        th = theta + theta_pre*t
        points.append([r*np.cos(th), r*np.sin(th), z])

    theta += theta_pre

    add()

    # =========================
    # MAIN LOOP
    # =========================

    length = 0

    while length < lunghezza_mm:

        theta += theta_step
        z += direction * dz * theta_step

        if direction == 1 and z >= spalla_mm:
            z = spalla_mm
            direction = -1
            r += passo_radiale

        elif direction == -1 and z <= 0:
            z = 0
            direction = 1
            r += passo_radiale

        add()

        if len(points) > 1:
            p0 = np.array(points[-2])
            p1 = np.array(points[-1])
            length += np.linalg.norm(p1 - p0)

    path = np.array(points)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza):

    pts = json.dumps(points.tolist())

    return f"""
    <div style="width:100%;height:{altezza}px;" id="v"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

    <script>
    const container = document.getElementById("v");
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const pts = {pts}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

    const curve = new THREE.CatmullRomCurve3(pts);

    const tube = new THREE.TubeGeometry(curve, 2000, {d_tubo/2}, 32, false);
    const mesh = new THREE.Mesh(tube, new THREE.MeshNormalMaterial());

    scene.add(mesh);

    camera.position.set(500,500,500);
    camera.lookAt(0,0,0);

    function animate(){{
        requestAnimationFrame(animate);
        renderer.render(scene,camera);
    }}
    animate();
    </script>
    """

# =========================
# UI
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    diam = st.number_input(t["diam_aspo"], 450.0)
    spalla = st.number_input(t["spalla"], 95.0)

with col2:
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    iso = st.number_input(t["isolamento"], 7.0)
    lung = st.number_input(t["lunghezza"], 50.0)

with col3:
    passo = st.number_input(t["passo_assiale"], 20.0)
    inc = st.number_input(t["incremento"], 20.0)

path, meta = build_coil(
    diam, spalla, lung,
    COPPER_SIZES_MM[rame],
    iso,
    passo,
    inc,
    0,0
)

components.html(build_viewer_html(path, meta["DiametroTubo"], 700), height=700)

st.write(meta)
