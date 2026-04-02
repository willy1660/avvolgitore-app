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
        "pre_rot": "Pre-rotazione mandrino (°)",
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
        "pre_rot": "Mandrel pre-rotation (°)",
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
# UTILS
# =========================

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline(points, target_length):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= target_length:
        return points
    idx = np.searchsorted(cum, target_length) - 1
    p0, p1 = points[idx], points[idx + 1]
    alpha = (target_length - cum[idx]) / np.linalg.norm(p1 - p0)
    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

def smoothstep01(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)

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
    pre_rot_deg
):
    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    r = d_aspo_mm/2 + d_tubo/2
    z = 0
    theta = 0
    direction = 1

    points = []

    def add():
        points.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    # PRE-ROTATION
    if pre_rot_deg > 0:
        theta_pre = np.deg2rad(pre_rot_deg)
        steps = max(8, int(pre_rot_deg/5))
        for i in range(steps):
            theta_i = theta + theta_pre*(i/steps)
            points.append([r*np.cos(theta_i), r*np.sin(theta_i), z])
        theta += theta_pre

    theta_step = np.deg2rad(4)
    dz = passo_assiale/(2*np.pi)

    while polyline_length(np.array(points)) < lunghezza_mm:

        theta += theta_step
        z += direction*dz*theta_step

        if direction == 1 and z >= spalla_mm:
            z = spalla_mm
            r += passo_radiale
            direction = -1

        elif direction == -1 and z <= 0:
            z = 0
            r += passo_radiale
            direction = 1

        add()

    path = trim_polyline(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    return path, {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext
    }

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo/2

    return f"""
    <div style="width:100%;height:{altezza}px;" id="viewer"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const pts = {points_json}.map(p => new THREE.Vector3(p[0],p[1],p[2]));
    const curve = new THREE.CatmullRomCurve3(pts);

    const tube = new THREE.TubeGeometry(curve, 2000, {r_tubo}, 32, false);
    const mesh = new THREE.Mesh(tube, new THREE.MeshStandardMaterial());
    scene.add(mesh);

    camera.position.set(500,500,500);

    function animate(){{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene,camera);
    }}
    animate();
    </script>
    """

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input(t["diam_aspo"], 450.0)
    spalla = st.number_input(t["spalla"], 95.0)

with colB:
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], 7.0)
    lunghezza = st.number_input(t["lunghezza"], 50.0)

with colC:
    passo = st.number_input(t["passo_assiale"], 20.0)
    inc = st.number_input(t["incremento"], 20.0)
    rit_min = st.number_input(t["rit_min"], 180.0)
    rit_max = st.number_input(t["rit_max"], 180.0)

with colD:
    pre_rot = st.number_input(t["pre_rot"], 180.0)
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], False)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

path, meta = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    COPPER_SIZES_MM[rame],
    spessore,
    passo,
    inc,
    rit_min,
    rit_max,
    pre_rot
)

components.html(build_viewer_html(path, meta["DiametroTubo"], altezza, anim, vel), height=altezza)

st.write(meta)
