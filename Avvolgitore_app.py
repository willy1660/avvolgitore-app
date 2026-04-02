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

if "Italiano" in lang_option:
    st.session_state.lang = "IT"
else:
    st.session_state.lang = "EN"

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
# UTILS
# =========================

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline(points, target_length):
    if len(points) < 2:
        return points

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    idx = max(0, min(idx, len(points) - 2))

    p0, p1 = points[idx], points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len < EPS:
        return points[:idx + 1]

    alpha = (target_length - cum[idx]) / seg_len
    alpha = max(0.0, min(1.0, alpha))

    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

# =========================
# GEOMETRY FIXED
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
    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    r = d_aspo_mm / 2 + d_tubo / 2
    z = 0.0
    theta = 0.0
    direction = 1

    dz_dtheta = passo_assiale / (2 * np.pi)
    theta_step = np.deg2rad(4)

    points = []

    def add():
        points.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    while True:

        # HELIX
        while True:
            theta += theta_step
            z += direction * dz_dtheta * theta_step

            if direction == 1 and z >= spalla_mm:
                z = spalla_mm
                add()
                break

            if direction == -1 and z <= 0:
                z = 0
                add()
                break

            add()

            if polyline_length(np.array(points)) >= lunghezza_mm:
                return trim_polyline(np.array(points), lunghezza_mm), {}

        # RADIAL STEP (INSTANT)
        r += passo_radiale
        add()

        # DWELL (CONSTANT RADIUS)
        rit = ritardo_max_deg if direction == 1 else ritardo_min_deg

        if rit > 0:
            steps = int(max(1, rit / 4))
            step = np.deg2rad(rit / steps)

            for _ in range(steps):
                theta += step
                add()

        # CHANGE DIRECTION
        direction *= -1

        if polyline_length(np.array(points)) >= lunghezza_mm:
            return trim_polyline(np.array(points), lunghezza_mm), {}

# =========================
# VIEWER (igual)
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0

    html = f"""
    <div style="width:100%;height:{altezza}px;">
    <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);
    const renderer = new THREE.WebGLRenderer({{ antialias:true }});

    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const rawPoints = {points_json};
    const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    const curve = new THREE.CatmullRomCurve3(vectors);

    const geometry = new THREE.TubeGeometry(curve, 1500, {r_tubo}, 32, false);
    const material = new THREE.MeshStandardMaterial({{color:0xe6e6e6}});

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5,5,5);
    scene.add(light);

    camera.position.set(300,300,200);

    function animate(){{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene,camera);
    }}
    animate();
    </script>
    """
    return html

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=50.0)
    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0)
    passo_radiale = st.number_input(t["incremento"], value=20.0)
    rit_min = st.number_input(t["rit_min"], value=180.0)
    rit_max = st.number_input(t["rit_max"], value=180.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)

# =========================
# RUN
# =========================

path, _ = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore,
    passo_assiale,
    passo_radiale,
    rit_min,
    rit_max,
)

html = build_viewer_html(path, d_rame + 2*spessore, altezza, False, 1.0)

components.html(html, height=altezza)
