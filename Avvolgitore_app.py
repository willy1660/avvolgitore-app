import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
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
# UTILS
# =========================

def polyline_length(points):
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
# GEOMETRY (INTACTE)
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

    r = d_aspo_mm/2 + d_tubo/2
    z = 0
    theta = 0
    direction = 1

    pts = []

    while len(pts) < 6000:
        theta += 0.05
        z += direction * passo_assiale * 0.02

        if z > spalla_mm:
            z = spalla_mm
            direction = -1
            r += passo_radiale

        if z < 0:
            z = 0
            direction = 1

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        pts.append([x,y,z])

    path = np.array(pts)

    diam_ext = 2*(np.max(np.sqrt(path[:,0]**2+path[:,1]**2)) + d_tubo/2)

    return path, {"DiametroTubo": d_tubo, "DiametroEsterno": diam_ext}

# =========================
# VIEWER (FIX CAPES)
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    return f"""
    <div style="width:100%;height:{altezza}px;">
    <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(800,800,800);
    scene.add(light);

    const pts = {points_json};
    const vec = pts.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    // 🔥 CLAVEEE → NO suavitzar
    const curve = new THREE.CatmullRomCurve3(vec, false, "centripetal", 0.0);

    const geom = new THREE.TubeGeometry(curve, 6000, {d_tubo/2}, 48, false);

    const mat = new THREE.MeshStandardMaterial({{
        color:0xcfd5db,
        roughness:0.6,
        metalness:0.1
    }});

    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    camera.position.set(600,600,400);
    controls.target.set(0,0,200);

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
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)

with colB:
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)

with colC:
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

path, meta = build_coil(
    diametro_aspo,
    95,
    50,
    COPPER_SIZES_MM[rame_label],
    spessore,
    passo_assiale,
    incremento,
    0,
    0
)

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita
)

components.html(html, height=altezza)

# =========================
# METRICS
# =========================

st.divider()

c1, c2 = st.columns(2)

c1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
c2.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")
