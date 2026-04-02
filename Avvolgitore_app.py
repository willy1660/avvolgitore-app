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
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    p0, p1 = points[idx], points[idx + 1]
    alpha = (target_length - cum[idx]) / (seg[idx] + EPS)

    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2 * np.pi))

# =========================
# GEOMETRY (FIX IMPORTANT)
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
    r_tubo = d_tubo / 2.0

    r = d_aspo_mm / 2 + r_tubo
    z_min = r_tubo
    z_max = spalla_mm - r_tubo

    z = z_min
    theta = 0.0
    direction = 1

    dz_dtheta = passo_assiale / (2 * np.pi)
    theta_step = np.deg2rad(4)

    points = []

    def add_point(r, theta, z):
        points.append([r * np.cos(theta), r * np.sin(theta), z])

    add_point(r, theta, z)

    while True:

        if len(points) > 2 and polyline_length(np.array(points)) >= lunghezza_mm:
            break

        # HELIX
        while True:
            theta_prev = theta
            z_prev = z

            theta += theta_step
            z += direction * dz_dtheta * theta_step

            if direction == 1 and z >= z_max:
                frac = (z_max - z_prev) / (z - z_prev + EPS)
                theta = theta_prev + frac * (theta - theta_prev)
                z = z_max
                add_point(r, theta, z)
                break

            if direction == -1 and z <= z_min:
                frac = (z_min - z_prev) / (z - z_prev + EPS)
                theta = theta_prev + frac * (theta - theta_prev)
                z = z_min
                add_point(r, theta, z)
                break

            add_point(r, theta, z)

        # =========================
        # 🔥 RITARDO FIXAT
        # =========================

        at_top = direction == 1
        ritardo_deg = ritardo_max_deg if at_top else ritardo_min_deg
        theta_dwell = np.deg2rad(ritardo_deg)

        if theta_dwell > EPS:

            dwell_steps = max(8, int(np.ceil(ritardo_deg / 4)))
            theta_step_dwell = theta_dwell / dwell_steps

            theta_start = theta
            r_start = r

            for _ in range(dwell_steps):

                theta += theta_step_dwell

                frac = (theta - theta_start) / theta_dwell
                frac = max(0.0, min(1.0, frac))

                r_curr = r_start + passo_radiale * frac

                add_point(r_curr, theta, z)

            r = r_start + passo_radiale

        else:
            r += passo_radiale
            add_point(r, theta, z)

        direction *= -1

    path = np.array(points)
    path = trim_polyline(path, lunghezza_mm)

    r_path = np.sqrt(path[:, 0]**2 + path[:, 1]**2)
    r_max = np.max(r_path)

    return path, {
        "DiametroTubo": d_tubo,
        "DiametroEsterno": 2 * (r_max + r_tubo),
        "VolteTotali": compute_total_turns(path),
    }

# =========================
# VIEWER (mateix que abans)
# =========================

def build_viewer(points, d_tubo, height):
    pts = json.dumps(points.tolist())

    return f"""
    <div id="viewer" style="width:100%;height:{height}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100000);
    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth, {height});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const pts = {pts}.map(p => new THREE.Vector3(p[0],p[1],p[2]));

    class Curve extends THREE.Curve {{
        constructor(points) {{ super(); this.points = points; }}
        getPoint(t) {{
            const i = Math.floor(t*(this.points.length-1));
            return this.points[i];
        }}
    }}

    const curve = new Curve(pts);

    const tube = new THREE.TubeGeometry(curve, 2000, {d_tubo/2}, 32, false);
    const mesh = new THREE.Mesh(tube, new THREE.MeshStandardMaterial({{color:0xffffff}}));

    scene.add(mesh);

    const light = new THREE.DirectionalLight(0xffffff,1);
    light.position.set(10,10,10);
    scene.add(light);

    camera.position.set(500,500,300);

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
    rit1 = st.number_input(t["rit_min"], 180.0)
    rit2 = st.number_input(t["rit_max"], 180.0)

d_rame = COPPER_SIZES_MM[rame]

path, meta = build_coil(
    diam, spalla, lung,
    d_rame, iso,
    passo, inc,
    rit1, rit2
)

components.html(build_viewer(path, meta["DiametroTubo"], 700), height=700)

st.write(meta)
