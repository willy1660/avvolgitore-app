import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1,7])

logo_path = os.path.join(os.path.dirname(__file__), "New Logo PDM - rame.png")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)

with col_title:
    st.markdown("# Avvolgimento")

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
    idx = max(0, min(idx, len(points)-2))

    p0, p1 = points[idx], points[idx+1]
    alpha = (target_length - cum[idx]) / np.linalg.norm(p1 - p0)

    return np.vstack([points[:idx+1], p0 + alpha*(p1-p0)])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2*np.pi))

def hermite_scalar(y0, y1, m0, m1, u):
    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2
    return h00*y0 + h10*m0 + h01*y1 + h11*m1

def smoothstep01(u):
    return 0.5 - 0.5*np.cos(np.pi*u)

# =========================
# MODEL CONTINU REAL
# =========================

def build_coil(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
    ritardo_top_deg,
    ritardo_bottom_deg,
):

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2*spessore_guaina_mm

    dz_dtheta = passo_assiale / (2*np.pi)

    r = d_aspo_mm/2 + d_tubo/2
    r0 = r

    theta = 0
    z = 0

    direction = 1  # +1 puja, -1 baixa

    points = []

    def add_point(theta, r, z):
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        points.append([x,y,z])

    while True:

        if len(points) > 2 and polyline_length(np.array(points)) >= lunghezza_mm:
            break

        # =========================
        # RUN LINEAL
        # =========================

        for _ in range(200):

            theta += 0.05
            z += direction * dz_dtheta * 0.05

            add_point(theta, r, z)

            if z >= spalla_mm or z <= 0:
                break

        # =========================
        # TURN
        # =========================

        if direction == 1:
            ritardo = ritardo_top_deg
        else:
            ritardo = ritardo_bottom_deg

        if ritardo < 1e-6:
            # 🔥 canvi sec REAL → sense kink artificial
            direction *= -1
            r += passo_radiale
            continue

        theta_turn = np.deg2rad(ritardo)
        steps = int(200 * (ritardo/360 + 0.1))

        for i in range(steps):

            t = i / steps
            theta += theta_turn / steps

            if direction == 1:
                z_local = hermite_scalar(
                    spalla_mm - 1,
                    spalla_mm,
                    1,
                    0,
                    min(t*2,1)
                ) if t < 0.5 else hermite_scalar(
                    spalla_mm,
                    spalla_mm - 1,
                    0,
                    -1,
                    (t-0.5)*2
                )
            else:
                z_local = hermite_scalar(
                    1,
                    0,
                    -1,
                    0,
                    min(t*2,1)
                ) if t < 0.5 else hermite_scalar(
                    0,
                    1,
                    0,
                    1,
                    (t-0.5)*2
                )

            z = z_local
            r_current = r + passo_radiale * smoothstep01(t)

            add_point(theta, r_current, z)

        r += passo_radiale
        direction *= -1

    path = trim_polyline(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    meta = {
        "DiametroTubo": d_tubo,
        "DiametroEsterno": diam_ext,
        "Capes": int((r_max-r0)/passo_radiale)+1,
        "VolteTotali": compute_total_turns(path)
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo/2

    return f"""
<div style="width:100%;height:{altezza}px;">
<div id="viewer" style="width:100%;height:100%;"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100000);

const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(window.innerWidth, {altezza});
document.getElementById("viewer").appendChild(renderer.domElement);

const rawPoints = {points_json};
const vectors = rawPoints.map(p => new THREE.Vector3(p[0],p[1],p[2]));

const curve = new THREE.CatmullRomCurve3(vectors);

const geometry = new THREE.TubeGeometry(curve, 2000, {r_tubo}, 32, false);

const material = new THREE.MeshBasicMaterial({{ color:0xffffff }});
const mesh = new THREE.Mesh(geometry, material);

scene.add(mesh);

camera.position.z = 500;

function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}}

animate();
</script>
"""

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input("Ø Aspo (mm)", 450.0)
    spalla = st.number_input("Spalla (mm)", 95.0)

with colB:
    rame_label = st.selectbox("Ø Rame", list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input("Spessore isolamento", 7.0)
    lunghezza = st.number_input("Lunghezza (m)", 50.0)

    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    passo_assiale = st.number_input("Passo assiale", 20.0)
    incremento_strato = st.number_input("Incremento strato", 20.0)

    ritardo_top = st.number_input("Ritardo alto (°)", 180.0)
    ritardo_bottom = st.number_input("Ritardo basso (°)", 180.0)

with colD:
    altezza = st.number_input("Viewer height", 700)

# =========================
# BUILD
# =========================

path, meta = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    passo_assiale,
    incremento_strato,
    ritardo_top,
    ritardo_bottom,
)

html = build_viewer_html(path, meta["DiametroTubo"], altezza)
components.html(html, height=altezza)

# =========================
# METRICS
# =========================

st.write(meta)
