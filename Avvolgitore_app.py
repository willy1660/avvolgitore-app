import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os
import sys

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================================================
# HEADER
# =========================================================
col_logo, col_title = st.columns([1,6])

with col_logo:
    logo_path = resource_path("New Logo PDM - rame.png")
    st.image(logo_path, width=120)

with col_title:
    st.title("Avvolgimento")

# =========================================================
# DATI
# =========================================================
COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

# =========================================================
# UTILITÀ
# =========================================================
def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline_to_length(points, target):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= target:
        return points
    idx = np.searchsorted(cum, target, side="right") - 1
    idx = max(0, min(idx, len(points) - 2))
    p0 = points[idx]
    p1 = points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)
    alpha = (target - cum[idx]) / seg_len
    p_cut = p0 + alpha * (p1 - p0)
    return np.vstack([points[:idx + 1], p_cut])

# =========================================================
# GEOMETRIA
# =========================================================
def build_coil_centerline(
    d_aspo_mm, spalla_mm, lunghezza_m,
    d_rame_mm, spessore_guaina_mm,
    compressione_pct, gap_axiale_mm,
    incremento_strato, ritardo_min_deg, ritardo_max_deg):

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    passo_radiale = incremento_strato if incremento_strato > 0 else d_tubo * (1 - compressione_pct / 100)
    passo_assiale = d_tubo + gap_axiale_mm

    r0 = d_aspo_mm/2 + d_tubo/2
    r = r0

    z0, z1 = 0, spalla_mm
    theta = 0
    points = []

    while True:

        dz = z1 - z0
        giri = max(abs(dz) / passo_assiale, 0.1)

        dtheta = 2 * math.pi * giri
        t = np.linspace(0, dtheta, max(100, int(giri * 120)))

        x = r * np.cos(theta + t)
        y = r * np.sin(theta + t)
        z = z0 + dz * t / dtheta

        layer = np.column_stack([x,y,z])
        if len(points) > 0:
            layer = layer[1:]
        points.extend(layer.tolist())

        theta += dtheta

        # RITARDO
        if ritardo_max_deg > 0:
            rit = np.random.uniform(ritardo_min_deg, ritardo_max_deg)
            rit_rad = math.radians(rit)

            t = np.linspace(0, rit_rad, 20)
            x = r * np.cos(theta + t)
            y = r * np.sin(theta + t)
            z = np.full_like(t, z1)

            delay = np.column_stack([x,y,z])[1:]
            points.extend(delay.tolist())

            theta += rit_rad

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

        r += passo_radiale
        z0, z1 = z1, z0

    return trim_polyline_to_length(np.array(points), lunghezza_mm), d_tubo

# =========================================================
# VIEWER FIX DEFINITIU
# =========================================================
def build_viewer_html(points, d_tubo, altezza):

    points_json = json.dumps(points.tolist())
    r = d_tubo/2

    return f"""
<div id="viewer" style="width:100%;height:{altezza}px;"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const container = document.getElementById("viewer")

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000)

const renderer = new THREE.WebGLRenderer({{antialias:true}})
container.appendChild(renderer.domElement)

function resize(){{
    const w = container.clientWidth
    const h = container.clientHeight
    renderer.setSize(w, h)
    camera.aspect = w/h
    camera.updateProjectionMatrix()
}}
resize()

window.addEventListener('resize', resize)

const controls = new THREE.OrbitControls(camera, renderer.domElement)

const pts = {points_json}
const vectors = pts.map(p => new THREE.Vector3(p[0],p[1],p[2]))

const curve = new THREE.CatmullRomCurve3(vectors)

const mesh = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 1000, {r}, 24, false),
    new THREE.MeshStandardMaterial({{color:0xe6e6e6}})
)

scene.add(mesh)

scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1))

camera.position.set(500,500,300)
controls.update()

function animate(){{
    requestAnimationFrame(animate)
    controls.update()
    renderer.render(scene, camera)
}}

animate()

</script>
"""

# =========================================================
# UI
# =========================================================
c1,c2,c3,c4,c5 = st.columns(5)

with c1: diametro_aspo = st.number_input("Diametro aspo (mm)",450.0)
with c2: spalla = st.number_input("Spalla (mm)",95.0)
with c3: lunghezza = st.number_input("Lunghezza (m)",50.0)
with c4: rame_label = st.selectbox("Diametro rame",list(COPPER_SIZES_MM.keys()))
with c5: spessore_guaina = st.number_input("Spessore guaina (mm)",7.0)

c6,c7,c8,c9 = st.columns(4)

with c6: compressione = st.slider("Compressione %",0.0,20.0,0.0)
with c7: gap = st.number_input("Gap axiale (mm)",0.0)
with c8: incremento_strato = st.number_input("Incremento strato (mm)",0.0)
with c9: altezza = st.slider("Altezza viewer",400,900,700)

c10,c11 = st.columns(2)
with c10: ritardo_min = st.number_input("Ritardo MIN (deg)",0.0)
with c11: ritardo_max = st.number_input("Ritardo MAX (deg)",0.0)

# =========================================================
# RUN
# =========================================================
path, d_tubo = build_coil_centerline(
    diametro_aspo, spalla, lunghezza,
    COPPER_SIZES_MM[rame_label],
    spessore_guaina,
    compressione, gap,
    incremento_strato,
    ritardo_min, ritardo_max
)

components.html(build_viewer_html(path, d_tubo, altezza), height=altezza)
