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

EPS = 1e-9

# =========================================================
# UTILITÀ
# =========================================================

def polyline_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline_to_length(points, target):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target:
        return points

    idx = np.searchsorted(cum, target) - 1
    p0 = points[idx]
    p1 = points[idx + 1]

    alpha = (target - cum[idx]) / np.linalg.norm(p1 - p0)
    p_cut = p0 + alpha * (p1 - p0)

    return np.vstack([points[:idx + 1], p_cut])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    return np.sum(np.abs(np.diff(theta))) / (2*np.pi)

def points_to_sldcrv(points):
    return "\n".join(f"{p[0]} {p[1]} {p[2]}" for p in points).encode()

# =========================================================
# GEOMETRIA
# =========================================================

def build_coil_centerline(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    compressione_pct,
    gap_axiale_mm,
    ritardo_inv_max,
    ritardo_inv_min,
):

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    passo_radiale = d_tubo * (1 - compressione_pct / 100)
    passo_assiale = d_tubo + gap_axiale_mm

    r = d_aspo_mm/2 + d_tubo/2
    z0, z1 = 0, spalla_mm
    theta = 0
    points = []

    while True:

        dz = z1 - z0
        giri = max(abs(dz)/passo_assiale, 0.1)
        dtheta = 2*np.pi*giri

        t = np.linspace(0, dtheta, int(giri*120)+50)

        theta_vals = theta + t
        z_vals = z0 + dz * t / dtheta

        x = r * np.cos(theta_vals)
        y = r * np.sin(theta_vals)

        layer = np.column_stack([x,y,z_vals])

        if len(points)>0:
            layer = layer[1:]

        points.extend(layer.tolist())
        theta += dtheta

        # ==============================
        # RITARDI
        # ==============================

        ritardo = math.radians(ritardo_inv_max if z1 > z0 else ritardo_inv_min)

        if ritardo > 0:
            t_delay = np.linspace(0, ritardo, 30)
            theta_vals = theta + t_delay
            z_vals = np.full_like(theta_vals, z1)

            x = r * np.cos(theta_vals)
            y = r * np.sin(theta_vals)

            points.extend(np.column_stack([x,y,z_vals])[1:].tolist())
            theta += ritardo

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

        r += passo_radiale
        z0, z1 = z1, z0

    path = trim_polyline_to_length(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))

    return path, {
        "DiametroTubo": d_tubo,
        "PassoRadiale": passo_radiale,
        "PassoAssiale": passo_assiale,
        "DiametroEsterno": 2*(r_max + d_tubo/2),
        "VolteTotali": compute_total_turns(path)
    }

# =========================================================
# VIEWER ORIGINAL
# =========================================================

def build_viewer_html(points,d_tubo,altezza,animazione,velocita):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo/2

    html = f"""
<div style="width:100%;height:{altezza}px;" id="viewer"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const container = document.getElementById("viewer")

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight,0.1,100000)

const renderer = new THREE.WebGLRenderer({{antialias:true}})
renderer.setSize(container.clientWidth,container.clientHeight)
container.appendChild(renderer.domElement)

const controls = new THREE.OrbitControls(camera,renderer.domElement)

const light = new THREE.DirectionalLight(0xffffff,1)
light.position.set(5,5,5)
scene.add(light)

const points = {points_json}.map(p=>new THREE.Vector3(p[0],p[1],p[2]))

class Curve extends THREE.Curve {{
    constructor(points){{super();this.points=points;}}
    getPoint(t){{
        const i = Math.floor(t*(this.points.length-1))
        return this.points[i]
    }}
}}

const curve = new Curve(points)

const tubularSegments = points.length
const geometry = new THREE.TubeGeometry(curve,tubularSegments,{r_tubo},16,false)

const material = new THREE.MeshStandardMaterial({{color:0xffffff}})
const mesh = new THREE.Mesh(geometry,material)
scene.add(mesh)

let progress = 0

function animate(){{
requestAnimationFrame(animate)

if({str(animazione).lower()}){{
progress += {velocita} * 0.002
if(progress > 1) progress = 1

mesh.geometry.setDrawRange(0, progress * geometry.index.count)
}}

controls.update()
renderer.render(scene,camera)
}}

camera.position.set(500,500,300)

animate()

</script>
"""
    return html

# =========================================================
# UI
# =========================================================

c1,c2,c3,c4,c5=st.columns(5)

with c1:
    diametro_aspo=st.number_input("Diametro aspo (mm)",value=450.0)

with c2:
    spalla=st.number_input("Spalla (mm)",value=95.0)

with c3:
    lunghezza=st.number_input("Lunghezza (m)",value=50.0)

with c4:
    rame_label=st.selectbox("Diametro rame",list(COPPER_SIZES_MM.keys()))

with c5:
    spessore_guaina=st.number_input("Spessore guaina (mm)",value=7.0)

c6,c7,c8,c9,c10=st.columns(5)

with c6:
    compressione=st.slider("Compressione %",0.0,20.0,0.0)

with c7:
    gap=st.number_input("Gap axiale (mm)",value=0.0)

with c8:
    ritardo_inv_max=st.number_input("Ritardo inv max (°)",value=0.0)

with c9:
    ritardo_inv_min=st.number_input("Ritardo inv min (°)",value=0.0)

with c10:
    altezza=st.slider("Altezza viewer",400,900,700)

c11,c12=st.columns(2)

with c11:
    animazione=st.checkbox("Animazione avvolgimento",True)

with c12:
    velocita=st.slider("Velocità animazione",0.1,5.0,1.0)

d_rame=COPPER_SIZES_MM[rame_label]

path,meta=build_coil_centerline(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    compressione,
    gap,
    ritardo_inv_max,
    ritardo_inv_min
)

html = build_viewer_html(path, meta["DiametroTubo"], altezza, animazione, velocita)

components.html(html,height=altezza)

st.write(meta)
