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

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    dtheta = np.diff(theta)
    return np.sum(np.abs(dtheta)) / (2*np.pi)

def points_to_sldcrv(points):
    return "\n".join(f"{p[0]} {p[1]} {p[2]}" for p in points).encode()

# =========================================================
# GEOMETRIA BOBINA
# =========================================================
def build_coil_centerline(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    compressione_pct,
    gap_axiale_mm,
    incremento_strato,
    ritardo_min_deg,
    ritardo_max_deg,
):

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    if incremento_strato > 0:
        passo_radiale = incremento_strato
    else:
        passo_radiale = d_tubo * (1 - compressione_pct / 100)

    passo_assiale = d_tubo + gap_axiale_mm

    r0 = d_aspo_mm/2 + d_tubo/2
    r = r0

    z0 = 0
    z1 = spalla_mm

    theta = 0
    points = []

    transition_turn = 0.2
    transition_pts = 40

    while True:

        dz = z1 - z0
        giri = max(abs(dz) / passo_assiale, 0.1)

        dtheta = 2 * math.pi * giri
        n = max(100, int(giri * 120))

        t = np.linspace(0,dtheta,n)
        theta_vals = theta + t
        z_vals = z0 + dz * t / dtheta

        x = r * np.cos(theta_vals)
        y = r * np.sin(theta_vals)

        layer = np.column_stack([x,y,z_vals])

        if len(points)>0:
            layer = layer[1:]

        points.extend(layer.tolist())
        theta += dtheta

        # RITARDO
        if ritardo_max_deg > 0:
            ritardo_deg = np.random.uniform(ritardo_min_deg, ritardo_max_deg)
            ritardo_rad = math.radians(ritardo_deg)

            t_delay = np.linspace(0, ritardo_rad, 30)
            theta_vals = theta + t_delay
            z_vals = np.full_like(theta_vals, z1)

            x = r * np.cos(theta_vals)
            y = r * np.sin(theta_vals)

            delay = np.column_stack([x,y,z_vals])[1:]
            points.extend(delay.tolist())

            theta += ritardo_rad

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

        r_next = r + passo_radiale

        t = np.linspace(0,2*math.pi*transition_turn,transition_pts)
        s = 0.5 - 0.5*np.cos(np.linspace(0,math.pi,transition_pts))

        r_vals = r + (r_next-r)*s
        theta_vals = theta + t
        z_vals = np.full_like(theta_vals,z1)

        x = r_vals*np.cos(theta_vals)
        y = r_vals*np.sin(theta_vals)

        transition = np.column_stack([x,y,z_vals])[1:]
        points.extend(transition.tolist())

        theta += 2*math.pi*transition_turn
        r = r_next
        z0,z1 = z1,z0

    path = trim_polyline_to_length(np.array(points),lunghezza_mm)

    total_turns = compute_total_turns(path)
    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    capes = max(1, int(round((r_max - r0)/passo_radiale))+1)

    return path, {
        "DiametroTubo":d_tubo,
        "PassoRadiale":passo_radiale,
        "PassoAssiale":passo_assiale,
        "DiametroEsterno":diam_ext,
        "LunghezzaM":polyline_length(path)/1000,
        "Capes":capes,
        "VolteTotali":total_turns,
        "VoltePerCapa":total_turns / capes,
    }

# =========================================================
# VIEWER
# =========================================================
def build_viewer_html(points,d_tubo,altezza,animazione,velocita):

    pts = points.tolist()
    points_json = json.dumps(pts)
    r_tubo = d_tubo/2
    tubular_segments = max(300,len(pts))

    html = f"""
<div style="position:relative;width:100%;height:{altezza}px;">
<img src="{logo_path}" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:60%;opacity:0.05;">
<div id="viewer" style="width:100%;height:100%;"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight,0.1,100000)
const renderer = new THREE.WebGLRenderer({{antialias:true}})
renderer.setSize(window.innerWidth,window.innerHeight)

document.getElementById("viewer").appendChild(renderer.domElement)

const controls = new THREE.OrbitControls(camera,renderer.domElement)

const rawPoints = {points_json}
const vectors = rawPoints.map(p=>new THREE.Vector3(p[0],p[1],p[2]))

class CurvePath extends THREE.Curve{{
constructor(points){{super();this.points=points;}}
getPoint(t){{
const f=t*(this.points.length-1)
const i=Math.floor(f)
const t2=f-i
return new THREE.Vector3().lerpVectors(this.points[i],this.points[i+1],t2)
}}
}}

const curve = new CurvePath(vectors)
const tube = new THREE.Mesh(
new THREE.TubeGeometry(curve,{tubular_segments},{r_tubo},32,false),
new THREE.MeshStandardMaterial({{color:0xe6e6e6}})
)

scene.add(tube)
camera.position.set(500,500,300)

function animate(){{
requestAnimationFrame(animate)
controls.update()
renderer.render(scene,camera)
}}

animate()
</script>
"""
    return html

# =========================================================
# UI
# =========================================================
c1,c2,c3,c4,c5=st.columns(5)

with c1: diametro_aspo=st.number_input("Diametro aspo (mm)",450.0)
with c2: spalla=st.number_input("Spalla (mm)",95.0)
with c3: lunghezza=st.number_input("Lunghezza (m)",50.0)
with c4: rame_label=st.selectbox("Diametro rame",list(COPPER_SIZES_MM.keys()))
with c5: spessore_guaina=st.number_input("Spessore guaina (mm)",7.0)

c6,c7,c8,c9=st.columns(4)

with c6: compressione=st.slider("Compressione %",0.0,20.0,0.0)
with c7: gap=st.number_input("Gap axiale (mm)",0.0)
with c8: incremento_strato=st.number_input("Incremento strato (mm)",0.0)
with c9: altezza=st.slider("Altezza viewer",400,900,700)

c10,c11=st.columns(2)
with c10: animazione=st.checkbox("Animazione",True)
with c11: velocita=st.slider("Velocità",0.1,5.0,1.0)

c12,c13=st.columns(2)
with c12: ritardo_min=st.number_input("Ritardo MIN (deg)",0.0)
with c13: ritardo_max=st.number_input("Ritardo MAX (deg)",0.0)

path,meta=build_coil_centerline(
diametro_aspo,spalla,lunghezza,
COPPER_SIZES_MM[rame_label],
spessore_guaina,compressione,gap,
incremento_strato,ritardo_min,ritardo_max
)

components.html(build_viewer_html(path,meta["DiametroTubo"],altezza,animazione,velocita),height=altezza)

# =========================================================
# METRICS
# =========================================================
st.divider()

for k,v in meta.items():
    st.write(k,":",round(v,2) if isinstance(v,float) else v)

if meta["DiametroEsterno"] > 750:
    st.warning("Bobina fuori pallet")
