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

col_logo, col_title = st.columns([1, 6])

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
    if seg_len < EPS:
        return points[:idx + 1]

    alpha = (target - cum[idx]) / seg_len
    p_cut = p0 + alpha * (p1 - p0)

    return np.vstack([points[:idx + 1], p_cut])


def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    dtheta = np.diff(theta)
    return np.sum(np.abs(dtheta)) / (2 * np.pi)


def points_to_sldcrv(points):
    lines = []
    for p in points:
        lines.append(f"{p[0]} {p[1]} {p[2]}")
    return "\n".join(lines).encode()


def cylindrical_point(r, theta, z):
    return np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=float)


def helix_tangent_xyz(r, theta, dz_dtheta):
    return np.array([
        -r * math.sin(theta),
         r * math.cos(theta),
         dz_dtheta
    ], dtype=float)


def hermite_curve(p0, p1, m0, m1, n_pts):
    u = np.linspace(0.0, 1.0, n_pts)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2

    pts = (
        h00[:, None] * p0[None, :]
        + h10[:, None] * m0[None, :]
        + h01[:, None] * p1[None, :]
        + h11[:, None] * m1[None, :]
    )
    return pts

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
    ritardo_min_deg,
    ritardo_max_deg,
):

    lunghezza_mm = lunghezza_m * 1000.0

    d_tubo = d_rame_mm + 2.0 * spessore_guaina_mm
    passo_radiale = d_tubo * (1.0 - compressione_pct / 100.0)
    passo_assiale = d_tubo + gap_axiale_mm

    if passo_radiale <= EPS:
        passo_radiale = EPS
    if passo_assiale <= EPS:
        passo_assiale = EPS

    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0
    r = r0

    z0 = 0.0
    z1 = spalla_mm

    theta = 0.0
    points = []

    base_transition_turn = 0.20
    min_deg = min(ritardo_min_deg, ritardo_max_deg)
    max_deg = max(ritardo_min_deg, ritardo_max_deg)

    while True:
        dz = z1 - z0
        giri = abs(dz) / passo_assiale
        giri = max(giri, 0.1)

        dtheta_layer = 2.0 * math.pi * giri
        dz_dtheta_in = dz / dtheta_layer

        n_layer = max(100, int(giri * 120))
        t = np.linspace(0.0, dtheta_layer, n_layer)

        theta_vals = theta + t
        z_vals = z0 + dz * t / dtheta_layer

        x = r * np.cos(theta_vals)
        y = r * np.sin(theta_vals)

        layer = np.column_stack([x, y, z_vals])

        if len(points) > 0:
            layer = layer[1:]

        if len(points) == 0:
            points = layer.tolist()
        else:
            points.extend(layer.tolist())

        pts_np = np.array(points)
        if polyline_length(pts_np) >= lunghezza_mm:
            break

        # -------------------------------------------------
        # TRANSICIÓ SUAU ENTRE CAPES
        # -------------------------------------------------
        ritardo_deg = 0.0
        if max_deg > 0:
            ritardo_deg = np.random.uniform(min_deg, max_deg)

        extra_turn = math.radians(ritardo_deg) / (2.0 * math.pi)
        total_transition_turn = base_transition_turn + extra_turn
        dtheta_transition = 2.0 * math.pi * total_transition_turn

        r_next = r + passo_radiale

        # punt final de la capa actual
        p0 = cylindrical_point(r, theta + dtheta_layer, z1)

        # punt inicial de la capa següent
        theta_next_start = theta + dtheta_layer + dtheta_transition
        p1 = cylindrical_point(r_next, theta_next_start, z1)

        # pendent axial de la capa següent (canvi de signe)
        dz_next = z0 - z1
        giri_next = abs(dz_next) / passo_assiale
        giri_next = max(giri_next, 0.1)
        dtheta_next = 2.0 * math.pi * giri_next
        dz_dtheta_out = dz_next / dtheta_next

        # tangents 3D consistents amb les hèlixs entrant i sortint
        t0 = helix_tangent_xyz(r, theta + dtheta_layer, dz_dtheta_in)
        t1 = helix_tangent_xyz(r_next, theta_next_start, dz_dtheta_out)

        # escalat de tangents per a Hermite
        m0 = t0 * dtheta_transition
        m1 = t1 * dtheta_transition

        n_trans = max(40, int(total_transition_turn * 220))
        transition = hermite_curve(p0, p1, m0, m1, n_trans)

        transition = transition[1:]
        points.extend(transition.tolist())

        theta = theta_next_start
        r = r_next
        z0, z1 = z1, z0

        pts_np = np.array(points)
        if polyline_length(pts_np) >= lunghezza_mm:
            break

    path = np.array(points)
    path = trim_polyline_to_length(path, lunghezza_mm)

    total_turns = compute_total_turns(path)
    r_max = np.max(np.sqrt(path[:, 0]**2 + path[:, 1]**2))
    diam_ext = 2.0 * (r_max + d_tubo / 2.0)

    capes = int(round((r_max - r0) / passo_radiale)) + 1
    capes = max(1, capes)

    voltes_per_capa = total_turns / capes

    meta = {
        "DiametroTubo": d_tubo,
        "PassoRadiale": passo_radiale,
        "PassoAssiale": passo_assiale,
        "DiametroEsterno": diam_ext,
        "LunghezzaM": polyline_length(path) / 1000.0,
        "Capes": capes,
        "VolteTotali": total_turns,
        "VoltePerCapa": voltes_per_capa,
    }

    return path, meta

# =========================================================
# VIEWER THREEJS
# =========================================================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = max(300, len(pts))

    html = f"""
<div style="position:relative;width:100%;height:{altezza}px;">
<img src="{logo_path}"
style="
position:absolute;
top:50%;
left:50%;
transform:translate(-50%,-50%);
width:60%;
opacity:0.05;
pointer-events:none;
z-index:0;
">

<div id="viewer"
style="width:100%;height:100%;position:absolute;z-index:1;">
</div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const container = document.getElementById("viewer")

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000)

const renderer = new THREE.WebGLRenderer({{antialias:true}})
renderer.setSize(container.clientWidth, container.clientHeight)
container.appendChild(renderer.domElement)

const controls = new THREE.OrbitControls(camera, renderer.domElement)

scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 0.55))

const light = new THREE.DirectionalLight(0xffffff, 0.38)
light.position.set(5, 5, 5)
scene.add(light)

const rawPoints = {points_json}
const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]))

class CurvePath extends THREE.Curve {{

constructor(points) {{
    super()
    this.points = points
}}

getPoint(t) {{

    const n = this.points.length
    const f = t * (n - 1)

    const i = Math.floor(f)
    const i0 = Math.max(0, Math.min(i, n - 2))
    const i1 = i0 + 1

    const tt = f - i0

    return new THREE.Vector3().lerpVectors(this.points[i0], this.points[i1], tt)
}}

}}

const curve = new CurvePath(vectors)

let progress = 0

const tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 40, false)

const tubeMat = new THREE.MeshStandardMaterial({{
color:0xe6e6e6,
roughness:0.92
}})

const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat)
scene.add(tubeMesh)

// ==============================
// CAP PLANES
// ==============================

function createCap(position, dir, color) {{

const geometry = new THREE.CircleGeometry({r_tubo}, 32)

const material = new THREE.MeshBasicMaterial({{
color: color,
side: THREE.DoubleSide
}})

const cap = new THREE.Mesh(geometry, material)

const up = new THREE.Vector3(0,0,1)
const d = dir.clone().normalize()

if (d.length() > 0) {{
    const quat = new THREE.Quaternion().setFromUnitVectors(up, d)
    cap.quaternion.copy(quat)
}}

cap.position.copy(position)
scene.add(cap)

}}

// CAP VERD (entrada)
if (vectors.length >= 2) {{
    const start = vectors[0]
    const dirStart = vectors[1].clone().sub(vectors[0]).multiplyScalar(-1)
    createCap(start, dirStart, 0x00ff00)

    // CAP VERMELL (sortida)
    const end = vectors[vectors.length-1]
    const dirEnd = vectors[vectors.length-1].clone().sub(vectors[vectors.length-2])
    createCap(end, dirEnd, 0xff0000)
}}

// ==============================
// CAMERA
// ==============================

const box = new THREE.Box3().setFromPoints(vectors)
const center = new THREE.Vector3()
box.getCenter(center)

camera.position.set(center.x + 600, center.y + 600, center.z + 300)
camera.lookAt(center)

controls.target.copy(center)

if ({str(animazione).lower()}) {{
    tubeMesh.geometry.setDrawRange(0, 0)
}}

function animate() {{

requestAnimationFrame(animate)

if ({str(animazione).lower()}) {{

    progress += {velocita} * 0.002

    if (progress > 1) progress = 1

    if (tubeGeom.index) {{
        tubeMesh.geometry.setDrawRange(0, Math.floor(progress * tubeGeom.index.count))
    }}
}}

controls.update()
renderer.render(scene, camera)

}}

animate()

</script>
"""
    return html

# =========================================================
# UI
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    diametro_aspo = st.number_input("Diametro aspo (mm)", value=450.0)

with c2:
    spalla = st.number_input("Spalla (mm)", value=95.0)

with c3:
    lunghezza = st.number_input("Lunghezza (m)", value=50.0)

with c4:
    rame_label = st.selectbox("Diametro rame", list(COPPER_SIZES_MM.keys()))

with c5:
    spessore_guaina = st.number_input("Spessore guaina (mm)", value=7.0)

c6, c7, c8, c9 = st.columns(4)

with c6:
    compressione = st.slider("Compressione %", 0.0, 20.0, 0.0)

with c7:
    gap = st.number_input("Gap axiale (mm)", value=0.0)

with c8:
    ritardo_min = st.number_input("Ritardo inversione min (°)", value=0.0)

with c9:
    ritardo_max = st.number_input("Ritardo inversione max (°)", value=0.0)

c10, c11, c12 = st.columns(3)

with c10:
    animazione = st.checkbox("Animazione avvolgimento", True)

with c11:
    velocita = st.slider("Velocità animazione", 0.1, 5.0, 1.0)

with c12:
    altezza = st.slider("Altezza viewer", 400, 900, 700)

d_rame = COPPER_SIZES_MM[rame_label]

path, meta = build_coil_centerline(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    compressione,
    gap,
    ritardo_min,
    ritardo_max
)

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita
)

components.html(html, height=altezza)

# =========================================================
# METRICHE
# =========================================================

st.divider()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Diametro tubo", f"{meta['DiametroTubo']:.2f} mm")

with m2:
    st.metric("Passo radiale", f"{meta['PassoRadiale']:.2f} mm")

with m3:
    st.metric("Passo assiale", f"{meta['PassoAssiale']:.2f} mm")

with m4:
    st.metric("Diametro esterno max", f"{meta['DiametroEsterno']:.1f} mm")

m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric("Strati", meta["Capes"])

with m6:
    st.metric("Spire", f"{meta['VoltePerCapa']:.2f}")

with m7:
    st.metric("Giri totali", f"{meta['VolteTotali']:.2f}")

with m8:
    st.download_button(
        "Scarica centerline SLDCRV",
        data=points_to_sldcrv(path),
        file_name="coil_centerline.sldcrv",
        mime="text/plain"
    )

if meta["DiametroEsterno"] > 750:
    st.warning("Diametro esterno superiore a 750 mm. La bobina potrebbe uscire dal pallet.")
