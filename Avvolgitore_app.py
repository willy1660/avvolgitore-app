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

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2*spessore_guaina_mm

    passo_assiale = max(passo_assiale, EPS)
    passo_radiale = max(passo_radiale, EPS)

    r0 = d_aspo_mm/2 + d_tubo/2
    r = r0

    z0, z1 = 0.0, spalla_mm
    theta = 0

    points = []

    base_transition_turn = 0.18

    while True:

        dz = z1 - z0
        turns = max(abs(dz)/passo_assiale, 0.1)
        dtheta = 2*np.pi*turns
        dz_dtheta_in = dz / dtheta

        t = np.linspace(0, dtheta, int(turns*200)+80)

        theta_vals = theta + t
        z_vals = z0 + dz*(t/dtheta)

        x = r*np.cos(theta_vals)
        y = r*np.sin(theta_vals)

        layer = np.column_stack([x,y,z_vals])

        if len(points) > 0:
            layer = layer[1:]

        points.extend(layer.tolist())

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

        ritardo = np.random.uniform(ritardo_min_deg, ritardo_max_deg)
        extra_turn = ritardo / 360.0

        total_turn = base_transition_turn + extra_turn
        dtheta_trans = 2*np.pi*total_turn

        r_next = r + passo_radiale

        dz_next = z0 - z1
        turns_next = max(abs(dz_next)/passo_assiale, 0.1)
        dtheta_next = 2*np.pi*turns_next
        dz_dtheta_out = dz_next / dtheta_next

        t_trans = np.linspace(0, dtheta_trans, int(total_turn*240)+60)
        u = t_trans / dtheta_trans

        theta_trans = theta + dtheta + t_trans

        s = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, len(t_trans)))
        r_trans = r + (r_next - r)*s

        z_trans = hermite_scalar(
            z1, z1,
            dz_dtheta_in*dtheta_trans,
            dz_dtheta_out*dtheta_trans,
            u
        )

        x = r_trans*np.cos(theta_trans)
        y = r_trans*np.sin(theta_trans)

        points.extend(np.column_stack([x,y,z_trans])[1:].tolist())

        theta += dtheta + dtheta_trans
        r = r_next
        z0, z1 = z1, z0

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

    path = trim_polyline(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    capes = int((r_max - r0)/passo_radiale) + 1
    turns_tot = compute_total_turns(path)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext,
        "Capes": capes,
        "VolteTotali": turns_tot,
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = min(4000, max(800, int(len(pts)*0.5)))

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
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 0.7));

const light = new THREE.DirectionalLight(0xffffff, 0.5);
light.position.set(5,5,5);
scene.add(light);

const rawPoints = {points_json};
const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

class CurvePath extends THREE.Curve {{
  constructor(points) {{
    super();
    this.points = points;
  }}
  getPoint(t) {{
    const n = this.points.length;
    const f = t*(n-1);
    const i = Math.floor(f);
    const i0 = Math.max(0, Math.min(i, n-2));
    const i1 = i0+1;
    const tt = f-i0;
    return new THREE.Vector3().lerpVectors(this.points[i0], this.points[i1], tt);
  }}
}}

const curve = new CurvePath(vectors);

let tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 48, false);
tubeGeom = tubeGeom.toNonIndexed();

const tubeMesh = new THREE.Mesh(
  tubeGeom,
  new THREE.MeshStandardMaterial({{
    color:0xe6e6e6,
    roughness:0.85,
    metalness:0.1
  }})
);

scene.add(tubeMesh);

// CAPS
function createCap(position, direction, color) {{
  const geometry = new THREE.CircleGeometry({r_tubo}, 32);
  const material = new THREE.MeshBasicMaterial({{color:color, side:THREE.DoubleSide}});
  const cap = new THREE.Mesh(geometry, material);

  const up = new THREE.Vector3(0,0,1);
  const quat = new THREE.Quaternion().setFromUnitVectors(up, direction.clone().normalize());

  cap.quaternion.copy(quat);
  cap.position.copy(position);

  scene.add(cap);
}}

if (vectors.length >= 2) {{
  createCap(vectors[0], vectors[1].clone().sub(vectors[0]).multiplyScalar(-1), 0x00ff00);
  createCap(vectors[vectors.length-1], vectors[vectors.length-1].clone().sub(vectors[vectors.length-2]), 0xff0000);
}}

// CAMERA
const box = new THREE.Box3().setFromPoints(vectors);
const center = new THREE.Vector3();
box.getCenter(center);

const size = new THREE.Vector3();
box.getSize(size);

const dist = Math.max(size.x,size.y,size.z)*1.8;

camera.position.set(center.x+dist, center.y+dist, center.z+dist*0.6);
camera.lookAt(center);
controls.target.copy(center);

// ANIMACIÓ REAL
let progress = 0;
const total = tubeGeom.attributes.position.count;

if ({str(animazione).lower()}) {{
  tubeGeom.setDrawRange(0,0);
}} else {{
  tubeGeom.setDrawRange(0,total);
}}

function animate(){{
  requestAnimationFrame(animate);

  if ({str(animazione).lower()}) {{
    progress += {velocita} * 0.002;
    if (progress > 1) progress = 1;

    const visible = Math.floor(progress * total);
    tubeGeom.setDrawRange(0, visible);
  }}

  controls.update();
  renderer.render(scene,camera);
}}

animate();
</script>
"""
    return html

# =========================
# UI COMPACTA EN COLUMNES
# =========================

colA, colB, colC, colD = st.columns(4)

# -------------------------
# 🟦 BOBINA
# -------------------------
with colA:
    st.markdown("#### 🟦 Bobina")

    diametro_aspo = st.number_input("Ø Aspo (mm)", 450.0)
    spalla = st.number_input("Spalla (mm)", 95.0)

# -------------------------
# 🟩 TUBO
# -------------------------
with colB:
    st.markdown("#### 🟩 Tubo")

    rame_label = st.selectbox("Ø Rame (in")", list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input("Spessore isolamento (mm)", 7.0)
    lunghezza = st.number_input("Lunghezza rotolo (m)", 50.0)

    d_rame = COPPER_SIZES_MM[rame_label]
    d_tubo = d_rame + 2*spessore_guaina

# -------------------------
# 🟧 AVVOLGIMENTO
# -------------------------
with colC:
    st.markdown("#### 🟧 Avvolg.")

    passo_assiale = st.number_input("Passo assiale (mm)", value=float(d_tubo))
    incremento_strato = st.number_input("Incremento strato (mm)", value=float(d_tubo))

    ritardo_min = st.number_input("Ritardo min (º)", 0.0, 720.0, 360.0)
    ritardo_max = st.number_input("Ritardo max (º)", 0.0, 720.0, 360.0)

# -------------------------
# ⚙️ VIEWER
# -------------------------
with colD:
    st.markdown("#### ⚙️ Viewer")

    altezza = st.slider("Altezza", 400, 900, 700)
    animazione = st.checkbox("Animazione", False)
    velocita = st.slider("Velocità", 0.1, 5.0, 1.0)

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
    ritardo_min,
    ritardo_max,
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
# METRICS + WARNING
# =========================

st.divider()

m1,m2,m3,m4 = st.columns(4)

m1.metric("Diametro tubo", f"{meta['DiametroTubo']:.2f} mm")
m2.metric("Passo assiale", f"{meta['PassoAssiale']:.2f} mm")
m3.metric("Incremento strato", f"{meta['IncrementoStrato']:.2f} mm")
m4.metric("Diametro esterno", f"{meta['DiametroEsterno']:.1f} mm")

if meta["DiametroEsterno"] > 750:
    st.warning("⚠️ Diametro esterno superiore a 750 mm. La bobina potrebbe uscire dal pallet.")
