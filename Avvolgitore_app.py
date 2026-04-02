import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

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


def trim_polyline(points, target):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    if cum[-1] <= target:
        return points

    i = np.searchsorted(cum, target) - 1
    i = max(0, min(i, len(points)-2))

    p0, p1 = points[i], points[i+1]
    t = (target - cum[i]) / (cum[i+1] - cum[i] + EPS)
    cut = p0 + t*(p1-p0)

    return np.vstack([points[:i+1], cut])


def compute_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    return np.sum(np.abs(np.diff(theta))) / (2*np.pi)

# =========================
# GEOMETRY
# =========================

def build_coil(
    d_aspo,
    spalla,
    lunghezza_m,
    d_rame,
    guaina,
    compressione,
    gap,
    rit_min,
    rit_max
):

    lunghezza = lunghezza_m * 1000

    d_tubo = d_rame + 2*guaina
    passo_rad = d_tubo * (1 - compressione/100)
    passo_ass = d_tubo + gap

    r = d_aspo/2 + d_tubo/2
    z0, z1 = 0, spalla

    theta = 0
    pts = []

    while True:

        dz = z1 - z0
        giri = abs(dz)/passo_ass
        dtheta = 2*np.pi*giri

        n = max(120, int(giri*150))
        t = np.linspace(0, dtheta, n)

        theta_vals = theta + t
        z_vals = z0 + dz*(t/dtheta)

        x = r*np.cos(theta_vals)
        y = r*np.sin(theta_vals)

        layer = np.column_stack([x,y,z_vals])

        if pts:
            layer = layer[1:]

        pts.extend(layer.tolist())

        if polyline_length(np.array(pts)) >= lunghezza:
            break

        # ======================
        # TRANSICIÓ REAL
        # ======================

        rit = 0
        if rit_max > 0:
            rit = np.random.uniform(rit_min, rit_max)

        extra = rit/360
        turn = 0.2 + extra
        dth = 2*np.pi*turn

        r_next = r + passo_rad

        n = 60
        t = np.linspace(0, dth, n)

        # easing suau
        s = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, n))

        th_vals = theta + t
        r_vals = r + (r_next - r)*s
        z_vals = np.full_like(th_vals, z1)

        x = r_vals*np.cos(th_vals)
        y = r_vals*np.sin(th_vals)

        trans = np.column_stack([x,y,z_vals])[1:]
        pts.extend(trans.tolist())

        theta += dth
        r = r_next
        z0, z1 = z1, z0

        if polyline_length(np.array(pts)) >= lunghezza:
            break

    pts = np.array(pts)
    pts = trim_polyline(pts, lunghezza)

    turns = compute_turns(pts)
    rmax = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))

    meta = {
        "DiametroTubo": d_tubo,
        "PassoRadiale": passo_rad,
        "PassoAssiale": passo_ass,
        "DiametroEsterno": 2*(rmax + d_tubo/2),
        "LunghezzaM": polyline_length(pts)/1000,
        "VolteTotali": turns,
        "Capes": int((rmax - (d_aspo/2))/passo_rad)+1
    }

    return pts, meta

# =========================
# VIEWER
# =========================

def viewer(points, d_tubo, altezza, anim, vel):

    pts = json.dumps(points.tolist())

    return f"""
<div id="v" style="width:100%;height:{altezza}px"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const pts = {pts}.map(p => new THREE.Vector3(p[0],p[1],p[2]))

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(45,1,0.1,100000)

const renderer = new THREE.WebGLRenderer({{antialias:true}})
renderer.setSize(window.innerWidth, {altezza})
document.getElementById("v").appendChild(renderer.domElement)

const controls = new THREE.OrbitControls(camera, renderer.domElement)

class C extends THREE.Curve {{
    constructor(p) {{ super(); this.p=p }}
    getPoint(t) {{
        const i = t*(this.p.length-1)
        const i0 = Math.floor(i)
        const i1 = Math.min(i0+1,this.p.length-1)
        return this.p[i0].clone().lerp(this.p[i1], i-i0)
    }}
}}

const curve = new C(pts)

const geo = new THREE.TubeGeometry(curve, pts.length, {d_tubo/2}, 32, false)
const mat = new THREE.MeshStandardMaterial({{color:0xdddddd}})
const mesh = new THREE.Mesh(geo, mat)
scene.add(mesh)

scene.add(new THREE.HemisphereLight(0xffffff,0x444444,0.7))

const box = new THREE.Box3().setFromPoints(pts)
const c = new THREE.Vector3()
box.getCenter(c)

camera.position.set(c.x+600,c.y+600,c.z+300)
controls.target.copy(c)

let p=0

function animate(){{
requestAnimationFrame(animate)

if({str(anim).lower()}){{
    p+= {vel}*0.002
    if(p>1)p=1
    mesh.geometry.setDrawRange(0, p*geo.index.count)
}}

controls.update()
renderer.render(scene,camera)
}}

animate()

</script>
"""

# =========================
# UI
# =========================

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    d_aspo = st.number_input("Diametro aspo", value=450.0)

with c2:
    spalla = st.number_input("Spalla", value=95.0)

with c3:
    lunghezza = st.number_input("Lunghezza (m)", value=50.0)

with c4:
    rame = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))

with c5:
    guaina = st.number_input("Guaina", value=7.0)

c6,c7,c8,c9 = st.columns(4)

with c6:
    comp = st.slider("Compressione",0.0,20.0,0.0)

with c7:
    gap = st.number_input("Gap",0.0)

with c8:
    rmin = st.number_input("Ritardo min",0.0)

with c9:
    rmax = st.number_input("Ritardo max",0.0)

c10,c11,c12 = st.columns(3)

with c10:
    anim = st.checkbox("Animazione",True)

with c11:
    vel = st.slider("Velocità",0.1,5.0,1.0)

with c12:
    h = st.slider("Altezza",400,900,700)

# =========================
# RUN
# =========================

pts, meta = build_coil(
    d_aspo,
    spalla,
    lunghezza,
    COPPER_SIZES_MM[rame],
    guaina,
    comp,
    gap,
    rmin,
    rmax
)

components.html(viewer(pts, meta["DiametroTubo"], h, anim, vel), height=h)

# =========================
# METRICS
# =========================

st.divider()

m1,m2,m3,m4 = st.columns(4)

m1.metric("Diametro tubo", f"{meta['DiametroTubo']:.2f} mm")
m2.metric("Passo radiale", f"{meta['PassoRadiale']:.2f} mm")
m3.metric("Passo assiale", f"{meta['PassoAssiale']:.2f} mm")
m4.metric("Diametro esterno", f"{meta['DiametroEsterno']:.1f} mm")
