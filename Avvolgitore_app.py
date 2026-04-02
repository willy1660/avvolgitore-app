import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

EPS = 1e-9

def polyline_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p)>1 else 0


def trim(p, L):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    if cum[-1] <= L:
        return p

    i = np.searchsorted(cum, L) - 1
    p0, p1 = p[i], p[i+1]
    t = (L - cum[i]) / (cum[i+1] - cum[i] + EPS)

    return np.vstack([p[:i+1], p0 + t*(p1-p0)])


def turns(p):
    th = np.unwrap(np.arctan2(p[:,1], p[:,0]))
    return np.sum(np.abs(np.diff(th))) / (2*np.pi)


# =========================================================
# GEOMETRY FIXED
# =========================================================

def build_coil(d_aspo, spalla, Lm, d_rame, guaina, comp, gap, rmin, rmax):

    L = Lm * 1000

    d = d_rame + 2*guaina
    pr = d * (1 - comp/100)
    pa = d + gap

    r = d_aspo/2 + d/2
    z0, z1 = 0, spalla

    theta = 0
    pts = []

    while True:

        dz = z1 - z0
        giri = abs(dz)/pa
        dth_layer = 2*np.pi*giri

        n = max(120, int(giri*150))
        t = np.linspace(0, dth_layer, n)

        th = theta + t
        z = z0 + dz*(t/dth_layer)

        x = r*np.cos(th)
        y = r*np.sin(th)

        layer = np.column_stack([x,y,z])

        if pts:
            layer = layer[1:]

        pts.extend(layer.tolist())

        if polyline_length(np.array(pts)) >= L:
            break

        # =========================
        # TRANSICIÓ CORRECTA
        # =========================

        rit = 0
        if rmax > 0:
            rit = np.random.uniform(rmin, rmax)

        turn = 0.2 + rit/360
        dth = 2*np.pi*turn

        r_next = r + pr

        n = 60
        t = np.linspace(0, dth, n)

        s = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, n))

        th = theta + dth_layer + t   # 🔴 FIX CLAU

        r_vals = r + (r_next - r)*s
        z_vals = np.full_like(th, z1)

        x = r_vals*np.cos(th)
        y = r_vals*np.sin(th)

        trans = np.column_stack([x,y,z_vals])[1:]
        pts.extend(trans.tolist())

        # 🔴 FIX CLAU
        theta = theta + dth_layer + dth

        r = r_next
        z0, z1 = z1, z0

        if polyline_length(np.array(pts)) >= L:
            break

    pts = np.array(pts)
    pts = trim(pts, L)

    ttot = turns(pts)
    rmax = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))

    meta = {
        "DiametroTubo": d,
        "PassoRadiale": pr,
        "PassoAssiale": pa,
        "DiametroEsterno": 2*(rmax + d/2),
        "LunghezzaM": polyline_length(pts)/1000,
        "VolteTotali": ttot,
    }

    return pts, meta


# =========================================================
# VIEWER (igual que abans)
# =========================================================

def viewer(points, d, h, anim, vel):

    pts = json.dumps(points.tolist())

    return f"""
<div id="v" style="width:100%;height:{h}px"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const pts = {pts}.map(p => new THREE.Vector3(p[0],p[1],p[2]))

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x000000)

const camera = new THREE.PerspectiveCamera(45,1,0.1,100000)

const renderer = new THREE.WebGLRenderer({{antialias:true}})
renderer.setSize(window.innerWidth, {h})
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

const geo = new THREE.TubeGeometry(curve, pts.length, {d/2}, 32, false)
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


# =========================================================
# UI
# =========================================================

c1,c2,c3,c4,c5 = st.columns(5)

with c1: d_aspo = st.number_input("Diametro aspo",450.0)
with c2: spalla = st.number_input("Spalla",95.0)
with c3: L = st.number_input("Lunghezza",50.0)
with c4: rame = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))
with c5: guaina = st.number_input("Guaina",7.0)

c6,c7,c8,c9 = st.columns(4)

with c6: comp = st.slider("Compressione",0.0,20.0,0.0)
with c7: gap = st.number_input("Gap",0.0)
with c8: rmin = st.number_input("Ritardo min",0.0)
with c9: rmax = st.number_input("Ritardo max",0.0)

c10,c11,c12 = st.columns(3)

with c10: anim = st.checkbox("Animazione",True)
with c11: vel = st.slider("Velocità",0.1,5.0,1.0)
with c12: h = st.slider("Altezza",400,900,700)

pts, meta = build_coil(
    d_aspo,
    spalla,
    L,
    COPPER_SIZES_MM[rame],
    guaina,
    comp,
    gap,
    rmin,
    rmax
)

components.html(viewer(pts, meta["DiametroTubo"], h, anim, vel), height=h)

st.metric("Diametro esterno", f"{meta['DiametroEsterno']:.1f} mm")
