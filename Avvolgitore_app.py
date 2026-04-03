import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
# =========================

lang = "IT"

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
    if len(points) < 2:
        return points

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    idx = max(0, min(idx, len(points) - 2))

    p0, p1 = points[idx], points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len < EPS:
        return points[:idx + 1]

    alpha = (target_length - cum[idx]) / seg_len
    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

# =========================
# COIL MODEL
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo_ax, passo_rad, rit_min, rit_max):

    L = lunghezza * 1000.0
    d_tubo = d_rame + 2 * spessore
    r_tubo = d_tubo / 2

    r0 = d_aspo / 2 + r_tubo
    z_min = r_tubo
    z_max = spalla - r_tubo

    theta = 0.0
    z = z_min
    r = r0

    dz = passo_ax / (2*np.pi)
    dtheta = np.deg2rad(4)

    direction = +1
    dwell = 0
    pending_rad = False

    pts = []

    def add():
        pts.append([
            r*np.cos(theta),
            r*np.sin(theta),
            z
        ])

    add()

    while polyline_length(np.array(pts)) < L:

        theta -= dtheta  # sentit horari

        if dwell > 0:
            dwell -= np.rad2deg(dtheta)
            add()
            continue

        if pending_rad:
            r += passo_rad
            pending_rad = False

        z += direction * dz * dtheta

        if direction == 1 and z >= z_max:
            z = z_max
            dwell = rit_max
            pending_rad = True
            direction = -1

        elif direction == -1 and z <= z_min:
            z = z_min
            dwell = rit_min
            pending_rad = True
            direction = +1

        add()

    path = np.array(pts)
    path = trim_polyline(path, L)

    path[:,2] -= spalla/2

    return path, d_tubo

# =========================
# VIEWER
# =========================

def build_html(points, d_tubo, altura, anim, speed, d_aspo, spalla):

    pts = json.dumps(points.tolist())

    r_tubo = d_tubo/2
    r_mandrel = d_aspo/2
    flange = r_mandrel + 40

    anim_js = "true" if anim else "false"

    return f"""
    <div style="width:100%;height:{altura}px;" id="viewer"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 1, 10000);
    camera.position.set(800,-900,300);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth, {altura});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const pts = {pts};
    const vec = pts.map(p=> new THREE.Vector3(p[0],p[1],p[2]));

    class Curve extends THREE.Curve {{
        constructor(p){{ super(); this.p=p; }}
        getPoint(t){{
            let i = Math.floor(t*(this.p.length-1));
            let a = this.p[i];
            let b = this.p[Math.min(i+1,this.p.length-1)];
            return new THREE.Vector3().lerpVectors(a,b,t*(this.p.length-1)-i);
        }}
    }}

    const curve = new Curve(vec);

    const geom = new THREE.TubeGeometry(curve, vec.length, {r_tubo}, 32);
    const mat = new THREE.MeshStandardMaterial({{color:0xffffff}});
    const mesh = new THREE.Mesh(geom,mat);
    scene.add(mesh);

    const cyl = new THREE.Mesh(
        new THREE.CylinderGeometry({r_mandrel},{r_mandrel},{spalla},64),
        new THREE.MeshStandardMaterial({{color:0x666666,transparent:true,opacity:0.4}})
    );
    cyl.rotation.x = Math.PI/2;
    scene.add(cyl);

    const base = new THREE.Mesh(
        new THREE.CylinderGeometry({flange},{flange},6,64),
        new THREE.MeshStandardMaterial({{color:0x2e69b9}})
    );
    base.rotation.x = Math.PI/2;
    base.position.z = -{spalla}/2-3;
    scene.add(base);

    const light = new THREE.HemisphereLight(0xffffff,0x222222,1);
    scene.add(light);

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

col1,col2,col3 = st.columns(3)

with col1:
    d_aspo = st.number_input("Ø Aspo", value=450.0)
    spalla = st.number_input("Spalla", value=95.0)

with col2:
    rame = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))
    spess = st.number_input("Spessore", value=7.0)
    lung = st.number_input("Lunghezza", value=50.0)

with col3:
    passo_ax = st.number_input("Passo assiale", value=20.0)
    passo_rad = st.number_input("Incremento", value=20.0)
    rit_min = st.number_input("Rit base", value=180.0)
    rit_max = st.number_input("Rit top", value=180.0)

d_rame = COPPER_SIZES_MM[rame]

# =========================
# RUN
# =========================

path, d_tubo = build_coil(
    d_aspo, spalla, lung,
    d_rame, spess,
    passo_ax, passo_rad,
    rit_min, rit_max
)

html = build_html(path, d_tubo, 700, False, 1.0, d_aspo, spalla)

components.html(html, height=700)
