# =========================
# AVVOLGIMENTO PRO - OPTIMIZED
# =========================

import os
import glob
import json
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

# =========================
# UI
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    d_aspo = st.number_input("Ø Aspo (mm)", 200.0, 1200.0, 450.0)
    spalla = st.number_input("Spalla (mm)", 50.0, 300.0, 95.0)

with col2:
    rame = st.selectbox("Ø Rame", list(COPPER_SIZES_MM.keys()))
    spess = st.number_input("Guaina (mm)", 0.0, 20.0, 7.0)
    lunghezza = st.number_input("Lunghezza (m)", 1.0, 200.0, 30.0)

with col3:
    passo = st.number_input("Passo assiale", 5.0, 50.0, 20.0)
    incremento = st.number_input("Incremento strato", 5.0, 50.0, 20.0)
    rit = st.number_input("Ritardo (°)", 0.0, 360.0, 180.0)

with col4:
    anim = st.checkbox("Animazione", True)
    vel = st.slider("Velocità", 0.5, 5.0, 1.0)
    aspo_mode = st.selectbox("Aspo", ["visible", "transparent", "hidden"])

d_tubo = COPPER_SIZES_MM[rame] + 2 * spess

# =========================
# SIMULATION (FAST HYBRID)
# =========================

def simulate_fast():
    R = d_aspo / 2
    Rt = d_tubo / 2
    H = spalla
    max_len = lunghezza * 1000

    pts = []
    theta = 0.0
    z = Rt
    r = R + Rt

    deposited = 0.0
    dir = 1
    layer = 0

    step_deg = 5
    step_rad = np.deg2rad(step_deg)

    def pt(th, rad, z):
        a = -th + np.pi
        return np.array([rad*np.cos(a), rad*np.sin(a), z])

    pts.append(pt(theta, r, z))

    while deposited < max_len:

        theta -= step_rad

        # FIRST LAYER → real simulation
        if layer == 0:
            z += dir * passo * (step_deg / 360)

            if z >= H - Rt:
                z = H - Rt
                layer += 1
                r += incremento
                dir = -1

            elif z <= Rt:
                z = Rt
                layer += 1
                r += incremento
                dir = 1

        else:
            # FAST MODE → no delay simulation
            z += dir * passo * (step_deg / 360)

            if z >= H - Rt or z <= Rt:
                dir *= -1
                r += incremento

        new = pt(theta, r, z)
        seg = np.linalg.norm(new - pts[-1])

        if deposited + seg > max_len:
            break

        pts.append(new)
        deposited += seg

    return np.array(pts)

points = simulate_fast()

# =========================
# METRICS
# =========================

def metrics(points):
    r = np.sqrt(points[:,0]**2 + points[:,1]**2)
    diam = 2*(np.max(r) + d_tubo/2)

    xy = points[:,:2]
    diff = xy[:,None,:] - xy[None,:,:]
    span = np.sqrt(np.max(np.sum(diff**2, axis=2))) + d_tubo

    length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) / 1000

    return diam, span, length

diam, span, length = metrics(points)

# =========================
# VIEWER
# =========================

def viewer():

    return f"""
    <div id="v" style="width:100%;height:700px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

    <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth/700, 1, 10000);
    camera.position.set(-600,-800,500);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth,700);
    document.getElementById("v").appendChild(renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff,1);
    light.position.set(500,500,500);
    scene.add(light);

    // =================
    // COIL
    // =================

    const pts = {json.dumps(points.tolist())}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, pts.length*2, {d_tubo/2}, 12, false);
    const mat = new THREE.MeshStandardMaterial({{color:0xffffff}});
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);

    // =================
    // START / END
    // =================

    const s = new THREE.Mesh(new THREE.SphereGeometry(6), new THREE.MeshBasicMaterial({{color:0x00ff00}}));
    s.position.copy(pts[0]);
    scene.add(s);

    const e = new THREE.Mesh(new THREE.SphereGeometry(6), new THREE.MeshBasicMaterial({{color:0xffaa00}}));
    e.position.copy(pts[pts.length-1]);
    scene.add(e);

    // =================
    // ASP0
    // =================

    if ("{aspo_mode}" !== "hidden") {{

        const matA = new THREE.MeshStandardMaterial({{
            color:0xff3333,
            transparent: "{aspo_mode}"==="transparent",
            opacity: "{aspo_mode}"==="transparent"?0.2:1
        }});

        const cyl = new THREE.Mesh(
            new THREE.CylinderGeometry({d_aspo/2},{d_aspo/2},{spalla},64),
            matA
        );
        cyl.rotation.x = Math.PI/2;
        cyl.position.z = {spalla}/2;
        scene.add(cyl);
    }}

    function animate(){{
        requestAnimationFrame(animate);
        renderer.render(scene,camera);
    }}

    animate();
    </script>
    """

components.html(viewer(), height=700)

# =========================
# METRICS UI
# =========================

c1, c2, c3 = st.columns(3)

c1.metric("Diametro", f"{diam:.1f} mm")
c2.metric("Ingombro XY", f"{span:.1f} mm")
c3.metric("Lunghezza reale", f"{length:.2f} m")

if span > 750:
    st.warning("⚠️ No entra en pallet")
