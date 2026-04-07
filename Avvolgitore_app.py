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
# GEOMETRY (NO TOCAT)
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo, incremento, rit_b, rit_t, gradi_start, pinza):
    pts = []
    r = d_aspo/2 + (d_rame + 2*spessore)/2
    z_min, z_max = 0, spalla
    z = z_min
    theta = 0
    dir = 1
    delay = 0
    pending = False

    for _ in range(20000):
        theta += np.deg2rad(4)

        if delay > 0:
            delay -= 4
        else:
            if pending:
                r += incremento
                pending = False

            z += dir * (passo/(2*np.pi)) * np.deg2rad(4)

            if z >= z_max:
                z = z_max
                delay = rit_t
                pending = True
                dir = -1

            elif z <= z_min:
                z = z_min
                delay = rit_b
                pending = True
                dir = 1

        x = r*np.cos(theta)
        y = r*np.sin(theta)
        pts.append([x,y,z])

        if len(pts)>2:
            if np.sum(np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1)) > lunghezza*1000:
                break

    pts = np.array(pts)
    pts[:,2] -= spalla/2
    return pts

# =========================
# VIEWER FIXAT (EIXOS CORRECTES)
# =========================

def viewer(d_aspo, spalla, d_tubo):

    return f"""
    <div id="viewer" style="width:100%;height:700px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    setTimeout(() => {{

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 10000);
        camera.position.set(400, -600, 300);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(window.innerWidth, 700);
        document.getElementById("viewer").appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // ASPO
        // =====================

        const R = {d_aspo}/2;
        const H = {spalla};

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, H, 64),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        scene.add(mandrel);

        const flangeR = R + 120;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        base.rotation.x = Math.PI/2;
        base.position.z = -H/2 - 3;
        scene.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = H/2 + 3;
        scene.add(top);

        // =====================
        // GUIDATUBO (ARA EN Y)
        // =====================

        const R_tube = {d_tubo}/2;

        const guideY = -(R + R_tube);   // 🔥 CORRECTE
        const guideZ = -H/2 + R_tube;

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30,20,20),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        guide.position.set(0, guideY, guideZ);
        scene.add(guide);

        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(6,6,20,16),
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );

        nozzle.rotation.x = Math.PI/2;
        nozzle.position.set(0, guideY + 15, guideZ);
        scene.add(nozzle);

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff,0.8));

        const light = new THREE.DirectionalLight(0xffffff,0.6);
        light.position.set(500,-500,800);
        scene.add(light);

        function animate(){{
            requestAnimationFrame(animate);
            renderer.render(scene,camera);
        }}

        animate();

    }}, 50);
    </script>
    """

# =========================
# UI ORIGINAL (INTACTE)
# =========================

colA, colB = st.columns(2)

with colA:
    diametro_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with colB:
    rame = st.selectbox("Ø Rame", list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input("Isolamento (mm)", value=7.0)
    d_rame = COPPER_SIZES_MM[rame]
    d_tubo = d_rame + 2*spessore

components.html(viewer(diametro_aspo, spalla, d_tubo), height=700)
