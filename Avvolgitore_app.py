import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

st.title("🌀 Avvolgimento")

# =========================
# INPUTS
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    diam_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with col2:
    d_tubo = st.number_input("Ø Tubo (mm)", value=22.0)
    lunghezza = st.number_input("Lunghezza (m)", value=30.0)

with col3:
    passo = st.number_input("Passo assiale", value=20.0)
    incremento = st.number_input("Incremento strato", value=20.0)
    rit_b = st.number_input("Ritardo base", value=180.0)
    rit_t = st.number_input("Ritardo spalla", value=180.0)

altezza = st.slider("Viewer height", 400, 900, 700)
vel = st.slider("Velocitat", 0.1, 5.0, 1.0)

# =========================
# VIEWER
# =========================

def viewer():

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:black;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth/window.innerHeight, 0.1, 10000);
    camera.position.set(-500,-700,400);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth, {altezza});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const R = {diam_aspo}/2;
    const H = {spalla};
    const Rt = {d_tubo}/2;

    const machine = new THREE.Group();
    scene.add(machine);

    const mat = new THREE.MeshStandardMaterial({{color:0xff3333}});

    const mandrel = new THREE.Mesh(
        new THREE.CylinderGeometry(R,R,H,80), mat
    );
    mandrel.rotation.x = Math.PI/2;
    mandrel.position.z = H/2;
    machine.add(mandrel);

    const guide = new THREE.Mesh(
        new THREE.BoxGeometry(20,20,20),
        new THREE.MeshStandardMaterial({{color:0x0044ff}})
    );
    scene.add(guide);

    let guideY = R + Rt;
    let guideZ = Rt;
    let dir = 1;
    let delay = 0;

    let points = [];
    let total = 0;
    let maxLen = {lunghezza}*1000;

    let tubeMesh = null;

    function updateTubeMesh() {{
        if (points.length < 2) return;

        if (tubeMesh) scene.remove(tubeMesh);

        const curve = new THREE.CatmullRomCurve3(points);

        const geometry = new THREE.TubeGeometry(curve, 200, Rt, 8, false);

        tubeMesh = new THREE.Mesh(
            geometry,
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );

        scene.add(tubeMesh);
    }}

    function getPoint() {{
        const theta = -machine.rotation.z;

        const x = guideY * Math.cos(theta);
        const y = guideY * Math.sin(theta);

        return new THREE.Vector3(x,y,guideZ);
    }}

    function addPoint(p) {{
        if(points.length>0){{
            let d = p.distanceTo(points[points.length-1]);
            if(total+d>maxLen) return;
            total+=d;
        }}
        points.push(p);
        updateTubeMesh();
    }}

    scene.add(new THREE.AmbientLight(0xffffff,0.8));

    function animate(){{
        requestAnimationFrame(animate);

        machine.rotation.z -= 0.02 * {vel};

        if(delay>0) delay--;
        else{{
            guideZ += dir * {passo} * 0.02 * {vel};

            if(guideZ >= H-Rt){{
                guideZ = H-Rt;
                guideY += {incremento};
                delay = {rit_t};
                dir = -1;
            }}

            if(guideZ <= Rt){{
                guideZ = Rt;
                guideY += {incremento};
                delay = {rit_b};
                dir = 1;
            }}
        }}

        guide.position.set(-(R+80),guideY,guideZ);

        addPoint(getPoint());

        controls.update();
        renderer.render(scene,camera);
    }}

    animate();
    </script>
    """

components.html(viewer(), height=altezza)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3 = st.columns(3)
m1.metric("Ø Tubo", f"{d_tubo} mm")
m2.metric("Passo", f"{passo} mm")
m3.metric("Incremento", f"{incremento} mm")
