import os
import glob
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

lang_option = st.selectbox(
    "🌍 Language",
    ["🇮🇹 Italiano", "🇺🇸 English (US)"],
    index=0 if st.session_state.lang == "IT" else 1
)

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
lang = st.session_state.lang

TEXTS = {
    "IT": {
        "title": "Avvolgimento",
        "bobina": "🟦 Bobina",
        "tubo": "🟩 Tubo",
        "avvolg": "🟧 Avvolgimento",
        "viewer": "⚙️ Viewer",
        "diam_aspo": "Ø Aspo (mm)",
        "spalla": "Spalla (mm)",
        "rame": "Ø Rame",
        "isolamento": "Spessore guaina (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm/rev)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
    },
    "EN": {
        "title": "Coiling",
        "bobina": "🟦 Coil",
        "tubo": "🟩 Tube",
        "avvolg": "🟧 Winding",
        "viewer": "⚙️ Viewer",
        "diam_aspo": "Spool diameter (mm)",
        "spalla": "Width (mm)",
        "rame": "Copper size",
        "isolamento": "Foam thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm/rev)",
        "incremento": "Layer increment (mm)",
        "rit_min": "Bottom delay (°)",
        "rit_max": "Top delay (°)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
    }
}

t = TEXTS[lang]

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
# FIXED PARAMS
# =========================

gradi_start = 0.0

# =========================
# UTILS
# =========================

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def deposited_point(theta, radius, z):
    return np.array([
        radius * np.cos(-theta + np.pi),
        radius * np.sin(-theta + np.pi),
        z
    ])

# =========================
# SIMULATION (SMOOTH)
# =========================

def simulate(
    d_aspo, spalla, d_tubo,
    passo, incremento,
    rit_b, rit_t,
    lunghezza_m
):
    max_len = lunghezza_m * 1000
    Rt = d_tubo / 2
    H = spalla

    theta = 0
    radius = d_aspo/2 + Rt
    z = Rt

    direction = 1
    mode = "axial"

    turn_progress = 0
    turn_delay = 0
    r0 = radius
    r1 = radius

    pts = [deposited_point(theta, radius, z)]
    L = 0

    for _ in range(300000):
        prev = pts[-1]
        theta -= np.deg2rad(4)

        if mode == "axial":
            z += direction * passo * (4/360)

            if z >= H - Rt:
                z = H - Rt
                mode = "turn"
                turn_delay = rit_t
                turn_progress = 0
                r0 = radius
                r1 = radius + incremento

            elif z <= Rt:
                z = Rt
                mode = "turn"
                turn_delay = rit_b
                turn_progress = 0
                r0 = radius
                r1 = radius + incremento

        else:
            turn_progress += 4
            s = smoothstep(turn_progress / max(turn_delay,1))
            radius = r0 + s*(r1-r0)

            if turn_progress >= turn_delay:
                radius = r1
                direction *= -1
                mode = "axial"

        new_p = deposited_point(theta, radius, z)
        seg = np.linalg.norm(new_p-prev)

        if seg > 0.4:
            pts.append(new_p)
            L += seg

        if L >= max_len:
            break

    return np.array(pts)

# =========================
# VIEWER (FIXED)
# =========================

def viewer(points, height):

    return f"""
    <div id="viewer" style="width:100%;height:{height}px;background:black;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(40, container.clientWidth/container.clientHeight, 0.1, 10000);
    camera.position.set(-500,-700,400);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const light = new THREE.DirectionalLight(0xffffff,1);
    light.position.set(300,-300,400);
    scene.add(light);

    const mat = new THREE.LineBasicMaterial({{color:0xffffff}});

    const pts = {json.dumps(points.tolist())}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat);
    scene.add(line);

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

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=50.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=360.0)
    rit_t = st.number_input(t["rit_max"], value=360.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)

# =========================
# BUILD
# =========================

d_tubo = d_rame + 2*spessore

points = simulate(
    diametro_aspo,
    spalla,
    d_tubo,
    passo,
    incremento,
    rit_b,
    rit_t,
    lunghezza
)

components.html(viewer(points, altezza), height=altezza)
