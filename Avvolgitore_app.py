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
        "gradi_start": "Gradi iniziali (°)",
        "pinza": "Lunghezza tratto libero (m)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
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
        "gradi_start": "Initial degrees (°)",
        "pinza": "Free straight length (m)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Outer diameter",
    }
}

t = TEXTS[lang]

COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

# =========================
# LOGO
# =========================

def find_logo():
    files = glob.glob("*.png")
    return files[0] if files else None

logo = find_logo()

if logo:
    c1, c2 = st.columns([1,5])
    with c1:
        st.image(logo)
    with c2:
        st.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

# =========================
# GEOMETRY
# =========================

def deposited_point(theta, radius, z):
    a = -theta + np.pi
    return np.array([radius*np.cos(a), radius*np.sin(a), z])

# =========================
# SIMULATION (AMB DEPOSICIÓ)
# =========================

def simulate(d_aspo, spalla, d_tubo, passo, incremento, lunghezza):

    R = d_aspo/2
    Rt = d_tubo/2
    H = spalla
    max_len = lunghezza*1000

    theta = 0
    radius = R + Rt
    z = Rt

    pts = []
    length = 0
    direction = 1

    alpha = 0.25  # 🔥 factor de deposició

    pts.append(deposited_point(theta, radius, z))

    for _ in range(200000):

        prev = pts[-1]
        theta -= np.deg2rad(3)

        z += direction * passo / 120

        if z >= H-Rt:
            z = H-Rt
            radius += incremento
            direction = -1

        if z <= Rt:
            z = Rt
            radius += incremento
            direction = 1

        target = deposited_point(theta, radius, z)

        # 🔥 DEPOSICIÓ REAL
        new = prev + alpha * (target - prev)

        # snap final (evita flotació)
        if np.linalg.norm(target - new) < Rt * 0.05:
            new = target.copy()

        seg = np.linalg.norm(new-prev)

        if length + seg > max_len:
            break

        pts.append(new)
        length += seg

    return np.array(pts)

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    d_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spess = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)

d_tubo = d_rame + 2*spess

points = simulate(d_aspo, spalla, d_tubo, passo, incremento, lunghezza)

# =========================
# VIEWER (mateix)
# =========================

components.html(f"""
<div id="c" style="width:100%;height:{altezza}px"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>

const container = document.getElementById("c");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(40, container.clientWidth/{altezza}, 1, 10000);
camera.position.set(-600,-800,500);

const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(container.clientWidth,{altezza});
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);

const light = new THREE.DirectionalLight(0xffffff,1);
light.position.set(500,500,500);
scene.add(light);

const aspo = new THREE.Mesh(
    new THREE.CylinderGeometry({d_aspo/2},{d_aspo/2},{spalla},64),
    new THREE.MeshStandardMaterial({{color:0xff3333}})
);
aspo.rotation.x = Math.PI/2;
aspo.position.z = {spalla}/2;
scene.add(aspo);

const pts = {json.dumps(points.tolist())}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

const curve = new THREE.CatmullRomCurve3(pts);
const geo = new THREE.TubeGeometry(curve, pts.length*2, {d_tubo/2}, 12, false);

const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{color:0xffffff}}));
scene.add(mesh);

function animate(){{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}}

animate();

</script>
""", height=altezza)

# =========================
# METRICS
# =========================

st.divider()

m1,m2,m3,m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
m4.metric(t["metric4"], f"{np.max(np.sqrt(points[:,0]**2+points[:,1]**2))*2:.1f} mm")
