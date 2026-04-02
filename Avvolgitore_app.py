import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# 🌍 LANGUAGE
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
        "isolamento": "Spessore isolamento (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "pre_rot": "Pre-rotazione mandrino (°)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
        "warning": "⚠️ Diametro esterno superiore a 750 mm."
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
        "isolamento": "Insulation thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm)",
        "incremento": "Layer increment (mm)",
        "rit_min": "Bottom delay (°)",
        "rit_max": "Top delay (°)",
        "pre_rot": "Mandrel pre-rotation (°)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Outer diameter",
        "warning": "⚠️ Outer diameter exceeds 750 mm."
    }
}

t = TEXTS[lang]

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1, 7])

logo_path = os.path.join(os.path.dirname(__file__), "New Logo PDM - rame.png")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)

with col_title:
    st.markdown(f"# {t['title']}")

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
    pre_rot_deg,
):
    lunghezza_mm = lunghezza_m * 1000.0
    d_tubo = d_rame_mm + 2.0 * spessore_guaina_mm

    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0
    r = r0

    z = 0.0
    theta = 0.0
    direction = 1

    theta_step = np.deg2rad(4.0)
    dz_dtheta = passo_assiale / (2.0 * np.pi)

    points = []

    def add():
        points.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    # =========================
    # PRE-ROTATION (MANDRÍ)
    # =========================
    if pre_rot_deg > 0:
        theta_pre = np.deg2rad(pre_rot_deg)
        steps = max(8, int(pre_rot_deg / 5))

        for i in range(1, steps + 1):
            u = i / steps
            th = theta + theta_pre * u
            points.append([r*np.cos(th), r*np.sin(th), z])

        theta += theta_pre

    length = 0

    while length < lunghezza_mm:

        theta += theta_step
        z += direction * dz_dtheta * theta_step

        if direction == 1 and z >= spalla_mm:
            z = spalla_mm
            direction = -1
            r += passo_radiale

        elif direction == -1 and z <= 0:
            z = 0
            direction = 1
            r += passo_radiale

        new_point = [r*np.cos(theta), r*np.sin(theta), z]

        if len(points) > 0:
            prev = np.array(points[-1])
            length += np.linalg.norm(np.array(new_point) - prev)

        points.append(new_point)

    path = np.array(points)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext
    }

    return path, meta

# =========================
# VIEWER (NO TOCAT)
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = min(4000, max(800, int(len(pts) * 0.5)))

    html = f"""<div style="width:100%;height:{altezza}px;">
    <div id="viewer" style="width:100%;height:100%;"></div></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>
    <script>
    const container=document.getElementById("viewer");
    const scene=new THREE.Scene();
    scene.background=new THREE.Color(0x000000);
    const camera=new THREE.PerspectiveCamera(45,container.clientWidth/container.clientHeight,0.1,100000);
    const renderer=new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.clientWidth,container.clientHeight);
    container.appendChild(renderer.domElement);
    const controls=new THREE.OrbitControls(camera,renderer.domElement);
    const rawPoints={points_json};
    const vectors=rawPoints.map(p=>new THREE.Vector3(p[0],p[1],p[2]));
    const curve=new THREE.CatmullRomCurve3(vectors);
    const tube=new THREE.TubeGeometry(curve,{tubular_segments},{r_tubo},32,false);
    const mesh=new THREE.Mesh(tube,new THREE.MeshStandardMaterial());
    scene.add(mesh);
    camera.position.set(500,500,500);
    function animate(){{requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}}
    animate();
    </script>"""
    return html

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input(t["diam_aspo"], 450.0)
    spalla = st.number_input(t["spalla"], 95.0)

with colB:
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], 7.0)
    lunghezza = st.number_input(t["lunghezza"], 50.0)
    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    passo_assiale = st.number_input(t["passo_assiale"], 20.0)
    incremento_strato = st.number_input(t["incremento"], 20.0)
    ritardo_min = st.number_input(t["rit_min"], 180.0)
    ritardo_max = st.number_input(t["rit_max"], 180.0)

with colD:
    pre_rot = st.number_input(t["pre_rot"], 180.0)  # 🔥 NOU INPUT
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

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
    pre_rot
)

components.html(
    build_viewer_html(path, meta["DiametroTubo"], altezza, animazione, velocita),
    height=altezza
)

st.write(meta)
