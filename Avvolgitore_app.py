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
        "aspo_mode": "Aspo",
        "aspo_visible": "Visibile",
        "aspo_transparent": "Trasparente",
        "aspo_hidden": "Nascosto",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro radiale max",
        "metric5": "Ingombro max XY",
        "metric6": "Lunghezza avvolta",
        "warning": "⚠️ Ingombro max XY superiore a 750 mm."
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
        "aspo_mode": "Spool",
        "aspo_visible": "Visible",
        "aspo_transparent": "Transparent",
        "aspo_hidden": "Hidden",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Max radial diameter",
        "metric5": "Max XY span",
        "metric6": "Wound length",
        "warning": "⚠️ Max XY span exceeds 750 mm."
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
# FIXED VALUES (REMOVED UI)
# =========================

gradi_start = 0.0
guide_offset_x = 150.0

# =========================
# LOGO
# =========================

def find_logo():
    for f in glob.glob("*.png"):
        return f
    return None

logo_path = find_logo()

if logo_path:
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(logo_path, use_container_width=True)
    with c2:
        st.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

# =========================
# UTILS
# =========================

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def deposited_point(theta, radius, z):
    return np.array([
        radius * np.cos(-theta + np.pi),
        radius * np.sin(-theta + np.pi),
        z
    ])

# =========================
# SIMULATION (SMOOTH FIX)
# =========================

def simulate_winding(
    d_aspo, spalla, d_tubo, passo,
    incremento, rit_b, rit_t, lunghezza_m
):
    max_len = lunghezza_m * 1000.0
    Rt = d_tubo / 2.0
    H = spalla

    theta = 0.0
    radius = d_aspo/2 + Rt
    z = Rt

    direction = 1
    mode = "axial"

    turn_progress = 0
    turn_delay = 0
    r0 = radius
    r1 = radius

    points = [deposited_point(theta, radius, z)]
    deposited_len = 0

    for _ in range(500000):

        prev = points[-1]
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

            # 🔥 SUAVITZAT CLAU
            s = smoothstep(turn_progress / max(turn_delay, 1))
            radius = r0 + s * (r1 - r0)

            if turn_progress >= turn_delay:
                radius = r1
                direction *= -1
                mode = "axial"

        new_p = deposited_point(theta, radius, z)
        seg = np.linalg.norm(new_p - prev)

        if seg > 0.4:
            points.append(new_p)
            deposited_len += seg

        if deposited_len >= max_len:
            break

    return np.array(points)

# =========================
# VIEWER (INTOCABLE)
# =========================

def viewer(points, h):
    return f"""
    <div id="viewer" style="width:100%;height:{h}px;background:black;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, window.innerWidth/{h}, 0.1, 10000);
    camera.position.set(400,-600,400);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth,{h});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const mat = new THREE.LineBasicMaterial({{color:0xffffff}});
    const pts = {json.dumps(points.tolist())}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo,mat);
    scene.add(line);

    function animate(){{
        requestAnimationFrame(animate);
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
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

d_tubo = d_rame + 2*spessore

points = simulate_winding(
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

st.write("Points:", len(points))
