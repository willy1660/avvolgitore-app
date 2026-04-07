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
        "warning": "⚠️ Outer diameter exceeds 750 mm."
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
# GEOMETRY
# =========================

def deposited_point(theta, radius, z):
    tube_theta = -theta + np.pi
    x = radius * np.cos(tube_theta)
    y = radius * np.sin(tube_theta)
    return np.array([x, y, z], dtype=float)

# =========================
# SIMULATION (FIXED)
# =========================

def simulate(
    d_aspo, spalla, d_tubo, passo, incremento,
    rit_b, rit_t, lunghezza_m, gradi_start
):
    R = d_aspo / 2
    Rt = d_tubo / 2
    H = spalla
    max_len = lunghezza_m * 1000

    theta = np.deg2rad(gradi_start)
    radius = R + Rt
    z = Rt

    points = []
    deposited = 0

    direction = 1
    layer = 0

    deg_step = 3
    alpha = 0.25

    p = deposited_point(theta, radius, z)
    points.append(p)

    for _ in range(300000):

        prev = points[-1]
        theta -= np.deg2rad(deg_step)

        z += direction * passo * (deg_step / 360)

        if z >= H - Rt:
            z = H - Rt
            radius += incremento
            direction = -1
            layer += 1

        elif z <= Rt:
            z = Rt
            radius += incremento
            direction = 1
            layer += 1

        # 🔥 CONTACTE CORRECTE AMB ROTACIÓ
        contact = deposited_point(theta, radius, z)

        # 🔥 DEPOSICIÓ
        new_p = prev + alpha * (contact - prev)

        if np.linalg.norm(contact - new_p) < Rt * 0.05:
            new_p = contact

        seg = np.linalg.norm(new_p - prev)

        if deposited + seg > max_len:
            break

        points.append(new_p)
        deposited += seg

    return np.array(points)

# =========================
# METRICS
# =========================

def metrics(points):
    r = np.sqrt(points[:,0]**2 + points[:,1]**2)
    diam = 2*(np.max(r))

    xy = points[:,:2]
    diff = xy[:,None,:] - xy[None,:,:]
    span = np.sqrt(np.max(np.sum(diff**2, axis=2)))

    return diam, span

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
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)
    gradi_start = st.number_input(t["gradi_start"], value=30.0)
    pinza = st.number_input(t["pinza"], value=0.3)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

d_tubo = d_rame + 2 * spessore

points = simulate(
    diametro_aspo, spalla, d_tubo, passo,
    incremento, rit_b, rit_t, lunghezza, gradi_start
)

diam, span = metrics(points)

# =========================
# VIEWER (SIMPLE FINAL)
# =========================

def viewer():
    return f"""
    <div id="v" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

    <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth/{altezza}, 1, 10000);
    camera.position.set(-600,-800,500);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth,{altezza});
    document.getElementById("v").appendChild(renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff,1);
    light.position.set(500,500,500);
    scene.add(light);

    const pts = {json.dumps(points.tolist())}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, pts.length*2, {d_tubo/2}, 12, false);
    const mat = new THREE.MeshStandardMaterial({{color:0xffffff}});
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);

    function animate(){{
        requestAnimationFrame(animate);
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

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
m4.metric(t["metric4"], f"{diam:.1f} mm")

if span > 750:
    st.warning(t["warning"])
