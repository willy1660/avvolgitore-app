import os
import glob
import json
import math
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
        "aspo_mode": "Aspo",
        "aspo_visible": "Visibile",
        "aspo_transparent": "Trasparente",
        "aspo_hidden": "Nascosto",
        "guide_offset_x": "Offset guidatubo (mm)",
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
        "gradi_start": "Initial degrees (°)",
        "pinza": "Free straight length (m)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "aspo_mode": "Spool",
        "aspo_visible": "Visible",
        "aspo_transparent": "Transparent",
        "aspo_hidden": "Hidden",
        "guide_offset_x": "Guide offset (mm)",
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
# LOGO
# =========================

def find_logo():
    candidates = [
        "New Logo PDM – rame.png",
        "New Logo PDM - rame.png",
        "new_logo_pdm_rame.png",
        "logo.png",
        "logo.svg",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
    ]
    for name in candidates:
        if os.path.exists(name):
            return name
    for pattern in ("*.png", "*.svg", "*.jpg", "*.jpeg", "*.webp"):
        files = glob.glob(pattern)
        if files:
            return files[0]
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
# SIMULATION (NO TOCAR LOGICA)
# =========================

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x*x*(3-2*x)

def deposited_point(theta, radius, z):
    t = -theta + np.pi
    return np.array([radius*np.cos(t), radius*np.sin(t), z])

def simulate(d_aspo, spalla, d_tubo, passo, incremento, lunghezza_m):
    R = d_aspo/2
    Rt = d_tubo/2
    H = spalla

    theta = 0
    radius = R + Rt
    z = Rt
    dir = 1

    pts = [deposited_point(theta, radius, z)]
    total = 0
    max_len = lunghezza_m*1000

    for _ in range(200000):
        prev = pts[-1]
        theta += 0.05

        z += dir * passo * 0.05 / 360

        if z >= H-Rt:
            z = H-Rt
            dir = -1
            radius += incremento

        if z <= Rt:
            z = Rt
            dir = 1
            radius += incremento

        p = deposited_point(theta, radius, z)
        seg = np.linalg.norm(p-prev)

        if seg < Rt*0.2:
            continue

        if total+seg >= max_len:
            break

        pts.append(p)
        total += seg

    return np.array(pts)

def compute_metrics(points, d_tubo):
    r = np.sqrt(points[:,0]**2 + points[:,1]**2)
    diam = 2*(np.max(r)+d_tubo/2)
    span = np.max(r)*2 + d_tubo
    length = np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))/1000
    return diam, span, length

# =========================
# UI INPUTS
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

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)
    guide_offset = st.number_input(t["guide_offset_x"], value=120.0)

d_tubo = d_rame + 2*spessore

points = simulate(diametro_aspo, spalla, d_tubo, passo, incremento, lunghezza)
diam, span, length = compute_metrics(points, d_tubo)

# =========================
# VIEWER (FIX IMPORTANT)
# =========================

def viewer():
    pts_json = json.dumps(points.tolist())

    return f"""
    <div id="v" style="height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const pts = {pts_json}.map(p => new THREE.Vector3(p[0],p[1],p[2]));

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth/window.innerHeight, 0.1, 10000);
    camera.position.set(-500,-800,400);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth,{altezza});
    document.getElementById("v").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const R = {diametro_aspo}/2;
    const Rt = {d_tubo}/2;
    const H = {spalla};

    const red = new THREE.MeshStandardMaterial({{color:0xff3333}});
    const blue = new THREE.MeshStandardMaterial({{color:0x0044ff}});
    const white = new THREE.MeshStandardMaterial({{color:0xffffff}});

    // ASP0
    const mandrel = new THREE.Mesh(new THREE.CylinderGeometry(R,R,H,64), red);
    mandrel.rotation.x = Math.PI/2;
    mandrel.position.z = H/2;
    scene.add(mandrel);

    // GUIDATUBO FIX
    const guide = new THREE.Mesh(new THREE.BoxGeometry(30,20,20), blue);
    scene.add(guide);

    // FIX TOTAL
    const guideX = -(R + {guide_offset});
    const guideY = 0;

    // BOBINA
    const curve = new THREE.CatmullRomCurve3(pts);
    const geo = new THREE.TubeGeometry(curve, pts.length*2, Rt, 10, false);
    const mesh = new THREE.Mesh(geo, white);
    scene.add(mesh);

    // TUB RECTE FINAL
    const last = pts[pts.length-1];
    const guideP = new THREE.Vector3(guideX, guideY, last.z);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([guideP, last]);
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({{color:0x00ffff}}));
    scene.add(line);

    guide.position.copy(guideP);

    scene.add(new THREE.AmbientLight(0xffffff,0.8));

    function animate(){{
        requestAnimationFrame(animate);
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

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
m4.metric(t["metric4"], f"{diam:.1f} mm")
m5.metric(t["metric5"], f"{span:.1f} mm")
m6.metric(t["metric6"], f"{length:.3f} m")
if metrics["max_xy_span"] > 750:
    st.warning(t["warning"])
