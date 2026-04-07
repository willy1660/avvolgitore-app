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
        "aspo_mode": "Aspo",
        "aspo_visible": "Visibile",
        "aspo_transparent": "Trasparente",
        "aspo_hidden": "Nascosto",
        "guide_offset_x": "Offset guidatubo X (mm)",
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
        "guide_offset_x": "Guide offset X (mm)",
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
    for f in glob.glob("*.png")+glob.glob("*.jpg")+glob.glob("*.svg"):
        return f
    return None

logo = find_logo()
if logo:
    c1,c2 = st.columns([1,5])
    c1.image(logo)
    c2.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, d_tubo, passo, incremento, lunghezza, altezza):

    return f"""
<div id="v" style="width:100%;height:{altezza}px;background:black;"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(40, window.innerWidth/{altezza}, 0.1, 10000);
camera.position.set(-600,-800,500);

const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(window.innerWidth,{altezza});
document.getElementById("v").appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff,0.8));
const light = new THREE.DirectionalLight(0xffffff,0.6);
light.position.set(500,-500,500);
scene.add(light);

const R = {d_aspo}/2;
const Rt = {d_tubo}/2;
const H = {spalla};

let theta = 0;
let guideZ = Rt;
let direction = 1;
let guideRadius = R + Rt;

let deposited = [];
let activeLayer = [];

let mesh = null;

function worldToLocal(p) {{
    const c = Math.cos(-theta);
    const s = Math.sin(-theta);
    return new THREE.Vector3(p.x*c - p.y*s, p.x*s + p.y*c, p.z);
}}

function localToWorld(p) {{
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return new THREE.Vector3(p.x*c - p.y*s, p.x*s + p.y*c, p.z);
}}

function computeContact(g,t) {{

    const pts = activeLayer.length>20 ? activeLayer : deposited;

    let best=null;
    let bestS=999;

    const line = t.clone().sub(g);
    const len2 = line.lengthSq();

    for(let i=0;i<pts.length;i++){{

        const p = pts[i];

        if (Math.abs(p.length()-t.length())>Rt*0.8) continue;
        if ((p.z-g.z)*direction < -Rt) continue;

        const v = p.clone().sub(g);
        const s = v.dot(line)/len2;

        if(s<=0||s>=1) continue;
        if(s>0.6) continue;

        const proj = g.clone().add(line.clone().multiplyScalar(s));
        const perp = proj.distanceTo(p);

        if(perp<=Rt*0.35){{
            if(s<bestS){{
                bestS=s;
                best=p.clone();
            }}
        }}
    }}

    return best || t;
}}

function rebuild(){{
    if(mesh) scene.remove(mesh);
    if(deposited.length<2) return;

    const curve = new THREE.CatmullRomCurve3(deposited);
    const geo = new THREE.TubeGeometry(curve, deposited.length*2, Rt, 8, false);

    mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{color:0xffffff}}));
    scene.add(mesh);
}}

function animate(){{
    requestAnimationFrame(animate);

    theta -= 0.03;

    guideZ += direction * {passo} * 0.01;

    if(guideZ > H-Rt){{
        guideZ = H-Rt;
        direction = -1;
        guideRadius += {incremento};
        activeLayer = []; // 🔥 reset capa
    }}

    if(guideZ < Rt){{
        guideZ = Rt;
        direction = 1;
        guideRadius += {incremento};
        activeLayer = []; // 🔥 reset capa
    }}

    const guideW = new THREE.Vector3(-(guideRadius+80), guideRadius, guideZ);
    const targetW = new THREE.Vector3(0, guideRadius, guideZ);

    const gL = worldToLocal(guideW);
    const tL = worldToLocal(targetW);

    const contactL = computeContact(gL,tL);
    const contactW = localToWorld(contactL);

    deposited.push(contactL);
    activeLayer.push(contactL);

    if(deposited.length % 5 === 0) rebuild();

    controls.update();
    renderer.render(scene,camera);
}}

animate();
</script>
"""

# =========================
# UI
# =========================

c1,c2,c3 = st.columns(3)

with c1:
    d_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with c2:
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    d_rame = COPPER_SIZES_MM[rame]

with c3:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)

d_tubo = d_rame + 2*spessore

components.html(viewer(d_aspo, spalla, d_tubo, passo, incremento, 30, 700), height=700)
