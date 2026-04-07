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
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro radiale max",
        "metric5": "Ingombro max XY",
        "metric6": "Lunghezza avvolta"
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
        "passo_assiale": "Axial pitch",
        "incremento": "Layer increment",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Max diameter",
        "metric5": "Max XY",
        "metric6": "Length"
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

logo = None
for f in glob.glob("*.png")+glob.glob("*.jpg"):
    logo = f
    break

if logo:
    c1,c2 = st.columns([1,5])
    c1.image(logo)
    c2.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

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
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with c3:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)

d_tubo = d_rame + 2*spessore

# =========================
# VIEWER
# =========================

html = f"""
<div id="viewer" style="width:100%;height:700px;background:black;"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(40, window.innerWidth/700, 0.1, 10000);
camera.position.set(-600,-800,500);

const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(window.innerWidth,700);
document.getElementById("viewer").appendChild(renderer.domElement);

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
        activeLayer = [];
    }}

    if(guideZ < Rt){{
        guideZ = Rt;
        direction = 1;
        guideRadius += {incremento};
        activeLayer = [];
    }}

    const guideW = new THREE.Vector3(-(guideRadius+80), guideRadius, guideZ);
    const targetW = new THREE.Vector3(0, guideRadius, guideZ);

    const gL = worldToLocal(guideW);
    const tL = worldToLocal(targetW);

    const contactL = computeContact(gL,tL);

    deposited.push(contactL);
    activeLayer.push(contactL);

    if(deposited.length % 5 === 0) rebuild();

    controls.update();
    renderer.render(scene,camera);
}}

animate();
</script>
"""

components.html(html, height=700)

# =========================
# METRICS
# =========================

radial = np.sqrt(0**2 + 0**2)
diam = d_aspo + 2*d_tubo

m1,m2,m3,m4,m5,m6 = st.columns(6)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f}")
m3.metric(t["metric3"], f"{incremento:.2f}")
m4.metric(t["metric4"], f"{diam:.1f}")
m5.metric(t["metric5"], f"{diam:.1f}")
m6.metric(t["metric6"], f"{lunghezza:.2f}")
