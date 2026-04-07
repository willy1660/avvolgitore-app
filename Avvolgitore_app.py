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
    for f in glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.svg"):
        return f
    return None

logo = find_logo()
if logo:
    c1, c2 = st.columns([1,5])
    c1.image(logo)
    c2.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

# =========================
# SIMULATION (igual que abans)
# =========================

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def compute_metrics(points, d_tubo):
    if len(points) == 0:
        return {"diam_radiale":0,"max_xy_span":0,"wound_length_m":0}

    radial = np.sqrt(points[:,0]**2 + points[:,1]**2)
    max_r = np.max(radial)

    diam = 2*(max_r + d_tubo/2)

    return {
        "diam_radiale": diam,
        "max_xy_span": diam,
        "wound_length_m": polyline_length(points)/1000
    }

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t,
           lunghezza, altezza, anim, vel, gradi_start, pinza,
           final_points, aspo_mode, guide_offset_x):

    return f"""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<div id="v" style="width:100%;height:{altezza}px;background:black;"></div>

<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, window.innerWidth/{altezza}, 0.1, 10000);
camera.position.set(-500,-700,400);

const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(window.innerWidth,{altezza});
document.getElementById("v").appendChild(renderer.domElement);

const R = {d_aspo}/2;
const Rt = {d_tubo}/2;
const H = {spalla};
let theta = 0;

let depositedLocalPoints = [];
let direction = 1;

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

function computeContactPointLocal(g, t) {{

    if (depositedLocalPoints.length < 40) return t;

    let best=null;
    let bestS=999;

    const line = t.clone().sub(g);
    const len2 = line.lengthSq();

    const startIdx = Math.max(0, depositedLocalPoints.length - 400);

    for(let i=startIdx;i<depositedLocalPoints.length;i++){{

        const p = depositedLocalPoints[i];

        // 🔥 mateixa capa
        if (Math.abs(p.length() - t.length()) > Rt*0.8) continue;

        // 🔥 direcció correcta
        if ((p.z - g.z)*direction < -Rt) continue;

        const v = p.clone().sub(g);
        const s = v.dot(line)/len2;

        if(s<=0||s>=1) continue;
        if(s>0.6) continue;

        const proj = g.clone().add(line.clone().multiplyScalar(s));
        const perp = proj.distanceTo(p);

        if(perp <= Rt*0.35){{
            if(s<bestS){{
                bestS=s;
                best=p.clone();
            }}
        }}
    }}

    return best || t;
}}

function animate(){{
    requestAnimationFrame(animate);

    theta -= 0.02;

    const radius = R + Rt;
    const z = H/2;

    const guide = new THREE.Vector3(-(radius+80), radius, z);
    const target = new THREE.Vector3(0,radius,z);

    const gL = worldToLocal(guide);
    const tL = worldToLocal(target);

    const contactL = computeContactPointLocal(gL, tL);
    const contactW = localToWorld(contactL);

    depositedLocalPoints.push(contactL);

    renderer.render(scene,camera);
}}

animate();
</script>
"""

# =========================
# UI
# =========================

c1,c2,c3,c4 = st.columns(4)

with c1:
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with c2:
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with c3:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)
    gradi_start = st.number_input(t["gradi_start"], value=30.0)
    pinza = st.number_input(t["pinza"], value=0.3)

with c4:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)
    guide_offset_x = st.number_input(t["guide_offset_x"], value=80.0)

d_tubo = d_rame + 2*spessore

points = np.array([[0,0,0]])

metrics = compute_metrics(points, d_tubo)

components.html(
    viewer(
        diametro_aspo, spalla, d_tubo, passo, incremento,
        rit_b, rit_t, lunghezza, altezza, anim, vel,
        gradi_start, pinza, points.tolist(), "visible", guide_offset_x
    ),
    height=altezza
)

st.divider()

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric(t["metric1"], f"{d_tubo:.2f}")
m2.metric(t["metric2"], f"{passo:.2f}")
m3.metric(t["metric3"], f"{incremento:.2f}")
m4.metric(t["metric4"], f"{metrics['diam_radiale']:.1f}")
m5.metric(t["metric5"], f"{metrics['max_xy_span']:.1f}")
m6.metric(t["metric6"], f"{metrics['wound_length_m']:.2f}")
