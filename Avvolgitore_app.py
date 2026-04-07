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
        "guide_offset_x": "Offset guidatubo (mm)",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro radiale max",
        "metric5": "Ingombro max XY",
        "metric6": "Lunghezza avvolta",
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
        "rit_min": "Bottom delay",
        "rit_max": "Top delay",
        "gradi_start": "Start angle",
        "pinza": "Free length",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "guide_offset_x": "Guide offset",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Max diameter",
        "metric5": "Max XY span",
        "metric6": "Length",
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
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"#### {t['bobina']}")
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)
    guide_offset = st.number_input(t["guide_offset_x"], value=80.0)

d_tubo = d_rame + 2 * spessore

# =========================
# VIEWER (FIX DEFINITIU)
# =========================

def viewer():
    return f"""
    <div id="v" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
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
    const offset = {guide_offset};
    const speed = {vel};

    const red = new THREE.MeshStandardMaterial({{color:0xff3333}});
    const blue = new THREE.MeshStandardMaterial({{color:0x0044ff}});
    const white = new THREE.MeshStandardMaterial({{color:0xffffff}});

    const mandrel = new THREE.Mesh(new THREE.CylinderGeometry(R,R,H,64), red);
    mandrel.rotation.x = Math.PI/2;
    mandrel.position.z = H/2;
    scene.add(mandrel);

    const guide = new THREE.Mesh(new THREE.BoxGeometry(30,20,20), blue);
    scene.add(guide);

    scene.add(new THREE.AmbientLight(0xffffff,0.8));

    let theta = 0;
    let z = Rt;
    let radius = R + Rt;
    let dir = 1;

    let pts = [];
    pts.push(new THREE.Vector3(radius,0,z));

    let mesh = null;

    function buildMesh(){{
        if(mesh) scene.remove(mesh);
        if(pts.length<2) return;

        const curve = new THREE.CatmullRomCurve3(pts);
        const geo = new THREE.TubeGeometry(curve, pts.length*2, Rt, 10, false);
        mesh = new THREE.Mesh(geo, white);
        scene.add(mesh);
    }}

    function getContact(){{
        const t = -theta + Math.PI;
        return new THREE.Vector3(
            radius*Math.cos(t),
            radius*Math.sin(t),
            z
        );
    }}

    function getGuide(contact){{
        const tx = -contact.y;
        const ty = contact.x;
        const len = Math.sqrt(tx*tx+ty*ty)+1e-9;

        return new THREE.Vector3(
            contact.x + (tx/len)*offset,
            contact.y + (ty/len)*offset,
            contact.z
        );
    }}

    function step(){{
        theta += 0.03*speed;

        z += dir * {passo} * 0.03 / 360;

        if(z > H-Rt){{ z=H-Rt; dir=-1; radius+={incremento}; }}
        if(z < Rt){{ z=Rt; dir=1; radius+={incremento}; }}

        const contact = getContact();
        const g = getGuide(contact);

        guide.position.copy(g);

        const last = pts[pts.length-1];
        if(contact.distanceTo(last) > Rt*0.3){{
            pts.push(contact.clone());
            buildMesh();
        }}

        const line = new THREE.BufferGeometry().setFromPoints([g, contact]);
        const l = new THREE.Line(line, new THREE.LineBasicMaterial({{color:0x00ffff}}));
        scene.add(l);
        setTimeout(()=>scene.remove(l),16);
    }}

    function animate(){{
        requestAnimationFrame(animate);
        step();
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

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
