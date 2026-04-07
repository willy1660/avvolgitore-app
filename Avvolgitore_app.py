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
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità"
    },
    "EN": {
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
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed"
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
# VIEWER (HARDWARE ONLY)
# =========================

def viewer(d_aspo, spalla, d_tubo, altezza, anim, vel):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    setTimeout(() => {{

        const container = document.getElementById("viewer");

        const w = container.clientWidth;
        const h = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 10000);
        camera.position.set(-400, -900, 300);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(w, h);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const r = {d_aspo}/2;
        const h_m = {spalla};

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(r, r, h_m, 64),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        machine.add(mandrel);

        const flangeR = r + 120;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        base.rotation.x = Math.PI/2;
        base.position.z = -h_m/2 - 3;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = h_m/2 + 3;
        machine.add(top);

        // =====================
        // GUIDATUBO
        // =====================

        const rTube = {d_tubo}/2;
        const guideX = -(r + rTube);

        // base groga
        const yellow = new THREE.Mesh(
            new THREE.BoxGeometry(150, 20, 20),
            new THREE.MeshStandardMaterial({{color:0xffff00}})
        );
        yellow.position.set(guideX - 100, 0, -h_m/2 - 10);
        scene.add(yellow);

        // columna negra
        const column = new THREE.Mesh(
            new THREE.BoxGeometry(20, 20, h_m + 150),
            new THREE.MeshStandardMaterial({{color:0x111111}})
        );
        column.position.set(guideX - 20, 0, 0);
        scene.add(column);

        // carro
        const guide = new THREE.Group();
        scene.add(guide);

        const block = new THREE.Mesh(
            new THREE.BoxGeometry(30, 20, 20),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        guide.add(block);

        const arm = new THREE.Mesh(
            new THREE.CylinderGeometry(5,5,100,16),
            new THREE.MeshStandardMaterial({{color:0xaaaaaa}})
        );
        arm.rotation.z = Math.PI/2;
        arm.position.x = 50;
        guide.add(arm);

        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(6,6,20,16),
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );
        nozzle.rotation.z = Math.PI/2;
        nozzle.position.x = 100;
        guide.add(nozzle);

        guide.position.set(guideX - 100, 0, 0);

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff,0.8));

        const light = new THREE.DirectionalLight(0xffffff,0.6);
        light.position.set(500,-500,800);
        scene.add(light);

        // =====================
        // ANIMATION
        // =====================

        let t = 0;

        function animate(){{
            requestAnimationFrame(animate);

            machine.rotation.z -= 0.01 * {vel if anim else 0};

            t += 0.02;
            guide.position.z = Math.sin(t) * (h_m/2);

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }}, 50);
    </script>
    """

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
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

d_tubo = d_rame + 2*spessore

components.html(
    viewer(diametro_aspo, spalla, d_tubo, altezza, anim, vel),
    height=altezza
)
