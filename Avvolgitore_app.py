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
        "isolamento": "Spessore isolamento (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "gradi_start": "Gradi iniziali (°)",
        "pinza": "Lunghezza pinza (m)",
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
        "isolamento": "Insulation thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm)",
        "incremento": "Layer increment (mm)",
        "rit_min": "Bottom delay (°)",
        "rit_max": "Top delay (°)",
        "gradi_start": "Initial degrees (°)",
        "pinza": "Clamp length (m)",
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
# GEOMETRY (governat per màquina)
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo, incremento, rit_b, rit_t):
    d_tubo = d_rame + 2*spessore
    r = d_aspo/2 + d_tubo/2

    z_min, z_max = -spalla/2, spalla/2
    z = z_min

    theta = 0
    dir = 1
    delay = 0
    pending = False

    pts = []

    for _ in range(40000):
        theta -= np.deg2rad(4)  # sentit corregit

        if delay > 0:
            delay -= 4
        else:
            if pending:
                r += incremento
                pending = False

            z += dir * (passo/(2*np.pi)) * np.deg2rad(4)

            if z >= z_max:
                z = z_max
                delay = rit_t
                pending = True
                dir = -1

            elif z <= z_min:
                z = z_min
                delay = rit_b
                pending = True
                dir = 1

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        pts.append([x,y,z])

        if len(pts)>2:
            if np.sum(np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1)) > lunghezza*1000:
                break

    return np.array(pts)

# =========================
# VIEWER
# =========================

def viewer(points, d_aspo, spalla, d_tubo, altezza, anim, vel):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe6e6e6);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 10000);
    camera.position.set(-400, -900, 300);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth, {altezza});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    // =====================
    // ASPO (VERMELL)
    // =====================

    const machine = new THREE.Group();
    scene.add(machine);

    const r = {d_aspo}/2;
    const h = {spalla};

    const redMat = new THREE.MeshStandardMaterial({{color:0xff3333}});

    const mandrel = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, h, 64),
        redMat
    );
    mandrel.rotation.x = Math.PI/2;
    machine.add(mandrel);

    // discos (spalla)
    const flangeR = r + 120;

    const base = new THREE.Mesh(
        new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
        redMat
    );
    base.rotation.x = Math.PI/2;
    base.position.z = -h/2 - 3;
    machine.add(base);

    const top = new THREE.Mesh(
        new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
        redMat
    );
    top.rotation.x = Math.PI/2;
    top.position.z = h/2 + 3;
    machine.add(top);

    // =====================
    // GUIA TUBO
    // =====================

    const rTube = {d_tubo}/2;

    // posició tangent
    const guideX = -(r + rTube);

    // =====================
    // EIX HORITZONTAL (GROC)
    // =====================

    const yellow = new THREE.Mesh(
        new THREE.BoxGeometry(150, 20, 20),
        new THREE.MeshStandardMaterial({{color:0xffff00}})
    );
    yellow.position.set(guideX - 100, 0, -h/2 - 10);
    scene.add(yellow);

    // =====================
    // COLUMNA (NEGRE)
    // =====================

    const column = new THREE.Mesh(
        new THREE.BoxGeometry(20, 20, h + 150),
        new THREE.MeshStandardMaterial({{color:0x111111}})
    );
    column.position.set(guideX - 20, 0, 0);
    scene.add(column);

    // =====================
    // GUIDATUBO (BLAU)
    // =====================

    const guide = new THREE.Group();
    scene.add(guide);

    // carro blau
    const block = new THREE.Mesh(
        new THREE.BoxGeometry(30, 20, 20),
        new THREE.MeshStandardMaterial({{color:0x0044ff}})
    );
    guide.add(block);

    // braç cap a l’aspo
    const armLen = Math.abs(guideX - (guideX - 20));

    const arm = new THREE.Mesh(
        new THREE.CylinderGeometry(5,5,100,16),
        new THREE.MeshStandardMaterial({{color:0xaaaaaa}})
    );
    arm.rotation.z = Math.PI/2;
    arm.position.x = 50;
    guide.add(arm);

    // nozzle
    const nozzle = new THREE.Mesh(
        new THREE.CylinderGeometry(6,6,20,16),
        new THREE.MeshStandardMaterial({{color:0xffffff}})
    );
    nozzle.rotation.z = Math.PI/2;
    nozzle.position.x = 100;
    guide.add(nozzle);

    // posició inicial
    guide.position.set(guideX - 100, 0, 0);

    // =====================
    // LIGHT
    // =====================

    scene.add(new THREE.AmbientLight(0xffffff,0.8));

    const dlight = new THREE.DirectionalLight(0xffffff,0.6);
    dlight.position.set(500,-500,800);
    scene.add(dlight);

    // =====================
    // ANIMATION
    // =====================

    let t = 0;

    function animate(){{
        requestAnimationFrame(animate);

        // gir aspo (correcte)
        machine.rotation.z -= 0.01 * {vel if anim else 0};

        // moviment vertical guidatubo (simulació)
        t += 0.02;
        guide.position.z = Math.sin(t) * (h/2);

        controls.update();
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
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

pts = build_coil(diametro_aspo, spalla, lunghezza, d_rame, spessore, passo, incremento, rit_b, rit_t)

components.html(viewer(pts, diametro_aspo, spalla, d_rame+2*spessore, altezza, anim, vel), height=altezza)

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_rame+2*spessore:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")

rmax = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))
m4.metric(t["metric4"], f"{2*rmax:.1f} mm")
