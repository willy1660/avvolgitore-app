import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

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
        "gradi_start": "Initial degrees (°)",
        "pinza": "Clamp length (m)",
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
# GEOMETRY (PER METRICS)
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo, incremento, rit_b, rit_t, gradi_start, pinza):
    pts = []
    r = d_aspo/2 + (d_rame + 2*spessore)/2
    z_min, z_max = 0, spalla
    z = z_min
    theta = 0
    direction = 1
    delay = 0
    pending = False

    for _ in range(20000):
        theta += np.deg2rad(4)

        if delay > 0:
            delay -= 4
        else:
            if pending:
                r += incremento
                pending = False

            z += direction * (passo/(2*np.pi)) * np.deg2rad(4)

            if z >= z_max:
                z = z_max
                delay = rit_t
                pending = True
                direction = -1

            elif z <= z_min:
                z = z_min
                delay = rit_b
                pending = True
                direction = 1

        x = r*np.cos(theta)
        y = r*np.sin(theta)
        pts.append([x, y, z])

        if len(pts) > 2:
            if np.sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)) > lunghezza * 1000:
                break

    pts = np.array(pts)
    pts[:, 2] -= spalla/2
    return pts

# =========================
# VIEWER (HARDWARE ONLY, FIXED AXES)
# X = left-right
# Y = depth
# Z = vertical
# =========================

def viewer(d_aspo, spalla, d_tubo, altezza, anim, vel):
    anim_js = "true" if anim else "false"

    return f"""
    <div id="viewer_root" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const host = document.getElementById("viewer_root");
        host.innerHTML = "";

        const W = Math.max(host.clientWidth, 600);
        const H = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 10000);
        camera.position.set(-520, -760, 260);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, H);
        host.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, 0);

        // =====================
        // PARAMETERS
        // =====================

        const R = {float(d_aspo)} / 2.0;
        const Hs = {float(spalla)};
        const Rt = {float(d_tubo)} / 2.0;

        const flangeR = Math.max(R + 120, 260);
        const flangeTh = 6;

        const zBase = -Hs / 2.0;
        const zTop = Hs / 2.0;

        // Tangency position for the tube:
        // guide offset is in Y, not X
        const tangentY = -(R + Rt);
        const tangentZ = zBase + Rt;

        // Hardware placement in X
        const postX = -(R + 85);
        const blockX = postX - 38;
        const nozzleX = -(R + 8);

        // =====================
        // LIGHTS
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.9));

        const key = new THREE.DirectionalLight(0xffffff, 0.8);
        key.position.set(-500, -500, 800);
        scene.add(key);

        const fill = new THREE.DirectionalLight(0xffffff, 0.35);
        fill.position.set(400, 200, 500);
        scene.add(fill);

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0xff2f2f,
            roughness: 0.55,
            metalness: 0.08
        }});

        const hubMat = new THREE.MeshStandardMaterial({{
            color: 0xc63c3c,
            roughness: 0.65,
            metalness: 0.04
        }});

        const hub = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, Hs, 80),
            hubMat
        );
        hub.rotation.x = Math.PI / 2;
        machine.add(hub);

        const lowerFlange = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 80),
            redMat
        );
        lowerFlange.rotation.x = Math.PI / 2;
        lowerFlange.position.z = zBase - flangeTh / 2;
        machine.add(lowerFlange);

        const upperFlange = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 80),
            redMat
        );
        upperFlange.rotation.x = Math.PI / 2;
        upperFlange.position.z = zTop + flangeTh / 2;
        machine.add(upperFlange);

        // =====================
        // HARDWARE WITHOUT RAILS
        // =====================

        const blackMat = new THREE.MeshStandardMaterial({{
            color: 0x1b1b1b,
            roughness: 0.82,
            metalness: 0.12
        }});

        const blueMat = new THREE.MeshStandardMaterial({{
            color: 0x2448d8,
            roughness: 0.55,
            metalness: 0.10
        }});

        const steelMat = new THREE.MeshStandardMaterial({{
            color: 0xcfcfcf,
            roughness: 0.35,
            metalness: 0.72
        }});

        const whiteSteelMat = new THREE.MeshStandardMaterial({{
            color: 0xdedede,
            roughness: 0.42,
            metalness: 0.60
        }});

        // Vertical black post
        const post = new THREE.Mesh(
            new THREE.BoxGeometry(22, 22, Hs + 150),
            blackMat
        );
        post.position.set(postX, tangentY, 22);
        scene.add(post);

        // Blue guide block
        const guideBlock = new THREE.Mesh(
            new THREE.BoxGeometry(28, 20, 18),
            blueMat
        );
        guideBlock.position.set(blockX, tangentY, tangentZ);
        scene.add(guideBlock);

        // Short rear cylinder on the block
        const rearRoll = new THREE.Mesh(
            new THREE.CylinderGeometry(6.5, 6.5, 22, 20),
            whiteSteelMat
        );
        rearRoll.rotation.z = Math.PI / 2;
        rearRoll.position.set(blockX - 18, tangentY, tangentZ);
        scene.add(rearRoll);

        // Arm from block to nozzle
        const armStart = new THREE.Vector3(blockX + 10, tangentY, tangentZ + 2);
        const armEnd   = new THREE.Vector3(nozzleX, tangentY, tangentZ + 22);

        const armVec = new THREE.Vector3().subVectors(armEnd, armStart);
        const armLen = armVec.length();

        const arm = new THREE.Mesh(
            new THREE.CylinderGeometry(5.2, 5.2, armLen, 20),
            steelMat
        );
        arm.position.copy(new THREE.Vector3().addVectors(armStart, armEnd).multiplyScalar(0.5));
        arm.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            armVec.clone().normalize()
        );
        scene.add(arm);

        // Nozzle
        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(7, 7, 22, 20),
            whiteSteelMat
        );
        nozzle.position.copy(armEnd);
        nozzle.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            armVec.clone().normalize()
        );
        scene.add(nozzle);

        // Optional tiny tube segment showing tangency point
        const tubeSegLen = Math.max(18, Rt * 1.8);
        const tubeSeg = new THREE.Mesh(
            new THREE.CylinderGeometry(Rt, Rt, tubeSegLen, 18),
            new THREE.MeshStandardMaterial({{
                color: 0xffffff,
                roughness: 0.65,
                metalness: 0.15
            }})
        );
        tubeSeg.rotation.x = Math.PI / 2;
        tubeSeg.position.set(0, -(R + Rt), zBase + Rt);
        scene.add(tubeSeg);

        // =====================
        // ANIMATION
        // =====================

        const animEnabled = {anim_js};
        const speed = {float(vel)};

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled) {{
                machine.rotation.z -= 0.01 * speed;
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
            const w = Math.max(host.clientWidth, 600);
            const h = Math.max(host.clientHeight, 400);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }});
    }})();
    </script>
    """

# =========================
# UI (ORIGINAL)
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
    gradi_start = st.number_input(t["gradi_start"], value=30.0)
    pinza = st.number_input(t["pinza"], value=0.3)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], False)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

pts = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore,
    passo,
    incremento,
    rit_b,
    rit_t,
    gradi_start,
    pinza
)

d_tubo = d_rame + 2 * spessore

components.html(
    viewer(diametro_aspo, spalla, d_tubo, altezza, anim, vel),
    height=altezza
)

# =========================
# METRICS (ORIGINAL)
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_rame + 2 * spessore:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")

rmax = np.max(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2))
m4.metric(t["metric4"], f"{2 * (rmax):.1f} mm")

if 2 * rmax > 750:
    st.warning(t["warning"])
