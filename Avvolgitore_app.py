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

COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza, altezza, anim, vel):
    anim_js = "true" if anim else "false"

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{

        const el = document.getElementById("viewer");
        el.innerHTML = "";

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, el.clientWidth/el.clientHeight, 0.1, 10000);
        camera.position.set(-500, -700, 400);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(el.clientWidth, el.clientHeight);
        el.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // PARAMS
        // =====================

        const R = {d_aspo}/2;
        const H = {spalla};
        const Rt = {d_tubo}/2;

        const passo = {passo};
        const incremento = {incremento};
        const ritB = {rit_b};
        const ritT = {rit_t};

        const maxLen = {lunghezza} * 1000;

        const guideX = -(R + 80);

        let guideY = R + Rt;
        let guideZ = Rt;

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, H, 80),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        mandrel.position.z = H/2;
        machine.add(mandrel);

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(R+120, R+120, 6, 80),
            red
        );
        base.rotation.x = Math.PI/2;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(R+120, R+120, 6, 80),
            red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = H;
        machine.add(top);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30,20,20),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        scene.add(guide);

        // =====================
        // TUB
        // =====================

        let points = [];
        let totalLength = 0;
        let finished = false;

        const geometry = new THREE.BufferGeometry();
        const material = new THREE.LineBasicMaterial({{color:0xffffff}});
        const line = new THREE.Line(geometry, material);
        scene.add(line);

        function currentTubePoint() {{

            const theta = machine.rotation.z;

            const x = guideY * Math.cos(theta);
            const y = guideY * Math.sin(theta);

            return new THREE.Vector3(x, y, guideZ);
        }}

        function addPoint(p) {{
            if (points.length > 0) {{
                const prev = points[points.length - 1];
                const d = p.distanceTo(prev);

                if (totalLength + d > maxLen) {{
                    const remain = maxLen - totalLength;
                    const dir = p.clone().sub(prev).normalize();
                    const finalP = prev.clone().add(dir.multiplyScalar(remain));
                    points.push(finalP);
                    finished = true;
                }} else {{
                    points.push(p);
                    totalLength += d;
                }}
            }} else {{
                points.push(p);
            }}

            geometry.setFromPoints(points);
        }}

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff,0.8));
        const light = new THREE.DirectionalLight(0xffffff,0.6);
        light.position.set(500,-500,800);
        scene.add(light);

        // =====================
        // MOTION
        // =====================

        let dir = 1;
        let delay = 0;

        function animate(){{
            requestAnimationFrame(animate);

            if ({anim_js} && !finished) {{

                // ROTACIÓ CORRECTA
                machine.rotation.z -= 0.02 * {vel};

                if (delay > 0) {{
                    delay -= 1;
                }} else {{

                    guideZ += dir * passo * 0.02 * {vel};

                    if (guideZ >= H - Rt) {{
                        guideZ = H - Rt;
                        guideY += incremento;
                        delay = ritT;
                        dir = -1;
                    }}

                    if (guideZ <= Rt) {{
                        guideZ = Rt;
                        guideY += incremento;
                        delay = ritB;
                        dir = 1;
                    }}
                }}
            }}

            guide.position.set(guideX, guideY, guideZ);

            addPoint(currentTubePoint());

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }})();
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
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

d_tubo = d_rame + 2 * spessore

components.html(
    viewer(
        diametro_aspo,
        spalla,
        d_tubo,
        passo,
        incremento,
        rit_b,
        rit_t,
        lunghezza,
        altezza,
        anim,
        vel
    ),
    height=altezza
)
