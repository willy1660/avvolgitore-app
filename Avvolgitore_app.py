import os
import glob
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
        "passo_assiale": "Passo assiale (mm)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
        "warning": "⚠️ Diametro esterno superiore a 750 mm."
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
    for f in ["New Logo PDM – rame.png", "logo.png"]:
        if os.path.exists(f):
            return f
    return None

logo = find_logo()

if logo:
    c1, c2 = st.columns([1,5])
    c1.image(logo)
    c2.title("Avvolgimento")
else:
    st.title("Avvolgimento")

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza, altezza, vel):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:black;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth/{altezza}, 0.1, 10000);
    camera.position.set(-500,-700,400);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth,{altezza});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const R = {d_aspo}/2;
    const H = {spalla};
    const Rt = {d_tubo}/2;
    const maxLen = {lunghezza}*1000;

    const machine = new THREE.Group();
    scene.add(machine);

    const mat = new THREE.MeshStandardMaterial({{color:0xff3333}});

    const mandrel = new THREE.Mesh(
        new THREE.CylinderGeometry(R,R,H,80), mat
    );
    mandrel.rotation.x = Math.PI/2;
    mandrel.position.z = H/2;
    machine.add(mandrel);

    const base = new THREE.Mesh(
        new THREE.CylinderGeometry(R+120,R+120,6,80), mat
    );
    base.rotation.x = Math.PI/2;
    machine.add(base);

    const top = new THREE.Mesh(
        new THREE.CylinderGeometry(R+120,R+120,6,80), mat
    );
    top.rotation.x = Math.PI/2;
    top.position.z = H;
    machine.add(top);

    const guide = new THREE.Mesh(
        new THREE.BoxGeometry(30,20,20),
        new THREE.MeshStandardMaterial({{color:0x0044ff}})
    );
    scene.add(guide);

    let guideX = -(R+80);
    let guideY = R+Rt;
    let guideZ = Rt;

    let points = [];
    let total = 0;
    let finished = false;

    let tubeMesh = null;

    function contactPoint(theta){{
        const t = -theta + Math.PI;
        return new THREE.Vector3(
            guideY*Math.cos(t),
            guideY*Math.sin(t),
            guideZ
        );
    }}

    function rebuildMesh(){{
        if(points.length < 2) return;

        const display = points.slice();

        const last = points[points.length-1];
        const guidePoint = new THREE.Vector3(guideX,guideY,guideZ);

        // 🔥 TRAM RECTE REAL
        const dir = new THREE.Vector3().subVectors(guidePoint,last);
        const dist = dir.length();
        const steps = Math.max(3, Math.floor(dist/(Rt*0.8)));

        for(let i=1;i<=steps;i++){{
            const t = i/steps;
            const p = new THREE.Vector3().lerpVectors(last,guidePoint,t);
            display.push(p);
        }}

        if(tubeMesh) scene.remove(tubeMesh);

        const curve = new THREE.CatmullRomCurve3(display);
        const geo = new THREE.TubeGeometry(curve,200,Rt,10,false);

        tubeMesh = new THREE.Mesh(
            geo,
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );

        scene.add(tubeMesh);
    }}

    let theta = 0;
    let dir = 1;
    let delay = 0;

    scene.add(new THREE.AmbientLight(0xffffff,0.8));

    function animate(){{
        requestAnimationFrame(animate);

        if(!finished){{
            theta -= 0.02*{vel};
            machine.rotation.z = theta;

            if(delay>0) delay--;
            else{{
                guideZ += dir*{passo}*0.02*{vel};

                if(guideZ >= H-Rt){{
                    guideZ = H-Rt;
                    guideY += {incremento};
                    delay = {rit_t};
                    dir = -1;
                }}

                if(guideZ <= Rt){{
                    guideZ = Rt;
                    guideY += {incremento};
                    delay = {rit_b};
                    dir = 1;
                }}
            }}

            const p = contactPoint(theta);

            if(points.length>0){{
                const d = p.distanceTo(points[points.length-1]);
                if(total + d > maxLen) finished = true;
                else total += d;
            }}

            points.push(p);
            rebuildMesh();
        }}

        guide.position.set(guideX,guideY,guideZ);

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
    diam_aspo = st.number_input("Ø Aspo", value=450.0)
    spalla = st.number_input("Spalla", value=95.0)

with colB:
    rame = st.selectbox("Ø Rame", list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input("Spessore guaina", value=7.0)
    lunghezza = st.number_input("Lunghezza", value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    passo = st.number_input("Passo", value=20.0)
    incremento = st.number_input("Incremento", value=20.0)
    rit_b = st.number_input("Ritardo base", value=180.0)
    rit_t = st.number_input("Ritardo spalla", value=180.0)

with colD:
    altezza = st.slider("Altezza", 400, 900, 700)
    vel = st.slider("Velocità", 0.1, 5.0, 1.0)

d_tubo = d_rame + 2*spessore

components.html(
    viewer(diam_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza, altezza, vel),
    height=altezza
)

# =========================
# METRICS
# =========================

st.divider()

m1,m2,m3,m4 = st.columns(4)

m1.metric("Ø Tubo", f"{d_tubo:.2f} mm")
m2.metric("Passo", f"{passo:.2f} mm")
m3.metric("Incremento", f"{incremento:.2f} mm")

outer = diam_aspo + 2*incremento*5
m4.metric("Ø Esterno", f"{outer:.1f} mm")

if outer > 750:
    st.warning("⚠️ Diàmetre massa gran")
