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
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "aspo_mode": "Aspo",
        "aspo_visible": "Visibile",
        "aspo_transparent": "Trasparente",
        "aspo_hidden": "Nascosto",
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
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "aspo_mode": "Spool",
        "aspo_visible": "Visible",
        "aspo_transparent": "Transparent",
        "aspo_hidden": "Hidden",
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
# UTILS
# =========================

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

# =========================
# SIMULATION
# =========================

def simulate(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza):
    R = d_aspo / 2
    Rt = d_tubo / 2
    H = spalla

    theta = 0
    radius = R + Rt
    z = Rt
    direction = 1

    mode = "axial"
    turnProgress = 0
    turnDelay = 0
    turnStart = radius
    turnEnd = radius
    turnZ = z
    layer = 0

    pts = []
    total_len = lunghezza * 1000
    current_len = 0

    for _ in range(200000):

        deg = 2
        theta -= np.deg2rad(deg)

        if layer == 0:

            if mode == "axial":
                z += direction * passo * (deg / 360)

                if z >= H - Rt:
                    z = H - Rt
                    mode = "turn"
                    turnProgress = 0
                    turnDelay = rit_t
                    turnStart = radius
                    turnEnd = radius + incremento
                    turnZ = z

                if z <= Rt:
                    z = Rt
                    mode = "turn"
                    turnProgress = 0
                    turnDelay = rit_b
                    turnStart = radius
                    turnEnd = radius + incremento
                    turnZ = z

            else:
                turnProgress += deg
                s = smoothstep(turnProgress / max(turnDelay, 1e-6))
                radius = turnStart + s * (turnEnd - turnStart)
                z = turnZ

                if turnProgress >= turnDelay:
                    radius = turnEnd
                    mode = "axial"
                    direction *= -1
                    layer = 1

        else:
            z += direction * passo * (deg / 360)

            if z >= H - Rt:
                z = H - Rt
                radius += incremento
                direction = -1

            if z <= Rt:
                z = Rt
                radius += incremento
                direction = 1

        x = radius * np.cos(-theta + np.pi)
        y = radius * np.sin(-theta + np.pi)

        p = np.array([x, y, z])

        if len(pts) > 0:
            seg = np.linalg.norm(p - pts[-1])
            current_len += seg
            if current_len > total_len:
                break

        pts.append(p)

    return np.array(pts)

# =========================
# METRICS
# =========================

def compute_metrics(points, d_tubo):
    radial = np.sqrt(points[:,0]**2 + points[:,1]**2)
    diam = 2*(np.max(radial) + d_tubo/2)
    xy = points[:,:2]
    span = np.max(np.linalg.norm(xy[:,None,:]-xy[None,:,:], axis=2))
    return diam, span, polyline_length(points)/1000

# =========================
# VIEWER
# =========================

def viewer(points, d_aspo, spalla, d_tubo, altezza):
    pts_json = json.dumps(points.tolist())

    return f"""
    <div id="v" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (()=>{{
        const pts = {pts_json}.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

        const scene=new THREE.Scene();
        scene.background=new THREE.Color(0x0f1115);

        const camera=new THREE.PerspectiveCamera(40,1,0.1,10000);
        camera.position.set(-900,-900,600);

        const renderer=new THREE.WebGLRenderer({{antialias:true}});
        const host=document.getElementById("v");
        renderer.setSize(host.clientWidth,host.clientHeight);
        host.appendChild(renderer.domElement);

        const controls=new THREE.OrbitControls(camera,renderer.domElement);
        controls.target.set(0,0,{spalla}/2);

        scene.add(new THREE.AmbientLight(0xffffff,0.4));
        const l=new THREE.DirectionalLight(0xffffff,1);
        l.position.set(500,-500,800);
        scene.add(l);

        const grid=new THREE.GridHelper(2000,40);
        grid.rotation.x=Math.PI/2;
        scene.add(grid);

        if(pts.length>2){{
            const curve=new THREE.CatmullRomCurve3(pts);
            const geo=new THREE.TubeGeometry(curve,pts.length*2,{d_tubo}/2,12,false);
            const mat=new THREE.MeshStandardMaterial({{color:0xffffff}});
            scene.add(new THREE.Mesh(geo,mat));
        }}

        function animate(){{
            requestAnimationFrame(animate);
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

st.markdown(f"## {t['title']}")

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"#### {t['bobina']}")
    d_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=50.0)

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=360.0)
    rit_t = st.number_input(t["rit_max"], value=360.0)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)

d_tubo = COPPER_SIZES_MM[rame] + 2*spessore

points = simulate(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza)

components.html(viewer(points, d_aspo, spalla, d_tubo, altezza), height=altezza)

# =========================
# METRICS
# =========================

diam, span, length = compute_metrics(points, d_tubo)

m1,m2,m3,m4,m5,m6 = st.columns(6)

m1.metric(t["metric1"], f"{d_tubo:.1f} mm")
m2.metric(t["metric2"], f"{passo:.1f}")
m3.metric(t["metric3"], f"{incremento:.1f}")
m4.metric(t["metric4"], f"{diam:.1f}")
m5.metric(t["metric5"], f"{span:.1f}")
m6.metric(t["metric6"], f"{length:.2f} m")

if span > 750:
    st.warning(t["warning"])
