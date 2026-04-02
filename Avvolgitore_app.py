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
    ["🇮🇹 Italiano", "🇺🇸 English"],
    index=0 if st.session_state.lang == "IT" else 1
)

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
lang = st.session_state.lang

TEXTS = {
    "IT": {
        "title": "Avvolgimento",
        "viewer": "⚙️ Viewer",
        "grid": "Griglia",
        "axes": "Assi",
        "light": "Light mode",
        "transp": "Trasparenza tubo",
        "progress": "Progresso",
    },
    "EN": {
        "title": "Coiling",
        "viewer": "⚙️ Viewer",
        "grid": "Grid",
        "axes": "Axes",
        "light": "Light mode",
        "transp": "Tube transparency",
        "progress": "Progress",
    }
}

t = TEXTS[lang]

# =========================
# HEADER
# =========================

st.markdown(f"# {t['title']}")

# =========================
# SIMPLE DATA (per prova)
# =========================

points = np.array([[np.cos(i)*200, np.sin(i)*200, i*2] for i in np.linspace(0, 40, 2000)])

# =========================
# VIEWER CONTROLS UI
# =========================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    show_grid = st.checkbox(t["grid"], True)
with col2:
    show_axes = st.checkbox(t["axes"], True)
with col3:
    light_mode = st.checkbox(t["light"], False)
with col4:
    transparency = st.slider(t["transp"], 0.0, 0.9, 0.0)
with col5:
    progress_manual = st.slider(t["progress"], 0.0, 1.0, 1.0)

# =========================
# VIEWER HTML
# =========================

def build_viewer():
    return f"""
    <div style="width:100%;height:700px;border-radius:16px;overflow:hidden;">
    <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>

    const scene = new THREE.Scene();
    scene.background = new THREE.Color({ "0xffffff" if light_mode else "0x0b0d10" });

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.setSize(window.innerWidth,700);

    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // LIGHT
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(500,500,500);
    scene.add(dir);

    // GEOMETRY
    const pts = {json.dumps(points.tolist())};
    const vec = pts.map(p => new THREE.Vector3(p[0],p[1],p[2]));

    class Curve extends THREE.Curve {{
        constructor(points) {{ super(); this.points = points; }}
        getPoint(t) {{
            const i = Math.floor(t*(this.points.length-1));
            return this.points[i];
        }}
    }}

    const curve = new Curve(vec);

    const geom = new THREE.TubeGeometry(curve, 3000, 8, 48, false);

    const mat = new THREE.MeshStandardMaterial({{
        color:0xe6e6e6,
        transparent:true,
        opacity:{1-transparency},
        roughness:0.7
    }});

    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    // GRID
    {"scene.add(new THREE.GridHelper(2000,40));" if show_grid else ""}

    // AXES
    {"scene.add(new THREE.AxesHelper(200));" if show_axes else ""}

    // CAMERA
    camera.position.set(600,600,400);
    controls.target.set(0,0,200);

    // PROGRESS
    const total = geom.attributes.position.count;
    geom.setDrawRange(0, Math.floor(total * {progress_manual}));

    function animate(){{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene,camera);
    }}

    animate();

    </script>
    """

components.html(build_viewer(), height=720)
