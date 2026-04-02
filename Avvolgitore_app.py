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
        "bobina": "🟦 Bobina",
        "tubo": "🟩 Tubo",
        "avvolg": "🟧 Avvolgimento",
        "viewer": "⚙️ Viewer",
        "grid": "Griglia",
        "axes": "Assi",
        "light": "Light mode",
        "transp": "Trasparenza",
        "progress": "Progresso",
        "metric1": "Diametro tubo",
        "metric4": "Diametro esterno",
    },
    "EN": {
        "title": "Coiling",
        "bobina": "🟦 Coil",
        "tubo": "🟩 Tube",
        "avvolg": "🟧 Winding",
        "viewer": "⚙️ Viewer",
        "grid": "Grid",
        "axes": "Axes",
        "light": "Light mode",
        "transp": "Transparency",
        "progress": "Progress",
        "metric1": "Tube diameter",
        "metric4": "Outer diameter",
    }
}

t = TEXTS[lang]

# =========================
# HEADER
# =========================

st.markdown(f"# {t['title']}")

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

EPS = 1e-9

# =========================
# GEOMETRY
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo_assiale, passo_radiale):
    lunghezza_mm = lunghezza * 1000
    d_tubo = d_rame + 2 * spessore

    r = d_aspo/2 + d_tubo/2
    z = 0
    theta = 0

    pts = []

    while len(pts) < 3000:
        theta += 0.1
        z += passo_assiale * 0.01

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        pts.append([x,y,z])

    path = np.array(pts)

    diam_ext = 2*(np.max(np.sqrt(path[:,0]**2+path[:,1]**2)) + d_tubo/2)

    return path, {"DiametroTubo":d_tubo, "DiametroEsterno":diam_ext}

# =========================
# VIEWER
# =========================

def build_viewer(points, d_tubo, altezza, show_grid, show_axes, light_mode, transparency, progress):

    return f"""
    <div style="width:100%;height:{altezza}px;border-radius:16px;overflow:hidden;">
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
    renderer.setSize(window.innerWidth,{altezza});

    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    light.position.set(500,500,500);
    scene.add(light);

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

    const geom = new THREE.TubeGeometry(curve, 4000, {d_tubo/2}, 48, false);

    const mat = new THREE.MeshStandardMaterial({{
        color:0xe6e6e6,
        transparent:true,
        opacity:{1-transparency},
        roughness:0.7
    }});

    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    {"scene.add(new THREE.GridHelper(2000,40));" if show_grid else ""}
    {"scene.add(new THREE.AxesHelper(200));" if show_axes else ""}

    camera.position.set(600,600,400);
    controls.target.set(0,0,200);

    const total = geom.attributes.position.count;
    geom.setDrawRange(0, Math.floor(total * {progress}));

    function animate(){{
        requestAnimationFrame(animate);
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
    diametro_aspo = st.number_input("Ø Aspo", value=450.0)

with colB:
    rame = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input("Isolamento", value=7.0)

with colC:
    passo_assiale = st.number_input("Passo", value=20.0)

with colD:
    st.markdown("### Viewer")
    show_grid = st.checkbox(t["grid"], True)
    show_axes = st.checkbox(t["axes"], True)
    light_mode = st.checkbox(t["light"], False)
    transparency = st.slider(t["transp"], 0.0, 0.9, 0.0)
    progress = st.slider(t["progress"], 0.0, 1.0, 1.0)

# =========================
# BUILD
# =========================

path, meta = build_coil(
    diametro_aspo,
    100,
    50,
    COPPER_SIZES_MM[rame],
    spessore,
    passo_assiale,
    20
)

html = build_viewer(
    path,
    meta["DiametroTubo"],
    700,
    show_grid,
    show_axes,
    light_mode,
    transparency,
    progress
)

components.html(html, height=700)

# =========================
# METRICS
# =========================

st.divider()

c1, c2 = st.columns(2)

c1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
c2.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")
