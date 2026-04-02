import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# 🌍 LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

lang_option = st.selectbox(
    "🌍 Language",
    ["🇮🇹 Italiano", "🇺🇸 English (US)"],
    index=0 if st.session_state.lang == "IT" else 1
)

if "Italiano" in lang_option:
    st.session_state.lang = "IT"
else:
    st.session_state.lang = "EN"

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
# HEADER
# =========================

col_logo, col_title = st.columns([1, 7])

logo_path = os.path.join(os.path.dirname(__file__), "New Logo PDM - rame.png")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)

with col_title:
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
# UTILS
# =========================

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline(points, target_length):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    p0, p1 = points[idx], points[idx + 1]
    alpha = (target_length - cum[idx]) / (np.linalg.norm(p1 - p0) + EPS)

    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

# =========================
# GEOMETRY FIX
# =========================

def build_coil(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
    ritardo_min_deg,
    ritardo_max_deg,
):
    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm
    R = d_tubo / 2

    z_min = R
    z_max = spalla_mm - R

    r = d_aspo_mm / 2 + R

    z = z_min
    theta = 0
    direction = 1

    step = np.deg2rad(4)
    dz = passo_assiale / (2 * np.pi)

    pts = []

    def add():
        pts.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    while polyline_length(np.array(pts)) < lunghezza_mm:
        theta += step
        z += direction * dz * step

        if z >= z_max:
            z = z_max
            add()
            r += passo_radiale
            direction = -1
            continue

        if z <= z_min:
            z = z_min
            add()
            r += passo_radiale
            direction = 1
            continue

        add()

    pts = np.array(pts)
    pts = trim_polyline(pts, lunghezza_mm)

    r_path = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    diam_ext = 2*np.max(r_path + R)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext,
    }

    return pts, meta

# =========================
# VIEWER ORIGINAL (FIXED)
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2
    tubular_segments = min(4000, max(800, int(len(pts)*0.5)))

    html = f"""
    <div style="width:100%;height:{altezza}px;">
    <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff,0x2a2a2a,0.7));

    const rawPoints = {points_json};
    const vectors = rawPoints.map(p => new THREE.Vector3(p[0],p[1],p[2]));

    class CurvePath extends THREE.Curve {{
      constructor(points){{ super(); this.points=points; }}
      getPoint(t){{
        const n=this.points.length;
        const f=t*(n-1);
        const i=Math.floor(f);
        const i0=Math.max(0,Math.min(i,n-2));
        const i1=i0+1;
        const tt=f-i0;
        return new THREE.Vector3().lerpVectors(this.points[i0],this.points[i1],tt);
      }}
    }}

    const curve=new CurvePath(vectors);

    let tubeGeom=new THREE.TubeGeometry(curve,{tubular_segments},{r_tubo},48,false);
    tubeGeom=tubeGeom.toNonIndexed();

    const mesh=new THREE.Mesh(tubeGeom,new THREE.MeshStandardMaterial({{color:0xe6e6e6}}));
    scene.add(mesh);

    const box=new THREE.Box3().setFromPoints(vectors);
    const center=new THREE.Vector3();
    box.getCenter(center);

    const size=new THREE.Vector3();
    box.getSize(size);

    const dist=Math.max(size.x,size.y,size.z)*1.8;

    camera.position.set(center.x+dist,center.y+dist,center.z+dist*0.6);
    camera.lookAt(center);
    controls.target.copy(center);

    function animate(){{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene,camera);
    }}

    animate();
    </script>
    """
    return html

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=50.0)
    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0)
    incremento_strato = st.number_input(t["incremento"], value=20.0)
    ritardo_min = st.number_input(t["rit_min"], value=180.0)
    ritardo_max = st.number_input(t["rit_max"], value=180.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# RUN
# =========================

path, meta = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    passo_assiale,
    incremento_strato,
    ritardo_min,
    ritardo_max,
)

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita
)

components.html(html, height=altezza)

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
m2.metric(t["metric2"], f"{meta['PassoAssiale']:.2f} mm")
m3.metric(t["metric3"], f"{meta['IncrementoStrato']:.2f} mm")
m4.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")

if meta["DiametroEsterno"] > 750:
    st.warning(t["warning"])
