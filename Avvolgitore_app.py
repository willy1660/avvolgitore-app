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

def smoothstep(u):
    return 0.5 - 0.5*np.cos(np.pi*u)

# =========================
# GEOMETRY FIX + SMOOTH
# =========================

def build_coil(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
):
    L = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2*spessore_guaina_mm
    R = d_tubo / 2

    # FIX físic
    z_min = R
    z_max = spalla_mm - R
    r = d_aspo_mm/2 + R

    theta = 0
    z = z_min
    direction = 1

    step = np.deg2rad(4)
    dz = passo_assiale/(2*np.pi)

    pts = []

    def add():
        pts.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    while True:
        if len(pts) > 2:
            if np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1).sum() >= L:
                break

        while True:
            theta += step
            z += direction*dz*step
            add()

            if direction==1 and z>=z_max:
                z=z_max
                break
            if direction==-1 and z<=z_min:
                z=z_min
                break

        # 🔥 TRANSICIÓ SUAU RADIAL
        steps = 20
        r_start = r
        r_end = r + passo_radiale

        for i in range(1,steps+1):
            theta += step
            u = i/steps
            r = r_start + (r_end - r_start)*smoothstep(u)
            add()

        r = r_end
        direction *= -1

    pts = np.array(pts)

    # trim
    seg = np.linalg.norm(np.diff(pts,axis=0),axis=1)
    cum = np.concatenate([[0],np.cumsum(seg)])
    idx = np.searchsorted(cum,L)-1
    pts = pts[:idx+1]

    # diàmetre real
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
# VIEWER FIX ANIMACIÓ
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):

    pts = points.tolist()
    r_tubo = d_tubo/2

    html = f"""
    <div style="width:100%;height:{altezza}px;">
    <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight,0.1,100000);
    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(window.innerWidth, {altezza});
    document.getElementById("viewer").appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const pts = {json.dumps(pts)};
    const vectors = pts.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

    class CurvePath extends THREE.Curve {{
        constructor(points){{super();this.points=points;}}
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

    let geom=new THREE.TubeGeometry(curve, 2000, {r_tubo}, 32, false);
    geom=geom.toNonIndexed();

    const mesh=new THREE.Mesh(
        geom,
        new THREE.MeshStandardMaterial({{color:0xe6e6e6}})
    );

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

    let progress=0;
    const total=geom.attributes.position.count;

    if ({str(animazione).lower()}) {{
        geom.setDrawRange(0,0);
    }}

    function animate(){{
        requestAnimationFrame(animate);

        if ({str(animazione).lower()}) {{
            progress += {velocita}*0.002;
            if(progress>1) progress=1;
            geom.setDrawRange(0, Math.floor(progress*total));
        }}

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
    diametro_aspo = st.number_input("Ø Aspo", value=450.0)
    spalla = st.number_input("Spalla", value=95.0)

with colB:
    rame_label = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input("Guaina", value=7.0)
    lunghezza = st.number_input("Lunghezza", value=50.0)
    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    passo_assiale = st.number_input("Passo assiale", value=20.0)
    incremento_strato = st.number_input("Passo radiale", value=20.0)

with colD:
    altezza = st.slider("Altezza", 400, 900, 700)
    animazione = st.checkbox("Animazione", False)
    velocita = st.slider("Velocità", 0.1, 5.0, 1.0)

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
)

html = build_viewer_html(path, meta["DiametroTubo"], altezza, animazione, velocita)

components.html(html, height=altezza)

st.write(meta)
