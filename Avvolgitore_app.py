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
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
    }
}

t = TEXTS[lang]

# =========================
# HEADER (recuperat)
# =========================

col_logo, col_title = st.columns([1, 7])

logo_path = os.path.join(os.path.dirname(__file__), "New Logo PDM - rame.png")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)

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

def smoothstep(u):
    return 0.5 - 0.5*np.cos(np.pi*u)

# =========================
# GEOMETRY (FIX + RITARDO)
# =========================

def build_coil(d_aspo, spalla, Lm, d_rame, sp, passo_assiale, passo_radiale, rit_min, rit_max):

    L = Lm * 1000
    d_tubo = d_rame + 2*sp
    R = d_tubo/2

    z_min = R
    z_max = spalla - R
    r = d_aspo/2 + R

    theta = 0
    z = z_min
    dir = 1

    step = np.deg2rad(1)
    dz = passo_assiale/(2*np.pi)

    pts = []

    def add():
        pts.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    while True:

        if len(pts) > 2:
            if np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1).sum() >= L:
                break

        # helicoide
        while True:
            theta += step
            z += dir*dz*step
            add()

            if dir==1 and z>=z_max:
                z=z_max
                break
            if dir==-1 and z<=z_min:
                z=z_min
                break

        # =====================
        # RITARDO (dwell real)
        # =====================

        rit = rit_max if dir==1 else rit_min
        theta_dwell = np.deg2rad(rit)

        if theta_dwell > 0:

            steps = max(6, int(rit/2))
            r0 = r
            r1 = r + passo_radiale

            for i in range(steps):
                theta += theta_dwell/steps
                u = (i+1)/steps
                r = r0 + (r1-r0)*smoothstep(u)
                add()

            r = r1

        else:
            r += passo_radiale

        dir *= -1

    pts = np.array(pts)

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
# VIEWER ORIGINAL (restaurat)
# =========================

def build_viewer_html(points, d_tubo, h, anim, speed):

    pts = json.dumps(points.tolist())
    r = d_tubo/2

    return f"""
    <div style="width:100%;height:{h}px;" id="viewer"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight,0.1,100000);

    const renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1));

    const light = new THREE.DirectionalLight(0xffffff,0.8);
    light.position.set(5,5,5);
    scene.add(light);

    const pts = {pts};
    const v = pts.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

    class C extends THREE.Curve {{
     getPoint(t){{
      const n=v.length;
      const f=t*(n-1);
      const i=Math.floor(f);
      const a=v[Math.max(0,Math.min(i,n-2))];
      const b=v[Math.max(1,Math.min(i+1,n-1))];
      return new THREE.Vector3().lerpVectors(a,b,f-i);
     }}
    }}

    const curve=new C();

    let g=new THREE.TubeGeometry(curve,2000,{r},32,false);
    g=g.toNonIndexed();

    const m=new THREE.MeshStandardMaterial({{
        color:0xdedede,
        roughness:0.9,
        metalness:0.05
    }});

    const mesh=new THREE.Mesh(g,m);
    scene.add(mesh);

    // CAPS (recuperats)
    function cap(pos,dir,color){{
        const geo=new THREE.CircleGeometry({r},32);
        const mat=new THREE.MeshBasicMaterial({{color:color,side:THREE.DoubleSide}});
        const c=new THREE.Mesh(geo,mat);

        const up=new THREE.Vector3(0,0,1);
        const q=new THREE.Quaternion().setFromUnitVectors(up,dir.clone().normalize());
        c.quaternion.copy(q);

        c.position.copy(pos);
        scene.add(c);
    }}

    if(v.length>1){{
        cap(v[0],v[1].clone().sub(v[0]).multiplyScalar(-1),0x00ff00);
        cap(v[v.length-1],v[v.length-1].clone().sub(v[v.length-2]),0xff0000);
    }}

    const box=new THREE.Box3().setFromPoints(v);
    const c=new THREE.Vector3();
    box.getCenter(c);

    const size=new THREE.Vector3();
    box.getSize(size);

    const d=Math.max(size.x,size.y,size.z)*1.8;

    camera.position.set(c.x+d,c.y+d,c.z+d*0.6);
    camera.lookAt(c);
    controls.target.copy(c);

    let p=0;
    const total=g.attributes.position.count;

    if ({str(anim).lower()}) g.setDrawRange(0,0);
    else g.setDrawRange(0,total);

    function loop(){{
        requestAnimationFrame(loop);

        if ({str(anim).lower()}){{
            p+= {speed}*0.002;
            if(p>1)p=1;
            g.setDrawRange(0,Math.floor(p*total));
        }}

        controls.update();
        renderer.render(scene,camera);
    }}

    loop();
    </script>
    """

# =========================
# UI (recuperada)
# =========================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown(f"#### {t['bobina']}")
    d_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with c2:
    st.markdown(f"#### {t['tubo']}")
    size = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    sp = st.number_input(t["isolamento"], value=7.0)
    L = st.number_input(t["lunghezza"], value=50.0)
    d_rame = COPPER_SIZES_MM[size]

with c3:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0)
    step_r = st.number_input(t["incremento"], value=20.0)
    rit_min = st.number_input(t["rit_min"], value=180.0)
    rit_max = st.number_input(t["rit_max"], value=180.0)

with c4:
    st.markdown(f"#### {t['viewer']}")
    h = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], False)
    speed = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# RUN
# =========================

p, m = build_coil(d_aspo, spalla, L, d_rame, sp, passo, step_r, rit_min, rit_max)

components.html(build_viewer_html(p, m["DiametroTubo"], h, anim, speed), height=h)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{m['DiametroTubo']:.2f} mm")
m2.metric(t["metric2"], f"{m['PassoAssiale']:.2f} mm")
m3.metric(t["metric3"], f"{m['IncrementoStrato']:.2f} mm")
m4.metric(t["metric4"], f"{m['DiametroEsterno']:.1f} mm")
