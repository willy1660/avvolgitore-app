import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

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
# GEOMETRY (per mètriques)
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
# VIEWER FINAL
# =========================

def viewer(d_aspo, spalla, d_tubo, altezza, anim, vel):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{

        const el = document.getElementById("viewer");
        el.innerHTML = "";

        const w = el.clientWidth;
        const h = el.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, w/h, 0.1, 10000);
        camera.position.set(-500, -700, 250);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(w,h);
        el.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // PARAMS
        // =====================

        const R = {d_aspo}/2;
        const H = {spalla};
        const Rt = {d_tubo}/2;

        // 🔥 POSICIÓ CORRECTA
        const guideY = (R + Rt);
        const guideZ = -H/2 + Rt;

        // 🔥 OFFSET X PER NO COL·LISIÓ
        const guideX = -(R + 80);

        // =====================
        // ASPO
        // =====================

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, H, 80),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        scene.add(mandrel);

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(R+120, R+120, 6, 80),
            red
        );
        base.rotation.x = Math.PI/2;
        base.position.z = -H/2 - 3;
        scene.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(R+120, R+120, 6, 80),
            red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = H/2 + 3;
        scene.add(top);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30,20,20),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        guide.position.set(guideX, guideY, guideZ);
        scene.add(guide);

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff,0.8));

        const light = new THREE.DirectionalLight(0xffffff,0.6);
        light.position.set(500,-500,800);
        scene.add(light);

        function animate(){{
            requestAnimationFrame(animate);

            if ({'true' if anim else 'false'}) {{
                mandrel.rotation.z -= 0.01 * {vel};
            }}

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }})();
    </script>
    """

# =========================
# UI ORIGINAL
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with colB:
    rame = st.selectbox("Ø Rame", list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input("Isolamento (mm)", value=7.0)
    lunghezza = st.number_input("Lunghezza (m)", value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    passo = st.number_input("Passo", value=20.0)
    incremento = st.number_input("Incremento", value=20.0)
    rit_b = st.number_input("Ritardo base", value=180.0)
    rit_t = st.number_input("Ritardo top", value=180.0)
    gradi_start = st.number_input("Start", value=30.0)
    pinza = st.number_input("Pinza", value=0.3)

with colD:
    altezza = st.slider("Altezza", 400, 900, 700)
    anim = st.checkbox("Animazione", False)
    vel = st.slider("Velocità", 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

pts = build_coil(diametro_aspo, spalla, lunghezza, d_rame, spessore, passo, incremento, rit_b, rit_t, gradi_start, pinza)

d_tubo = d_rame + 2*spessore

components.html(viewer(diametro_aspo, spalla, d_tubo, altezza, anim, vel), height=altezza)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Diametro tubo", f"{d_rame+2*spessore:.2f} mm")
m2.metric("Passo", f"{passo:.2f} mm")
m3.metric("Incremento", f"{incremento:.2f} mm")

rmax = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))
m4.metric("Diametro esterno", f"{2*(rmax):.1f} mm")
