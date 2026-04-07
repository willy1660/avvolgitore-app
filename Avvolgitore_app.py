import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

st.title("🌀 Avvolgimento")

# =========================
# INPUTS
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with colB:
    d_tubo = st.number_input("Ø Tubo (mm)", value=22.0)
    lunghezza = st.number_input("Lunghezza (m)", value=30.0)

with colC:
    passo = st.number_input("Passo assiale (mm)", value=20.0)
    incremento = st.number_input("Incremento strato (mm)", value=20.0)
    rit_b = st.number_input("Ritardo base", value=180.0)
    rit_t = st.number_input("Ritardo spalla", value=180.0)

with colD:
    altezza = st.slider("Viewer height", 400, 900, 700)
    anim = st.checkbox("Animació", True)
    vel = st.slider("Velocitat", 0.1, 5.0, 1.0)

# =========================
# VIEWER
# =========================

def viewer():

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:black;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, window.innerWidth/{altezza}, 0.1, 10000);
        camera.position.set(-500,-700,400);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(window.innerWidth, {altezza});
        document.getElementById("viewer").appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // PARAMS
        // =====================

        const R = {diametro_aspo}/2;
        const H = {spalla};
        const Rt = {d_tubo}/2;

        const passo = {passo};
        const incremento = {incremento};
        const ritB = {rit_b};
        const ritT = {rit_t};
        const maxLen = {lunghezza} * 1000;

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R,R,H,80),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        mandrel.position.z = H/2;
        machine.add(mandrel);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(25,25,25),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        scene.add(guide);

        let guideX = -(R + 80);
        let guideY = R + Rt;
        let guideZ = Rt;

        // =====================
        // TUB
        // =====================

        let points = [];
        let total = 0;

        const geometry = new THREE.BufferGeometry();
        const material = new THREE.LineBasicMaterial({{color:0xffffff}});
        const line = new THREE.Line(geometry, material);
        scene.add(line);

        function currentPoint() {{
            // 🔥 CLAU: antihorari respecte aspo horari
            const theta = -machine.rotation.z + Math.PI;

            const x = guideY * Math.cos(theta);
            const y = guideY * Math.sin(theta);

            return new THREE.Vector3(x,y,guideZ);
        }}

        function addPoint(p){{
            if(points.length>0){{
                const d = p.distanceTo(points[points.length-1]);
                if(total + d > maxLen) return;
                total += d;
            }}
            points.push(p);
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

            if({str(anim).lower()}){{

                // aspo horari
                machine.rotation.z -= 0.02 * {vel};

                if(delay>0) delay--;
                else{{
                    guideZ += dir * passo * 0.02 * {vel};

                    if(guideZ >= H - Rt){{
                        guideZ = H - Rt;
                        guideY += incremento;
                        delay = ritT;
                        dir = -1;
                    }}

                    if(guideZ <= Rt){{
                        guideZ = Rt;
                        guideY += incremento;
                        delay = ritB;
                        dir = 1;
                    }}
                }}
            }}

            guide.position.set(guideX, guideY, guideZ);

            addPoint(currentPoint());

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }})();
    </script>
    """

components.html(viewer(), height=altezza)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3 = st.columns(3)
m1.metric("Ø Tubo", f"{d_tubo:.2f} mm")
m2.metric("Passo", f"{passo:.2f} mm")
m3.metric("Incremento", f"{incremento:.2f} mm")
