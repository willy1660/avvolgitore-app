import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LOGO + TITLE
# =========================

if os.path.exists("New Logo PDM – rame.png"):
    col1, col2 = st.columns([1,5])
    col1.image("New Logo PDM – rame.png")
    col2.title("Avvolgimento")
else:
    st.title("Avvolgimento")

# =========================
# INPUTS (UI ORIGINAL)
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diam_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with colB:
    rame_map = {"1/4":6.35,"3/8":9.52,"1/2":12.70,"5/8":15.88,"3/4":19.05,"7/8":22.23}
    rame = st.selectbox("Ø Rame", list(rame_map.keys()))
    spessore = st.number_input("Spessore guaina (mm)", value=7.0)
    lunghezza = st.number_input("Lunghezza rotolo (m)", value=30.0)

with colC:
    passo = st.number_input("Passo assiale (mm)", value=20.0)
    incremento = st.number_input("Incremento strato (mm)", value=20.0)
    rit_b = st.number_input("Ritardo base", value=180.0)
    rit_t = st.number_input("Ritardo spalla", value=180.0)

with colD:
    altezza = st.slider("Altezza", 400, 900, 700)
    vel = st.slider("Velocità", 0.1, 5.0, 1.0)

d_tubo = rame_map[rame] + 2*spessore

# =========================
# VIEWER
# =========================

def viewer():
    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:black;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    setTimeout(() => {{

        const container = document.getElementById("viewer");
        const w = container.clientWidth;
        const h = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, w/h, 0.1, 10000);
        camera.position.set(-500,-700,400);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(w,h);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        const R = {diam_aspo}/2;
        const H = {spalla};
        const Rt = {d_tubo}/2;
        const maxLen = {lunghezza}*1000;

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R,R,H,80), red
        );
        mandrel.rotation.x = Math.PI/2;
        mandrel.position.z = H/2;
        machine.add(mandrel);

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(R+120,R+120,6,80), red
        );
        base.rotation.x = Math.PI/2;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(R+120,R+120,6,80), red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = H;
        machine.add(top);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(25,25,25),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        scene.add(guide);

        let guideX = -(R+80);
        let guideY = R+Rt;
        let guideZ = Rt;

        // =====================
        // TUB
        // =====================

        let pts = [];
        let total = 0;
        let finished = false;
        let mesh = null;

        function contact(theta){{
            const t = -theta + Math.PI;
            return new THREE.Vector3(
                guideY*Math.cos(t),
                guideY*Math.sin(t),
                guideZ
            );
        }}

        function rebuild(){{
            if(pts.length<2) return;

            const display = pts.slice();
            const last = pts[pts.length-1];
            const guideP = new THREE.Vector3(guideX,guideY,guideZ);

            // 🔥 TRAM RECTE
            const steps = 6;
            for(let i=1;i<=steps;i++){{
                display.push(new THREE.Vector3().lerpVectors(last,guideP,i/steps));
            }}

            if(mesh) scene.remove(mesh);

            const curve = new THREE.CatmullRomCurve3(display);
            const geo = new THREE.TubeGeometry(curve,200,Rt,10,false);

            mesh = new THREE.Mesh(geo,new THREE.MeshStandardMaterial({{color:0xffffff}}));
            scene.add(mesh);
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

                    if(guideZ>=H-Rt){{
                        guideZ = H-Rt;
                        guideY += {incremento};
                        delay = {rit_t};
                        dir = -1;
                    }}

                    if(guideZ<=Rt){{
                        guideZ = Rt;
                        guideY += {incremento};
                        delay = {rit_b};
                        dir = 1;
                    }}
                }}

                const p = contact(theta);

                if(pts.length>0){{
                    const d = p.distanceTo(pts[pts.length-1]);
                    if(total+d > maxLen) finished = true;
                    else total += d;
                }}

                pts.push(p);
                rebuild();
            }}

            guide.position.set(guideX,guideY,guideZ);

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }},100);
    </script>
    """

components.html(viewer(), height=altezza)

# =========================
# METRICS
# =========================

st.divider()

m1,m2,m3,m4 = st.columns(4)

m1.metric("Ø Tubo", f"{d_tubo:.2f} mm")
m2.metric("Passo", f"{passo:.2f} mm")
m3.metric("Incremento", f"{incremento:.2f} mm")

outer = diam_aspo + incremento*4
m4.metric("Ø Esterno", f"{outer:.1f} mm")

if outer > 750:
    st.warning("⚠️ Diàmetre massa gran")
