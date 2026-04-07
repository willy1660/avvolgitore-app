import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

def viewer(d_aspo, spalla, d_tubo, altezza):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    setTimeout(() => {{

        const el = document.getElementById("viewer");
        const w = el.clientWidth;
        const h = el.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, w/h, 0.1, 10000);
        camera.position.set(-500, -600, 250);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(w,h);
        el.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // ASPO
        // =====================

        const R = {d_aspo}/2;
        const H = {spalla};

        const machine = new THREE.Group();
        scene.add(machine);

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, H, 64),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        machine.add(mandrel);

        const flangeR = R + 120;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        base.rotation.x = Math.PI/2;
        base.position.z = -H/2 - 3;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = H/2 + 3;
        machine.add(top);

        // =====================
        // GUIDATUBO (FÍSICAMENT CORRECTE)
        // =====================

        const R_tube = {d_tubo}/2;

        // 👉 POSICIÓ REAL
        const guideX = -(R + R_tube);      // tangència radial
        const guideZ = -H/2 + R_tube;      // tangència amb base

        // bloc blau (cos)
        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30,20,20),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        guide.position.set(guideX - 20, 0, guideZ);
        scene.add(guide);

        // nozzle (sortida tub)
        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(6,6,20,16),
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );

        // orientació radial cap a l’aspo
        nozzle.rotation.z = Math.PI/2;
        nozzle.position.set(guideX, 0, guideZ);
        scene.add(nozzle);

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff,0.8));

        const light = new THREE.DirectionalLight(0xffffff,0.6);
        light.position.set(500,-500,800);
        scene.add(light);

        // =====================
        // ANIMATION
        // =====================

        function animate(){{
            requestAnimationFrame(animate);

            machine.rotation.z -= 0.01;

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }}, 50);
    </script>
    """

# =========================
# UI
# =========================

col1, col2 = st.columns(2)

with col1:
    diametro_aspo = st.number_input("Ø Aspo (mm)", value=450.0)
    spalla = st.number_input("Spalla (mm)", value=95.0)

with col2:
    rame = st.selectbox("Ø Rame", ["1/4","3/8","1/2","5/8","3/4","7/8"])
    spessore = st.number_input("Isolamento", value=7.0)
    d_map = {"1/4":6.35,"3/8":9.52,"1/2":12.7,"5/8":15.88,"3/4":19.05,"7/8":22.23}
    d_rame = d_map[rame]
    d_tubo = d_rame + 2*spessore

altezza = st.slider("Altezza viewer", 400, 900, 700)

components.html(
    viewer(diametro_aspo, spalla, d_tubo, altezza),
    height=altezza
)
