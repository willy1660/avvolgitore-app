import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, altezza):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    setTimeout(() => {{

        const container = document.getElementById("viewer");
        const w = container.clientWidth;
        const h = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 10000);
        camera.position.set(-400, -800, 250);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(w,h);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const r = {d_aspo}/2;
        const h_m = {spalla};

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(r, r, h_m, 64),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        machine.add(mandrel);

        const flangeR = r + 120;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        base.rotation.x = Math.PI/2;
        base.position.z = -h_m/2 - 3;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, 6, 64),
            red
        );
        top.rotation.x = Math.PI/2;
        top.position.z = h_m/2 + 3;
        machine.add(top);

        // =====================
        // GUIDATUBO REAL
        // =====================

        const guideX = -(r + 20);
        const guideY = -120;

        // BASE GROC (rail)
        const rail = new THREE.Mesh(
            new THREE.BoxGeometry(200, 20, 20),
            new THREE.MeshStandardMaterial({{color:0xffff00}})
        );
        rail.position.set(guideX - 80, guideY, -h_m/2 - 10);
        scene.add(rail);

        // CARRO BLAU
        const carriage = new THREE.Mesh(
            new THREE.BoxGeometry(30,20,20),
            new THREE.MeshStandardMaterial({{color:0x0033ff}})
        );
        carriage.position.set(guideX, guideY, -h_m/2);
        scene.add(carriage);

        // BRAÇ GRIS (des del carro)
        const arm = new THREE.Mesh(
            new THREE.CylinderGeometry(6,6,160,16),
            new THREE.MeshStandardMaterial({{color:0xaaaaaa}})
        );
        arm.position.set(guideX + 70, guideY + 40, 0);
        arm.rotation.z = -Math.PI/4;
        scene.add(arm);

        // NOZZLE
        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(6,6,20,16),
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );
        nozzle.position.set(guideX + 120, guideY + 70, 0);
        nozzle.rotation.z = Math.PI/2;
        scene.add(nozzle);

        // COLUMNA NEGRA (entre carro i aspo)
        const column = new THREE.Mesh(
            new THREE.BoxGeometry(20,20,h_m+120),
            new THREE.MeshStandardMaterial({{color:0x111111}})
        );
        column.position.set(guideX + 40, guideY + 20, 0);
        scene.add(column);

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
    altezza = st.slider("Altezza", 400, 900, 700)

components.html(viewer(diametro_aspo, spalla, altezza), height=altezza)
