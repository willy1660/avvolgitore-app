import streamlit as st
import streamlit.components.v1 as components

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

TEXTS = {
    "IT": {
        "bobina": "🟦 Bobina",
        "tubo": "🟩 Tubo",
        "avvolg": "🟧 Avvolgimento",
        "viewer": "⚙️ Viewer",
        "diam_aspo": "Ø Aspo (mm)",
        "spalla": "Spalla (mm)",
        "rame": "Ø Rame",
        "isolamento": "Spessore isolamento (mm)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
    },
    "EN": {
        "bobina": "🟦 Coil",
        "tubo": "🟩 Tube",
        "avvolg": "🟧 Winding",
        "viewer": "⚙️ Viewer",
        "diam_aspo": "Spool diameter (mm)",
        "spalla": "Width (mm)",
        "rame": "Copper size",
        "isolamento": "Insulation thickness (mm)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
    }
}

t = TEXTS[lang]

COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, d_tubo, altezza, anim, vel):
    anim_js = "true" if anim else "false"

    return f"""
    <div id="viewer_host" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const host = document.getElementById("viewer_host");
        host.innerHTML = "";

        const W = Math.max(host.clientWidth, 600);
        const H = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 10000);
        camera.position.set(-520, -760, 210);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, H);
        host.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(40, 0, 0);

        // =====================
        // PARAMETERS
        // =====================

        const dAspo = {float(d_aspo)};
        const spalla = {float(spalla)};
        const dTubo = {float(d_tubo)};
        const rMandrel = dAspo / 2.0;

        const flangeR = rMandrel + 120;
        const flangeTh = 6;

        const zMin = -spalla / 2.0;
        const zMax =  spalla / 2.0;

        // =====================
        // LIGHTS
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.85));

        const key = new THREE.DirectionalLight(0xffffff, 0.65);
        key.position.set(-500, -400, 700);
        scene.add(key);

        const fill = new THREE.DirectionalLight(0xffffff, 0.35);
        fill.position.set(400, 200, 500);
        scene.add(fill);

        // =====================
        // ASPO GROUP
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0xff2a2a,
            roughness: 0.55,
            metalness: 0.08
        }});

        const hubMat = new THREE.MeshStandardMaterial({{
            color: 0xc93b3b,
            roughness: 0.65,
            metalness: 0.04
        }});

        const hub = new THREE.Mesh(
            new THREE.CylinderGeometry(rMandrel, rMandrel, spalla, 80),
            hubMat
        );
        hub.rotation.x = Math.PI / 2;
        machine.add(hub);

        const lowerFlange = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 80),
            redMat
        );
        lowerFlange.rotation.x = Math.PI / 2;
        lowerFlange.position.z = zMin - flangeTh / 2;
        machine.add(lowerFlange);

        const upperFlange = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 80),
            redMat
        );
        upperFlange.rotation.x = Math.PI / 2;
        upperFlange.position.z = zMax + flangeTh / 2;
        machine.add(upperFlange);

        // =====================
        // HARDWARE GEOMETRY
        // target: like your reference image
        // =====================

        // right end of yellow rail, close to the spool
        const postX = -(rMandrel + 22);
        const railY = 0;
        const railZ = zMin - 18;

        // yellow rail: left-right
        const railLen = 185;
        const railHeight = 16;
        const railDepth = 18;

        const yellowMat = new THREE.MeshStandardMaterial({{
            color: 0xc8c83a,
            roughness: 0.9,
            metalness: 0.03
        }});

        const blackMat = new THREE.MeshStandardMaterial({{
            color: 0x1a1a1a,
            roughness: 0.82,
            metalness: 0.15
        }});

        const blueMat = new THREE.MeshStandardMaterial({{
            color: 0x2346d8,
            roughness: 0.55,
            metalness: 0.1
        }});

        const steelMat = new THREE.MeshStandardMaterial({{
            color: 0xcfcfcf,
            roughness: 0.35,
            metalness: 0.72
        }});

        const whiteSteelMat = new THREE.MeshStandardMaterial({{
            color: 0xdcdcdc,
            roughness: 0.4,
            metalness: 0.62
        }});

        // rail starts left and ends at post
        const rail = new THREE.Mesh(
            new THREE.BoxGeometry(railLen, railDepth, railHeight),
            yellowMat
        );
        rail.position.set(postX - railLen / 2 + 10, railY, railZ);
        scene.add(rail);

        // vertical black post at right end of rail
        const post = new THREE.Mesh(
            new THREE.BoxGeometry(22, 22, spalla + 145),
            blackMat
        );
        post.position.set(postX, railY, (zMin + zMax) / 2 + 22);
        scene.add(post);

        // carriage rides on rail, slightly left of post
        const carriageX0 = postX - 68;
        const carriage = new THREE.Group();
        scene.add(carriage);

        // blue block
        const blueBlock = new THREE.Mesh(
            new THREE.BoxGeometry(28, 20, 18),
            blueMat
        );
        blueBlock.position.set(carriageX0, railY, railZ + 10);
        scene.add(blueBlock);

        // short metallic roller coming out left
        const leftRoller = new THREE.Mesh(
            new THREE.CylinderGeometry(6.2, 6.2, 26, 20),
            whiteSteelMat
        );
        leftRoller.rotation.z = Math.PI / 2;
        leftRoller.position.set(carriageX0 - 18, railY, railZ + 10);
        scene.add(leftRoller);

        // diagonal arm from carriage to tangent point near lower flange
        // approximate the visual reference
        const armStart = new THREE.Vector3(carriageX0 + 8, railY, railZ + 16);
        const armEnd   = new THREE.Vector3(-(rMandrel + dTubo/2 + 6), railY, zMin + 24);

        const armVec = new THREE.Vector3().subVectors(armEnd, armStart);
        const armLen = armVec.length();

        const arm = new THREE.Mesh(
            new THREE.CylinderGeometry(5.2, 5.2, armLen, 20),
            steelMat
        );
        arm.position.copy(new THREE.Vector3().addVectors(armStart, armEnd).multiplyScalar(0.5));
        arm.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            armVec.clone().normalize()
        );
        scene.add(arm);

        // nozzle at end of arm, near spool tangent
        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(7, 7, 22, 20),
            whiteSteelMat
        );
        nozzle.position.copy(armEnd);
        nozzle.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            armVec.clone().normalize()
        );
        scene.add(nozzle);

        // small support from post toward arm area, to compact the assembly visually
        const braceStart = new THREE.Vector3(postX, railY, railZ + 18);
        const braceEnd   = new THREE.Vector3(postX - 6, railY, zMin + 34);
        const braceVec = new THREE.Vector3().subVectors(braceEnd, braceStart);
        const braceLen = braceVec.length();

        const brace = new THREE.Mesh(
            new THREE.BoxGeometry(10, 10, braceLen),
            blackMat
        );
        brace.position.copy(new THREE.Vector3().addVectors(braceStart, braceEnd).multiplyScalar(0.5));
        brace.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 0, 1),
            braceVec.clone().normalize()
        );
        scene.add(brace);

        // =====================
        // ANIMATION
        // =====================

        const animEnabled = {anim_js};
        const speed = {float(vel)};

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled) {{
                machine.rotation.z -= 0.01 * speed;
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
            const w = Math.max(host.clientWidth, 600);
            const h = Math.max(host.clientHeight, 400);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }});
    }})();
    </script>
    """

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"#### {t['bobina']}")
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    st.write("—")

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

d_tubo = d_rame + 2 * spessore

components.html(
    viewer(diametro_aspo, spalla, d_tubo, altezza, anim, vel),
    height=altezza
)
