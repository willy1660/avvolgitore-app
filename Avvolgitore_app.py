import json
import numpy as np
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
        "gradi_start": "Gradi iniziali (°)",
        "pinza": "Lunghezza pinza (m)",
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
        "gradi_start": "Initial degrees (°)",
        "pinza": "Clamp length (m)",
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
# METRICS GEOMETRY
# Només per conservar les mètriques
# =========================

def build_coil_for_metrics(
    d_aspo, spalla, lunghezza, d_rame, spessore,
    passo, incremento, rit_b, rit_t, gradi_start, pinza
):
    pts = []
    r = d_aspo / 2 + (d_rame + 2 * spessore) / 2
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

            z += direction * (passo / (2 * np.pi)) * np.deg2rad(4)

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

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y, z])

        if len(pts) > 2:
            length_now = np.sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))
            if length_now > lunghezza * 1000:
                break

    pts = np.array(pts)
    pts[:, 2] -= spalla / 2
    return pts

# =========================
# VIEWER
# X = esquerra / dreta
# Y = profunditat
# Z = vertical
# =========================

def viewer(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza, altezza, anim, vel):
    anim_js = "true" if anim else "false"

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const el = document.getElementById("viewer");
        el.innerHTML = "";

        const w = Math.max(el.clientWidth, 600);
        const h = Math.max(el.clientHeight, 400);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 10000);
        camera.position.set(-500, -700, 400);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(w, h);
        el.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, {spalla}/2);

        // =====================
        // PARAMS
        // =====================

        const R = {float(d_aspo)} / 2.0;
        const H = {float(spalla)};
        const Rt = {float(d_tubo)} / 2.0;

        const passo = {float(passo)};
        const incremento = {float(incremento)};
        const ritB = {float(rit_b)};
        const ritT = {float(rit_t)};
        const maxLen = {float(lunghezza)} * 1000.0;

        // Posició inicial correcta del guidatubo
        // offset en X per evitar col·lisió
        // offset en Y per tangència del tub
        // Z inicial = 0 + Rt
        const guideX = -(R + 80.0);
        let guideY = (R + Rt);
        let guideZ = Rt;

        // =====================
        // MATERIALS
        // =====================

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0xff3333,
            roughness: 0.55,
            metalness: 0.08
        }});

        const blueMat = new THREE.MeshStandardMaterial({{
            color: 0x0044ff,
            roughness: 0.55,
            metalness: 0.10
        }});

        const tubeMat = new THREE.LineBasicMaterial({{
            color: 0xffffff
        }});

        // =====================
        // ASPO GROUP
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, H, 80),
            redMat
        );
        mandrel.rotation.x = Math.PI / 2;
        mandrel.position.z = H / 2;
        machine.add(mandrel);

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(R + 120, R + 120, 6, 80),
            redMat
        );
        base.rotation.x = Math.PI / 2;
        base.position.z = 0;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(R + 120, R + 120, 6, 80),
            redMat
        );
        top.rotation.x = Math.PI / 2;
        top.position.z = H;
        machine.add(top);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30, 20, 20),
            blueMat
        );
        scene.add(guide);

        // =====================
        // TUB / ROTLLO
        // =====================

        let points = [];
        let totalLength = 0.0;
        let finished = false;

        const lineGeometry = new THREE.BufferGeometry();
        const line = new THREE.Line(lineGeometry, tubeMat);
        scene.add(line);

        function currentTubePoint() {{
            // El punt on el tub toca el rotllo és al mateix Y del guidatubo,
            // però en X = 0 perquè la tangència és respecte l’aspo.
            return new THREE.Vector3(0, guideY, guideZ);
        }}

        function updateTubeGeometry() {{
            lineGeometry.setFromPoints(points);
        }}

        function tryAddPoint(p) {{
            if (points.length === 0) {{
                points.push(p.clone());
                updateTubeGeometry();
                return;
            }}

            const prev = points[points.length - 1];
            const dx = p.x - prev.x;
            const dy = p.y - prev.y;
            const dz = p.z - prev.z;
            const d = Math.sqrt(dx*dx + dy*dy + dz*dz);

            if (d <= 1e-9) return;

            const remaining = maxLen - totalLength;

            if (remaining <= 0) {{
                finished = true;
                return;
            }}

            if (d <= remaining) {{
                points.push(p.clone());
                totalLength += d;
                updateTubeGeometry();
            }} else {{
                const alpha = remaining / d;
                const px = prev.x + alpha * dx;
                const py = prev.y + alpha * dy;
                const pz = prev.z + alpha * dz;

                points.push(new THREE.Vector3(px, py, pz));
                totalLength = maxLen;
                finished = true;
                updateTubeGeometry();
            }}
        }}

        // Punt inicial
        guide.position.set(guideX, guideY, guideZ);
        tryAddPoint(currentTubePoint());

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const light = new THREE.DirectionalLight(0xffffff, 0.6);
        light.position.set(500, -500, 800);
        scene.add(light);

        // =====================
        // STATE MACHINE
        // Guidatubo + Aspo manen la simulació
        // =====================

        let dir = 1;      // +1 puja, -1 baixa
        let delay = 0;    // retard discret

        function animate() {{
            requestAnimationFrame(animate);

            if ({anim_js} && !finished) {{

                // gira tot l’aspo sobre eix Y
                machine.rotation.y -= 0.02 * {float(vel)};

                if (delay > 0) {{
                    delay -= 1;
                }} else {{
                    // moviment axial
                    guideZ += dir * passo * 0.02 * {float(vel)};

                    // arribada a spalla superior
                    if (guideZ >= H - Rt) {{
                        guideZ = H - Rt;
                        guideY += incremento;
                        delay = ritT;
                        dir = -1;
                    }}

                    // arribada a base
                    if (guideZ <= Rt) {{
                        guideZ = Rt;
                        guideY += incremento;
                        delay = ritB;
                        dir = 1;
                    }}
                }}

                guide.position.set(guideX, guideY, guideZ);
                tryAddPoint(currentTubePoint());
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
            const nw = Math.max(el.clientWidth, 600);
            const nh = Math.max(el.clientHeight, 400);
            camera.aspect = nw / nh;
            camera.updateProjectionMatrix();
            renderer.setSize(nw, nh);
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
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)
    gradi_start = st.number_input(t["gradi_start"], value=30.0)
    pinza = st.number_input(t["pinza"], value=0.3)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

pts_metrics = build_coil_for_metrics(
    diametro_aspo, spalla, lunghezza,
    d_rame, spessore, passo, incremento,
    rit_b, rit_t, gradi_start, pinza
)

d_tubo = d_rame + 2 * spessore

components.html(
    viewer(
        diametro_aspo,
        spalla,
        d_tubo,
        passo,
        incremento,
        rit_b,
        rit_t,
        lunghezza,
        altezza,
        anim,
        vel
    ),
    height=altezza
)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_rame + 2 * spessore:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")

rmax = np.max(np.sqrt(pts_metrics[:, 0]**2 + pts_metrics[:, 1]**2))
m4.metric(t["metric4"], f"{2 * rmax:.1f} mm")

if 2 * rmax > 750:
    st.warning(t["warning"])
