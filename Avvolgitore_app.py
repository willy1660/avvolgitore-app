import os
import glob
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
        "isolamento": "Spessore guaina (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm/rev)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "gradi_start": "Gradi iniziali (°)",
        "pinza": "Lunghezza tratto libero (m)",
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
        "isolamento": "Foam thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm/rev)",
        "incremento": "Layer increment (mm)",
        "rit_min": "Bottom delay (°)",
        "rit_max": "Top delay (°)",
        "gradi_start": "Initial degrees (°)",
        "pinza": "Free straight length (m)",
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
# LOGO
# =========================

def find_logo():
    candidates = [
        "New Logo PDM – rame.png",
        "New Logo PDM - rame.png",
        "new_logo_pdm_rame.png",
        "logo.png",
        "logo.svg",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
    ]
    for name in candidates:
        if os.path.exists(name):
            return name
    for pattern in ("*.png", "*.svg", "*.jpg", "*.jpeg", "*.webp"):
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None

logo_path = find_logo()

if logo_path:
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(logo_path, use_container_width=True)
    with c2:
        st.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

# =========================
# SIMULATION
# =========================

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def deposited_point(theta: float, radius: float, z: float) -> np.ndarray:
    tube_theta = -theta + np.pi
    x = radius * np.cos(tube_theta)
    y = radius * np.sin(tube_theta)
    return np.array([x, y, z], dtype=float)

def simulate_outer_diameter(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    lunghezza_m: float,
    gradi_start: float,
):
    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla
    max_len = lunghezza_m * 1000.0

    theta = np.deg2rad(gradi_start)
    radius = R + Rt
    z = Rt

    points = []
    deposited_len = 0.0

    direction = 1
    mode = "axial"
    turn_progress = 0.0
    turn_delay = 0.0
    turn_start_radius = radius
    turn_end_radius = radius
    turn_z = z

    deg_step = 3.0
    rad_step = np.deg2rad(deg_step)

    p0 = deposited_point(theta, radius, z)
    points.append(p0)

    for _ in range(300000):
        prev = points[-1]
        theta -= rad_step

        if mode == "axial":
            z += direction * passo * (deg_step / 360.0)

            if z >= H - Rt:
                z = H - Rt
                mode = "turn"
                turn_progress = 0.0
                turn_delay = max(rit_t, 0.0)
                turn_start_radius = radius
                turn_end_radius = radius + incremento
                turn_z = z

            elif z <= Rt:
                z = Rt
                mode = "turn"
                turn_progress = 0.0
                turn_delay = max(rit_b, 0.0)
                turn_start_radius = radius
                turn_end_radius = radius + incremento
                turn_z = z

        else:
            if turn_delay <= 0.0:
                radius = turn_end_radius
                mode = "axial"
                direction *= -1
            else:
                turn_progress += deg_step
                s = smoothstep(turn_progress / turn_delay)
                radius = turn_start_radius + s * (turn_end_radius - turn_start_radius)
                z = turn_z

                if turn_progress >= turn_delay:
                    radius = turn_end_radius
                    mode = "axial"
                    direction *= -1

        new_p = deposited_point(theta, radius, z)
        seg = float(np.linalg.norm(new_p - prev))

        if seg < max(0.4, Rt * 0.08):
            continue

        if deposited_len + seg >= max_len:
            remain = max_len - deposited_len
            if seg > 1e-9:
                alpha = remain / seg
                final_p = prev + alpha * (new_p - prev)
                points.append(final_p)
            break

        points.append(new_p)
        deposited_len += seg

    pts = np.array(points)
    radial = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    max_centerline_r = float(np.max(radial)) if len(radial) else (R + Rt)
    outer_diameter = 2.0 * (max_centerline_r + Rt)
    return outer_diameter, pts

# =========================
# VIEWER
# =========================

def viewer(
    d_aspo,
    spalla,
    d_tubo,
    passo,
    incremento,
    rit_b,
    rit_t,
    lunghezza,
    altezza,
    anim,
    vel,
    gradi_start,
    pinza,
    final_points
):
    anim_js = "true" if anim else "false"
    final_points_json = json.dumps(final_points)

    return f"""
    <div id="viewer_root" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const host = document.getElementById("viewer_root");
        host.innerHTML = "";

        const W = Math.max(host.clientWidth, 600);
        const Hview = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(38, W / Hview, 0.1, 20000);
        camera.position.set(-520, -760, 420);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, Hview);
        host.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, {spalla}/2);

        // =====================
        // PARAMS
        // =====================

        const R = {float(d_aspo)} / 2.0;
        const Rt = {float(d_tubo)} / 2.0;
        const Hs = {float(spalla)};
        const passo = {float(passo)};
        const incremento = {float(incremento)};
        const ritB = {float(rit_b)};
        const ritT = {float(rit_t)};
        const maxLen = {float(lunghezza)} * 1000.0;
        const speed = {float(vel)};
        const animEnabled = {anim_js};
        const straightLen = Math.max(50.0, {float(pinza)} * 1000.0);
        const finalPointsRaw = {final_points_json};

        // guidatubo offset com abans
        const guideOffsetX = 80.0;
        let guideRadius = R + Rt;
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

        const tubeMat = new THREE.MeshStandardMaterial({{
            color: 0xffffff,
            roughness: 0.65,
            metalness: 0.15
        }});

        const startMat = new THREE.MeshStandardMaterial({{
            color: 0x00ff88,
            roughness: 0.45,
            metalness: 0.12
        }});

        const endMat = new THREE.MeshStandardMaterial({{
            color: 0xffcc00,
            roughness: 0.45,
            metalness: 0.12
        }});

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, Hs, 96),
            redMat
        );
        mandrel.rotation.x = Math.PI / 2;
        mandrel.position.z = Hs / 2.0;
        machine.add(mandrel);

        const flangeR = R + 120.0;
        const flangeTh = 6.0;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 96),
            redMat
        );
        base.rotation.x = Math.PI / 2;
        base.position.z = 0.0;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 96),
            redMat
        );
        top.rotation.x = Math.PI / 2;
        top.position.z = Hs;
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
        // TUBE MODEL
        // =====================

        let depositedPoints = [];
        let depositedLength = 0.0;
        let finished = false;

        let rollMesh = null;
        let freeMesh = null;
        let startMarker = null;
        let endMarker = null;
        let lastRebuildCount = 0;

        function smoothstep(x) {{
            x = Math.max(0.0, Math.min(1.0, x));
            return x * x * (3.0 - 2.0 * x);
        }}

        function currentGuidePoint() {{
            return new THREE.Vector3(-(guideRadius + guideOffsetX), guideRadius, guideZ);
        }}

        function tangentPoint() {{
            return new THREE.Vector3(0.0, guideRadius, guideZ);
        }}

        function currentDepositedPoint(thetaMachine) {{
            const tubeTheta = -thetaMachine + Math.PI;
            const x = guideRadius * Math.cos(tubeTheta);
            const y = guideRadius * Math.sin(tubeTheta);
            return new THREE.Vector3(x, y, guideZ);
        }}

        function buildTubeMesh(points, radialSegments = 12) {{
            if (!points || points.length < 2) return null;
            const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.1);
            const tubularSegments = Math.max(32, Math.min(2400, points.length * 2));
            const geo = new THREE.TubeGeometry(curve, tubularSegments, Rt, radialSegments, false);
            return new THREE.Mesh(geo, tubeMat);
        }}

        function createMarker(point, material) {{
            const g = new THREE.SphereGeometry(Math.max(4, Rt * 0.9), 18, 18);
            const m = new THREE.Mesh(g, material);
            m.position.copy(point);
            scene.add(m);
            return m;
        }}

        function clearObj(obj) {{
            if (!obj) return;
            scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }}

        function rebuildMeshes(guidePoint, depositedPoint) {{
            if (rollMesh) {{
                clearObj(rollMesh);
                rollMesh = null;
            }}
            if (freeMesh) {{
                clearObj(freeMesh);
                freeMesh = null;
            }}
            if (startMarker) {{
                scene.remove(startMarker);
                startMarker = null;
            }}
            if (endMarker) {{
                scene.remove(endMarker);
                endMarker = null;
            }}

            if (depositedPoints.length >= 2) {{
                rollMesh = buildTubeMesh(depositedPoints, 12);
                if (rollMesh) scene.add(rollMesh);
            }}

            // tram lliure com abans: recte + corba suau fins al punt actual
            const preGuide = new THREE.Vector3(
                guidePoint.x + straightLen,
                guidePoint.y,
                guidePoint.z
            );

            const freePts = [
                depositedPoint.clone(),
                preGuide,
                guidePoint.clone()
            ];

            freeMesh = buildTubeMesh(freePts, 12);
            if (freeMesh) scene.add(freeMesh);

            if (depositedPoints.length >= 1) {{
                startMarker = createMarker(depositedPoints[0], startMat);
                endMarker = createMarker(depositedPoints[depositedPoints.length - 1], endMat);
            }}
        }}

        function buildStaticFinalView() {{
            guide.visible = false;

            const finalPts = finalPointsRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

            if (finalPts.length >= 2) {{
                rollMesh = buildTubeMesh(finalPts, 12);
                if (rollMesh) scene.add(rollMesh);

                startMarker = createMarker(finalPts[0], startMat);
                endMarker = createMarker(finalPts[finalPts.length - 1], endMat);
            }}
        }}

        // =====================
        // INITIAL STATE
        // =====================

        let thetaMachine = THREE.MathUtils.degToRad({float(gradi_start)});
        machine.rotation.z = thetaMachine;

        depositedPoints.push(currentDepositedPoint(thetaMachine));

        // =====================
        // LIGHTS
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const dLight = new THREE.DirectionalLight(0xffffff, 0.7);
        dLight.position.set(500, -500, 800);
        scene.add(dLight);

        // =====================
        // MOTION STATE MACHINE
        // =====================

        let direction = 1;
        let mode = "axial";
        let turnProgress = 0.0;
        let turnDelay = 0.0;
        let turnStartRadius = guideRadius;
        let turnEndRadius = guideRadius;
        let turnZ = guideZ;

        function advanceMechanics() {{
            const degPerFrame = 2.0 * speed;
            const radPerFrame = THREE.MathUtils.degToRad(degPerFrame);

            thetaMachine -= radPerFrame;
            machine.rotation.z = thetaMachine;

            if (mode === "axial") {{
                guideZ += direction * passo * (degPerFrame / 360.0);

                if (guideZ >= Hs - Rt) {{
                    guideZ = Hs - Rt;
                    mode = "turn";
                    turnProgress = 0.0;
                    turnDelay = Math.max(ritT, 0.0);
                    turnStartRadius = guideRadius;
                    turnEndRadius = guideRadius + incremento;
                    turnZ = guideZ;
                }} else if (guideZ <= Rt) {{
                    guideZ = Rt;
                    mode = "turn";
                    turnProgress = 0.0;
                    turnDelay = Math.max(ritB, 0.0);
                    turnStartRadius = guideRadius;
                    turnEndRadius = guideRadius + incremento;
                    turnZ = guideZ;
                }}
            }} else {{
                if (turnDelay <= 0.0) {{
                    guideRadius = turnEndRadius;
                    mode = "axial";
                    direction *= -1;
                }} else {{
                    turnProgress += degPerFrame;
                    const s = smoothstep(turnProgress / turnDelay);
                    guideRadius = turnStartRadius + s * (turnEndRadius - turnStartRadius);
                    guideZ = turnZ;

                    if (turnProgress >= turnDelay) {{
                        guideRadius = turnEndRadius;
                        mode = "axial";
                        direction *= -1;
                    }}
                }}
            }}
        }}

        function addDepositedPoint(newPoint) {{
            const prev = depositedPoints[depositedPoints.length - 1];
            const seg = newPoint.distanceTo(prev);

            if (seg < Math.max(0.8, Rt * 0.10)) {{
                return newPoint;
            }}

            if (depositedLength + seg <= maxLen) {{
                depositedPoints.push(newPoint.clone());
                depositedLength += seg;
                return newPoint;
            }}

            let lo = 0.0;
            let hi = 1.0;

            for (let i = 0; i < 28; i++) {{
                const mid = 0.5 * (lo + hi);
                const candidate = prev.clone().lerp(newPoint, mid);
                const trialLen = depositedLength + prev.distanceTo(candidate);

                if (trialLen < maxLen) lo = mid;
                else hi = mid;
            }}

            const finalPoint = prev.clone().lerp(newPoint, lo);
            depositedPoints.push(finalPoint);
            depositedLength += prev.distanceTo(finalPoint);
            finished = true;
            return finalPoint;
        }}

        if (!animEnabled) {{
            buildStaticFinalView();
        }}

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled) {{
                const guidePoint = currentGuidePoint();

                if (!finished) {{
                    advanceMechanics();
                }}

                const currentGuide = currentGuidePoint();
                const depositedPointRaw = currentDepositedPoint(thetaMachine);
                const depositedPoint = (!finished)
                    ? addDepositedPoint(depositedPointRaw)
                    : depositedPoints[depositedPoints.length - 1];

                guide.visible = true;
                guide.position.copy(currentGuide);

                if (
                    rollMesh === null ||
                    freeMesh === null ||
                    depositedPoints.length !== lastRebuildCount ||
                    finished
                ) {{
                    rebuildMeshes(currentGuide, depositedPoint);
                    lastRebuildCount = depositedPoints.length;
                }}
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
            const nw = Math.max(host.clientWidth, 600);
            const nh = Math.max(host.clientHeight, 400);
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

d_tubo = d_rame + 2 * spessore

diam_esterno, final_points = simulate_outer_diameter(
    diametro_aspo,
    spalla,
    d_tubo,
    passo,
    incremento,
    rit_b,
    rit_t,
    lunghezza,
    gradi_start,
)

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
        vel,
        gradi_start,
        pinza,
        final_points.tolist(),
    ),
    height=altezza
)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
m4.metric(t["metric4"], f"{diam_esterno:.1f} mm")

if diam_esterno > 750:
    st.warning(t["warning"])
