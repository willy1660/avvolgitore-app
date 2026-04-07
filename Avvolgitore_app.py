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
        "aspo_mode": "Aspo",
        "aspo_visible": "Visibile",
        "aspo_transparent": "Trasparente",
        "aspo_hidden": "Nascosto",
        "guide_offset_x": "Offset guidatubo X (mm)",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro radiale max",
        "metric5": "Ingombro max XY",
        "metric6": "Lunghezza avvolta",
        "warning": "⚠️ Ingombro max XY superiore a 750 mm."
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
        "aspo_mode": "Spool",
        "aspo_visible": "Visible",
        "aspo_transparent": "Transparent",
        "aspo_hidden": "Hidden",
        "guide_offset_x": "Guide offset X (mm)",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Max radial diameter",
        "metric5": "Max XY span",
        "metric6": "Wound length",
        "warning": "⚠️ Max XY span exceeds 750 mm."
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
# GEOMETRY / SIMULATION
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

def simulate_first_layer(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    gradi_start: float,
    deg_step: float = 3.0,
):
    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla

    theta = np.deg2rad(gradi_start)
    radius = R + Rt
    z = Rt

    points = [deposited_point(theta, radius, z)]
    rad_step = np.deg2rad(deg_step)

    direction = 1
    mode = "axial"
    turn_progress = 0.0
    turn_delay = 0.0
    turn_start_radius = radius
    turn_end_radius = radius
    turn_z = z
    first_layer_done = False

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
                first_layer_done = True
            else:
                turn_progress += deg_step
                s = smoothstep(turn_progress / turn_delay)
                radius = turn_start_radius + s * (turn_end_radius - turn_start_radius)
                z = turn_z

                if turn_progress >= turn_delay:
                    radius = turn_end_radius
                    mode = "axial"
                    direction *= -1
                    first_layer_done = True

        new_p = deposited_point(theta, radius, z)
        seg = float(np.linalg.norm(new_p - prev))

        if seg >= max(0.4, Rt * 0.08):
            points.append(new_p)

        if first_layer_done:
            break

    return {
        "points": np.array(points, dtype=float),
        "theta_end": theta,
        "radius_end": radius,
        "z_end": z,
        "direction_end": direction,
    }

def simulate_winding_hybrid(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    lunghezza_m: float,
    gradi_start: float,
    deg_step_first: float = 3.0,
    deg_step_fast: float = 6.0,
):
    max_len = lunghezza_m * 1000.0
    Rt = d_tubo / 2.0
    H = spalla

    first = simulate_first_layer(
        d_aspo=d_aspo,
        spalla=spalla,
        d_tubo=d_tubo,
        passo=passo,
        incremento=incremento,
        rit_b=rit_b,
        rit_t=rit_t,
        gradi_start=gradi_start,
        deg_step=deg_step_first,
    )

    points = first["points"].tolist()
    deposited_len = polyline_length(first["points"])

    if deposited_len >= max_len:
        pts = np.array(points, dtype=float)
        return pts, max_len

    theta = first["theta_end"]
    radius = first["radius_end"]
    z = first["z_end"]
    direction = first["direction_end"]

    step_deg = deg_step_fast
    rad_step = np.deg2rad(step_deg)

    for _ in range(500000):
        prev = np.array(points[-1], dtype=float)
        theta -= rad_step
        z += direction * passo * (step_deg / 360.0)

        hit_top = z >= H - Rt
        hit_bottom = z <= Rt

        if hit_top:
            z = H - Rt
            radius += incremento
            direction = -1
        elif hit_bottom:
            z = Rt
            radius += incremento
            direction = 1

        new_p = deposited_point(theta, radius, z)
        seg = float(np.linalg.norm(new_p - prev))

        if seg < max(0.4, Rt * 0.08):
            continue

        if deposited_len + seg >= max_len:
            remain = max_len - deposited_len
            if seg > 1e-9:
                alpha = remain / seg
                final_p = prev + alpha * (new_p - prev)
                points.append(final_p.tolist())
                deposited_len += float(np.linalg.norm(final_p - prev))
            break

        points.append(new_p.tolist())
        deposited_len += seg

    pts = np.array(points, dtype=float)
    return pts, deposited_len

def compute_max_xy_span(points: np.ndarray, d_tubo: float) -> float:
    if len(points) < 2:
        return float(d_tubo)

    xy = points[:, :2]

    max_samples = 1200
    if len(xy) > max_samples:
        idx = np.linspace(0, len(xy) - 1, max_samples).astype(int)
        xy = xy[idx]

    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    max_centerline_span = float(np.sqrt(np.max(dist2)))
    return max_centerline_span + d_tubo

def compute_metrics(points: np.ndarray, d_tubo: float):
    if len(points) == 0:
        return {
            "diam_radiale": 0.0,
            "max_xy_span": 0.0,
            "wound_length_m": 0.0,
        }

    radial = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    max_centerline_r = float(np.max(radial))
    diam_radiale = 2.0 * (max_centerline_r + d_tubo / 2.0)
    max_xy_span = compute_max_xy_span(points, d_tubo)
    wound_length_m = polyline_length(points) / 1000.0

    return {
        "diam_radiale": diam_radiale,
        "max_xy_span": max_xy_span,
        "wound_length_m": wound_length_m,
    }

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
    final_points,
    aspo_mode,
    guide_offset_x,
):
    anim_js = "true" if anim else "false"
    final_points_json = json.dumps(final_points)
    aspo_mode_json = json.dumps(aspo_mode)

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
        const guideOffsetX = {float(guide_offset_x)};
        const finalPointsRaw = {final_points_json};
        const aspoMode = {aspo_mode_json};

        // =====================
        // MATERIALS
        // =====================

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0xff3333,
            roughness: 0.55,
            metalness: 0.08,
            transparent: aspoMode === "transparent",
            opacity: aspoMode === "transparent" ? 0.18 : 1.0
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

        const freeTubeMat = new THREE.MeshStandardMaterial({{
            color: 0x9fe7ff,
            roughness: 0.55,
            metalness: 0.08
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
        // ASPO GROUP
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

        const flangeR = R + 150.0;
        const flangeTh = 4.0;

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

        machine.visible = aspoMode !== "hidden";

        // bobina animada com a fill del grup que gira
        const rollGroup = new THREE.Group();
        machine.add(rollGroup);

        // =====================
        // GUIDATUBO (en world, no gira)
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(80, 60, 60),
            blueMat
        );
        scene.add(guide);

        // =====================
        // LIGHTS
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.82));

        const dLight1 = new THREE.DirectionalLight(0xffffff, 0.72);
        dLight1.position.set(500, -500, 800);
        scene.add(dLight1);

        const dLight2 = new THREE.DirectionalLight(0xffffff, 0.30);
        dLight2.position.set(-600, 250, 300);
        scene.add(dLight2);

        // =====================
        // HELPERS
        // =====================

        function smoothstep(x) {{
            x = Math.max(0.0, Math.min(1.0, x));
            return x * x * (3.0 - 2.0 * x);
        }}

        // punt de contacte en world: sempre tangent superior
        function contactPointWorld(radius, z) {{
            return new THREE.Vector3(0.0, radius, z);
        }}

        // guidatubo fix en X, i Y = radius per fer l'entrada horitzontal tangent
        function guidePointWorld(radius, z) {{
            return new THREE.Vector3(
                -(radius + guideOffsetX),
                radius,
                z
            );
        }}

        // world -> local del grup que gira
        function worldToMachineLocal(pWorld, thetaMachine) {{
            const c = Math.cos(-thetaMachine);
            const s = Math.sin(-thetaMachine);
            return new THREE.Vector3(
                pWorld.x * c - pWorld.y * s,
                pWorld.x * s + pWorld.y * c,
                pWorld.z
            );
        }}

        function buildTubeMeshFromPoints(points, radialSegments = 12, material = tubeMat) {{
            if (!points || points.length < 2) return null;
            const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.1);
            const tubularSegments = Math.max(24, Math.min(2200, points.length * 2));
            const geo = new THREE.TubeGeometry(curve, tubularSegments, Rt, radialSegments, false);
            return new THREE.Mesh(geo, material);
        }}

        function createMarker(point, material, parentObj = scene) {{
            const g = new THREE.SphereGeometry(Math.max(4, Rt * 0.9), 18, 18);
            const m = new THREE.Mesh(g, material);
            m.position.copy(point);
            parentObj.add(m);
            return m;
        }}

        function clearObj(obj, parentObj = scene) {{
            if (!obj) return;
            parentObj.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }}

        // =====================
        // STATIC FINAL VIEW
        // =====================

        let rollMesh = null;
        let freeMesh = null;
        let startMarker = null;
        let endMarker = null;

        function buildStaticFinalView() {{
            guide.visible = false;

            const finalPts = finalPointsRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

            if (finalPts.length >= 2) {{
                rollMesh = buildTubeMeshFromPoints(finalPts, 12, tubeMat);
                if (rollMesh) scene.add(rollMesh);

                startMarker = createMarker(finalPts[0], startMat, scene);
                endMarker = createMarker(finalPts[finalPts.length - 1], endMat, scene);
            }}
        }}

        // =====================
        // ANIM STATE
        // =====================

        let depositedLocalPoints = [];
        let depositedLength = 0.0;
        let finished = false;
        let lastRebuildCount = -1;

        let thetaMachine = THREE.MathUtils.degToRad({float(gradi_start)});
        let guideRadius = R + Rt;
        let guideZ = Rt;

        let direction = 1;
        let mode = "axial";
        let turnProgress = 0.0;
        let turnDelay = 0.0;
        let turnStartRadius = guideRadius;
        let turnEndRadius = guideRadius;
        let turnZ = guideZ;
        let layerIndex = 0;

        machine.rotation.z = thetaMachine;

        if (animEnabled) {{
            const c0w = contactPointWorld(guideRadius, guideZ);
            const c0l = worldToMachineLocal(c0w, thetaMachine);
            depositedLocalPoints.push(c0l);
            guide.position.copy(guidePointWorld(guideRadius, guideZ));
        }} else {{
            buildStaticFinalView();
        }}

        function rebuildAnimatedMeshes(contactWorld) {{
            if (rollMesh) {{
                clearObj(rollMesh, rollGroup);
                rollMesh = null;
            }}
            if (freeMesh) {{
                clearObj(freeMesh, scene);
                freeMesh = null;
            }}
            if (startMarker) {{
                clearObj(startMarker, rollGroup);
                startMarker = null;
            }}
            if (endMarker) {{
                clearObj(endMarker, rollGroup);
                endMarker = null;
            }}

            if (depositedLocalPoints.length >= 2) {{
                rollMesh = buildTubeMeshFromPoints(depositedLocalPoints, 12, tubeMat);
                if (rollMesh) rollGroup.add(rollMesh);
            }}

            const guideWorld = guidePointWorld(guideRadius, guideZ);
            const freePts = [guideWorld, contactWorld];
            freeMesh = buildTubeMeshFromPoints(freePts, 10, freeTubeMat);
            if (freeMesh) scene.add(freeMesh);

            guide.position.copy(guideWorld);
            guide.visible = true;

            if (depositedLocalPoints.length >= 1) {{
                startMarker = createMarker(depositedLocalPoints[0], startMat, rollGroup);
                endMarker = createMarker(
                    depositedLocalPoints[depositedLocalPoints.length - 1],
                    endMat,
                    rollGroup
                );
            }}
        }}

        function addDepositedPoint(contactLocal) {{
            const prev = depositedLocalPoints[depositedLocalPoints.length - 1];
            const seg = contactLocal.distanceTo(prev);

            if (seg < Math.max(0.8, Rt * 0.10)) return;

            if (depositedLength + seg <= maxLen) {{
                depositedLocalPoints.push(contactLocal.clone());
                depositedLength += seg;
                return;
            }}

            const remain = maxLen - depositedLength;
            if (seg > 1e-9 && remain > 0) {{
                const trim = remain / seg;
                const finalPoint = prev.clone().lerp(contactLocal, trim);
                depositedLocalPoints.push(finalPoint);
                depositedLength += prev.distanceTo(finalPoint);
            }}
            finished = true;
        }}

        function advanceMechanics() {{
            const degPerFrame = 2.0 * speed;
            const radPerFrame = THREE.MathUtils.degToRad(degPerFrame);

            thetaMachine -= radPerFrame;
            machine.rotation.z = thetaMachine;

            if (layerIndex === 0) {{
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
                        layerIndex = 1;
                    }} else {{
                        turnProgress += degPerFrame;
                        const s = smoothstep(turnProgress / turnDelay);
                        guideRadius = turnStartRadius + s * (turnEndRadius - turnStartRadius);
                        guideZ = turnZ;

                        if (turnProgress >= turnDelay) {{
                            guideRadius = turnEndRadius;
                            mode = "axial";
                            direction *= -1;
                            layerIndex = 1;
                        }}
                    }}
                }}
            }} else {{
                guideZ += direction * passo * (degPerFrame / 360.0);

                if (guideZ >= Hs - Rt) {{
                    guideZ = Hs - Rt;
                    guideRadius += incremento;
                    direction = -1;
                    layerIndex += 1;
                }} else if (guideZ <= Rt) {{
                    guideZ = Rt;
                    guideRadius += incremento;
                    direction = 1;
                    layerIndex += 1;
                }}
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled && !finished) {{
                advanceMechanics();

                const cWorld = contactPointWorld(guideRadius, guideZ);
                const cLocal = worldToMachineLocal(cWorld, thetaMachine);

                addDepositedPoint(cLocal);

                if (depositedLocalPoints.length !== lastRebuildCount || finished) {{
                    rebuildAnimatedMeshes(cWorld);
                    lastRebuildCount = depositedLocalPoints.length;
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
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0, step=10.0)
    spalla = st.number_input(t["spalla"], value=95.0, step=1.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0, step=1.0)
    lunghezza = st.number_input(t["lunghezza"], value=50.0, step=5.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0, step=0.5)
    incremento = st.number_input(t["incremento"], value=20.0, step=0.5)
    rit_b = st.number_input(t["rit_min"], value=360.0, step=1.0)
    rit_t = st.number_input(t["rit_max"], value=360.0, step=1.0)
    gradi_start = st.number_input(t["gradi_start"], value=00.0, step=1.0)
    pinza = st.number_input(t["pinza"], value=0.0, step=0.1)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)
    aspo_mode_label = st.selectbox(
        t["aspo_mode"],
        [t["aspo_visible"], t["aspo_transparent"], t["aspo_hidden"]],
        index=0
    )
    guide_offset_x = st.number_input(t["guide_offset_x"], value=150.0, step=1.0)

if aspo_mode_label == t["aspo_visible"]:
    aspo_mode = "visible"
elif aspo_mode_label == t["aspo_transparent"]:
    aspo_mode = "transparent"
else:
    aspo_mode = "hidden"

# =========================
# BUILD
# =========================

d_tubo = d_rame + 2.0 * spessore

points, deposited_len_mm = simulate_winding_hybrid(
    d_aspo=diametro_aspo,
    spalla=spalla,
    d_tubo=d_tubo,
    passo=passo,
    incremento=incremento,
    rit_b=rit_b,
    rit_t=rit_t,
    lunghezza_m=lunghezza,
    gradi_start=gradi_start,
    deg_step_first=3.0,
    deg_step_fast=6.0,
)

metrics = compute_metrics(points, d_tubo)

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
        points.tolist(),
        aspo_mode,
        guide_offset_x,
    ),
    height=altezza
)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
m4.metric(t["metric4"], f"{metrics['diam_radiale']:.1f} mm")
m5.metric(t["metric5"], f"{metrics['max_xy_span']:.1f} mm")
m6.metric(t["metric6"], f"{metrics['wound_length_m']:.3f} m")

if metrics["max_xy_span"] > 750:
    st.warning(t["warning"])
