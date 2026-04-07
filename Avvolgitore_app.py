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

def guide_point(radius: float, z: float, guide_offset_x: float) -> np.ndarray:
    return np.array([-(radius + guide_offset_x), radius, z], dtype=float)

def compute_contact_point_python(
    guide_p: np.ndarray,
    target_p: np.ndarray,
    deposited_points: list,
    d_tubo: float,
    search_back: int = 2200,
):
    """
    Contacte físic simplificat:
    - primera capa: target ideal sobre l'aspo
    - capes superiors: primer punt de la capa existent que intercepta aproximadament
      la recta guide -> target
    """
    if len(deposited_points) < 30:
        return target_p.copy()

    pts = np.array(deposited_points[-search_back:], dtype=float)
    if len(pts) == 0:
        return target_p.copy()

    line = target_p - guide_p
    line_len2 = float(np.dot(line, line))
    if line_len2 < 1e-9:
        return target_p.copy()

    # proximitat axial per evitar tocar voltes llunyanes
    zmask = np.abs(pts[:, 2] - target_p[2]) <= d_tubo * 0.9
    pts = pts[zmask]
    if len(pts) == 0:
        return target_p.copy()

    best_idx = -1
    best_s = None

    for i, p in enumerate(pts):
        v = p - guide_p
        s = float(np.dot(v, line) / line_len2)  # 0..1 sobre el segment
        if s <= 0.0 or s >= 1.0:
            continue

        proj = guide_p + s * line
        perp = float(np.linalg.norm(p - proj))

        if perp <= d_tubo * 0.75:
            if best_s is None or s < best_s:
                best_s = s
                best_idx = i

    if best_idx >= 0:
        return pts[best_idx].copy()

    return target_p.copy()

def simulate_winding_deposition_hybrid(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    lunghezza_m: float,
    gradi_start: float,
    guide_offset_x: float,
    deg_step_first: float = 3.0,
    deg_step_fast: float = 6.0,
    alpha_first: float = 0.34,
    alpha_fast: float = 0.22,
):
    max_len = lunghezza_m * 1000.0
    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla

    theta = np.deg2rad(gradi_start)
    radius = R + Rt
    z = Rt

    deposited_points = [deposited_point(theta, radius, z).tolist()]
    deposited_len = 0.0

    direction = 1
    mode = "axial"
    turn_progress = 0.0
    turn_delay = 0.0
    turn_start_radius = radius
    turn_end_radius = radius
    turn_z = z
    layer_index = 0

    for _ in range(500000):
        prev = np.array(deposited_points[-1], dtype=float)

        step_deg = deg_step_first if layer_index == 0 else deg_step_fast
        alpha_dep = alpha_first if layer_index == 0 else alpha_fast

        theta -= np.deg2rad(step_deg)

        if layer_index == 0:
            if mode == "axial":
                z += direction * passo * (step_deg / 360.0)

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
                    layer_index = 1
                else:
                    turn_progress += step_deg
                    s = smoothstep(turn_progress / turn_delay)
                    radius = turn_start_radius + s * (turn_end_radius - turn_start_radius)
                    z = turn_z

                    if turn_progress >= turn_delay:
                        radius = turn_end_radius
                        mode = "axial"
                        direction *= -1
                        layer_index = 1
        else:
            z += direction * passo * (step_deg / 360.0)

            if z >= H - Rt:
                z = H - Rt
                radius += incremento
                direction = -1
                layer_index += 1
            elif z <= Rt:
                z = Rt
                radius += incremento
                direction = 1
                layer_index += 1

        guide_p = guide_point(radius, z, guide_offset_x)
        target_p = deposited_point(theta, radius, z)
        contact_p = compute_contact_point_python(
            guide_p=guide_p,
            target_p=target_p,
            deposited_points=deposited_points,
            d_tubo=d_tubo,
        )

        new_p = prev + alpha_dep * (contact_p - prev)

        if np.linalg.norm(new_p - contact_p) < Rt * 0.05:
            new_p = contact_p.copy()

        seg = float(np.linalg.norm(new_p - prev))

        if seg < max(0.4, Rt * 0.08):
            continue

        if deposited_len + seg >= max_len:
            remain = max_len - deposited_len
            if seg > 1e-9:
                trim = remain / seg
                final_p = prev + trim * (new_p - prev)
                deposited_points.append(final_p.tolist())
                deposited_len += float(np.linalg.norm(final_p - prev))
            break

        deposited_points.append(new_p.tolist())
        deposited_len += seg

    pts = np.array(deposited_points, dtype=float)
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
        const straightLen = Math.max(50.0, {float(pinza)} * 1000.0);
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

        machine.visible = aspoMode !== "hidden";

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30, 20, 20),
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

        function guidePointFor(radius, z) {{
            return new THREE.Vector3(
                -(radius + guideOffsetX),
                radius,
                z
            );
        }}

        function currentDepositedPoint(thetaMachine, radius, z) {{
            const tubeTheta = -thetaMachine + Math.PI;
            const x = radius * Math.cos(tubeTheta);
            const y = radius * Math.sin(tubeTheta);
            return new THREE.Vector3(x, y, z);
        }}

        function buildTubeMeshFromPoints(points, radialSegments = 12, material = tubeMat) {{
            if (!points || points.length < 2) return null;
            const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.1);
            const tubularSegments = Math.max(24, Math.min(2200, points.length * 2));
            const geo = new THREE.TubeGeometry(curve, tubularSegments, Rt, radialSegments, false);
            return new THREE.Mesh(geo, material);
        }}

        function createMarker(point, material) {{
            const g = new THREE.SphereGeometry(Math.max(4, Rt * 0.9), 18, 18);
            const m = new THREE.Mesh(g, material);
            m.position.copy(point);
            scene.add(m);
            return m;
        }}

        function buildFreePathPoints(guideP, contactP) {{
            return [guideP, contactP];
        }}

        function computeContactPoint(guideP, targetP) {{
            if (depositedPoints.length < 30) return targetP.clone();

            const recentCount = Math.min(2200, depositedPoints.length);
            const line = targetP.clone().sub(guideP);
            const lineLen2 = line.lengthSq();
            if (lineLen2 < 1e-9) return targetP.clone();

            let best = null;
            let bestS = Infinity;

            for (let i = depositedPoints.length - recentCount; i < depositedPoints.length; i += 2) {{
                if (i < 0) continue;
                const p = depositedPoints[i];
                if (Math.abs(p.z - targetP.z) > Rt * 1.8) continue;

                const v = p.clone().sub(guideP);
                const s = v.dot(line) / lineLen2;
                if (s <= 0.0 || s >= 1.0) continue;

                const proj = guideP.clone().add(line.clone().multiplyScalar(s));
                const perp = proj.distanceTo(p);

                if (perp <= Rt * 1.5) {{
                    if (s < bestS) {{
                        bestS = s;
                        best = p.clone();
                    }}
                }}
            }}

            return best ? best : targetP.clone();
        }}

        // =====================
        // STATIC FINAL VIEW
        // =====================

        let rollMesh = null;
        let freeMesh = null;
        let startMarker = null;
        let endMarker = null;

        function clearObj(obj) {{
            if (!obj) return;
            scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }}

        function buildStaticFinalView() {{
            guide.visible = false;

            const finalPoints = finalPointsRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

            if (finalPoints.length >= 2) {{
                rollMesh = buildTubeMeshFromPoints(finalPoints, 12, tubeMat);
                if (rollMesh) scene.add(rollMesh);

                startMarker = createMarker(finalPoints[0], startMat);
                endMarker = createMarker(finalPoints[finalPoints.length - 1], endMat);
            }}
        }}

        // =====================
        // ANIM STATE
        // =====================

        let depositedPoints = [];
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
            depositedPoints.push(currentDepositedPoint(thetaMachine, guideRadius, guideZ));
            guide.position.copy(guidePointFor(guideRadius, guideZ));
        }} else {{
            buildStaticFinalView();
        }}

        function rebuildAnimatedMeshes(guideP, contactP) {{
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
                rollMesh = buildTubeMeshFromPoints(depositedPoints, 12, tubeMat);
                if (rollMesh) scene.add(rollMesh);
            }}

            const freePts = buildFreePathPoints(guideP, contactP);
            if (freePts.length >= 2) {{
                freeMesh = buildTubeMeshFromPoints(freePts, 10, freeTubeMat);
                if (freeMesh) scene.add(freeMesh);
            }}

            if (depositedPoints.length >= 1) {{
                startMarker = createMarker(depositedPoints[0], startMat);
                endMarker = createMarker(depositedPoints[depositedPoints.length - 1], endMat);
            }}
        }}

        function addDepositedPoint(guideP, contactPoint) {{
            const prev = depositedPoints[depositedPoints.length - 1];

            const alpha = (layerIndex === 0) ? 0.34 : 0.22;
            const newPoint = prev.clone().lerp(contactPoint, alpha);

            if (newPoint.distanceTo(contactPoint) < Rt * 0.05) {{
                newPoint.copy(contactPoint);
            }}

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

                const guideP = guidePointFor(guideRadius, guideZ);
                const targetIdeal = currentDepositedPoint(thetaMachine, guideRadius, guideZ);
                const contactP = computeContactPoint(guideP, targetIdeal);
                addDepositedPoint(guideP, contactP);

                guide.visible = true;
                guide.position.copy(guideP);

                if (depositedPoints.length !== lastRebuildCount || finished) {{
                    rebuildAnimatedMeshes(guideP, contactP);
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
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0, step=1.0)
    spalla = st.number_input(t["spalla"], value=95.0, step=1.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0, step=0.1)
    lunghezza = st.number_input(t["lunghezza"], value=30.0, step=0.1)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0, step=0.1)
    incremento = st.number_input(t["incremento"], value=20.0, step=0.1)
    rit_b = st.number_input(t["rit_min"], value=180.0, step=1.0)
    rit_t = st.number_input(t["rit_max"], value=180.0, step=1.0)
    gradi_start = st.number_input(t["gradi_start"], value=30.0, step=1.0)
    pinza = st.number_input(t["pinza"], value=0.3, step=0.05)

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
    guide_offset_x = st.number_input(t["guide_offset_x"], value=80.0, step=1.0)

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

points, deposited_len_mm = simulate_winding_deposition_hybrid(
    d_aspo=diametro_aspo,
    spalla=spalla,
    d_tubo=d_tubo,
    passo=passo,
    incremento=incremento,
    rit_b=rit_b,
    rit_t=rit_t,
    lunghezza_m=lunghezza,
    gradi_start=gradi_start,
    guide_offset_x=guide_offset_x,
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
