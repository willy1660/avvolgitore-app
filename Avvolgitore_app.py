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
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "aspo_mode": "Aspo",
        "aspo_visible": "Visibile",
        "aspo_transparent": "Trasparente",
        "aspo_hidden": "Nascosto",
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
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "aspo_mode": "Spool",
        "aspo_visible": "Visible",
        "aspo_transparent": "Transparent",
        "aspo_hidden": "Hidden",
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

EPS = 1e-9
gradi_start = 0.0
pinza = 0.0
guide_offset_x = 150.0

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

def deposit_point_world(radius: float, z: float) -> np.ndarray:
    return np.array([0.0, radius, z], dtype=float)

def world_to_spool_local(pt_world: np.ndarray, theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    x =  pt_world[0] * c + pt_world[1] * s
    y = -pt_world[0] * s + pt_world[1] * c
    return np.array([x, y, pt_world[2]], dtype=float)

def simulate_winding_center_plane_local(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    lunghezza_m: float,
    gradi_start: float,
    deg_step: float = 2.0,
):
    max_len = lunghezza_m * 1000.0
    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla

    theta = np.deg2rad(gradi_start)
    z = Rt
    current_layer_radius = R + Rt

    first_contact_world = deposit_point_world(current_layer_radius, z)
    first_local = world_to_spool_local(first_contact_world, theta)

    contact_world = [first_contact_world]
    deposited_local = [first_local]
    theta_values = [theta]
    radius_values = [current_layer_radius]
    z_values = [z]

    deposited_len = 0.0

    direction = 1
    mode = "axial"

    turn_progress = 0.0
    turn_delay = 0.0
    turn_z = z
    turn_start_radius = current_layer_radius
    turn_end_radius = current_layer_radius

    for _ in range(1200000):
        next_theta = theta - np.deg2rad(deg_step)

        next_z = z
        next_direction = direction
        next_mode = mode
        next_turn_progress = turn_progress
        next_turn_delay = turn_delay
        next_turn_z = turn_z
        next_turn_start_radius = turn_start_radius
        next_turn_end_radius = turn_end_radius
        next_radius = current_layer_radius

        if mode == "axial":
            next_z = z + direction * passo * (deg_step / 360.0)
            next_radius = current_layer_radius

            if next_z >= H - Rt:
                next_z = H - Rt
                next_mode = "turn"
                next_turn_progress = 0.0
                next_turn_delay = max(rit_t, 0.0)
                next_turn_z = next_z
                next_turn_start_radius = current_layer_radius
                next_turn_end_radius = current_layer_radius + max(0.0, incremento)

            elif next_z <= Rt:
                next_z = Rt
                next_mode = "turn"
                next_turn_progress = 0.0
                next_turn_delay = max(rit_b, 0.0)
                next_turn_z = next_z
                next_turn_start_radius = current_layer_radius
                next_turn_end_radius = current_layer_radius + max(0.0, incremento)

        else:
            next_z = next_turn_z

            if next_turn_delay <= 0.0:
                next_radius = next_turn_end_radius
                current_layer_radius = next_turn_end_radius
                next_mode = "axial"
                next_direction = -direction
            else:
                next_turn_progress = turn_progress + deg_step
                s = smoothstep(next_turn_progress / next_turn_delay)
                next_radius = next_turn_start_radius + s * (next_turn_end_radius - next_turn_start_radius)

                if next_turn_progress >= next_turn_delay:
                    next_radius = next_turn_end_radius
                    current_layer_radius = next_turn_end_radius
                    next_mode = "axial"
                    next_direction = -direction

        new_contact_world = deposit_point_world(next_radius, next_z)
        new_local = world_to_spool_local(new_contact_world, next_theta)

        prev_local = deposited_local[-1]
        seg = float(np.linalg.norm(new_local - prev_local))

        if seg < max(0.25, Rt * 0.05):
            theta = next_theta
            z = next_z
            direction = next_direction
            mode = next_mode
            turn_progress = next_turn_progress
            turn_delay = next_turn_delay
            turn_z = next_turn_z
            turn_start_radius = next_turn_start_radius
            turn_end_radius = next_turn_end_radius
            continue

        if deposited_len + seg >= max_len:
            remain = max_len - deposited_len
            if seg > EPS and remain > 0.0:
                a = remain / seg
                final_theta = theta + a * (next_theta - theta)
                final_z = z + a * (next_z - z)
                prev_r = radius_values[-1]
                final_r = prev_r + a * (next_radius - prev_r)

                final_contact_world = deposit_point_world(final_r, final_z)
                final_local = world_to_spool_local(final_contact_world, final_theta)

                contact_world.append(final_contact_world)
                deposited_local.append(final_local)
                theta_values.append(final_theta)
                radius_values.append(final_r)
                z_values.append(final_z)

                deposited_len += float(np.linalg.norm(final_local - prev_local))
            break

        contact_world.append(new_contact_world)
        deposited_local.append(new_local)
        theta_values.append(next_theta)
        radius_values.append(next_radius)
        z_values.append(next_z)
        deposited_len += seg

        theta = next_theta
        z = next_z
        direction = next_direction
        mode = next_mode
        turn_progress = next_turn_progress
        turn_delay = next_turn_delay
        turn_z = next_turn_z
        turn_start_radius = next_turn_start_radius
        turn_end_radius = next_turn_end_radius

    return (
        np.array(contact_world, dtype=float),
        np.array(deposited_local, dtype=float),
        np.array(theta_values, dtype=float),
        np.array(radius_values, dtype=float),
        np.array(z_values, dtype=float),
        deposited_len,
    )

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
    final_world_contacts,
    final_local_points,
    final_thetas,
    final_radii,
    final_zs,
    aspo_mode,
    guide_offset_x,
):
    anim_js = "true" if anim else "false"
    final_world_contacts_json = json.dumps(final_world_contacts)
    final_local_points_json = json.dumps(final_local_points)
    final_thetas_json = json.dumps(final_thetas)
    final_radii_json = json.dumps(final_radii)
    final_zs_json = json.dumps(final_zs)
    aspo_mode_json = json.dumps(aspo_mode)

    return f"""
    <div id="viewer_root" style="
        width:100%;
        height:{altezza}px;
        background:#0b0f14;
        border-radius:10px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,0.06);
        box-shadow:0 10px 24px rgba(0,0,0,0.30);
    "></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const host = document.getElementById("viewer_root");
        host.innerHTML = "";

        const W = Math.max(host.clientWidth, 600);
        const Hview = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0b0f14);

        const camera = new THREE.PerspectiveCamera(38, W / Hview, 0.1, 20000);
        camera.position.set(-520, -760, 420);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, Hview);
        renderer.outputEncoding = THREE.sRGBEncoding;
        host.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, {spalla}/2);

        const R = {float(d_aspo)} / 2.0;
        const Rt = {float(d_tubo)} / 2.0;
        const Hs = {float(spalla)};
        const speed = {float(vel)};
        const animEnabled = {anim_js};
        const guideOffsetX = {float(guide_offset_x)};
        const aspoMode = {aspo_mode_json};

        const finalWorldContactsRaw = {final_world_contacts_json};
        const finalLocalRaw = {final_local_points_json};
        const finalThetaRaw = {final_thetas_json};
        const finalRadiusRaw = {final_radii_json};
        const finalZRaw = {final_zs_json};

        const finalWorldContacts = finalWorldContactsRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const finalLocalPts = finalLocalRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0x6b7076,
            roughness: 0.78,
            metalness: 0.24,
            transparent: aspoMode === "transparent",
            opacity: aspoMode === "transparent" ? 0.18 : 1.0
        }});

        const blueMat = new THREE.MeshStandardMaterial({{
            color: 0x555b63,
            roughness: 0.82,
            metalness: 0.18
        }});

        const tubeMat = new THREE.MeshStandardMaterial({{
            color: 0xd6d9dd,
            roughness: 0.72,
            metalness: 0.08
        }});

        const freeTubeMat = new THREE.MeshStandardMaterial({{
            color: 0xb8bec6,
            roughness: 0.78,
            metalness: 0.04
        }});

        const startMat = new THREE.MeshStandardMaterial({{
            color: 0x3f7f56,
            roughness: 0.80,
            metalness: 0.06
        }});

        const endMat = new THREE.MeshStandardMaterial({{
            color: 0xa88437,
            roughness: 0.78,
            metalness: 0.06
        }});

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

        const rollGroup = new THREE.Group();
        machine.add(rollGroup);

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(80, 60, 60),
            blueMat
        );
        scene.add(guide);

        const guideFront = new THREE.Mesh(
            new THREE.BoxGeometry(18, 26, 26),
            new THREE.MeshStandardMaterial({{
                color: 0x8b9198,
                roughness: 0.76,
                metalness: 0.22
            }})
        );
        scene.add(guideFront);

        const guideNozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(Math.max(4.0, Rt * 0.28), Math.max(4.0, Rt * 0.28), 22, 20),
            new THREE.MeshStandardMaterial({{
                color: 0x9da3aa,
                roughness: 0.70,
                metalness: 0.26
            }})
        );
        guideNozzle.rotation.z = Math.PI / 2;
        scene.add(guideNozzle);

        scene.add(new THREE.AmbientLight(0xffffff, 0.72));

        const dLight1 = new THREE.DirectionalLight(0xffffff, 0.62);
        dLight1.position.set(500, -500, 800);
        scene.add(dLight1);

        const dLight2 = new THREE.DirectionalLight(0xdfe3e8, 0.18);
        dLight2.position.set(-600, 250, 300);
        scene.add(dLight2);

        const dLight3 = new THREE.DirectionalLight(0xffffff, 0.10);
        dLight3.position.set(0, -900, 500);
        scene.add(dLight3);

        function guidePointWorld(radius, z) {{
            return new THREE.Vector3(
                -(radius + guideOffsetX),
                radius,
                z
            );
        }}

        function disposeMaterial(mat) {{
            if (!mat) return;
            if (Array.isArray(mat)) {{
                mat.forEach(m => m && m.dispose && m.dispose());
            }} else if (mat.dispose) {{
                mat.dispose();
            }}
        }}

        function disposeObj(obj, parentObj = scene) {{
            if (!obj) return;
            parentObj.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            disposeMaterial(obj.material);
        }}

        function createDiscMarker(point, material, parentObj = scene) {{
            const g = new THREE.CylinderGeometry(
                Math.max(4, Rt * 0.8),
                Math.max(4, Rt * 0.8),
                Math.max(2.5, Rt * 0.32),
                18
            );
            const m = new THREE.Mesh(g, material);
            m.rotation.x = Math.PI / 2;
            m.position.copy(point);
            parentObj.add(m);
            return m;
        }}

        function makeTubeSegment(p0, p1, radius, material) {{
            const dir = new THREE.Vector3().subVectors(p1, p0);
            const len = dir.length();
            if (len < 1e-6) return null;

            const geo = new THREE.CylinderGeometry(radius, radius, len, 12, 1, false);
            const mesh = new THREE.Mesh(geo, material);

            const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
            mesh.position.copy(mid);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.clone().normalize());
            mesh.setRotationFromQuaternion(quat);

            return mesh;
        }}

        let depositedSegments = [];
        let depositedJoints = [];
        let freeMesh = null;
        let startMarker = null;
        let endMarker = null;

        let drawIndex = animEnabled ? 2 : finalLocalPts.length;
        let drawAccumulator = 0.0;
        let builtUntil = 1;

        function clearDeposited() {{
            for (const obj of depositedSegments) disposeObj(obj, rollGroup);
            for (const obj of depositedJoints) disposeObj(obj, rollGroup);
            depositedSegments = [];
            depositedJoints = [];
            builtUntil = 1;
        }}

        function rebuildDepositedUpTo(idx) {{
            clearDeposited();

            if (finalLocalPts.length === 0) return;

            const firstJoint = createDiscMarker(finalLocalPts[0], tubeMat, rollGroup);
            depositedJoints.push(firstJoint);

            for (let i = 1; i < idx; i++) {{
                const p0 = finalLocalPts[i - 1];
                const p1 = finalLocalPts[i];
                const seg = makeTubeSegment(p0, p1, Rt, tubeMat);
                if (seg) {{
                    rollGroup.add(seg);
                    depositedSegments.push(seg);
                }}

                const joint = createDiscMarker(p1, tubeMat, rollGroup);
                depositedJoints.push(joint);
            }}

            builtUntil = Math.max(1, idx - 1);
        }}

        function appendOneSegment() {{
            const nextI = builtUntil + 1;
            if (nextI >= drawIndex || nextI >= finalLocalPts.length) return;

            const p0 = finalLocalPts[nextI - 1];
            const p1 = finalLocalPts[nextI];

            const seg = makeTubeSegment(p0, p1, Rt, tubeMat);
            if (seg) {{
                rollGroup.add(seg);
                depositedSegments.push(seg);
            }}

            const joint = createDiscMarker(p1, tubeMat, rollGroup);
            depositedJoints.push(joint);

            builtUntil = nextI;
        }}

        function updateMarkersAndFreeTube() {{
            if (freeMesh) {{
                disposeObj(freeMesh, scene);
                freeMesh = null;
            }}
            if (startMarker) {{
                disposeObj(startMarker, rollGroup);
                startMarker = null;
            }}
            if (endMarker) {{
                disposeObj(endMarker, rollGroup);
                endMarker = null;
            }}

            const safeIndex = Math.max(2, Math.min(drawIndex, finalLocalPts.length));
            const i = safeIndex - 1;

            const currentTheta = finalThetaRaw[i];
            const currentRadius = finalRadiusRaw[i];
            const currentZ = finalZRaw[i];

            machine.rotation.z = currentTheta;

            startMarker = createDiscMarker(finalLocalPts[0], startMat, rollGroup);
            endMarker = createDiscMarker(finalLocalPts[i], endMat, rollGroup);

            const currentEndWorld = finalLocalPts[i].clone().applyAxisAngle(
                new THREE.Vector3(0, 0, 1), currentTheta
            );

            const guideWorld = guidePointWorld(currentRadius, currentZ);

            const seg = makeTubeSegment(guideWorld, currentEndWorld, Rt, freeTubeMat);
            if (seg) {{
                freeMesh = seg;
                scene.add(freeMesh);
            }}

            guide.position.copy(guideWorld);
            guide.visible = true;

            guideFront.position.set(
                guideWorld.x + 49,
                guideWorld.y,
                guideWorld.z
            );
            guideFront.visible = true;

            guideNozzle.position.set(
                guideWorld.x + 67,
                guideWorld.y,
                guideWorld.z
            );
            guideNozzle.visible = true;
        }}

        if (animEnabled) {{
            rebuildDepositedUpTo(drawIndex);
        }} else {{
            rebuildDepositedUpTo(finalLocalPts.length);
            drawIndex = finalLocalPts.length;
        }}
        updateMarkersAndFreeTube();

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled && drawIndex < finalLocalPts.length) {{
                drawAccumulator += Math.max(0.12, speed * 0.85);
                const stepNow = Math.floor(drawAccumulator);

                if (stepNow >= 1) {{
                    drawAccumulator -= stepNow;
                    const oldDrawIndex = drawIndex;
                    drawIndex = Math.min(finalLocalPts.length, drawIndex + stepNow);

                    for (let k = oldDrawIndex; k < drawIndex; k++) {{
                        appendOneSegment();
                    }}

                    updateMarkersAndFreeTube();
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

(
    world_contacts,
    local_points,
    theta_values,
    radius_values,
    z_values,
    deposited_len_mm,
) = simulate_winding_center_plane_local(
    d_aspo=diametro_aspo,
    spalla=spalla,
    d_tubo=d_tubo,
    passo=passo,
    incremento=incremento,
    rit_b=rit_b,
    rit_t=rit_t,
    lunghezza_m=lunghezza,
    gradi_start=gradi_start,
    deg_step=2.0,
)

metrics = compute_metrics(local_points, d_tubo)

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
        world_contacts.tolist(),
        local_points.tolist(),
        theta_values.tolist(),
        radius_values.tolist(),
        z_values.tolist(),
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
