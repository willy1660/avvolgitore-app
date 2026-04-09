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
        "tube_color_mode": "Colore tubo",
        "tube_gelwhite": "Gelwhite",
        "tube_gelblack": "Gelblack",
        "show_grid": "Mostra grid",
        "show_axes": "Mostra assi",
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
        "tube_color_mode": "Tube color",
        "tube_gelwhite": "Gelwhite",
        "tube_gelblack": "Gelblack",
        "show_grid": "Show grid",
        "show_axes": "Show axes",
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
guide_offset_x = 355.0

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
    x = pt_world[0] * c + pt_world[1] * s
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
    tube_color_mode,
    show_grid,
    show_axes,
):
    anim_js = "true" if anim else "false"
    final_local_points_json = json.dumps(final_local_points)
    final_thetas_json = json.dumps(final_thetas)
    final_radii_json = json.dumps(final_radii)
    final_zs_json = json.dumps(final_zs)
    aspo_mode_json = json.dumps(aspo_mode)
    tube_color_mode_json = json.dumps(tube_color_mode)
    show_grid_json = "true" if show_grid else "false"
    show_axes_json = "true" if show_axes else "false"

    bg = "#101317" if tube_color_mode == "gelwhite" else "#f6f6f4"

    return f"""
    <div id="viewer_root" style="
        width:100%;
        height:{altezza}px;
        background:{bg};
        border-radius:10px;
        overflow:hidden;
        border:1px solid rgba(0,0,0,0.08);
        box-shadow:0 10px 24px rgba(0,0,0,0.18);
    "></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/TrackballControls.js"></script>

    <script>
    (() => {{
        const host = document.getElementById("viewer_root");
        host.innerHTML = "";

        const W = Math.max(host.clientWidth, 600);
        const Hview = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();

        const tubeColorMode = {tube_color_mode_json};
        const gelwhite = tubeColorMode === "gelwhite";

        scene.background = new THREE.Color(gelwhite ? 0x101317 : 0xf6f6f4);

        const tubeBaseColor = gelwhite ? 0xd4d4d4 : 0x050505;
        const freeTubeColor = gelwhite ? 0xb8b8b8 : 0x0a0a0a;
        const activeTubeColor = gelwhite ? 0xe7e7e7 : 0x000000;

        const camera = new THREE.PerspectiveCamera(32, W / Hview, 0.1, 20000);
        camera.position.set(0, -25, 1150);

        const renderer = new THREE.WebGLRenderer({{
            antialias: true,
            powerPreference: "high-performance"
        }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
        renderer.setSize(W, Hview);
        renderer.outputEncoding = THREE.sRGBEncoding;
        host.appendChild(renderer.domElement);

        const controls = new THREE.TrackballControls(camera, renderer.domElement);
        controls.rotateSpeed = 3.8;
        controls.zoomSpeed = 0.8;
        controls.panSpeed = 0.1;
        controls.dynamicDampingFactor = 0.2;
        controls.staticMoving = false;
        controls.noPan = false;
        controls.noZoom = false;
        controls.noRotate = false;
        controls.target.set(0, 0, {spalla}/2);

        const R = {float(d_aspo)} / 2.0;
        const Rt = {float(d_tubo)} / 2.0;
        const Hs = {float(spalla)};
        const speed = {float(vel)};
        const animEnabled = {anim_js};
        const guideOffsetX = {float(guide_offset_x)};
        const aspoMode = {aspo_mode_json};
        const showGrid = {show_grid_json};
        const showAxes = {show_axes_json};

        const localRaw = {final_local_points_json};
        const thetaRaw = {final_thetas_json};
        const radiusRaw = {final_radii_json};
        const zRaw = {final_zs_json};

        const localPts = localRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

        // ==========================================
        // WAFFLE / DIAMOND KNURL TEXTURE FOR TUBE
        // ==========================================
        function makeWaffleKnurlTexture(size = 256) {{
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext("2d");

            ctx.fillStyle = "rgb(128,128,128)";
            ctx.fillRect(0, 0, size, size);

            const img = ctx.getImageData(0, 0, size, size);
            const data = img.data;

            const pitch = 24.0;      // mida del patró
            const lineWidth = 4.5;   // gruix de línia
            const depth = 130.0;      // profunditat del relleu

            for (let y = 0; y < size; y++) {{
                for (let x = 0; x < size; x++) {{
                    const u = x;
                    const v = y;

                    const d1 = Math.abs((((u + v) % pitch) + pitch) % pitch - pitch * 0.5);
                    const d2 = Math.abs((((u - v) % pitch) + pitch) % pitch - pitch * 0.5);

                    let value = 128;

                    if (d1 < lineWidth) value -= depth;
                    if (d2 < lineWidth) value -= depth;

                    const cell =
                        0.5 + 0.5 *
                        Math.cos((u + v) * Math.PI / pitch) *
                        Math.cos((u - v) * Math.PI / pitch);

                    value += (cell - 0.5) * 52.0;

                    value = Math.max(0, Math.min(255, Math.round(value)));

                    const i = (y * size + x) * 4;
                    data[i] = value;
                    data[i + 1] = value;
                    data[i + 2] = value;
                    data[i + 3] = 255;
                }}
            }}

            ctx.putImageData(img, 0, 0);

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(1.0, 3.0);
            return tex;
        }}

        function makeSteelTexture(size = 256) {{
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext("2d");

            const grad = ctx.createLinearGradient(0, 0, size, 0);
            grad.addColorStop(0.0, "#616870");
            grad.addColorStop(0.18, "#dadfe3");
            grad.addColorStop(0.36, "#7a8289");
            grad.addColorStop(0.58, "#c7ccd1");
            grad.addColorStop(0.82, "#6b727a");
            grad.addColorStop(1.0, "#e1e5e8");
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, size, size);

            for (let y = 0; y < size; y += 2) {{
                const a = 0.05 + Math.random() * 0.08;
                ctx.fillStyle = `rgba(255,255,255,${{a}})`;
                ctx.fillRect(0, y, size, 1);
            }}

            const img = ctx.getImageData(0, 0, size, size);
            for (let i = 0; i < img.data.length; i += 4) {{
                const n = Math.floor(Math.random() * 18) - 9;
                img.data[i] = Math.max(0, Math.min(255, img.data[i] + n));
                img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1] + n));
                img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2] + n));
            }}
            ctx.putImageData(img, 0, 0);

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(0.5, 0.5);
            return tex;
        }}

        const bumpTex = makeWaffleKnurlTexture(256);
        const steelTex = makeSteelTexture(256);

        const redMat = new THREE.MeshStandardMaterial({{
            color: gelwhite ? 0x676d74 : 0x7a7a7a,
            roughness: 0.84,
            metalness: 0.18,
            transparent: aspoMode === "transparent",
            opacity: aspoMode === "transparent" ? 0.18 : 1.0,
            depthWrite: aspoMode !== "transparent"
        }});

        const blueMat = new THREE.MeshStandardMaterial({{
            color: gelwhite ? 0x5e6670 : 0x737985,
            roughness: 0.86,
            metalness: 0.12
        }});

        const tubeMat = new THREE.MeshStandardMaterial({{
            color: tubeBaseColor,
            roughness: 1.0,
            metalness: 0.0,
            bumpMap: bumpTex,
            bumpScale: 3.0
        }});

        const activeTubeMat = new THREE.MeshStandardMaterial({{
            color: activeTubeColor,
            roughness: 1.0,
            metalness: 0.0,
            bumpMap: bumpTex,
            bumpScale: 3.0
        }});

        const freeTubeMat = new THREE.MeshStandardMaterial({{
            color: freeTubeColor,
            roughness: 1.0,
            metalness: 0.0,
            bumpMap: bumpTex,
            bumpScale: 3.0
        }});

        const steelMat = new THREE.MeshStandardMaterial({{
            color: 0xb8bec4,
            roughness: 0.35,
            metalness: 1.0,
            map: steelTex
        }});

        const steelDarkMat = new THREE.MeshStandardMaterial({{
            color: 0x8a9299,
            roughness: 0.35,
            metalness: 1.0,
            map: steelTex
        }});

        const markerStartMat = new THREE.MeshStandardMaterial({{
            color: 0x23a55a,
            roughness: 0.45,
            metalness: 0.02,
            emissive: 0x0b2013,
            emissiveIntensity: 0.12
        }});

        const markerEndMat = new THREE.MeshStandardMaterial({{
            color: 0xffb020,
            roughness: 0.40,
            metalness: 0.02,
            emissive: 0x2a1800,
            emissiveIntensity: 0.14
        }});

        const machine = new THREE.Group();
        scene.add(machine);

        const depositedGroup = new THREE.Group();
        machine.add(depositedGroup);

        const overlayGroup = new THREE.Group();
        scene.add(overlayGroup);

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

        mandrel.visible = aspoMode !== "hidden";
        base.visible = aspoMode !== "hidden";
        top.visible = aspoMode !== "hidden";

        const nozzleDiameter = 55.0;
        const oldNozzleDiameter = Math.max(4.0, Rt * 0.56);
        const guideScale = (nozzleDiameter / oldNozzleDiameter) * 0.74;

        const guideGroup = new THREE.Group();
        scene.add(guideGroup);

        const guideBarrel = new THREE.Mesh(
            new THREE.CylinderGeometry(20 * guideScale, 20 * guideScale, 44 * guideScale, 32, 1, false),
            steelDarkMat
        );
        guideBarrel.rotation.z = Math.PI / 2;
        guideBarrel.position.x = 0;
        guideGroup.add(guideBarrel);

        const guideShoulder = new THREE.Mesh(
            new THREE.CylinderGeometry(27 * guideScale, 20 * guideScale, 18 * guideScale, 32, 1, false),
            steelMat
        );
        guideShoulder.rotation.z = Math.PI / 2;
        guideShoulder.position.x = 22 * guideScale;
        guideGroup.add(guideShoulder);

        const guideTaper = new THREE.Mesh(
            new THREE.CylinderGeometry(12 * guideScale, 17 * guideScale, 22 * guideScale, 32, 1, false),
            steelMat
        );
        guideTaper.rotation.z = Math.PI / 2;
        guideTaper.position.x = 42 * guideScale;
        guideGroup.add(guideTaper);

        const guideNozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(nozzleDiameter / 2, nozzleDiameter / 2, 14 * guideScale, 36, 1, false),
            steelMat
        );
        guideNozzle.rotation.z = Math.PI / 2;
        guideNozzle.position.x = 58 * guideScale;
        guideGroup.add(guideNozzle);

        const guideBackCap = new THREE.Mesh(
            new THREE.CylinderGeometry(15 * guideScale, 15 * guideScale, 10 * guideScale, 28, 1, false),
            steelDarkMat
        );
        guideBackCap.rotation.z = Math.PI / 2;
        guideBackCap.position.x = -28 * guideScale;
        guideGroup.add(guideBackCap);

        scene.add(new THREE.AmbientLight(0xffffff, gelwhite ? 0.34 : 0.26));

        const hemi = new THREE.HemisphereLight(
            gelwhite ? 0xcfd8e2 : 0xffffff,
            gelwhite ? 0x1a1d20 : 0xd7d0c7,
            gelwhite ? 0.30 : 0.20
        );
        scene.add(hemi);

        const dLight1 = new THREE.DirectionalLight(0xffffff, gelwhite ? 0.30 : 0.18);
        dLight1.position.set(460, -380, 560);
        scene.add(dLight1);

        const dLight2 = new THREE.DirectionalLight(gelwhite ? 0xe2e8ef : 0xf1ede7, gelwhite ? 0.10 : 0.06);
        dLight2.position.set(-520, 220, 260);
        scene.add(dLight2);

        if (showGrid) {{
            const grid = new THREE.GridHelper(
                2600,
                32,
                gelwhite ? 0x707070 : 0x9a9a9a,
                gelwhite ? 0x2b2f33 : 0xdbdbdb
            );
            grid.rotation.x = Math.PI / 2;
            grid.position.z = 0;
            scene.add(grid);
        }}

        if (showAxes) {{
            const axes = new THREE.AxesHelper(350);
            scene.add(axes);
        }}

        function guidePointWorld(radius, z) {{
            return new THREE.Vector3(
                -(radius + guideOffsetX),
                radius,
                z
            );
        }}

        function localPointToWorld(ptLocal, theta) {{
            return ptLocal.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), theta);
        }}

        function lerp(a, b, t) {{
            return a + (b - a) * t;
        }}

        function lerpVec3(a, b, t) {{
            return new THREE.Vector3(
                lerp(a.x, b.x, t),
                lerp(a.y, b.y, t),
                lerp(a.z, b.z, t)
            );
        }}

        class PolylineCurve3 extends THREE.Curve {{
            constructor(points) {{
                super();
                this.points = points || [];
                this.arc = [0];
                this.totalLength = 0;

                for (let i = 1; i < this.points.length; i++) {{
                    const seg = this.points[i].distanceTo(this.points[i - 1]);
                    this.totalLength += seg;
                    this.arc.push(this.totalLength);
                }}
            }}

            getPoint(t) {{
                if (!this.points || this.points.length === 0) return new THREE.Vector3(0, 0, 0);
                if (this.points.length === 1 || this.totalLength <= 1e-9) return this.points[0].clone();

                const target = t * this.totalLength;
                let i = 1;
                while (i < this.arc.length && this.arc[i] < target) i++;

                if (i >= this.points.length) return this.points[this.points.length - 1].clone();

                const l0 = this.arc[i - 1];
                const l1 = this.arc[i];
                const p0 = this.points[i - 1];
                const p1 = this.points[i];
                const denom = Math.max(1e-9, l1 - l0);
                const a = (target - l0) / denom;

                return new THREE.Vector3(
                    p0.x + a * (p1.x - p0.x),
                    p0.y + a * (p1.y - p0.y),
                    p0.z + a * (p1.z - p0.z)
                );
            }}
        }}

        function disposeMaterial(mat) {{
            if (!mat) return;
            if (Array.isArray(mat)) mat.forEach(m => m && m.dispose && m.dispose());
            else if (mat.dispose) mat.dispose();
        }}

        function disposeObj(obj, parentObj = scene) {{
            if (!obj) return;
            parentObj.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            disposeMaterial(obj.material);
        }}

        function makeTubeMeshFromPoints(points, radius, material) {{
            if (!points || points.length < 2) return null;

            let totalLen = 0;
            for (let i = 1; i < points.length; i++) {{
                totalLen += points[i].distanceTo(points[i - 1]);
            }}

            const curve = new PolylineCurve3(points);
            const tubularSegments = Math.max(
                18,
                Math.min(2600, Math.floor(totalLen / Math.max(1.25, radius * 0.48)))
            );

            const geo = new THREE.TubeGeometry(curve, tubularSegments, radius, 14, false);
            geo.computeVertexNormals();
            return new THREE.Mesh(geo, material);
        }}

        function makeTubeSegment(p0, p1, radius, material) {{
            const dir = new THREE.Vector3().subVectors(p1, p0);
            const len = dir.length();
            if (len < 1e-6) return null;

            const geo = new THREE.CylinderGeometry(radius, radius, len, 16, 1, false);
            const mesh = new THREE.Mesh(geo, material);

            const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
            mesh.position.copy(mid);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.clone().normalize());
            mesh.setRotationFromQuaternion(quat);

            return mesh;
        }}

        function makeEndpointDisc(point, tangentDir, material, radiusScale = 0.92) {{
            const r = Math.max(7.0, Rt * radiusScale);
            const geo = new THREE.CylinderGeometry(r, r, Math.max(2.0, Rt * 0.22), 28);
            const mesh = new THREE.Mesh(geo, material);
            mesh.position.copy(point);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, tangentDir.clone().normalize());
            mesh.setRotationFromQuaternion(quat);

            return mesh;
        }}

        let depositedMesh = null;
        let freeMesh = null;
        let activeCoilMesh = null;
        let startMarker = null;
        let endMarker = null;

        let drawPos = animEnabled ? 1.0 : (localPts.length - 1);
        let lastRebuiltCompleted = -1;

        function rebuildDepositedMesh(completedIndex) {{
            if (completedIndex < 1) return;
            if (completedIndex === lastRebuiltCompleted) return;
            lastRebuiltCompleted = completedIndex;

            if (depositedMesh) {{
                disposeObj(depositedMesh, depositedGroup);
                depositedMesh = null;
            }}

            const pts = localPts.slice(0, completedIndex + 1);
            depositedMesh = makeTubeMeshFromPoints(pts, Rt, tubeMat);
            if (depositedMesh) depositedGroup.add(depositedMesh);
        }}

        function clearOverlay() {{
            if (freeMesh) {{
                disposeObj(freeMesh, overlayGroup);
                freeMesh = null;
            }}
            if (activeCoilMesh) {{
                disposeObj(activeCoilMesh, overlayGroup);
                activeCoilMesh = null;
            }}
            if (startMarker) {{
                disposeObj(startMarker, overlayGroup);
                startMarker = null;
            }}
            if (endMarker) {{
                disposeObj(endMarker, overlayGroup);
                endMarker = null;
            }}
        }}

        function updateOverlayContinuous() {{
            clearOverlay();
            if (localPts.length < 2) return;

            const maxPos = localPts.length - 1;
            const clampedPos = Math.max(1.0, Math.min(drawPos, maxPos));

            const i0 = Math.floor(clampedPos);
            const i1 = Math.min(i0 + 1, localPts.length - 1);
            const frac = clampedPos - i0;

            const theta = lerp(thetaRaw[i0], thetaRaw[i1], frac);
            const radius = lerp(radiusRaw[i0], radiusRaw[i1], frac);
            const z = lerp(zRaw[i0], zRaw[i1], frac);

            machine.rotation.z = theta;

            const activeLocalStart = localPts[i0];
            const activeLocalEnd = lerpVec3(localPts[i0], localPts[i1], frac);

            const startWorld = localPointToWorld(localPts[0], theta);
            const endWorld = localPointToWorld(activeLocalEnd, theta);

            const startTangentLocal = localPts[Math.min(1, localPts.length - 1)].clone().sub(localPts[0]);
            const endTangentLocal = activeLocalEnd.clone().sub(activeLocalStart);
            const startTangentWorld = startTangentLocal.clone().applyAxisAngle(new THREE.Vector3(0,0,1), theta);
            const endTangentWorld = endTangentLocal.clone().applyAxisAngle(new THREE.Vector3(0,0,1), theta);

            startMarker = makeEndpointDisc(startWorld, startTangentWorld, markerStartMat, 0.82);
            endMarker = makeEndpointDisc(endWorld, endTangentWorld.length() > 1e-6 ? endTangentWorld : startTangentWorld, markerEndMat, 0.96);
            overlayGroup.add(startMarker);
            overlayGroup.add(endMarker);

            if (animEnabled) {{
                if (frac > 1e-6 && i1 > i0) {{
                    const activeStartWorld = localPointToWorld(activeLocalStart, theta);
                    activeCoilMesh = makeTubeSegment(activeStartWorld, endWorld, Rt, activeTubeMat);
                    if (activeCoilMesh) overlayGroup.add(activeCoilMesh);
                }}

                const guideWorld = guidePointWorld(radius, z);
                freeMesh = makeTubeSegment(guideWorld, endWorld, Rt, freeTubeMat);
                if (freeMesh) overlayGroup.add(freeMesh);

                guideGroup.position.copy(guideWorld);
                guideGroup.visible = true;
            }} else {{
                guideGroup.visible = false;
            }}
        }}

        if (animEnabled) {{
            rebuildDepositedMesh(1);
        }} else {{
            rebuildDepositedMesh(localPts.length - 1);
            drawPos = localPts.length - 1;
        }}

        updateOverlayContinuous();

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled && drawPos < localPts.length - 1) {{
                const advance = 0.08 + Math.pow(speed, 2.35) * 1.1;
                const oldCompleted = Math.floor(drawPos);
                drawPos = Math.min(localPts.length - 1, drawPos + advance);
                const newCompleted = Math.floor(drawPos);

                if (newCompleted > oldCompleted) {{
                    rebuildDepositedMesh(newCompleted);
                }}

                updateOverlayContinuous();
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
            controls.handleResize();
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
    tube_color_mode_label = st.selectbox(
        t["tube_color_mode"],
        [t["tube_gelwhite"], t["tube_gelblack"]],
        index=0
    )
    show_grid = st.checkbox(t["show_grid"], True)
    show_axes = st.checkbox(t["show_axes"], False)

if aspo_mode_label == t["aspo_visible"]:
    aspo_mode = "visible"
elif aspo_mode_label == t["aspo_transparent"]:
    aspo_mode = "transparent"
else:
    aspo_mode = "hidden"

tube_color_mode = "gelwhite" if tube_color_mode_label == t["tube_gelwhite"] else "gelblack"

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
        tube_color_mode,
        show_grid,
        show_axes,
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
