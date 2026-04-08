import os
import glob
import json
import math
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
    # Punt de contacte instantani al pla x=0
    return np.array([0.0, radius, z], dtype=float)

def world_to_spool_local(pt_world: np.ndarray, theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    x = pt_world[0] * c + pt_world[1] * s
    y = -pt_world[0] * s + pt_world[1] * c
    return np.array([x, y, pt_world[2]], dtype=float)

def z_bin_index(z: float, bin_w: float) -> int:
    return int(np.floor(z / bin_w))

def circle_pair_candidate_radius(z0: float, z1: float, d_tubo: float) -> float | None:
    dz = abs(z0 - z1)
    half = dz * 0.5
    if half >= d_tubo:
        return None
    return math.sqrt(max(0.0, d_tubo * d_tubo - half * half))

def solve_deposit_radius_between_spires(
    z_target: float,
    r_cmd: float,
    deposited_local_points: list,
    base_radius: float,
    d_tubo: float,
    z_window: float,
):
    """
    Retorna el radi real dipositat per a aquest z_target.

    Idea:
    - El guidatubo imposa un radi màxim comandat r_cmd.
    - El tub ideal rígid es col·loca al menor radi admissible.
    - Es prioritza encaixar entre dues espires de la capa anterior.
    - Si no hi ha parella útil, es busca contacte amb una sola espira.
    - En cap cas es supera r_cmd.
    """
    if len(deposited_local_points) < 1:
        return min(r_cmd, base_radius)

    z_vals = np.array([p[2] for p in deposited_local_points], dtype=float)
    r_vals = np.array([np.linalg.norm(p[:2]) for p in deposited_local_points], dtype=float)

    # candidats propers en z
    mask = np.abs(z_vals - z_target) <= z_window
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return min(r_cmd, base_radius)

    # 1) prioritat: encaix entre dues espirals
    best_pair_radius = None

    # Limitem als punts més propers en radi comandat per evitar barrejar massa capes
    local_idx = idx[np.argsort(np.abs(r_vals[idx] - min(r_cmd, np.min(r_vals[idx]))))]
    local_idx = local_idx[:40]

    for i in range(len(local_idx)):
        for j in range(i + 1, len(local_idx)):
            a = local_idx[i]
            b = local_idx[j]

            ra = r_vals[a]
            rb = r_vals[b]

            # volem parelles relativament de la mateixa capa
            if abs(ra - rb) > d_tubo * 0.35:
                continue

            r_base = 0.5 * (ra + rb)
            rise = circle_pair_candidate_radius(z_vals[a], z_vals[b], d_tubo)
            if rise is None:
                continue

            cand = r_base + rise

            if cand <= r_cmd + 1e-9:
                if best_pair_radius is None or cand < best_pair_radius:
                    best_pair_radius = cand

    if best_pair_radius is not None:
        return max(base_radius, best_pair_radius)

    # 2) si no hi ha parella vàlida, contacte sobre una espira sola
    #    idealment sobre la més baixa possible però admissible
    single_candidates = []
    for k in local_idx:
        dz = abs(z_vals[k] - z_target)
        if dz >= d_tubo:
            continue
        rise = math.sqrt(max(0.0, d_tubo * d_tubo - dz * dz))
        cand = r_vals[k] + rise
        if cand <= r_cmd + 1e-9:
            single_candidates.append(cand)

    if single_candidates:
        return max(base_radius, min(single_candidates))

    # 3) si ni així, el tub queda on li permet el comandament però no per sota del mandrí
    return max(base_radius, r_cmd)

def simulate_winding_realistic(
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
    base_radius = R + Rt

    theta = np.deg2rad(gradi_start)
    z = Rt

    # comandament radial del guidatubo
    r_cmd = base_radius

    first_contact_world = deposit_point_world(base_radius, z)
    first_local = world_to_spool_local(first_contact_world, theta)

    contact_world = [first_contact_world]
    deposited_local = [first_local]
    theta_values = [theta]
    radius_values = [base_radius]
    z_values = [z]

    deposited_len = 0.0

    direction = 1
    mode = "axial"

    turn_progress = 0.0
    turn_delay = 0.0
    turn_z = z
    turn_start_cmd = r_cmd
    turn_end_cmd = r_cmd

    z_window = max(d_tubo * 1.25, passo * 1.5)

    for _ in range(1200000):
        next_theta = theta - np.deg2rad(deg_step)

        next_z = z
        next_direction = direction
        next_mode = mode
        next_turn_progress = turn_progress
        next_turn_delay = turn_delay
        next_turn_z = turn_z
        next_turn_start_cmd = turn_start_cmd
        next_turn_end_cmd = turn_end_cmd
        next_r_cmd = r_cmd

        if mode == "axial":
            next_z = z + direction * passo * (deg_step / 360.0)
            next_r_cmd = r_cmd

            if next_z >= H - Rt:
                next_z = H - Rt
                next_mode = "turn"
                next_turn_progress = 0.0
                next_turn_delay = max(rit_t, 0.0)
                next_turn_z = next_z
                next_turn_start_cmd = r_cmd
                next_turn_end_cmd = r_cmd + max(0.0, incremento)

            elif next_z <= Rt:
                next_z = Rt
                next_mode = "turn"
                next_turn_progress = 0.0
                next_turn_delay = max(rit_b, 0.0)
                next_turn_z = next_z
                next_turn_start_cmd = r_cmd
                next_turn_end_cmd = r_cmd + max(0.0, incremento)

        else:
            next_z = next_turn_z

            if next_turn_delay <= 0.0:
                next_r_cmd = next_turn_end_cmd
                next_mode = "axial"
                next_direction = -direction
            else:
                next_turn_progress = turn_progress + deg_step
                s = smoothstep(next_turn_progress / next_turn_delay)
                next_r_cmd = next_turn_start_cmd + s * (next_turn_end_cmd - next_turn_start_cmd)

                if next_turn_progress >= next_turn_delay:
                    next_r_cmd = next_turn_end_cmd
                    next_mode = "axial"
                    next_direction = -direction

        # radi real dipositat per acomodació geomètrica
        next_r_dep = solve_deposit_radius_between_spires(
            z_target=next_z,
            r_cmd=next_r_cmd,
            deposited_local_points=deposited_local,
            base_radius=base_radius,
            d_tubo=d_tubo,
            z_window=z_window,
        )

        new_contact_world = deposit_point_world(next_r_dep, next_z)
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
            turn_start_cmd = next_turn_start_cmd
            turn_end_cmd = next_turn_end_cmd
            r_cmd = next_r_cmd
            continue

        if deposited_len + seg >= max_len:
            remain = max_len - deposited_len
            if seg > EPS and remain > 0.0:
                a = remain / seg
                final_theta = theta + a * (next_theta - theta)
                final_z = z + a * (next_z - z)
                prev_r = radius_values[-1]
                final_r = prev_r + a * (next_r_dep - prev_r)

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
        radius_values.append(next_r_dep)
        z_values.append(next_z)
        deposited_len += seg

        theta = next_theta
        z = next_z
        direction = next_direction
        mode = next_mode
        turn_progress = next_turn_progress
        turn_delay = next_turn_delay
        turn_z = next_turn_z
        turn_start_cmd = next_turn_start_cmd
        turn_end_cmd = next_turn_end_cmd
        r_cmd = next_r_cmd

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

        class PolylineCurve3 extends THREE.Curve {{
            constructor(points) {{
                super();
                this.points = points || [];
                this.cumulative = [];
                this.totalLength = 0;

                if (this.points.length >= 2) {{
                    this.cumulative = [0];
                    for (let i = 1; i < this.points.length; i++) {{
                        this.totalLength += this.points[i].distanceTo(this.points[i - 1]);
                        this.cumulative.push(this.totalLength);
                    }}
                }}
            }}

            getPoint(t) {{
                if (!this.points || this.points.length === 0) return new THREE.Vector3();
                if (this.points.length === 1) return this.points[0].clone();

                const target = THREE.MathUtils.clamp(t, 0, 1) * this.totalLength;

                for (let i = 1; i < this.cumulative.length; i++) {{
                    if (target <= this.cumulative[i]) {{
                        const l0 = this.cumulative[i - 1];
                        const l1 = this.cumulative[i];
                        const segLen = Math.max(1e-9, l1 - l0);
                        const a = (target - l0) / segLen;
                        return this.points[i - 1].clone().lerp(this.points[i], a);
                    }}
                }}

                return this.points[this.points.length - 1].clone();
            }}
        }}

        function buildTubeMeshFromPolyline(points, radialSegments = 12, material = tubeMat) {{
            if (!points || points.length < 2) return null;

            let totalLen = 0;
            for (let i = 1; i < points.length; i++) {{
                totalLen += points[i].distanceTo(points[i - 1]);
            }}

            const tubularSegments = Math.max(8, Math.min(2500, Math.floor(totalLen / Math.max(1.5, Rt * 0.35))));
            const curve = new PolylineCurve3(points);
            const geo = new THREE.TubeGeometry(curve, tubularSegments, Rt, radialSegments, false);
            return new THREE.Mesh(geo, material);
        }}

        function createMarker(point, material, parentObj = scene) {{
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

        function disposeObj(obj, parentObj = scene) {{
            if (!obj) return;
            parentObj.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }}

        let rollMesh = null;
        let freeMesh = null;
        let startMarker = null;
        let endMarker = null;

        let drawIndex = animEnabled ? 2 : finalLocalPts.length;
        let drawAccumulator = 0.0;

        function rebuildView() {{
            if (rollMesh) {{
                disposeObj(rollMesh, rollGroup);
                rollMesh = null;
            }}
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
            const visibleLocal = finalLocalPts.slice(0, safeIndex);

            if (visibleLocal.length >= 2) {{
                rollMesh = buildTubeMeshFromPolyline(visibleLocal, 12, tubeMat);
                if (rollMesh) rollGroup.add(rollMesh);

                startMarker = createMarker(visibleLocal[0], startMat, rollGroup);
                endMarker = createMarker(visibleLocal[visibleLocal.length - 1], endMat, rollGroup);
            }}

            const i = safeIndex - 1;
            const currentTheta = finalThetaRaw[i];
            const currentRadius = finalRadiusRaw[i];
            const currentZ = finalZRaw[i];

            machine.rotation.z = currentTheta;

            const currentEndWorld = finalLocalPts[i].clone().applyAxisAngle(
                new THREE.Vector3(0, 0, 1), currentTheta
            );

            const guideWorld = guidePointWorld(currentRadius, currentZ);
            const freePts = [guideWorld, currentEndWorld];

            freeMesh = buildTubeMeshFromPolyline(freePts, 10, freeTubeMat);
            if (freeMesh) scene.add(freeMesh);

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

        rebuildView();

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled && drawIndex < finalLocalPts.length) {{
                drawAccumulator += Math.max(0.12, speed * 0.85);
                const stepNow = Math.floor(drawAccumulator);
                if (stepNow >= 1) {{
                    drawAccumulator -= stepNow;
                    drawIndex = Math.min(finalLocalPts.length, drawIndex + stepNow);
                    rebuildView();
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
) = simulate_winding_realistic(
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
