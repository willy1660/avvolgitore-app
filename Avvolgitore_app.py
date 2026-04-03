import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# 🌍 LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

lang_option = st.selectbox(
    "🌍 Language",
    ["🇮🇹 Italiano", "🇺🇸 English (US)"],
    index=0 if st.session_state.lang == "IT" else 1
)

if "Italiano" in lang_option:
    st.session_state.lang = "IT"
else:
    st.session_state.lang = "EN"

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
        "warning": "⚠️ Diametro esterno superiore a 750 mm. La bobina potrebbe uscire dal pallet."
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
        "warning": "⚠️ Outer diameter exceeds 750 mm. Coil may not fit on pallet."
    }
}

t = TEXTS[lang]

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1, 7])

logo_path = os.path.join(os.path.dirname(__file__), "New Logo PDM - rame.png")

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)

with col_title:
    st.markdown(f"# {t['title']}")

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

# =========================
# UTILS
# =========================

def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def compute_total_turns(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2 * np.pi))

def trim_polyline_and_aux(points: np.ndarray, aux_arrays: list[np.ndarray], target_length: float):
    if len(points) < 2:
        return points, aux_arrays

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points, aux_arrays

    idx = np.searchsorted(cum, target_length) - 1
    idx = max(0, min(idx, len(points) - 2))

    p0, p1 = points[idx], points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len < EPS:
        trimmed_points = points[:idx + 1]
        trimmed_aux = [a[:idx + 1] for a in aux_arrays]
        return trimmed_points, trimmed_aux

    alpha = (target_length - cum[idx]) / seg_len
    alpha = max(0.0, min(1.0, alpha))

    p_new = p0 + alpha * (p1 - p0)
    trimmed_points = np.vstack([points[:idx + 1], p_new])

    trimmed_aux = []
    for arr in aux_arrays:
        a0, a1 = arr[idx], arr[idx + 1]
        a_new = a0 + alpha * (a1 - a0)
        trimmed_aux.append(np.vstack([arr[:idx + 1], a_new]))

    return trimmed_points, trimmed_aux

def rot2(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y = float(xy[0]), float(xy[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=float)

def tangent_points_from_external_point(g_xy: np.ndarray, radius: float):
    gx, gy = float(g_xy[0]), float(g_xy[1])
    d2 = gx * gx + gy * gy
    radius = max(float(radius), EPS)

    if d2 <= radius * radius + EPS:
        d = math.sqrt(max(d2, EPS))
        return (
            np.array([radius * gx / d, radius * gy / d], dtype=float),
            np.array([radius * gx / d, radius * gy / d], dtype=float),
        )

    alpha = (radius * radius) / d2
    beta = radius * math.sqrt(d2 - radius * radius) / d2

    perp = np.array([-gy, gx], dtype=float)
    g = np.array([gx, gy], dtype=float)

    t1 = alpha * g + beta * perp
    t2 = alpha * g - beta * perp
    return t1, t2

def decimate_arrays(points: np.ndarray, guide_positions: np.ndarray, max_points: int = 1200):
    n = len(points)
    if n <= max_points:
        return points, guide_positions

    idx = np.linspace(0, n - 1, max_points).astype(int)
    idx = np.unique(idx)
    return points[idx], guide_positions[idx]

# =========================
# GEOMETRY - MANDRÍ + GUIDATUBO DRIVEN
# =========================

def build_coil(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
    ritardo_min_deg,
    ritardo_max_deg,
    gradi_start_deg,
    lunghezza_pinza_m,
):
    lunghezza_totale_mm = float(lunghezza_m) * 1000.0
    lunghezza_pinza_mm = max(0.0, float(lunghezza_pinza_m) * 1000.0)
    lunghezza_visibile_mm = max(0.0, lunghezza_totale_mm - lunghezza_pinza_mm)

    d_tubo = float(d_rame_mm) + 2.0 * float(spessore_guaina_mm)

    passo_assiale = max(float(passo_assiale), EPS)
    passo_radiale = max(float(passo_radiale), EPS)
    spalla_mm = max(float(spalla_mm), d_tubo + EPS)

    ritardo_bottom_deg = max(0.0, float(ritardo_min_deg))
    ritardo_top_deg = max(0.0, float(ritardo_max_deg))
    gradi_start_deg = max(0.0, float(gradi_start_deg))

    z_min = d_tubo / 2.0
    z_max = spalla_mm - d_tubo / 2.0
    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0

    guide_column_x = -(d_aspo_mm / 2.0 + 115.0)
    guide_head_x = guide_column_x - 75.0
    guide_head_y = +70.0

    theta_step_deg = 4.0
    theta_step = math.radians(theta_step_deg)
    dz_dtheta = passo_assiale / (2.0 * math.pi)

    theta = 0.0
    z_guide = z_min
    r_active = r0
    direction = +1

    path_points = []
    guide_positions = []
    prev_contact_world_xy = None

    def compute_contact_world(theta_world: float, z_world: float, radius_active: float):
        nonlocal prev_contact_world_xy

        g_world_xy = np.array([guide_head_x, guide_head_y], dtype=float)
        g_rot_xy = rot2(g_world_xy, -theta_world)

        t1_rot, t2_rot = tangent_points_from_external_point(g_rot_xy, radius_active)

        t1_world = rot2(t1_rot, theta_world)
        t2_world = rot2(t2_rot, theta_world)

        if prev_contact_world_xy is None:
            chosen = t1_world if t1_world[0] > t2_world[0] else t2_world
        else:
            d1 = np.linalg.norm(t1_world - prev_contact_world_xy)
            d2 = np.linalg.norm(t2_world - prev_contact_world_xy)
            chosen = t1_world if d1 <= d2 else t2_world

        prev_contact_world_xy = chosen.copy()

        contact_world = np.array([chosen[0], chosen[1], z_world], dtype=float)
        guide_body_world = np.array([guide_head_x, guide_head_y, z_world], dtype=float)
        return guide_body_world, contact_world

    def add_current_point(theta_world: float, z_world: float, radius_active: float):
        guide_body_world, contact_world = compute_contact_world(theta_world, z_world, radius_active)
        path_points.append(contact_world)
        guide_positions.append(guide_body_world)

    add_current_point(theta, z_guide, r_active)

    if gradi_start_deg > EPS and lunghezza_visibile_mm > EPS:
        start_steps = max(4, int(math.ceil(gradi_start_deg / theta_step_deg)))
        theta_step_start = math.radians(gradi_start_deg) / start_steps

        for _ in range(start_steps):
            theta += theta_step_start
            z_guide = z_min
            add_current_point(theta, z_guide, r_active)

            if len(path_points) > 2 and polyline_length(np.array(path_points, dtype=float)) >= lunghezza_visibile_mm:
                break

    WIND = 0
    TRANSITION = 1

    state = WIND
    trans_total_steps = 0
    trans_steps_remaining = 0
    trans_theta_step = 0.0
    trans_r_start = r_active
    trans_r_end = r_active
    trans_z_const = z_guide
    next_direction = direction

    while True:
        current_path = np.array(path_points, dtype=float)
        if len(current_path) > 2 and polyline_length(current_path) >= lunghezza_visibile_mm:
            break

        if state == WIND:
            theta += theta_step
            z_guide += direction * dz_dtheta * theta_step

            hit_top = direction == +1 and z_guide >= z_max
            hit_bottom = direction == -1 and z_guide <= z_min

            if hit_top or hit_bottom:
                z_guide = z_max if hit_top else z_min
                add_current_point(theta, z_guide, r_active)

                ritardo_deg = ritardo_top_deg if hit_top else ritardo_bottom_deg
                trans_total_steps = max(1, int(math.ceil(max(ritardo_deg, theta_step_deg) / theta_step_deg)))
                trans_steps_remaining = trans_total_steps
                trans_theta_step = math.radians(ritardo_deg) / trans_total_steps if ritardo_deg > EPS else 0.0
                trans_r_start = r_active
                trans_r_end = r_active + passo_radiale
                trans_z_const = z_guide
                next_direction = -direction
                state = TRANSITION
            else:
                add_current_point(theta, z_guide, r_active)

        else:
            done_steps = trans_total_steps - trans_steps_remaining + 1
            u = done_steps / trans_total_steps
            u = max(0.0, min(1.0, u))
            u_smooth = 0.5 - 0.5 * math.cos(math.pi * u)

            theta += trans_theta_step
            r_curr = trans_r_start + (trans_r_end - trans_r_start) * u_smooth
            add_current_point(theta, trans_z_const, r_curr)

            trans_steps_remaining -= 1

            if trans_steps_remaining <= 0:
                r_active = trans_r_end
                z_guide = trans_z_const
                direction = next_direction
                state = WIND

    path = np.array(path_points, dtype=float)
    guide_positions = np.array(guide_positions, dtype=float)

    if lunghezza_visibile_mm > EPS:
        path, [guide_positions] = trim_polyline_and_aux(
            path,
            [guide_positions],
            lunghezza_visibile_mm,
        )

    path[:, 2] -= spalla_mm / 2.0
    guide_positions[:, 2] -= spalla_mm / 2.0

    r_path = np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2)
    r_max = float(np.max(r_path)) if len(r_path) > 0 else r0
    diam_ext = 2.0 * (r_max + d_tubo / 2.0)

    capes = int(np.floor((r_max - r0) / passo_radiale)) + 1
    capes = max(capes, 1)

    turns_tot = compute_total_turns(path)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext,
        "Capes": capes,
        "VolteTotali": turns_tot,
        "Zmin": z_min - spalla_mm / 2.0,
        "Zmax": z_max - spalla_mm / 2.0,
        "MandrelHeight": spalla_mm,
        "LunghezzaVisibile": lunghezza_visibile_mm / 1000.0,
        "LunghezzaPinza": lunghezza_pinza_mm / 1000.0,
        "Rmax": r_max,
        "R0": r0,
        "GuidePositions": guide_positions.tolist(),
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita, d_aspo_mm, spalla_mm, r_max_mm, guide_positions):
    pts = points.tolist()
    points_json = json.dumps(pts)
    guide_json = json.dumps(guide_positions)

    r_tubo = d_tubo / 2.0
    r_mandrel = d_aspo_mm / 2.0
    flange_radius = max(r_max_mm + d_tubo * 1.2, r_mandrel + 55.0)

    tubular_segments = min(900, max(250, int(len(pts) * 0.35)))
    anim_str = "true" if animazione else "false"

    html = f"""
    <div style="width:100%;height:{altezza}px;">
      <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(
      42,
      container.clientWidth / container.clientHeight,
      0.1,
      100000
    );
    camera.up.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x202020, 0.88));

    const light1 = new THREE.DirectionalLight(0xffffff, 0.62);
    light1.position.set(900, -900, 1200);
    scene.add(light1);

    const light2 = new THREE.DirectionalLight(0xffffff, 0.25);
    light2.position.set(-1200, 500, 700);
    scene.add(light2);

    const rawPoints = {points_json};
    const guidePoints = {guide_json};

    const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    class CurvePath extends THREE.Curve {{
      constructor(points) {{
        super();
        this.points = points;
      }}
      getPoint(t) {{
        const n = this.points.length;
        if (n === 1) return this.points[0].clone();
        const f = t * (n - 1);
        const i = Math.floor(f);
        const i0 = Math.max(0, Math.min(i, n - 2));
        const i1 = i0 + 1;
        const tt = f - i0;
        return new THREE.Vector3().lerpVectors(this.points[i0], this.points[i1], tt);
      }}
    }}

    const curve = new CurvePath(vectors);

    const coilGroup = new THREE.Group();
    scene.add(coilGroup);

    let tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 18, false);
    tubeGeom = tubeGeom.toNonIndexed();

    const tubeMat = new THREE.MeshStandardMaterial({{
      color: 0xe6e6e6,
      roughness: 0.84,
      metalness: 0.08
    }});

    const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
    coilGroup.add(tubeMesh);

    const mandrelHeight = {spalla_mm};

    const mandrelGeom = new THREE.CylinderGeometry({r_mandrel}, {r_mandrel}, mandrelHeight, 48, 1, false);
    const mandrelMat = new THREE.MeshStandardMaterial({{
      color: 0x595959,
      roughness: 0.72,
      metalness: 0.45,
      transparent: true,
      opacity: 0.42
    }});
    const mandrelMesh = new THREE.Mesh(mandrelGeom, mandrelMat);
    mandrelMesh.rotation.x = Math.PI / 2.0;
    mandrelMesh.position.set(0, 0, 0);
    scene.add(mandrelMesh);

    const hubRadius = Math.max({r_mandrel} * 0.35, 28.0);
    const hubGeom = new THREE.CylinderGeometry(hubRadius, hubRadius, mandrelHeight, 32);
    const hubMat = new THREE.MeshStandardMaterial({{
      color: 0x2c5fa8,
      roughness: 0.82,
      metalness: 0.18
    }});
    const hubMesh = new THREE.Mesh(hubGeom, hubMat);
    hubMesh.rotation.x = Math.PI / 2.0;
    hubMesh.position.set(0, 0, 0);
    scene.add(hubMesh);

    const flangeRadius = {flange_radius};
    const flangeThickness = 6.0;

    const flangeMat = new THREE.MeshStandardMaterial({{
      color: 0x2e69b9,
      roughness: 0.86,
      metalness: 0.14
    }});

    const baseGeom = new THREE.CylinderGeometry(flangeRadius, flangeRadius, flangeThickness, 64);
    const baseMesh = new THREE.Mesh(baseGeom, flangeMat);
    baseMesh.rotation.x = Math.PI / 2.0;
    baseMesh.position.set(0, 0, -mandrelHeight / 2.0 - flangeThickness / 2.0);
    scene.add(baseMesh);

    const topGeom = new THREE.CylinderGeometry(flangeRadius, flangeRadius, flangeThickness, 64);
    const topMesh = new THREE.Mesh(topGeom, flangeMat);
    topMesh.rotation.x = Math.PI / 2.0;
    topMesh.position.set(0, 0, mandrelHeight / 2.0 + flangeThickness / 2.0);
    scene.add(topMesh);

    function createCap(position, direction, color) {{
      const geometry = new THREE.CircleGeometry({r_tubo}, 20);
      const material = new THREE.MeshBasicMaterial({{
        color: color,
        side: THREE.DoubleSide
      }});
      const cap = new THREE.Mesh(geometry, material);

      const up = new THREE.Vector3(0, 0, 1);
      const dir = direction.clone().normalize();

      if (dir.length() > 1e-9) {{
        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
        cap.quaternion.copy(quat);
      }}

      cap.position.copy(position);
      return cap;
    }}

    let startCap = null;
    let endCap = null;

    if (vectors.length >= 2) {{
      startCap = createCap(
        vectors[0],
        vectors[1].clone().sub(vectors[0]).multiplyScalar(-1),
        0x00ff00
      );
      coilGroup.add(startCap);

      endCap = createCap(
        vectors[vectors.length - 1],
        vectors[vectors.length - 1].clone().sub(vectors[vectors.length - 2]),
        0xff0000
      );
      coilGroup.add(endCap);
    }}

    const guideColumnX = -(flangeRadius + 115.0);

    const guideColumnGeom = new THREE.BoxGeometry(18, 18, mandrelHeight + 180.0);
    const guideColumnMat = new THREE.MeshStandardMaterial({{
      color: 0x5c5c5c,
      roughness: 0.78,
      metalness: 0.35
    }});
    const guideColumn = new THREE.Mesh(guideColumnGeom, guideColumnMat);
    guideColumn.position.set(guideColumnX, 0, 0);
    scene.add(guideColumn);

    const guideGroup = new THREE.Group();
    scene.add(guideGroup);

    const guideBodyGeom = new THREE.BoxGeometry(24, 20, 20);
    const guideBodyMat = new THREE.MeshStandardMaterial({{
      color: 0xf2f2f2,
      roughness: 0.78,
      metalness: 0.10
    }});
    const guideBody = new THREE.Mesh(guideBodyGeom, guideBodyMat);
    guideBody.position.set(0, 0, 0);
    guideGroup.add(guideBody);

    const guideNozzleLen = 18.0;
    const nozzleRadius = Math.max({r_tubo} * 0.95, 4.5);

    const nozzleGeom = new THREE.CylinderGeometry(nozzleRadius, nozzleRadius * 0.9, guideNozzleLen, 20);
    const nozzleMat = new THREE.MeshStandardMaterial({{
      color: 0xb0b0b0,
      roughness: 0.62,
      metalness: 0.42
    }});
    const nozzleMesh = new THREE.Mesh(nozzleGeom, nozzleMat);
    nozzleMesh.rotation.z = Math.PI / 2.0;
    nozzleMesh.position.set(12.0 + guideNozzleLen / 2.0, 0, 0);
    guideGroup.add(nozzleMesh);

    const ringGeom = new THREE.TorusGeometry(Math.max({r_tubo} * 0.9, 4.0), 1.0, 8, 20);
    const ringMat = new THREE.MeshStandardMaterial({{
      color: 0xe0e0e0,
      roughness: 0.58,
      metalness: 0.36
    }});
    const ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringMesh.rotation.y = Math.PI / 2.0;
    ringMesh.position.set(12.0 + guideNozzleLen, 0, 0);
    guideGroup.add(ringMesh);

    const feedGeom = new THREE.CylinderGeometry({r_tubo}, {r_tubo}, 1.0, 16, 1, false);
    const feedMesh = new THREE.Mesh(feedGeom, tubeMat);
    scene.add(feedMesh);

    function setCylinderBetween(mesh, p0, p1) {{
      const dir = new THREE.Vector3().subVectors(p1, p0);
      const len = dir.length();

      if (len < 1e-6) {{
        mesh.visible = false;
        return;
      }}

      mesh.visible = true;

      const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
      mesh.position.copy(mid);

      const yAxis = new THREE.Vector3(0, 1, 0);
      const q = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.clone().normalize());
      mesh.quaternion.copy(q);

      mesh.scale.set(1, len, 1);
    }}

    function getGuideOutletWorld() {{
      const outlet = new THREE.Vector3(12.0 + guideNozzleLen, 0, 0);
      return guideGroup.localToWorld(outlet);
    }}

    const floorGeom = new THREE.CircleGeometry(flangeRadius * 1.5, 64);
    const floorMat = new THREE.MeshBasicMaterial({{
      color: 0x0a0a0a,
      transparent: true,
      opacity: 0.52
    }});
    const floorMesh = new THREE.Mesh(floorGeom, floorMat);
    floorMesh.rotation.x = -Math.PI / 2.0;
    floorMesh.position.set(0, -flangeRadius - 120, -mandrelHeight / 2.0 - flangeThickness - 2);
    scene.add(floorMesh);

    const total = tubeGeom.attributes.position.count;
    let progress = {0.0 if animazione else 1.0};

    if ({anim_str}) {{
      tubeGeom.setDrawRange(0, 0);
    }} else {{
      tubeGeom.setDrawRange(0, total);
    }}

    function getCurrentIndex(progressValue) {{
      const n = Math.max(1, rawPoints.length);
      let idx = Math.floor(progressValue * (n - 1));
      idx = Math.max(0, Math.min(n - 1, idx));
      return idx;
    }}

    function updateMachine(progressValue) {{
      const idx = getCurrentIndex(progressValue);

      const contact = new THREE.Vector3(
        rawPoints[idx][0],
        rawPoints[idx][1],
        rawPoints[idx][2]
      );

      const guidePos = new THREE.Vector3(
        guidePoints[idx][0],
        guidePoints[idx][1],
        guidePoints[idx][2]
      );

      guideGroup.position.copy(guidePos);

      const outlet = getGuideOutletWorld();
      setCylinderBetween(feedMesh, outlet, contact);

      if (endCap) {{
        endCap.position.copy(contact);

        const tangentWorld = new THREE.Vector3().subVectors(contact, outlet).normalize();
        const up = new THREE.Vector3(0, 0, 1);
        if (tangentWorld.length() > 1e-9) {{
          const quat = new THREE.Quaternion().setFromUnitVectors(up, tangentWorld);
          endCap.quaternion.copy(quat);
        }}
      }}
    }}

    updateMachine(progress);

    camera.position.set(700, -980, 290);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);

    function animate() {{
      requestAnimationFrame(animate);

      if ({anim_str}) {{
        progress += {velocita} * 0.002;
        if (progress > 1.0) progress = 1.0;

        const visible = Math.max(2, Math.floor(progress * total));
        tubeGeom.setDrawRange(0, visible);
      }}

      updateMachine(progress);

      controls.update();
      renderer.render(scene, camera);
    }}

    animate();

    window.addEventListener("resize", () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});
    </script>
    """
    return html

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
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], value=7.0, step=0.1)
    lunghezza = st.number_input(t["lunghezza"], value=50.0, step=1.0)

    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0, step=0.1)
    incremento_strato = st.number_input(t["incremento"], value=20.0, step=0.1)
    ritardo_min = st.number_input(t["rit_min"], min_value=0.0, max_value=720.0, value=180.0, step=1.0)
    ritardo_max = st.number_input(t["rit_max"], min_value=0.0, max_value=720.0, value=180.0, step=1.0)
    gradi_start = st.number_input(t["gradi_start"], min_value=0.0, max_value=720.0, value=30.0, step=1.0)
    lunghezza_pinza = st.number_input(t["pinza"], min_value=0.0, value=0.30, step=0.01)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

path, meta = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    passo_assiale,
    incremento_strato,
    ritardo_min,
    ritardo_max,
    gradi_start,
    lunghezza_pinza,
)

guide_positions_np = np.array(meta["GuidePositions"], dtype=float)
path_view, guide_view = decimate_arrays(path, guide_positions_np, max_points=1200)

html = build_viewer_html(
    path_view,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita,
    diametro_aspo,
    spalla,
    meta["Rmax"],
    guide_view.tolist(),
)

components.html(html, height=altezza)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
m2.metric(t["metric2"], f"{meta['PassoAssiale']:.2f} mm")
m3.metric(t["metric3"], f"{meta['IncrementoStrato']:.2f} mm")
m4.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")

if meta["DiametroEsterno"] > 750:
    st.warning(t["warning"])
