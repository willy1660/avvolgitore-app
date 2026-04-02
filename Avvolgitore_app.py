import json
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

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline(points, target_length):
    if len(points) < 2:
        return points

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    idx = max(0, min(idx, len(points) - 2))

    p0, p1 = points[idx], points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len < EPS:
        return points[:idx + 1]

    alpha = (target_length - cum[idx]) / seg_len
    alpha = max(0.0, min(1.0, alpha))

    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

def compute_total_turns(points):
    if len(points) < 2:
        return 0.0
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2 * np.pi))

def smoothstep01(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)

# =========================
# GEOMETRY
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
):
    lunghezza_mm = float(lunghezza_m) * 1000.0
    d_tubo = float(d_rame_mm) + 2.0 * float(spessore_guaina_mm)

    passo_assiale = max(float(passo_assiale), EPS)
    passo_radiale = max(float(passo_radiale), EPS)
    spalla_mm = max(float(spalla_mm), EPS)

    ritardo_bottom_deg = max(0.0, min(360.0, float(ritardo_min_deg)))
    ritardo_top_deg = max(0.0, min(360.0, float(ritardo_max_deg)))

    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0
    r = r0

    z = 0.0
    theta = 0.0
    direction = 1

    theta_step_run = np.deg2rad(3.0)
    dz_dtheta = passo_assiale / (2.0 * np.pi)

    bridge_steps_zero_delay = 18

    points = []

    def add_point(theta_val, r_val, z_val):
        x = r_val * np.cos(theta_val)
        y = r_val * np.sin(theta_val)
        points.append([x, y, z_val])

    add_point(theta, r, z)

    pending_radial_shift = 0.0
    pending_bridge_steps = 0

    while True:
        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_mm:
            break

        while True:
            theta += theta_step_run

            if pending_bridge_steps > 0:
                bridge_idx = bridge_steps_zero_delay - pending_bridge_steps + 1
                u = bridge_idx / bridge_steps_zero_delay
                u_prev = (bridge_idx - 1) / bridge_steps_zero_delay
                dr = pending_radial_shift * (smoothstep01(u) - smoothstep01(u_prev))
                r += dr
                pending_bridge_steps -= 1

            z += direction * dz_dtheta * theta_step_run

            if direction == 1 and z >= spalla_mm:
                z = spalla_mm
                add_point(theta, r, z)
                break

            if direction == -1 and z <= 0.0:
                z = 0.0
                add_point(theta, r, z)
                break

            add_point(theta, r, z)

            if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_mm:
                break

        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_mm:
            break

        at_top = direction == 1
        ritardo_deg = ritardo_top_deg if at_top else ritardo_bottom_deg
        theta_dwell = np.deg2rad(ritardo_deg)

        if theta_dwell > EPS:
            dwell_steps = max(12, int(np.ceil(ritardo_deg / 3.0)))
            theta_step_dwell = theta_dwell / dwell_steps

            r_start = r
            r_end = r + passo_radiale
            z_const = spalla_mm if at_top else 0.0

            for i in range(1, dwell_steps + 1):
                theta += theta_step_dwell
                u = i / dwell_steps
                r_curr = r_start + passo_radiale * smoothstep01(u)
                add_point(theta, r_curr, z_const)

            r = r_end
        else:
            pending_radial_shift = passo_radiale
            pending_bridge_steps = bridge_steps_zero_delay

        direction *= -1

    path = np.array(points, dtype=float)
    path = trim_polyline(path, lunghezza_mm)

    r_path = np.sqrt(path[:, 0]**2 + path[:, 1]**2)
    r_max = float(np.max(r_path))
    diam_ext = 2.0 * (r_max + d_tubo / 2.0)

    capes = int((r_max - r0) / passo_radiale) + 1
    capes = max(capes, 1)

    turns_tot = compute_total_turns(path)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext,
        "Capes": capes,
        "VolteTotali": turns_tot,
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = min(8000, max(1500, int(len(pts) * 0.9)))
    radial_segments = 56

    html = f"""
    <div style="width:100%;height:{altezza}px;border-radius:16px;overflow:hidden;background:#0b0d10;">
      <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    scene.fog = new THREE.Fog(0x0b0d10, 1800, 7000);

    const camera = new THREE.PerspectiveCamera(
      42,
      container.clientWidth / container.clientHeight,
      0.1,
      100000
    );

    const renderer = new THREE.WebGLRenderer({{
      antialias: true,
      alpha: false,
      powerPreference: "high-performance"
    }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 0.9;
    controls.panSpeed = 0.7;
    controls.screenSpacePanning = true;
    controls.minDistance = 50;

    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambient);

    const hemi = new THREE.HemisphereLight(0xdde7ff, 0x20242c, 0.7);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(900, 900, 1200);
    key.castShadow = true;
    key.shadow.mapSize.width = 2048;
    key.shadow.mapSize.height = 2048;
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 6000;
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xbfd3ff, 0.35);
    fill.position.set(-900, -400, 600);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0xffffff, 0.25);
    rim.position.set(-300, 1200, 500);
    scene.add(rim);

    const rawPoints = {points_json};
    const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    class CurvePathLinear extends THREE.Curve {{
      constructor(points) {{
        super();
        this.points = points;
      }}
      getPoint(t) {{
        const n = this.points.length;
        const f = t * (n - 1);
        const i = Math.floor(f);
        const i0 = Math.max(0, Math.min(i, n - 2));
        const i1 = i0 + 1;
        const tt = f - i0;
        return new THREE.Vector3().lerpVectors(this.points[i0], this.points[i1], tt);
      }}
    }}

    const curve = new CurvePathLinear(vectors);

    let tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, {radial_segments}, false);
    tubeGeom.computeVertexNormals();

    const tubeMat = new THREE.MeshPhysicalMaterial({{
      color: 0xe2e5e9,
      roughness: 0.68,
      metalness: 0.08,
      clearcoat: 0.22,
      clearcoatRoughness: 0.65,
      reflectivity: 0.18
    }});

    const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
    tubeMesh.castShadow = true;
    tubeMesh.receiveShadow = true;
    scene.add(tubeMesh);

    function createCap(position, direction, color) {{
      const capGeom = new THREE.CircleGeometry({r_tubo}, 48);
      const capMat = new THREE.MeshStandardMaterial({{
        color: color,
        roughness: 0.55,
        metalness: 0.08,
        side: THREE.DoubleSide
      }});
      const cap = new THREE.Mesh(capGeom, capMat);

      const up = new THREE.Vector3(0, 0, 1);
      const dir = direction.clone().normalize();

      if (dir.length() > 1e-9) {{
        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
        cap.quaternion.copy(quat);
      }}

      cap.position.copy(position);
      cap.castShadow = true;
      cap.receiveShadow = true;
      scene.add(cap);
    }}

    if (vectors.length >= 2) {{
      createCap(
        vectors[0],
        vectors[1].clone().sub(vectors[0]).multiplyScalar(-1),
        0x2ecc71
      );
      createCap(
        vectors[vectors.length - 1],
        vectors[vectors.length - 1].clone().sub(vectors[vectors.length - 2]),
        0xe74c3c
      );
    }}

    const box = new THREE.Box3().setFromPoints(vectors);
    const center = new THREE.Vector3();
    box.getCenter(center);

    const size = new THREE.Vector3();
    box.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z);
    const radiusVisual = Math.max(size.x, size.y) * 0.5;

    const gridSize = Math.max(1200, Math.ceil(maxDim * 2.2 / 100) * 100);
    const gridDivisions = Math.max(12, Math.round(gridSize / 100));

    const grid = new THREE.GridHelper(gridSize, gridDivisions, 0x3b4452, 0x232933);
    grid.position.set(center.x, center.y, 0);
    grid.material.opacity = 0.28;
    grid.material.transparent = true;
    scene.add(grid);

    const axes = new THREE.AxesHelper(Math.max(120, radiusVisual * 0.35));
    axes.position.set(center.x, center.y, 0);
    scene.add(axes);

    const planeGeom = new THREE.PlaneGeometry(gridSize, gridSize);
    const planeMat = new THREE.ShadowMaterial({{
      opacity: 0.18
    }});
    const plane = new THREE.Mesh(planeGeom, planeMat);
    plane.receiveShadow = true;
    plane.position.set(center.x, center.y, 0);
    scene.add(plane);

    const dist = Math.max(maxDim * 1.55, 450);
    camera.position.set(
      center.x + dist * 0.95,
      center.y + dist * 0.95,
      center.z + dist * 0.42
    );
    camera.lookAt(center);
    controls.target.copy(center);

    controls.maxDistance = dist * 6.0;

    let progress = 0;
    const total = tubeGeom.index ? tubeGeom.index.count : tubeGeom.attributes.position.count;

    if ({str(animazione).lower()}) {{
      tubeGeom.setDrawRange(0, 0);
    }} else {{
      tubeGeom.setDrawRange(0, total);
    }}

    function animate() {{
      requestAnimationFrame(animate);

      if ({str(animazione).lower()}) {{
        progress += {velocita} * 0.0018;
        if (progress > 1) progress = 1;

        const visible = Math.floor(progress * total);
        tubeGeom.setDrawRange(0, visible);
      }}

      controls.update();
      renderer.render(scene, camera);
    }}

    function onResize() {{
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    }}

    window.addEventListener("resize", onResize);
    animate();
    onResize();
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
    ritardo_min = st.number_input(t["rit_min"], min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    ritardo_max = st.number_input(t["rit_max"], min_value=0.0, max_value=360.0, value=180.0, step=1.0)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 950, 720)
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
)

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita
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
