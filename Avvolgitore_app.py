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
        "attach_deg": "Aggancio iniziale (°)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
        "metric5": "Lunghezza richiesta",
        "metric6": "Lunghezza reale",
        "metric7": "Volte totali",
        "warning_spalla": "⚠️ La spalla és massa petita per al diàmetre del tub.",
        "warning_diam": "⚠️ Diametro esterno superiore a 750 mm. La bobina potrebbe uscire dal pallet."
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
        "attach_deg": "Initial attachment (°)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Outer diameter",
        "metric5": "Requested length",
        "metric6": "Real length",
        "metric7": "Total turns",
        "warning_spalla": "⚠️ Width is too small for tube diameter.",
        "warning_diam": "⚠️ Outer diameter exceeds 750 mm. Coil may not fit on pallet."
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
        st.image(logo_path, width=120)

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

def smoothstep01(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return 0.5 - 0.5 * np.cos(np.pi * u)

def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline(points: np.ndarray, target_length: float) -> np.ndarray:
    if len(points) < 2:
        return points

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    idx = max(0, min(idx, len(points) - 2))

    p0 = points[idx]
    p1 = points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len < EPS:
        return points[:idx + 1]

    alpha = (target_length - cum[idx]) / seg_len
    alpha = max(0.0, min(1.0, alpha))

    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])

def compute_total_turns(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2.0 * np.pi))

def add_point(points, x, y, z, total_len):
    if points:
        p_prev = np.array(points[-1], dtype=float)
        p_new = np.array([x, y, z], dtype=float)
        total_len += float(np.linalg.norm(p_new - p_prev))
    points.append([float(x), float(y), float(z)])
    return total_len

# =========================
# GEOMETRY
# =========================

def build_coil(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale_mm,
    passo_radiale_mm,
    ritardo_bottom_deg,
    ritardo_top_deg,
    attach_deg,
):
    lunghezza_target_mm = float(lunghezza_m) * 1000.0

    d_tubo = float(d_rame_mm) + 2.0 * float(spessore_guaina_mm)
    r_tubo = d_tubo / 2.0

    passo_assiale_mm = max(float(passo_assiale_mm), EPS)
    passo_radiale_mm = max(float(passo_radiale_mm), EPS)
    spalla_mm = float(spalla_mm)
    d_aspo_mm = float(d_aspo_mm)

    ritardo_bottom_deg = max(0.0, min(360.0, float(ritardo_bottom_deg)))
    ritardo_top_deg = max(0.0, min(360.0, float(ritardo_top_deg)))
    attach_deg = max(0.0, min(720.0, float(attach_deg)))

    if spalla_mm <= d_tubo:
        spalla_mm = d_tubo + 0.001

    # El tub toca base i spalla, no el centreline
    z_min = r_tubo
    z_max = spalla_mm - r_tubo

    # El tub toca el mandrí
    r = d_aspo_mm / 2.0 + r_tubo
    r0 = r

    z = z_min
    theta = 0.0
    direction = 1  # +1 puja, -1 baixa

    theta_step_run = np.deg2rad(1.0)
    dz_dtheta = passo_assiale_mm / (2.0 * np.pi)

    points = []
    total_len_mm = 0.0

    # Punt inicial
    total_len_mm = add_point(points, r * np.cos(theta), r * np.sin(theta), z, total_len_mm)

    # =========================
    # AGGANCIO INIZIALE
    # =========================
    theta_attach = np.deg2rad(attach_deg)
    steps_attach = max(1, int(np.ceil(max(attach_deg, 1.0) / 6.0)))

    for _ in range(steps_attach):
        if total_len_mm >= lunghezza_target_mm:
            break
        theta += theta_attach / steps_attach
        total_len_mm = add_point(points, r * np.cos(theta), r * np.sin(theta), z, total_len_mm)

    # =========================
    # MAIN LOOP
    # =========================
    while total_len_mm < lunghezza_target_mm:

        # RUN HELICOIDAL
        while True:
            if total_len_mm >= lunghezza_target_mm:
                break

            theta += theta_step_run
            z += direction * dz_dtheta * theta_step_run

            if direction == 1 and z >= z_max:
                z = z_max
                total_len_mm = add_point(points, r * np.cos(theta), r * np.sin(theta), z, total_len_mm)
                break

            if direction == -1 and z <= z_min:
                z = z_min
                total_len_mm = add_point(points, r * np.cos(theta), r * np.sin(theta), z, total_len_mm)
                break

            total_len_mm = add_point(points, r * np.cos(theta), r * np.sin(theta), z, total_len_mm)

        if total_len_mm >= lunghezza_target_mm:
            break

        # DWELL / RITARDO
        at_top = direction == 1
        ritardo_deg = ritardo_top_deg if at_top else ritardo_bottom_deg
        theta_dwell = np.deg2rad(ritardo_deg)

        if theta_dwell > EPS:
            dwell_steps = max(6, int(np.ceil(ritardo_deg / 2.0)))
            theta_step_dwell = theta_dwell / dwell_steps

            r_start = r
            r_end = r + passo_radiale_mm

            for i in range(1, dwell_steps + 1):
                if total_len_mm >= lunghezza_target_mm:
                    break

                theta += theta_step_dwell
                u = i / dwell_steps
                r_curr = r_start + (r_end - r_start) * smoothstep01(u)
                r = r_curr

                total_len_mm = add_point(points, r * np.cos(theta), r * np.sin(theta), z, total_len_mm)

            r = r_end
        else:
            r += passo_radiale_mm

        direction *= -1

    path = np.array(points, dtype=float)

    # Trim exacte a la longitud objectiu
    path = trim_polyline(path, lunghezza_target_mm)

    # Recalcular totes les magnituds sobre la geometria final retallada
    lunghezza_reale_mm = polyline_length(path)
    r_path = np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2)
    diam_ext = 2.0 * float(np.max(r_path + r_tubo))
    turns_tot = compute_total_turns(path)

    capes = int(max(1, np.floor((np.max(r_path) - r0) / passo_radiale_mm) + 1))

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale_mm,
        "IncrementoStrato": passo_radiale_mm,
        "DiametroEsterno": diam_ext,
        "LunghezzaRichiesta": lunghezza_target_mm / 1000.0,
        "LunghezzaReale": lunghezza_reale_mm / 1000.0,
        "VolteTotali": turns_tot,
        "Capes": capes,
        "SpallaEffettiva": spalla_mm,
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, d_aspo, spalla, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    r_aspo = d_aspo / 2.0
    tubular_segments = min(5000, max(1200, int(len(pts) * 0.55)))

    html = f"""
    <div style="width:100%;height:{altezza}px;">
      <div id="viewer" style="width:100%;height:100%;overflow:hidden;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");
    container.innerHTML = "";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const width = Math.max(container.clientWidth, 300);
    const height = Math.max(container.clientHeight, 300);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 0.85));

    const light1 = new THREE.DirectionalLight(0xffffff, 0.70);
    light1.position.set(5, 5, 8);
    scene.add(light1);

    const light2 = new THREE.DirectionalLight(0xffffff, 0.35);
    light2.position.set(-6, -4, 3);
    scene.add(light2);

    const rawPoints = {points_json};
    const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    class CurvePath extends THREE.Curve {{
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

    const curve = new CurvePath(vectors);

    let tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 48, false);
    tubeGeom = tubeGeom.toNonIndexed();

    const tubeMat = new THREE.MeshStandardMaterial({{
      color: 0xcfcfcf,
      roughness: 0.88,
      metalness: 0.04
    }});

    const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
    scene.add(tubeMesh);

    // CAPS
    function createCap(position, direction, color) {{
      const geometry = new THREE.CircleGeometry({r_tubo}, 32);
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
      scene.add(cap);
    }}

    if (vectors.length >= 2) {{
      createCap(
        vectors[0],
        vectors[1].clone().sub(vectors[0]).multiplyScalar(-1),
        0x00ff00
      );
      createCap(
        vectors[vectors.length - 1],
        vectors[vectors.length - 1].clone().sub(vectors[vectors.length - 2]),
        0xff0000
      );
    }}

    // MANDRÍ
    const mandrelMat = new THREE.MeshStandardMaterial({{
      color: 0x444444,
      roughness: 0.90,
      metalness: 0.15,
      transparent: true,
      opacity: 0.42
    }});

    const mandrel = new THREE.Mesh(
      new THREE.CylinderGeometry({r_aspo}, {r_aspo}, {spalla}, 64, 1, false),
      mandrelMat
    );

    // eix del cilindre -> Z
    mandrel.rotation.x = Math.PI / 2;
    // el mandrí ocupa z = [0, spalla], així que el centre és spalla/2
    mandrel.position.z = {spalla} / 2.0;

    scene.add(mandrel);

    // framing considerant també el mandrí
    const boxTube = new THREE.Box3().setFromPoints(vectors);
    const boxMandrel = new THREE.Box3().setFromObject(mandrel);
    const fullBox = boxTube.union(boxMandrel);

    const center = new THREE.Vector3();
    fullBox.getCenter(center);

    const size = new THREE.Vector3();
    fullBox.getSize(size);

    const dist = Math.max(size.x, size.y, size.z) * 1.8 + 50;

    camera.position.set(center.x + dist, center.y + dist, center.z + dist * 0.55);
    camera.lookAt(center);
    controls.target.copy(center);

    let progress = 0;
    const total = tubeGeom.attributes.position.count;

    if ({str(animazione).lower()}) {{
      tubeGeom.setDrawRange(0, 0);
    }} else {{
      tubeGeom.setDrawRange(0, total);
    }}

    function animate() {{
      requestAnimationFrame(animate);

      if ({str(animazione).lower()}) {{
        progress += {velocita} * 0.002;
        if (progress > 1) progress = 1;
        const visible = Math.floor(progress * total);
        tubeGeom.setDrawRange(0, visible);
      }}

      controls.update();
      renderer.render(scene, camera);
    }}

    animate();

    window.addEventListener("resize", () => {{
      const w = Math.max(container.clientWidth, 300);
      const h = Math.max(container.clientHeight, 300);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
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
    ritardo_min = st.number_input(t["rit_min"], min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    ritardo_max = st.number_input(t["rit_max"], min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    attach_deg = st.number_input(t["attach_deg"], min_value=0.0, max_value=720.0, value=180.0, step=5.0)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# WARNINGS
# =========================

d_tubo_check = d_rame + 2.0 * spessore_guaina
if spalla <= d_tubo_check:
    st.warning(t["warning_spalla"])

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
    attach_deg,
)

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    diametro_aspo,
    meta["SpallaEffettiva"],
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

m5, m6, m7 = st.columns(3)
m5.metric(t["metric5"], f"{meta['LunghezzaRichiesta']:.3f} m")
m6.metric(t["metric6"], f"{meta['LunghezzaReale']:.3f} m")
m7.metric(t["metric7"], f"{meta['VolteTotali']:.2f}")

if meta["DiametroEsterno"] > 750:
    st.warning(t["warning_diam"])
