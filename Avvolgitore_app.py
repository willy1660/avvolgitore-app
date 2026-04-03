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
    ritardo_min_deg,   # base
    ritardo_max_deg,   # spalla
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

    # =========================
    # MODEL FÍSIC
    # =========================
    # El tub toca base i spalla amb la seva superfície exterior
    # -> el centreline va de d_tubo/2 a spalla - d_tubo/2
    z_min = d_tubo / 2.0
    z_max = spalla_mm - d_tubo / 2.0

    # radi inicial: tub tocant el mandrí
    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0

    r = r0
    z = z_min
    theta = 0.0
    direction = 1  # +1 puja, -1 baixa

    # discretització
    theta_step_run = np.deg2rad(4.0)
    dz_dtheta = passo_assiale / (2.0 * np.pi)  # mm per radià

    # durant el retard, la rampa radial només passa al tram final
    radial_ramp_fraction = 0.20  # últim 20% del retard

    points = []

    def add_point(theta_val, r_val, z_val):
        x = r_val * np.cos(theta_val)
        y = r_val * np.sin(theta_val)
        points.append([x, y, z_val])

    add_point(theta, r, z)

    # =========================
    # START DWELL
    # gir inicial, z constant, r constant
    # =========================
    if gradi_start_deg > EPS and lunghezza_visibile_mm > EPS:
        start_steps = max(4, int(np.ceil(gradi_start_deg / 4.0)))
        theta_step_start = np.deg2rad(gradi_start_deg) / start_steps

        for _ in range(start_steps):
            theta += theta_step_start
            add_point(theta, r, z)

            if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
                break

    # =========================
    # MAIN LOOP
    # =========================
    while True:
        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
            break

        # =========================
        # RUN HELICOIDAL
        # Guidatubo imposa z i r; el mandrí imposa theta
        # =========================
        while True:
            theta += theta_step_run
            z += direction * dz_dtheta * theta_step_run

            if direction == 1 and z >= z_max:
                z = z_max
                add_point(theta, r, z)
                break

            if direction == -1 and z <= z_min:
                z = z_min
                add_point(theta, r, z)
                break

            add_point(theta, r, z)

            if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
                break

        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
            break

        # =========================
        # DWELL / RITARDO
        # guidatubo quiet axialment, mandrí gira
        # el canvi radial es concentra al tram final del retard
        # =========================
        at_top = direction == 1
        ritardo_deg = ritardo_top_deg if at_top else ritardo_bottom_deg
        theta_dwell = np.deg2rad(ritardo_deg)
        z_const = z_max if at_top else z_min

        if theta_dwell > EPS:
            dwell_steps = max(8, int(np.ceil(ritardo_deg / 4.0)))
            theta_step_dwell = theta_dwell / dwell_steps

            r_start = r
            r_end = r + passo_radiale

            ramp_steps = max(1, int(np.ceil(dwell_steps * radial_ramp_fraction)))
            flat_steps = max(0, dwell_steps - ramp_steps)

            for i in range(1, dwell_steps + 1):
                theta += theta_step_dwell

                if i <= flat_steps:
                    r_curr = r_start
                else:
                    j = i - flat_steps
                    u = j / ramp_steps
                    u = max(0.0, min(1.0, u))
                    r_curr = r_start + passo_radiale * u

                add_point(theta, r_curr, z_const)

                if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
                    break

            r = r_end
        else:
            # canvi radial immediat si no hi ha retard
            r += passo_radiale
            add_point(theta, r, z_const)

        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
            break

        # canvi de direcció
        direction *= -1

    path = np.array(points, dtype=float)

    if lunghezza_visibile_mm > EPS:
        path = trim_polyline(path, lunghezza_visibile_mm)

    r_path = np.sqrt(path[:, 0]**2 + path[:, 1]**2)
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
        "Zmin": z_min,
        "Zmax": z_max,
        "LunghezzaVisibile": lunghezza_visibile_mm / 1000.0,
        "LunghezzaPinza": lunghezza_pinza_mm / 1000.0,
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita, d_aspo_mm, spalla_mm):

    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    r_mandrel = d_aspo_mm / 2.0

    tubular_segments = min(4000, max(800, int(len(pts) * 0.5)))

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

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 0.7));

    const light = new THREE.DirectionalLight(0xffffff, 0.5);
    light.position.set(5,5,5);
    scene.add(light);

    const rawPoints = {points_json};
    const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    class CurvePath extends THREE.Curve {{
      constructor(points) {{
        super();
        this.points = points;
      }}
      getPoint(t) {{
        const n = this.points.length;
        const f = t*(n-1);
        const i = Math.floor(f);
        const i0 = Math.max(0, Math.min(i, n-2));
        const i1 = i0+1;
        const tt = f-i0;
        return new THREE.Vector3().lerpVectors(this.points[i0], this.points[i1], tt);
      }}
    }}

    const curve = new CurvePath(vectors);

    let tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 48, false);
    tubeGeom = tubeGeom.toNonIndexed();

    const tubeMesh = new THREE.Mesh(
      tubeGeom,
      new THREE.MeshStandardMaterial({{
        color:0xe6e6e6,
        roughness:0.85,
        metalness:0.1
      }})
    );

    scene.add(tubeMesh);

    // =========================
    // MANDRÍ
    // =========================
    const mandrelHeight = {spalla_mm};
    const mandrelGeom = new THREE.CylinderGeometry({r_mandrel}, {r_mandrel}, mandrelHeight, 64, 1, false);
    const mandrelMat = new THREE.MeshStandardMaterial({{
      color: 0x444444,
      roughness: 0.8,
      metalness: 0.4,
      transparent: true,
      opacity: 0.45
    }});
    const mandrelMesh = new THREE.Mesh(mandrelGeom, mandrelMat);
    mandrelMesh.position.set(0, mandrelHeight / 2.0, 0);
    scene.add(mandrelMesh);

    // =========================
    // BASE
    // =========================
    const baseRadius = Math.max({r_mandrel} + 80, 450);
    const baseThickness = 6;
    const baseGeom = new THREE.CylinderGeometry(baseRadius, baseRadius, baseThickness, 64);
    const baseMat = new THREE.MeshStandardMaterial({{
      color: 0x1f5aa6,
      roughness: 0.85,
      metalness: 0.15
    }});
    const baseMesh = new THREE.Mesh(baseGeom, baseMat);
    baseMesh.position.set(0, -baseThickness / 2.0, 0);
    scene.add(baseMesh);

    // =========================
    // SPALLA
    // =========================
    const topGeom = new THREE.CylinderGeometry(baseRadius, baseRadius, baseThickness, 64);
    const topMat = new THREE.MeshStandardMaterial({{
      color: 0x1f5aa6,
      roughness: 0.85,
      metalness: 0.15
    }});
    const topMesh = new THREE.Mesh(topGeom, topMat);
    topMesh.position.set(0, {spalla_mm} + baseThickness / 2.0, 0);
    scene.add(topMesh);

    function createCap(position, direction, color) {{
      const geometry = new THREE.CircleGeometry({r_tubo}, 32);
      const material = new THREE.MeshBasicMaterial({{color:color, side:THREE.DoubleSide}});
      const cap = new THREE.Mesh(geometry, material);

      const up = new THREE.Vector3(0,0,1);
      const dir = direction.clone().normalize();

      if (dir.length() > 1e-9) {{
        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
        cap.quaternion.copy(quat);
      }}

      cap.position.copy(position);
      scene.add(cap);
    }}

    if (vectors.length >= 2) {{
      createCap(vectors[0], vectors[1].clone().sub(vectors[0]).multiplyScalar(-1), 0x00ff00);
      createCap(vectors[vectors.length-1], vectors[vectors.length-1].clone().sub(vectors[vectors.length-2]), 0xff0000);
    }}

    // Convertim de Python (x,y,z) a Three.js (x,z,y) només visualment?
    // No. Mantenim y com alçada perquè el model actual ja està així.

    const box = new THREE.Box3().setFromObject(tubeMesh);
    box.expandByObject(mandrelMesh);
    box.expandByObject(baseMesh);
    box.expandByObject(topMesh);

    const center = new THREE.Vector3();
    box.getCenter(center);

    const size = new THREE.Vector3();
    box.getSize(size);

    const dist = Math.max(size.x, size.y, size.z) * 1.8 + 1.0;

    camera.position.set(center.x + dist, center.y + dist, center.z + dist * 0.6);
    camera.lookAt(center);
    controls.target.copy(center);

    let progress = 0;
    const total = tubeGeom.attributes.position.count;

    if ({str(animazione).lower()}) {{
      tubeGeom.setDrawRange(0, 0);
    }} else {{
      tubeGeom.setDrawRange(0, total);
    }}

    function animate(){{
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

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita,
    diametro_aspo,
    spalla
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
