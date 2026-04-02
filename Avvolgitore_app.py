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
        "ritardo": "Ritardo (°)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
        "metric5": "Volte totali",
        "metric6": "Strati",
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
        "ritardo": "Delay (°)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Outer diameter",
        "metric5": "Total turns",
        "metric6": "Layers",
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

def hermite_scalar(y0, y1, m0, m1, u):
    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2
    return h00*y0 + h10*m0 + h01*y1 + h11*m1

def smoothstep01(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)

def append_segment(points_list, segment):
    if segment is None or len(segment) == 0:
        return
    if len(points_list) == 0:
        points_list.extend(segment.tolist())
    else:
        points_list.extend(segment[1:].tolist())

def make_segment(theta0, duration, npts, r_vals, z_vals):
    tt = np.linspace(0.0, duration, npts)
    theta = theta0 + tt
    x = r_vals * np.cos(theta)
    y = r_vals * np.sin(theta)
    return np.column_stack([x, y, z_vals]), theta[-1]

def npts_from_turns(duration_rad, density_per_turn=240, min_pts=30):
    turns = max(duration_rad / (2 * np.pi), 0.01)
    return max(min_pts, int(turns * density_per_turn))

# =========================
# CONTINUOUS COIL MODEL
# =========================

def build_coil_continuous(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
    ritardo_deg,
):
    lunghezza_mm = lunghezza_m * 1000.0
    d_tubo = d_rame_mm + 2.0 * spessore_guaina_mm

    passo_assiale = max(float(passo_assiale), EPS)
    passo_radiale = max(float(passo_radiale), EPS)
    spalla_mm = max(float(spalla_mm), EPS)

    # ritardo real de màquina: 0..360°
    ritardo_deg = max(0.0, min(360.0, float(ritardo_deg)))
    theta_rev = np.deg2rad(ritardo_deg)

    # petit mínim numèric per evitar degeneració total si ritardo = 0
    theta_rev_eff = max(theta_rev, np.deg2rad(0.5))

    # pitch axial per radiant de rotació
    dz_dtheta = passo_assiale / (2.0 * np.pi)

    # radi inicial sobre línia central del tub
    r = d_aspo_mm / 2.0 + d_tubo / 2.0
    r0 = r

    # quant "arrodonim" els extrems axialment durant el gir
    # prou gran per suavitzar, però sense menjar-se tota la spalla
    blend = dz_dtheta * theta_rev_eff * 0.5
    blend = min(blend, spalla_mm * 0.45)
    blend = max(blend, 0.25)

    run_height = max(spalla_mm - 2.0 * blend, 0.0)
    theta_run = run_height / dz_dtheta if run_height > EPS else 0.0

    points = []
    theta = 0.0

    # Entrada suau inicial: de z=0 a z=blend, pendent 0 -> +dz_dtheta
    theta_entry = theta_rev_eff * 0.5
    n_entry = npts_from_turns(theta_entry, min_pts=24)
    u = np.linspace(0.0, 1.0, n_entry)
    z_entry = hermite_scalar(0.0, blend, 0.0, dz_dtheta * theta_entry, u)
    r_entry = np.full_like(z_entry, r)
    seg, theta = make_segment(theta, theta_entry, n_entry, r_entry, z_entry)
    append_segment(points, seg)

    target_reached = False
    turnarounds = 0

    while True:
        if polyline_length(np.array(points)) >= lunghezza_mm:
            target_reached = True
            break

        # ---------------------------------
        # RUN UP: blend -> spalla-blend
        # ---------------------------------
        if theta_run > EPS:
            n_run = npts_from_turns(theta_run, min_pts=40)
            tt = np.linspace(0.0, theta_run, n_run)
            z_run = blend + dz_dtheta * tt
            r_run = np.full_like(z_run, r)
            seg, theta = make_segment(theta, theta_run, n_run, r_run, z_run)
            append_segment(points, seg)

            if polyline_length(np.array(points)) >= lunghezza_mm:
                target_reached = True
                break

        # ---------------------------------
        # TOP TURNAROUND (continu)
        # z: (spalla-blend) -> spalla -> (spalla-blend)
        # r: r -> r + passo_radiale
        # ---------------------------------
        n_top = npts_from_turns(theta_rev_eff, min_pts=50)
        tt = np.linspace(0.0, theta_rev_eff, n_top)
        u = tt / theta_rev_eff
        u1 = np.clip(2.0 * u, 0.0, 1.0)
        u2 = np.clip(2.0 * u - 1.0, 0.0, 1.0)

        z_top = np.empty_like(u)

        mask1 = u <= 0.5
        z_top[mask1] = hermite_scalar(
            spalla_mm - blend,
            spalla_mm,
            dz_dtheta * (theta_rev_eff / 2.0),
            0.0,
            u1[mask1]
        )

        mask2 = ~mask1
        z_top[mask2] = hermite_scalar(
            spalla_mm,
            spalla_mm - blend,
            0.0,
            -dz_dtheta * (theta_rev_eff / 2.0),
            u2[mask2]
        )

        r_top = r + passo_radiale * smoothstep01(u)
        seg, theta = make_segment(theta, theta_rev_eff, n_top, r_top, z_top)
        append_segment(points, seg)

        r = r + passo_radiale
        turnarounds += 1

        if polyline_length(np.array(points)) >= lunghezza_mm:
            target_reached = True
            break

        # ---------------------------------
        # RUN DOWN: spalla-blend -> blend
        # ---------------------------------
        if theta_run > EPS:
            n_run = npts_from_turns(theta_run, min_pts=40)
            tt = np.linspace(0.0, theta_run, n_run)
            z_run = (spalla_mm - blend) - dz_dtheta * tt
            r_run = np.full_like(z_run, r)
            seg, theta = make_segment(theta, theta_run, n_run, r_run, z_run)
            append_segment(points, seg)

            if polyline_length(np.array(points)) >= lunghezza_mm:
                target_reached = True
                break

        # ---------------------------------
        # BOTTOM TURNAROUND (continu)
        # z: blend -> 0 -> blend
        # r: r -> r + passo_radiale
        # ---------------------------------
        n_bot = npts_from_turns(theta_rev_eff, min_pts=50)
        tt = np.linspace(0.0, theta_rev_eff, n_bot)
        u = tt / theta_rev_eff
        u1 = np.clip(2.0 * u, 0.0, 1.0)
        u2 = np.clip(2.0 * u - 1.0, 0.0, 1.0)

        z_bot = np.empty_like(u)

        mask1 = u <= 0.5
        z_bot[mask1] = hermite_scalar(
            blend,
            0.0,
            -dz_dtheta * (theta_rev_eff / 2.0),
            0.0,
            u1[mask1]
        )

        mask2 = ~mask1
        z_bot[mask2] = hermite_scalar(
            0.0,
            blend,
            0.0,
            dz_dtheta * (theta_rev_eff / 2.0),
            u2[mask2]
        )

        r_bot = r + passo_radiale * smoothstep01(u)
        seg, theta = make_segment(theta, theta_rev_eff, n_bot, r_bot, z_bot)
        append_segment(points, seg)

        r = r + passo_radiale
        turnarounds += 1

        if polyline_length(np.array(points)) >= lunghezza_mm:
            target_reached = True
            break

        if turnarounds > 10000:
            break

    path = np.array(points, dtype=float)
    path = trim_polyline(path, lunghezza_mm)

    r_path = np.sqrt(path[:, 0]**2 + path[:, 1]**2)
    r_max = float(np.max(r_path))
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
    }

    return path, meta

# =========================
# VIEWER
# =========================

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = min(5000, max(1200, int(len(pts) * 0.55)))

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
  45,
  container.clientWidth / container.clientHeight,
  0.1,
  100000
);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 0.75));

const light = new THREE.DirectionalLight(0xffffff, 0.55);
light.position.set(5, 5, 5);
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

const tubeMesh = new THREE.Mesh(
  tubeGeom,
  new THREE.MeshStandardMaterial({{
    color: 0xe6e6e6,
    roughness: 0.85,
    metalness: 0.1
  }})
);

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

const box = new THREE.Box3().setFromPoints(vectors);
const center = new THREE.Vector3();
box.getCenter(center);

const size = new THREE.Vector3();
box.getSize(size);

const dist = Math.max(size.x, size.y, size.z) * 1.8;

camera.position.set(center.x + dist, center.y + dist, center.z + dist * 0.6);
camera.lookAt(center);
controls.target.copy(center);

// Animation
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
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
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
    d_tubo = d_rame + 2 * spessore_guaina

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo_assiale = st.number_input(t["passo_assiale"], value=float(d_tubo), step=0.1)
    incremento_strato = st.number_input(t["incremento"], value=float(d_tubo), step=0.1)
    ritardo = st.slider(t["ritardo"], min_value=0, max_value=360, value=180, step=1)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

path, meta = build_coil_continuous(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    passo_assiale,
    incremento_strato,
    ritardo,
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

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
m2.metric(t["metric2"], f"{meta['PassoAssiale']:.2f} mm")
m3.metric(t["metric3"], f"{meta['IncrementoStrato']:.2f} mm")
m4.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")
m5.metric(t["metric5"], f"{meta['VolteTotali']:.1f}")
m6.metric(t["metric6"], f"{meta['Capes']}")

if meta["DiametroEsterno"] > 750:
    st.warning(t["warning"])
