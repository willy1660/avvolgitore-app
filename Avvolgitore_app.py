import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

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


def trim_polyline(points: np.ndarray, target_length: float) -> np.ndarray:
    if len(points) < 2:
        return points

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length, side="right") - 1
    idx = max(0, min(idx, len(points) - 2))

    p0 = points[idx]
    p1 = points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len < EPS:
        return points[:idx + 1]

    alpha = (target_length - cum[idx]) / seg_len
    p_cut = p0 + alpha * (p1 - p0)

    return np.vstack([points[:idx + 1], p_cut])


def compute_total_turns(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    dtheta = np.diff(theta)
    return float(np.sum(np.abs(dtheta)) / (2.0 * np.pi))


def points_to_sldcrv(points: np.ndarray) -> bytes:
    lines = [f"{p[0]} {p[1]} {p[2]}" for p in points]
    return "\n".join(lines).encode()


def hermite_scalar(y0: float, y1: float, m0: float, m1: float, u: np.ndarray) -> np.ndarray:
    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2
    return h00*y0 + h10*m0 + h01*y1 + h11*m1


# =========================
# GEOMETRY
# =========================

def build_coil(
    d_aspo_mm: float,
    spalla_mm: float,
    lunghezza_m: float,
    d_rame_mm: float,
    spessore_guaina_mm: float,
    compressione_pct: float,
    gap_axiale_mm: float,
    ritardo_min_deg: float,
    ritardo_max_deg: float,
):
    lunghezza_mm = lunghezza_m * 1000.0

    d_tubo = d_rame_mm + 2.0 * spessore_guaina_mm
    passo_radiale = d_tubo * (1.0 - compressione_pct / 100.0)
    passo_assiale = d_tubo + gap_axiale_mm

    passo_radiale = max(passo_radiale, EPS)
    passo_assiale = max(passo_assiale, EPS)

    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0
    r = r0

    z_start = 0.0
    z_end = spalla_mm
    theta = 0.0

    points = []

    # Aquesta part és la “sortida” cap a la nova capa després del dwell.
    # Ha de ser petita i estable. No depèn del ritardo real.
    blend_turn = 0.12
    dtheta_blend = 2.0 * np.pi * blend_turn

    while True:
        # -------------------------------------------------
        # 1) CAPA HELICOIDAL
        # -------------------------------------------------
        dz_layer = z_end - z_start
        giri_layer = max(abs(dz_layer) / passo_assiale, 0.1)
        dtheta_layer = 2.0 * np.pi * giri_layer

        n_layer = max(120, int(giri_layer * 150))
        t = np.linspace(0.0, dtheta_layer, n_layer)

        theta_vals = theta + t
        z_vals = z_start + dz_layer * (t / dtheta_layer)

        x = r * np.cos(theta_vals)
        y = r * np.sin(theta_vals)

        layer = np.column_stack([x, y, z_vals])

        if len(points) > 0:
            layer = layer[1:]

        points.extend(layer.tolist())

        pts_np = np.array(points)
        if polyline_length(pts_np) >= lunghezza_mm:
            break

        # -------------------------------------------------
        # 2) DWELL REAL DE MÀQUINA (RITARDO)
        # -------------------------------------------------
        rit_lo = min(ritardo_min_deg, ritardo_max_deg)
        rit_hi = max(ritardo_min_deg, ritardo_max_deg)

        ritardo_deg = 0.0
        if rit_hi > 0:
            ritardo_deg = np.random.uniform(rit_lo, rit_hi)

        dtheta_delay = math.radians(ritardo_deg)
        r_next = r + passo_radiale

        if dtheta_delay > EPS:
            n_delay = max(48, int(ritardo_deg * 0.35))
            t_delay = np.linspace(0.0, dtheta_delay, n_delay)

            # easing radial suau
            s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_delay))

            theta_delay = theta + dtheta_layer + t_delay
            r_delay = r + (r_next - r) * s

            # z pràcticament constant durant el dwell
            z_delay = np.full_like(theta_delay, z_end)

            x = r_delay * np.cos(theta_delay)
            y = r_delay * np.sin(theta_delay)

            delay_pts = np.column_stack([x, y, z_delay])[1:]
            points.extend(delay_pts.tolist())

        # -------------------------------------------------
        # 3) BLEND D'ENTRADA A LA NOVA CAPA
        # -------------------------------------------------
        # Després del dwell, la capa nova ha de començar amb una entrada
        # suau. Aquí fem una petita transició axial a radi constant r_next.
        theta_after_delay = theta + dtheta_layer + dtheta_delay

        dz_next_total = z_start - z_end
        giri_next_nom = max(abs(dz_next_total) / passo_assiale, 0.1)
        dtheta_next_nom = 2.0 * np.pi * giri_next_nom
        dz_dtheta_next = dz_next_total / dtheta_next_nom

        # El blend acaba amb la pendent de la nova hèlix
        z_blend_start = z_end
        z_blend_end = z_blend_start + dz_dtheta_next * dtheta_blend

        n_blend = 36
        t_blend = np.linspace(0.0, dtheta_blend, n_blend)
        u = t_blend / max(dtheta_blend, EPS)

        theta_blend = theta_after_delay + t_blend
        r_blend = np.full_like(theta_blend, r_next)

        # Hermite en Z:
        # inici amb pendent 0 (dwell)
        # final amb pendent dz_dtheta_next * dtheta_blend
        z_blend = hermite_scalar(
            y0=z_blend_start,
            y1=z_blend_end,
            m0=0.0,
            m1=dz_dtheta_next * dtheta_blend,
            u=u
        )

        x = r_blend * np.cos(theta_blend)
        y = r_blend * np.sin(theta_blend)

        blend_pts = np.column_stack([x, y, z_blend])[1:]
        points.extend(blend_pts.tolist())

        # Estat per a la següent capa
        theta = theta_after_delay + dtheta_blend
        r = r_next
        z_start = z_blend_end
        z_end = 0.0 if z_end > z_start else spalla_mm

        pts_np = np.array(points)
        if polyline_length(pts_np) >= lunghezza_mm:
            break

    path = np.array(points)
    path = trim_polyline(path, lunghezza_mm)

    total_turns = compute_total_turns(path)
    r_max = np.max(np.sqrt(path[:, 0]**2 + path[:, 1]**2))
    diam_ext = 2.0 * (r_max + d_tubo / 2.0)

    capes = int(round((r_max - r0) / passo_radiale)) + 1
    capes = max(1, capes)

    voltes_per_capa = total_turns / capes

    meta = {
        "DiametroTubo": d_tubo,
        "PassoRadiale": passo_radiale,
        "PassoAssiale": passo_assiale,
        "DiametroEsterno": diam_ext,
        "LunghezzaM": polyline_length(path) / 1000.0,
        "Capes": capes,
        "VolteTotali": total_turns,
        "VoltePerCapa": voltes_per_capa,
    }

    return path, meta


# =========================
# VIEWER
# =========================

def build_viewer_html(points: np.ndarray, d_tubo: float, altezza: int, animazione: bool, velocita: float) -> str:
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = max(300, len(pts))

    html = f"""
<div id="viewer-wrap" style="position:relative;width:100%;height:{altezza}px;">
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
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 0.60));

const light = new THREE.DirectionalLight(0xffffff, 0.40);
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

const tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 40, false);
const tubeMat = new THREE.MeshStandardMaterial({{
  color: 0xe6e6e6,
  roughness: 0.92
}});
const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
scene.add(tubeMesh);

// ==============================
// CAPS REALS
// ==============================

function createCap(position, direction, color) {{
  const geometry = new THREE.CircleGeometry({r_tubo}, 32);
  const material = new THREE.MeshBasicMaterial({{
    color: color,
    side: THREE.DoubleSide
  }});

  const cap = new THREE.Mesh(geometry, material);

  const up = new THREE.Vector3(0,0,1);
  const dir = direction.clone().normalize();

  if (dir.length() > 0) {{
    const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
    cap.quaternion.copy(quat);
  }}

  cap.position.copy(position);
  scene.add(cap);
}}

if (vectors.length >= 2) {{
  const start = vectors[0];
  const dirStart = vectors[1].clone().sub(vectors[0]).multiplyScalar(-1);
  createCap(start, dirStart, 0x00ff00);

  const end = vectors[vectors.length - 1];
  const dirEnd = vectors[vectors.length - 1].clone().sub(vectors[vectors.length - 2]);
  createCap(end, dirEnd, 0xff0000);
}}

const box = new THREE.Box3().setFromPoints(vectors);
const center = new THREE.Vector3();
box.getCenter(center);

const size = new THREE.Vector3();
box.getSize(size);
const maxDim = Math.max(size.x, size.y, size.z);

camera.position.set(center.x + maxDim, center.y + maxDim, center.z + maxDim * 0.6);
camera.lookAt(center);
controls.target.copy(center);

let progress = 0;

if ({str(animazione).lower()}) {{
  tubeMesh.geometry.setDrawRange(0, 0);
}}

function animate() {{
  requestAnimationFrame(animate);

  if ({str(animazione).lower()}) {{
    progress += {velocita} * 0.002;
    if (progress > 1) progress = 1;

    if (tubeGeom.index) {{
      tubeMesh.geometry.setDrawRange(0, Math.floor(progress * tubeGeom.index.count));
    }}
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

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    diametro_aspo = st.number_input("Diametro aspo (mm)", value=450.0)

with c2:
    spalla = st.number_input("Spalla (mm)", value=95.0)

with c3:
    lunghezza = st.number_input("Lunghezza (m)", value=50.0)

with c4:
    rame_label = st.selectbox("Diametro rame", list(COPPER_SIZES_MM.keys()))

with c5:
    spessore_guaina = st.number_input("Spessore guaina (mm)", value=7.0)

c6, c7, c8, c9 = st.columns(4)

with c6:
    compressione = st.slider("Compressione %", 0.0, 20.0, 0.0)

with c7:
    gap = st.number_input("Gap axiale (mm)", value=0.0)

with c8:
    ritardo_min = st.number_input("Ritardo inversione min (°)", value=360.0)

with c9:
    ritardo_max = st.number_input("Ritardo inversione max (°)", value=360.0)

c10, c11, c12 = st.columns(3)

with c10:
    animazione = st.checkbox("Animazione avvolgimento", True)

with c11:
    velocita = st.slider("Velocità animazione", 0.1, 5.0, 1.0)

with c12:
    altezza = st.slider("Altezza viewer", 400, 900, 700)

d_rame = COPPER_SIZES_MM[rame_label]

path, meta = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    compressione,
    gap,
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

with m1:
    st.metric("Diametro tubo", f"{meta['DiametroTubo']:.2f} mm")

with m2:
    st.metric("Passo radiale", f"{meta['PassoRadiale']:.2f} mm")

with m3:
    st.metric("Passo assiale", f"{meta['PassoAssiale']:.2f} mm")

with m4:
    st.metric("Diametro esterno", f"{meta['DiametroEsterno']:.1f} mm")

m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric("Strati", meta["Capes"])

with m6:
    st.metric("Spire", f"{meta['VoltePerCapa']:.2f}")

with m7:
    st.metric("Giri totali", f"{meta['VolteTotali']:.2f}")

with m8:
    st.download_button(
        "Scarica centerline SLDCRV",
        data=points_to_sldcrv(path),
        file_name="coil_centerline.sldcrv",
        mime="text/plain"
    )

if meta["DiametroEsterno"] > 750:
    st.warning("Diametro esterno superiore a 750 mm. La bobina potrebbe uscire dal pallet.")
