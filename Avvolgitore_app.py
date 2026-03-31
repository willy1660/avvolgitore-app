import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os
import sys

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================================================
# HEADER
# =========================================================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    logo_path = resource_path("New Logo PDM - rame.png")
    st.image(logo_path, width=120)

with col_title:
    st.title("Avvolgimento")

# =========================================================
# DATI
# =========================================================
COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

# =========================================================
# UTILITÀ
# =========================================================
def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline_to_length(points, target):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target:
        return points

    idx = np.searchsorted(cum, target, side="right") - 1
    idx = max(0, min(idx, len(points) - 2))

    p0 = points[idx]
    p1 = points[idx + 1]
    seg_len = np.linalg.norm(p1 - p0)

    if seg_len <= 1e-12:
        return points[:idx + 1]

    alpha = (target - cum[idx]) / seg_len
    p_cut = p0 + alpha * (p1 - p0)

    return np.vstack([points[:idx + 1], p_cut])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return np.sum(np.abs(np.diff(theta))) / (2 * np.pi)

def points_to_sldcrv(points):
    return "\n".join(f"{p[0]} {p[1]} {p[2]}" for p in points).encode()

# =========================================================
# GEOMETRIA
# =========================================================
def build_coil_centerline(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    compressione_pct,
    gap_axiale_mm,
    incremento_strato,
    ritardo_min_deg,
    ritardo_max_deg,
):
    lunghezza_mm = lunghezza_m * 1000.0
    d_tubo = d_rame_mm + 2.0 * spessore_guaina_mm

    if incremento_strato > 0:
        passo_radiale = incremento_strato
    else:
        passo_radiale = d_tubo * (1.0 - compressione_pct / 100.0)

    passo_assiale = d_tubo + gap_axiale_mm

    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0
    r = r0

    z0 = 0.0
    z1 = spalla_mm
    theta = 0.0

    points = []

    while True:
        dz = z1 - z0
        giri = max(abs(dz) / max(passo_assiale, 1e-9), 0.1)

        dtheta = 2.0 * math.pi * giri
        n_main = max(300, int(giri * 300))
        t = np.linspace(0.0, dtheta, n_main)

        # crescita radiale continua per evitare esglaons
        r_start = r
        r_end = r + passo_radiale
        r_interp = r_start + (r_end - r_start) * (t / dtheta)

        theta_vals = theta + t
        z_vals = z0 + dz * (t / dtheta)

        x = r_interp * np.cos(theta_vals)
        y = r_interp * np.sin(theta_vals)

        layer = np.column_stack([x, y, z_vals])

        if len(points) > 0:
            layer = layer[1:]

        points.extend(layer.tolist())

        theta += dtheta
        r = r_end

        # ritardo in inversione a r costante
        if ritardo_max_deg > 0:
            rit_min = min(ritardo_min_deg, ritardo_max_deg)
            rit_max = max(ritardo_min_deg, ritardo_max_deg)
            rit_deg = np.random.uniform(rit_min, rit_max)
            rit_rad = math.radians(rit_deg)

            n_delay = max(30, int(rit_deg / 2))
            t_delay = np.linspace(0.0, rit_rad, n_delay)

            theta_delay = theta + t_delay
            z_delay = np.full_like(theta_delay, z1)

            x_delay = r * np.cos(theta_delay)
            y_delay = r * np.sin(theta_delay)

            delay_pts = np.column_stack([x_delay, y_delay, z_delay])[1:]
            points.extend(delay_pts.tolist())

            theta += rit_rad

        pts_np = np.array(points)
        if polyline_length(pts_np) >= lunghezza_mm:
            break

        z0, z1 = z1, z0

    path = trim_polyline_to_length(np.array(points), lunghezza_mm)

    total_turns = compute_total_turns(path)
    r_max = np.max(np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2))
    diam_ext = 2.0 * (r_max + d_tubo / 2.0)

    capes = max(1, int(round((r_max - r0) / max(passo_radiale, 1e-9))) + 1)
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

# =========================================================
# VIEWER AMB ANIMACIÓ
# =========================================================
def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = json.dumps(points.tolist())
    r_tubo = d_tubo / 2.0

    return f"""
<div id="viewer-wrap" style="position:relative;width:100%;height:{altezza}px;background:black;overflow:hidden;">
  <div id="viewer" style="width:100%;height:100%;"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>
(function() {{
    const container = document.getElementById("viewer");
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    function resize() {{
        const w = Math.max(container.clientWidth, 100);
        const h = Math.max(container.clientHeight, 100);
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }}
    resize();

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.0));

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(400, 400, 500);
    scene.add(dirLight);

    const raw = {pts};
    const vectors = raw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

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

    const tubularSegments = Math.max(4000, vectors.length);
    const radialSegments = 32;

    const geom = new THREE.TubeGeometry(curve, tubularSegments, {r_tubo}, radialSegments, false);
    const mat = new THREE.MeshStandardMaterial({{
        color: 0xe6e6e6,
        roughness: 0.9,
        metalness: 0.0
    }});
    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    // taps d'inici i final
    function createCap(position, dir, color) {{
        const geometry = new THREE.CircleGeometry({r_tubo}, 40);
        const material = new THREE.MeshBasicMaterial({{
            color: color,
            side: THREE.DoubleSide
        }});
        const cap = new THREE.Mesh(geometry, material);

        const up = new THREE.Vector3(0, 0, 1);
        const direction = dir.clone().normalize();
        if (direction.length() > 0) {{
            const quat = new THREE.Quaternion().setFromUnitVectors(up, direction);
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
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const dist = maxDim * 1.6;

    camera.position.set(center.x + dist, center.y + dist, center.z + dist * 0.5);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();

    let progress = 0.0;

    if ({str(animazione).lower()}) {{
        if (geom.index) {{
            geom.setDrawRange(0, 0);
        }} else {{
            geom.setDrawRange(0, 0);
        }}
    }} else {{
        if (geom.index) {{
            geom.setDrawRange(0, geom.index.count);
        }} else {{
            geom.setDrawRange(0, geom.attributes.position.count);
        }}
        progress = 1.0;
    }}

    function render() {{
        controls.update();
        renderer.render(scene, camera);
    }}

    function animate() {{
        requestAnimationFrame(animate);

        if ({str(animazione).lower()}) {{
            progress += {velocita} * 0.002;
            if (progress > 1.0) progress = 1.0;

            if (geom.index) {{
                geom.setDrawRange(0, Math.floor(progress * geom.index.count));
            }} else {{
                geom.setDrawRange(0, Math.floor(progress * geom.attributes.position.count));
            }}
        }}

        render();
    }}

    const ro = new ResizeObserver(() => {{
        resize();
        render();
    }});
    ro.observe(container);

    window.addEventListener("resize", () => {{
        resize();
        render();
    }});

    animate();
}})();
</script>
"""

# =========================================================
# UI
# =========================================================
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
    incremento_strato = st.number_input("Incremento strato (mm)", value=0.0)

with c9:
    altezza = st.slider("Altezza viewer", 400, 900, 700)

c10, c11 = st.columns(2)

with c10:
    animazione = st.checkbox("Animazione", True)

with c11:
    velocita = st.slider("Velocità", 0.1, 5.0, 1.0)

c12, c13 = st.columns(2)

with c12:
    ritardo_min = st.number_input("Ritardo MIN (deg)", value=0.0)

with c13:
    ritardo_max = st.number_input("Ritardo MAX (deg)", value=0.0)

# =========================================================
# RUN
# =========================================================
d_rame = COPPER_SIZES_MM[rame_label]

path, meta = build_coil_centerline(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    compressione,
    gap,
    incremento_strato,
    ritardo_min,
    ritardo_max
)

components.html(
    build_viewer_html(path, meta["DiametroTubo"], altezza, animazione, velocita),
    height=altezza
)

# =========================================================
# METRICHE
# =========================================================
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
    st.metric("Strati", f"{meta['Capes']}")

with m6:
    st.metric("Spire per strato", f"{meta['VoltePerCapa']:.2f}")

with m7:
    st.metric("Giri totali", f"{meta['VolteTotali']:.2f}")

with m8:
    st.metric("Lunghezza sviluppata", f"{meta['LunghezzaM']:.2f} m")

if meta["DiametroEsterno"] > 750:
    st.warning("Diametro esterno superiore a 750 mm. La bobina potrebbe uscire dal pallet.")

st.download_button(
    "Scarica centerline SLDCRV",
    data=points_to_sldcrv(path),
    file_name="coil_centerline.sldcrv",
    mime="text/plain"
)
