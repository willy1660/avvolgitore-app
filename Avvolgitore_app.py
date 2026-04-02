import json
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1,4])

with col_logo:
    st.image("New Logo PDM - rame.png", use_container_width=True)

with col_title:
    st.title("Avvolgimento")

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

def compute_total_turns(points):
    if len(points) < 2:
        return 0.0
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    dtheta = np.diff(theta)
    return float(np.sum(np.abs(dtheta)) / (2.0 * np.pi))

def points_to_sldcrv(points):
    lines = [f"{p[0]} {p[1]} {p[2]}" for p in points]
    return "\n".join(lines).encode()

def hermite_scalar(y0, y1, m0, m1, u):
    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2
    return h00*y0 + h10*m0 + h01*y1 + h11*m1

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
    lunghezza_mm = lunghezza_m * 1000.0

    d_tubo = d_rame_mm + 2.0 * spessore_guaina_mm

    passo_radiale = max(passo_radiale, EPS)
    passo_assiale = max(passo_assiale, EPS)

    r0 = d_aspo_mm / 2.0 + d_tubo / 2.0
    r = r0

    z_start = 0.0
    z_end = spalla_mm
    theta = 0.0

    points = []

    base_transition_turn = 0.18

    while True:
        dz_layer = z_end - z_start
        giri_layer = max(abs(dz_layer) / passo_assiale, 0.1)
        dtheta_layer = 2.0 * np.pi * giri_layer
        dz_dtheta_in = dz_layer / dtheta_layer

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

        rit_lo = min(ritardo_min_deg, ritardo_max_deg)
        rit_hi = max(ritardo_min_deg, ritardo_max_deg)

        ritardo_deg = 0.0
        if rit_hi > 0:
            ritardo_deg = np.random.uniform(rit_lo, rit_hi)

        extra_turn = ritardo_deg / 360.0
        total_transition_turn = base_transition_turn + extra_turn
        dtheta_transition = 2.0 * np.pi * total_transition_turn

        r_next = r + passo_radiale

        dz_next_total = z_start - z_end
        giri_next_nom = max(abs(dz_next_total) / passo_assiale, 0.1)
        dtheta_next_nom = 2.0 * np.pi * giri_next_nom
        dz_dtheta_out = dz_next_total / dtheta_next_nom

        n_trans = max(48, int(total_transition_turn * 180))
        t_trans = np.linspace(0.0, dtheta_transition, n_trans)
        u = t_trans / max(dtheta_transition, EPS)

        theta_trans = theta + dtheta_layer + t_trans

        s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_trans))
        r_trans = r + (r_next - r) * s

        z_trans = hermite_scalar(
            y0=z_end,
            y1=z_end,
            m0=dz_dtheta_in * dtheta_transition,
            m1=dz_dtheta_out * dtheta_transition,
            u=u
        )

        x = r_trans * np.cos(theta_trans)
        y = r_trans * np.sin(theta_trans)

        trans_pts = np.column_stack([x, y, z_trans])[1:]
        points.extend(trans_pts.tolist())

        theta = theta + dtheta_layer + dtheta_transition
        r = r_next
        z_start, z_end = z_end, z_start

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
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext,
        "Capes": capes,
        "VolteTotali": total_turns,
        "VoltePerCapa": voltes_per_capa,
    }

    return path, meta

# =========================
# VIEWER (igual que tenies)
# =========================
# 👇 NO TOCAT (mantingut igual per no trencar res)

def build_viewer_html(points, d_tubo, altezza, animazione, velocita):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = max(300, len(pts))

    html = f"""
    <div id="viewer" style="width:100%;height:{altezza}px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 100000);
    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

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

    const tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, 40, false);
    const tubeMat = new THREE.MeshStandardMaterial({{ color:0xe6e6e6 }});
    const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
    scene.add(tubeMesh);

    function createCap(pos, dir, color) {{
      const g = new THREE.CircleGeometry({r_tubo}, 32);
      const m = new THREE.MeshBasicMaterial({{color:color, side:THREE.DoubleSide}});
      const cap = new THREE.Mesh(g, m);
      const up = new THREE.Vector3(0,0,1);
      const quat = new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize());
      cap.quaternion.copy(quat);
      cap.position.copy(pos);
      scene.add(cap);
    }}

    if(vectors.length>=2){{
      createCap(vectors[0], vectors[1].clone().sub(vectors[0]).multiplyScalar(-1), 0x00ff00);
      createCap(vectors[vectors.length-1], vectors[vectors.length-1].clone().sub(vectors[vectors.length-2]), 0xff0000);
    }}

    const box = new THREE.Box3().setFromPoints(vectors);
    const center = new THREE.Vector3();
    box.getCenter(center);

    camera.position.set(center.x+500, center.y+500, center.z+300);
    camera.lookAt(center);

    function animate(){{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene,camera);
    }}

    animate();
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

d_rame = COPPER_SIZES_MM[rame_label]
d_tubo = d_rame + 2 * spessore_guaina

c6, c7 = st.columns(2)

with c6:
    passo_assiale = st.number_input("Passo assiale (mm)", value=float(d_tubo))

with c7:
    incremento_strato = st.number_input("Incremento strato (mm)", value=float(d_tubo))

c8, c9 = st.columns(2)

with c8:
    ritardo_min = st.number_input("Ritardo min (°)", value=360.0)

with c9:
    ritardo_max = st.number_input("Ritardo max (°)", value=360.0)

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

html = build_viewer_html(path, meta["DiametroTubo"], 700, False, 1.0)
components.html(html, height=700)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Diametro tubo", f"{meta['DiametroTubo']:.2f} mm")
m2.metric("Passo assiale", f"{meta['PassoAssiale']:.2f} mm")
m3.metric("Incremento strato", f"{meta['IncrementoStrato']:.2f} mm")
m4.metric("Diametro esterno", f"{meta['DiametroEsterno']:.1f} mm")
