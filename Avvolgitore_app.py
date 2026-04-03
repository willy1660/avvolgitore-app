import math
import time
import json
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
    d = np.diff(points, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())

def trim_polyline(points: np.ndarray, target_length: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()

    if target_length <= 0:
        return points[:1].copy()

    out = [points[0]]
    acc = 0.0

    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]
        seg = float(np.linalg.norm(p1 - p0))

        if acc + seg <= target_length + EPS:
            out.append(p1)
            acc += seg
        else:
            rem = target_length - acc
            if seg > EPS and rem > 0:
                t = rem / seg
                pm = p0 + t * (p1 - p0)
                out.append(pm)
            break

    return np.array(out, dtype=float)

def sample_segment(p0, p1, n=20):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    t = np.linspace(0, 1, max(2, n))
    pts = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]
    return pts

def tangent_point_from_left_guide(guide_x: float, radius: float, z: float, side: int = 1):
    """
    Guide outlet at (guide_x, 0, z), with guide_x < -radius.
    Tangent point on cylinder x^2 + y^2 = radius^2 at same z.
    side = +1 or -1 chooses upper/lower tangent in top view.
    """
    L = abs(guide_x)
    R = radius

    if L <= R + 1e-9:
        L = R + 1.0

    xt = -(R * R) / L
    yt = side * (R * math.sqrt(max(L * L - R * R, 0.0)) / L)
    return np.array([xt, yt, z], dtype=float)

def helical_segment(radius, z0, z1, turns, theta0, n_per_turn=80):
    turns = max(turns, 1e-6)
    n = max(10, int(abs(turns) * n_per_turn) + 2)
    t = np.linspace(0, 1, n)
    theta = theta0 + 2 * np.pi * turns * t
    z = z0 + (z1 - z0) * t
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.column_stack([x, y, z])
    return pts, float(theta[-1])

def radial_transition_segment(radius0, radius1, z_const, theta0, transition_turns=0.35, n_per_turn=120):
    turns = max(transition_turns, 1e-4)
    n = max(12, int(turns * n_per_turn) + 2)
    t = np.linspace(0, 1, n)
    theta = theta0 + 2 * np.pi * turns * t
    r = radius0 + (radius1 - radius0) * t
    z = np.full_like(t, z_const)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack([x, y, z])
    return pts, float(theta[-1])

def make_box_edges(center, size_xyz):
    cx, cy, cz = center
    sx, sy, sz = size_xyz
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2

    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=float)

    edges_idx = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    segments = []
    for a, b in edges_idx:
        segments.append(corners[a].tolist())
        segments.append(corners[b].tolist())
    return np.array(segments, dtype=float)

def estimate_masses(length_m, copper_outer_mm, foam_thickness_mm, compression_pct):
    d_cu = copper_outer_mm / 1000.0
    foam_eff_th = max(foam_thickness_mm * (1 - compression_pct / 100.0), 0.0) / 1000.0
    d_outer = d_cu + 2 * foam_eff_th

    rho_cu = 8960.0
    rho_foam = 35.0

    area_cu = math.pi * (d_cu ** 2) / 4.0
    area_outer = math.pi * (d_outer ** 2) / 4.0
    area_foam = max(area_outer - area_cu, 0.0)

    m_cu = area_cu * length_m * rho_cu
    m_foam = area_foam * length_m * rho_foam
    return m_cu, m_foam, d_outer * 1000.0

# =========================
# MODEL
# =========================

def build_winding_model(
    d_aspo_mm,
    spalla_mm,
    length_m,
    copper_mm,
    foam_thickness_mm,
    compression_pct,
    gap_radiale_mm,
    quality,
    progress_pct,
):
    d_eff = copper_mm + 2.0 * foam_thickness_mm * (1.0 - compression_pct / 100.0)
    d_eff = max(d_eff, copper_mm)

    pitch_axial = d_eff
    radial_step = d_eff + gap_radiale_mm

    R_core_geom = d_aspo_mm / 2.0
    R_contact0 = R_core_geom + d_eff / 2.0

    zmin = -spalla_mm / 2.0 + d_eff / 2.0
    zmax =  spalla_mm / 2.0 - d_eff / 2.0
    z_span = max(zmax - zmin, d_eff)

    turns_per_pass = max(z_span / max(pitch_axial, 1e-6), 0.05)

    qmap = {
        "Bassa": 36,
        "Media": 72,
        "Alta": 120,
    }
    n_per_turn = qmap.get(quality, 72)

    guide_clearance = max(60.0, 2.2 * R_contact0)
    guide_x0 = -(R_contact0 + guide_clearance)

    points = []

    total_target_mm = length_m * 1000.0
    accumulated_mm = 0.0

    current_radius = R_contact0
    current_theta = np.pi
    direction = +1
    pass_idx = 0

    while accumulated_mm < total_target_mm - 1e-6:
        z_start = zmin if direction > 0 else zmax
        z_end = zmax if direction > 0 else zmin

        if pass_idx == 0:
            guide_x = guide_x0 - (current_radius - R_contact0)
            guide_pt = np.array([guide_x, 0.0, z_start], dtype=float)
            tan_pt = tangent_point_from_left_guide(guide_x, current_radius, z_start, side=1)

            feed_pts = sample_segment(guide_pt, tan_pt, n=24)
            helix_pts, current_theta = helical_segment(
                current_radius, z_start, z_end, turns_per_pass, current_theta, n_per_turn=n_per_turn
            )
            pts = np.vstack([feed_pts, helix_pts])
            accumulated_mm += polyline_length(pts)
            points.append(pts)
        else:
            helix_pts, current_theta = helical_segment(
                current_radius, z_start, z_end, turns_per_pass, current_theta, n_per_turn=n_per_turn
            )
            accumulated_mm += polyline_length(helix_pts)
            points.append(helix_pts)

        if accumulated_mm >= total_target_mm - 1e-6:
            break

        next_radius = current_radius + radial_step
        trans_pts, current_theta = radial_transition_segment(
            current_radius, next_radius, z_end, current_theta, transition_turns=0.33, n_per_turn=n_per_turn
        )
        accumulated_mm += polyline_length(trans_pts)
        points.append(trans_pts)

        current_radius = next_radius
        direction *= -1
        pass_idx += 1

    if len(points) == 0:
        centerline = np.zeros((0, 3), dtype=float)
    else:
        centerline = np.vstack(points)

    centerline = trim_polyline(centerline, total_target_mm)

    shown_length_mm = (progress_pct / 100.0) * polyline_length(centerline)
    shown_line = trim_polyline(centerline, shown_length_mm)

    if len(shown_line) >= 2:
        last = shown_line[-1]
        r_last = float(np.hypot(last[0], last[1]))
        z_last = float(last[2])
        guide_x_last = guide_x0 - max(r_last - R_contact0, 0.0)
        guide_pt_display = np.array([guide_x_last, 0.0, z_last], dtype=float)
        tangent_display = tangent_point_from_left_guide(guide_x_last, max(r_last, R_contact0), z_last, side=1)
    else:
        guide_pt_display = np.array([guide_x0, 0.0, zmin], dtype=float)
        tangent_display = tangent_point_from_left_guide(guide_x0, R_contact0, zmin, side=1)

    outer_radius_est = R_contact0
    if len(shown_line) > 0:
        outer_radius_est = max(float(np.max(np.hypot(shown_line[:, 0], shown_line[:, 1]))), R_contact0)

    total_length_m_actual = polyline_length(centerline) / 1000.0
    layers_est = max(int(round((outer_radius_est - R_contact0) / max(radial_step, 1e-6))) + 1, 1)
    ext_diam_est = 2.0 * outer_radius_est

    return {
        "centerline_full": centerline,
        "centerline_shown": shown_line,
        "guide_pt": guide_pt_display,
        "tangent_pt": tangent_display,
        "R_core_geom": R_core_geom,
        "R_contact0": R_contact0,
        "outer_radius_est": outer_radius_est,
        "zmin": zmin,
        "zmax": zmax,
        "d_eff": d_eff,
        "pitch_axial": pitch_axial,
        "radial_step": radial_step,
        "layers_est": layers_est,
        "ext_diam_est": ext_diam_est,
        "total_length_m_actual": total_length_m_actual,
    }

# =========================
# THREE.JS VIEWER
# =========================

def make_threejs_viewer_html(
    model,
    viewer_height=760,
    show_grid=True,
    show_axes=False,
    show_trajectory=True,
    taglio_z=None,
):
    line = model["centerline_shown"].copy()
    full_line = model["centerline_full"].copy()

    if taglio_z is not None and len(line) > 0:
        line = line[line[:, 2] <= taglio_z]
    if taglio_z is not None and len(full_line) > 0:
        full_line = full_line[full_line[:, 2] <= taglio_z]

    guide_pt = model["guide_pt"]
    tangent_pt = model["tangent_pt"]
    feed_pts = sample_segment(guide_pt, tangent_pt, n=20)

    guide_box_size = np.array([
        max(28.0, model["d_eff"] * 2.0),
        max(18.0, model["d_eff"] * 1.3),
        max(18.0, model["d_eff"] * 1.3),
    ], dtype=float)

    guide_box_edges = make_box_edges(guide_pt, guide_box_size)
    arm_start = guide_pt + np.array([guide_box_size[0] / 2.0, 0.0, 0.0], dtype=float)
    arm_end = np.array([-(model["R_core_geom"] + 12.0), 0.0, guide_pt[2]], dtype=float)

    outer_r = model["outer_radius_est"]
    R_core = model["R_core_geom"]
    d_eff = model["d_eff"]
    zmin = model["zmin"]
    zmax = model["zmax"]

    flange_outer = max(outer_r + d_eff * 1.2, R_core * 1.6)
    flange_th = max(d_eff * 0.9, 6.0)
    core_zmin = zmin - d_eff / 2.0
    core_zmax = zmax + d_eff / 2.0

    max_r = max(flange_outer, abs(float(guide_pt[0])) + 30.0)
    zmid = 0.5 * (core_zmin + core_zmax)
    zhalf = max(abs(core_zmax - core_zmin) / 2.0 + flange_th + 20.0, 40.0)

    scene_payload = {
        "line": line.tolist(),
        "full_line": full_line.tolist(),
        "feed_pts": feed_pts.tolist(),
        "guide_box_edges": guide_box_edges.tolist(),
        "guide_pt": guide_pt.tolist(),
        "tangent_pt": tangent_pt.tolist(),
        "arm_start": arm_start.tolist(),
        "arm_end": arm_end.tolist(),
        "R_core": float(R_core),
        "flange_outer": float(flange_outer),
        "flange_th": float(flange_th),
        "core_zmin": float(core_zmin),
        "core_zmax": float(core_zmax),
        "zmid": float(zmid),
        "zhalf": float(zhalf),
        "max_r": float(max_r),
        "show_grid": bool(show_grid),
        "show_axes": bool(show_axes),
        "show_trajectory": bool(show_trajectory),
        "height": int(viewer_height),
    }

    scene_json = json.dumps(scene_payload)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          overflow: hidden;
          background: #ffffff;
        }}
        #viewer {{
          width: 100%;
          height: {int(viewer_height)}px;
          display: block;
        }}
      </style>
    </head>
    <body>
      <div id="viewer"></div>

      <script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
      <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

      <script>
        const DATA = {scene_json};

        const container = document.getElementById("viewer");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        const width = container.clientWidth || 1200;
        const height = DATA.height || 760;

        const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 10000);
        camera.position.set(-2.25 * DATA.max_r, 1.45 * DATA.max_r, DATA.zmid + 1.15 * DATA.max_r);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0, DATA.zmid);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.screenSpacePanning = true;
        controls.update();

        scene.add(new THREE.AmbientLight(0xffffff, 0.90));

        const d1 = new THREE.DirectionalLight(0xffffff, 0.65);
        d1.position.set(1, 1, 2);
        scene.add(d1);

        const d2 = new THREE.DirectionalLight(0xffffff, 0.45);
        d2.position.set(-2, -1, 1);
        scene.add(d2);

        if (DATA.show_grid) {{
          const grid = new THREE.GridHelper(2.4 * DATA.max_r, 20, 0xd0d0d0, 0xe8e8e8);
          grid.rotation.x = Math.PI / 2;
          grid.position.z = DATA.core_zmin - DATA.flange_th - 10;
          scene.add(grid);
        }}

        if (DATA.show_axes) {{
          const axes = new THREE.AxesHelper(0.8 * DATA.max_r);
          scene.add(axes);
        }}

        function lineFromPoints(arr, color=0x111111, linewidth=2) {{
          if (!arr || arr.length < 2) return null;
          const pts = arr.map(p => new THREE.Vector3(p[0], p[1], p[2]));
          const geom = new THREE.BufferGeometry().setFromPoints(pts);
          const mat = new THREE.LineBasicMaterial({{ color }});
          return new THREE.Line(geom, mat);
        }}

        function lineSegmentsFromPoints(arr, color=0x333333) {{
          if (!arr || arr.length < 2) return null;
          const pts = arr.map(p => new THREE.Vector3(p[0], p[1], p[2]));
          const geom = new THREE.BufferGeometry().setFromPoints(pts);
          const mat = new THREE.LineBasicMaterial({{ color }});
          return new THREE.LineSegments(geom, mat);
        }}

        function makeTubePolyline(arr, radius, color=0x2f855a) {{
          if (!arr || arr.length < 2) return null;
          const pts = arr.map(p => new THREE.Vector3(p[0], p[1], p[2]));
          const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
          const tubularSegments = Math.max(64, Math.min(1200, pts.length * 2));
          const radialSegments = 10;
          const geom = new THREE.TubeGeometry(curve, tubularSegments, radius, radialSegments, false);
          const mat = new THREE.MeshStandardMaterial({{
            color,
            roughness: 0.55,
            metalness: 0.08
          }});
          return new THREE.Mesh(geom, mat);
        }}

        function makeCylinderZ(radius, height, color=0xbec5cc) {{
          const geom = new THREE.CylinderGeometry(radius, radius, height, 72, 1, false);
          const mat = new THREE.MeshStandardMaterial({{
            color,
            roughness: 0.70,
            metalness: 0.12
          }});
          const mesh = new THREE.Mesh(geom, mat);
          mesh.rotation.x = Math.PI / 2;
          return mesh;
        }}

        function makeSphere(pos, radius, color=0xcc4444) {{
          const geom = new THREE.SphereGeometry(radius, 16, 16);
          const mat = new THREE.MeshStandardMaterial({{
            color,
            roughness: 0.45,
            metalness: 0.10
          }});
          const mesh = new THREE.Mesh(geom, mat);
          mesh.position.set(pos[0], pos[1], pos[2]);
          return mesh;
        }}

        // Aspo core
        const coreHeight = DATA.core_zmax - DATA.core_zmin;
        const core = makeCylinderZ(DATA.R_core, coreHeight, 0xc8d0d8);
        core.position.set(0, 0, (DATA.core_zmin + DATA.core_zmax) / 2);
        scene.add(core);

        // Spalle
        const flangeColor = 0xaeb7c1;

        const flangeBottom = makeCylinderZ(DATA.flange_outer, DATA.flange_th, flangeColor);
        flangeBottom.position.set(0, 0, DATA.core_zmin - DATA.flange_th / 2);
        scene.add(flangeBottom);

        const flangeTop = makeCylinderZ(DATA.flange_outer, DATA.flange_th, flangeColor);
        flangeTop.position.set(0, 0, DATA.core_zmax + DATA.flange_th / 2);
        scene.add(flangeTop);

        // Full trajectory
        if (DATA.show_trajectory && DATA.full_line && DATA.full_line.length >= 2) {{
          const traj = lineFromPoints(DATA.full_line, 0x9aa5b1, 1);
          if (traj) scene.add(traj);
        }}

        // Main tube
        const tubeRadius = Math.max(1.0, 0.22 * (DATA.line.length ? 1 : 1));
        if (DATA.line && DATA.line.length >= 2) {{
          const mainTube = makeTubePolyline(DATA.line, Math.max(1.8, {max(1.8, float(model["d_eff"]) * 0.22):.4f}), 0x2e8b57);
          if (mainTube) scene.add(mainTube);
        }}

        // Straight feed segment
        if (DATA.feed_pts && DATA.feed_pts.length >= 2) {{
          const feedTube = makeTubePolyline(DATA.feed_pts, Math.max(1.6, {max(1.6, float(model["d_eff"]) * 0.18):.4f}), 0x3aa86b);
          if (feedTube) scene.add(feedTube);
        }}

        // Guide block
        const guideEdges = lineSegmentsFromPoints(DATA.guide_box_edges, 0x1f2937);
        if (guideEdges) scene.add(guideEdges);

        // Arm
        const armGeom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(DATA.arm_start[0], DATA.arm_start[1], DATA.arm_start[2]),
          new THREE.Vector3(DATA.arm_end[0], DATA.arm_end[1], DATA.arm_end[2])
        ]);
        const arm = new THREE.Line(
          armGeom,
          new THREE.LineBasicMaterial({{ color: 0x374151 }})
        );
        scene.add(arm);

        // Outlet & tangent markers
        scene.add(makeSphere(DATA.guide_pt, Math.max(1.8, {max(1.8, float(model["d_eff"]) * 0.14):.4f}), 0x111111));
        scene.add(makeSphere(DATA.tangent_pt, Math.max(1.5, {max(1.5, float(model["d_eff"]) * 0.11):.4f}), 0xcc3333));

        function fitFarPlane() {{
          const span = 6 * DATA.max_r + 4 * DATA.zhalf;
          camera.far = Math.max(5000, span);
          camera.updateProjectionMatrix();
        }}
        fitFarPlane();

        function onResize() {{
          const w = container.clientWidth || width;
          const h = DATA.height || height;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
          renderer.render(scene, camera);
        }}

        window.addEventListener("resize", onResize);

        function animate() {{
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        }}
        animate();
      </script>
    </body>
    </html>
    """
    return html

# =========================
# STATE FOR ANIMATION
# =========================

if "progress_anim" not in st.session_state:
    st.session_state.progress_anim = 100

# =========================
# UI
# =========================

st.title("Avvolgimento")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

with c1:
    d_aspo_mm = st.number_input("Ø Aspo (mm)", min_value=20.0, value=250.0, step=1.0)
    copper_label = st.selectbox("Tubo rame", list(COPPER_SIZES_MM.keys()), index=1)
    copper_mm = COPPER_SIZES_MM[copper_label]

with c2:
    spalla_mm = st.number_input("Spalla / Altezza utile (mm)", min_value=20.0, value=180.0, step=1.0)
    foam_thickness_mm = st.number_input("Spessore guaina (mm)", min_value=0.0, value=9.0, step=0.5)

with c3:
    length_m = st.number_input("Lunghezza target (m)", min_value=1.0, value=25.0, step=1.0)
    compression_pct = st.slider("Compressione (%)", min_value=0, max_value=60, value=15)

with c4:
    gap_radiale_mm = st.number_input("Gap radiale (mm)", min_value=0.0, value=0.5, step=0.1)
    quality = st.selectbox("Qualità viewer", ["Bassa", "Media", "Alta"], index=1)

st.markdown("---")

v1, v2, v3, v4, v5, v6 = st.columns([1, 1, 1, 1, 1, 1])

with v1:
    viewer_height = st.slider("Altezza viewer", 450, 1100, 760, 10)
with v2:
    animate = st.checkbox("Animazione", value=False)
with v3:
    anim_speed = st.slider("Velocità", 1, 20, 8, 1)
with v4:
    show_grid = st.checkbox("Grid", value=True)
with v5:
    show_axes = st.checkbox("Axes", value=False)
with v6:
    show_trajectory = st.checkbox("Traiettoria", value=True)

progress_manual = st.slider("Progress", 0, 100, int(st.session_state.progress_anim if animate else 100), 1)

clip_enabled = st.checkbox("Taglio Z", value=False)
taglio_z = None
if clip_enabled:
    taglio_z = st.slider("Quota taglio Z (mm)", -500.0, 500.0, 0.0, 1.0)

if animate:
    progress_pct = int(st.session_state.progress_anim)
else:
    progress_pct = int(progress_manual)

model = build_winding_model(
    d_aspo_mm=d_aspo_mm,
    spalla_mm=spalla_mm,
    length_m=length_m,
    copper_mm=copper_mm,
    foam_thickness_mm=foam_thickness_mm,
    compression_pct=compression_pct,
    gap_radiale_mm=gap_radiale_mm,
    quality=quality,
    progress_pct=progress_pct,
)

viewer_html = make_threejs_viewer_html(
    model=model,
    viewer_height=viewer_height,
    show_grid=show_grid,
    show_axes=show_axes,
    show_trajectory=show_trajectory,
    taglio_z=taglio_z,
)

components.html(viewer_html, height=viewer_height + 8, scrolling=False)

# =========================
# METRICS
# =========================

m_cu, m_foam, d_outer_eff_mm = estimate_masses(
    length_m=length_m,
    copper_outer_mm=copper_mm,
    foam_thickness_mm=foam_thickness_mm,
    compression_pct=compression_pct,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Diametro tubo eff. (mm)", f"{model['d_eff']:.2f}")
m2.metric("Passo radiale (mm)", f"{model['radial_step']:.2f}")
m3.metric("Raggio centerline iniziale (mm)", f"{model['R_contact0']:.2f}")
m4.metric("Strati radiali stimati", f"{model['layers_est']}")
m5.metric("Diametro esterno stimato (mm)", f"{model['ext_diam_est']:.2f}")
m6.metric("Lunghezza reale modello (m)", f"{model['total_length_m_actual']:.2f}")

m7, m8, m9, m10 = st.columns(4)
m7.metric("Massa rame stimata (kg)", f"{m_cu:.2f}")
m8.metric("Massa guaina stimata (kg)", f"{m_foam:.2f}")
m9.metric("Guida Z attuale (mm)", f"{model['guide_pt'][2]:.2f}")
m10.metric("Guida X attuale (mm)", f"{model['guide_pt'][0]:.2f}")

st.caption(
    "Modello físic simplificat: il tubo esce dal guidatubo con un tratto rettilineo e raggiunge "
    "la tangente reale del raggio esterno corrente. La prima posizione è impostata per essere tangente "
    "alla base del primo strato sull’aspo."
)

# =========================
# AUTO-ANIMATION RERUN
# =========================

if animate:
    step = max(1, anim_speed // 2)
    next_val = st.session_state.progress_anim + step
    if next_val > 100:
        next_val = 0
    st.session_state.progress_anim = next_val
    time.sleep(0.06)
    st.rerun()
else:
    st.session_state.progress_anim = progress_manual
