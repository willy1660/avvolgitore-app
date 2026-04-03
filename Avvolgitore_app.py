import math
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

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

def make_cylinder_surface(radius, zmin, zmax, n_theta=80, n_z=2):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(zmin, zmax, n_z)
    th, zz = np.meshgrid(theta, z)
    x = radius * np.cos(th)
    y = radius * np.sin(th)
    return x, y, zz

def make_disc_surface(r_inner, r_outer, z, n_theta=80, n_r=12):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    rr = np.linspace(r_inner, r_outer, n_r)
    th, r = np.meshgrid(theta, rr)
    x = r * np.cos(th)
    y = r * np.sin(th)
    z = np.full_like(x, z)
    return x, y, z

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
    ])

    edges_idx = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    xs, ys, zs = [], [], []
    for a, b in edges_idx:
        xs += [corners[a, 0], corners[b, 0], None]
        ys += [corners[a, 1], corners[b, 1], None]
        zs += [corners[a, 2], corners[b, 2], None]
    return xs, ys, zs

def estimate_masses(length_m, copper_outer_mm, foam_thickness_mm, compression_pct):
    d_cu = copper_outer_mm / 1000.0
    foam_eff_th = max(foam_thickness_mm * (1 - compression_pct / 100.0), 0.0) / 1000.0
    d_outer = d_cu + 2 * foam_eff_th

    rho_cu = 8960.0
    rho_foam = 35.0

    # Approx simple tube/cable style estimation
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
    # Effective outer diameter of insulated tube after compression
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

    # Guide geometry:
    # initial condition = line from guide to tangent point on first contact radius at base zmin
    guide_clearance = max(60.0, 2.2 * R_contact0)
    guide_x0 = -(R_contact0 + guide_clearance)

    points = []
    guide_positions = []

    total_target_mm = length_m * 1000.0
    accumulated_mm = 0.0

    current_radius = R_contact0
    current_theta = np.pi  # start visually on left side
    direction = +1
    current_z = zmin
    pass_idx = 0

    while accumulated_mm < total_target_mm - 1e-6:
        z_start = zmin if direction > 0 else zmax
        z_end = zmax if direction > 0 else zmin

        if pass_idx == 0:
            current_z = z_start

            # guide follows axial level and radial growth
            guide_x = guide_x0 - (current_radius - R_contact0)
            guide_pt = np.array([guide_x, 0.0, current_z], dtype=float)
            tan_pt = tangent_point_from_left_guide(guide_x, current_radius, current_z, side=1)

            feed_pts = sample_segment(guide_pt, tan_pt, n=24)
            helix_pts, current_theta = helical_segment(
                current_radius, z_start, z_end, turns_per_pass, current_theta, n_per_turn=n_per_turn
            )

            pts = np.vstack([feed_pts, helix_pts])

            accumulated_mm += polyline_length(pts)
            points.append(pts)
            guide_positions.append(guide_pt)

        else:
            helix_pts, current_theta = helical_segment(
                current_radius, z_start, z_end, turns_per_pass, current_theta, n_per_turn=n_per_turn
            )
            accumulated_mm += polyline_length(helix_pts)
            points.append(helix_pts)

        if accumulated_mm >= total_target_mm - 1e-6:
            break

        # Radial transition at end of pass
        next_radius = current_radius + radial_step

        trans_pts, current_theta = radial_transition_segment(
            current_radius, next_radius, z_end, current_theta, transition_turns=0.33, n_per_turn=n_per_turn
        )
        accumulated_mm += polyline_length(trans_pts)
        points.append(trans_pts)

        # guide position at end of pass / start next pass
        next_guide_x = guide_x0 - (next_radius - R_contact0)
        guide_positions.append(np.array([next_guide_x, 0.0, z_end], dtype=float))

        current_radius = next_radius
        current_z = z_end
        direction *= -1
        pass_idx += 1

    if len(points) == 0:
        centerline = np.zeros((0, 3), dtype=float)
    else:
        centerline = np.vstack(points)

    # Trim to requested target length
    centerline = trim_polyline(centerline, total_target_mm)

    # Progress
    shown_length_mm = (progress_pct / 100.0) * polyline_length(centerline)
    shown_line = trim_polyline(centerline, shown_length_mm)

    # Current displayed guide point:
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
# VIEWER
# =========================

def make_figure(model, viewer_height=760, show_grid=True, show_axes=False, show_trajectory=True, show_mesh=True, taglio_z=None):
    fig = go.Figure()

    R_core = model["R_core_geom"]
    zmin = model["zmin"]
    zmax = model["zmax"]
    outer_r = model["outer_radius_est"]
    guide_pt = model["guide_pt"]
    tangent_pt = model["tangent_pt"]
    line = model["centerline_shown"]

    flange_outer = max(outer_r + model["d_eff"] * 1.2, R_core * 1.6)
    flange_th = max(model["d_eff"] * 0.9, 6.0)
    core_zmin = zmin - model["d_eff"] / 2.0
    core_zmax = zmax + model["d_eff"] / 2.0

    # Core cylinder
    x, y, z = make_cylinder_surface(R_core, core_zmin, core_zmax, n_theta=90, n_z=2)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        showscale=False,
        opacity=0.85,
        hoverinfo="skip",
        name="Aspo"
    ))

    # Bottom flange
    xb, yb, zb = make_disc_surface(R_core, flange_outer, core_zmin - flange_th, n_theta=90, n_r=20)
    fig.add_trace(go.Surface(
        x=xb, y=yb, z=zb,
        showscale=False,
        opacity=0.85,
        hoverinfo="skip",
        name="Spalla inferiore"
    ))

    # Top flange
    xt, yt, zt = make_disc_surface(R_core, flange_outer, core_zmax + flange_th, n_theta=90, n_r=20)
    fig.add_trace(go.Surface(
        x=xt, y=yt, z=zt,
        showscale=False,
        opacity=0.85,
        hoverinfo="skip",
        name="Spalla superiore"
    ))

    # Tube centerline
    if len(line) >= 2:
        z_line = line[:, 2].copy()
        if taglio_z is not None:
            mask = z_line <= taglio_z
            xs = np.where(mask, line[:, 0], np.nan)
            ys = np.where(mask, line[:, 1], np.nan)
            zs = np.where(mask, line[:, 2], np.nan)
        else:
            xs, ys, zs = line[:, 0], line[:, 1], line[:, 2]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=8),
            name="Tubo"
        ))

    # Straight guide-to-tangent segment
    feed_pts = sample_segment(guide_pt, tangent_pt, n=20)
    fig.add_trace(go.Scatter3d(
        x=feed_pts[:, 0], y=feed_pts[:, 1], z=feed_pts[:, 2],
        mode="lines",
        line=dict(width=7, dash="solid"),
        name="Tratto rettilineo"
    ))

    # Guide block
    guide_box_size = (
        max(28.0, model["d_eff"] * 2.0),
        max(18.0, model["d_eff"] * 1.3),
        max(18.0, model["d_eff"] * 1.3),
    )
    bx, by, bz = make_box_edges(guide_pt, guide_box_size)
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(width=4),
        name="Guidatubo"
    ))

    # Arm from guide towards spool side
    arm_start = guide_pt + np.array([guide_box_size[0] / 2.0, 0.0, 0.0])
    arm_end = np.array([-(R_core + 12.0), 0.0, guide_pt[2]])
    fig.add_trace(go.Scatter3d(
        x=[arm_start[0], arm_end[0]],
        y=[arm_start[1], arm_end[1]],
        z=[arm_start[2], arm_end[2]],
        mode="lines",
        line=dict(width=5),
        name="Braccio guidatubo"
    ))

    # Outlet point + tangent point
    fig.add_trace(go.Scatter3d(
        x=[guide_pt[0]], y=[guide_pt[1]], z=[guide_pt[2]],
        mode="markers",
        marker=dict(size=5),
        name="Uscita guidatubo"
    ))
    fig.add_trace(go.Scatter3d(
        x=[tangent_pt[0]], y=[tangent_pt[1]], z=[tangent_pt[2]],
        mode="markers",
        marker=dict(size=4),
        name="Tangente"
    ))

    # Optional trajectory
    if show_trajectory and len(model["centerline_full"]) >= 2:
        tr = model["centerline_full"]
        fig.add_trace(go.Scatter3d(
            x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
            mode="lines",
            line=dict(width=2, dash="dot"),
            name="Traiettoria completa",
            opacity=0.35
        ))

    # Camera / aspect
    max_r = max(flange_outer, abs(guide_pt[0]) + 30.0)
    zmid = 0.5 * (core_zmin + core_zmax)
    zhalf = max(abs(core_zmax - core_zmin) / 2.0 + flange_th + 20.0, 40.0)

    fig.update_layout(
        height=viewer_height,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(
                visible=show_axes,
                showgrid=show_grid,
                range=[-max_r - 40, max_r + 40]
            ),
            yaxis=dict(
                visible=show_axes,
                showgrid=show_grid,
                range=[-max_r, max_r]
            ),
            zaxis=dict(
                visible=show_axes,
                showgrid=show_grid,
                range=[zmid - zhalf, zmid + zhalf]
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.1, z=1.3),
            camera=dict(
                eye=dict(x=-2.25, y=1.45, z=1.15)
            ),
        )
    )

    return fig

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

# Animation logic
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

fig = make_figure(
    model,
    viewer_height=viewer_height,
    show_grid=show_grid,
    show_axes=show_axes,
    show_trajectory=show_trajectory,
    show_mesh=True,
    taglio_z=taglio_z,
)

st.plotly_chart(fig, use_container_width=True)

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
