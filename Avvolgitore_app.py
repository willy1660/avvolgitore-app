import os
import glob
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

# =========================
# TEXTS
# =========================

TEXTS = {
    "IT": {
        "title": "Avvolgimento",
        "language": "🌍 Language",
        "bobina": "🟦 Bobina",
        "tubo": "🟩 Tubo",
        "avvolg": "🟧 Simulazione",
        "diam_aspo": "Ø Aspo (mm)",
        "spalla": "Spalla (mm)",
        "rame": "Ø Rame",
        "isolamento": "Spessore guaina (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm/rev)",
        "incremento": "Incremento strato (mm)",
        "rit_min": "Ritardo base (°)",
        "rit_max": "Ritardo spalla (°)",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro radiale max",
        "metric5": "Ingombro max XY",
        "metric6": "Lunghezza avvolta",
        "warning": "⚠️ Ingombro max XY superiore a 750 mm.",
        "play": "Play",
        "pause": "Pause",
        "fullscreen": "Fullscreen",
        "exit": "Exit",
        "progress": "Progresso",
        "speed": "Velocità",
        "spool": "Aspo",
        "visible": "Visibile",
        "transparent": "Trasparente",
        "hidden": "Nascosto",
        "tube_color": "Tubo",
        "gelwhite": "Gelwhite",
        "gelblack": "Gelblack",
        "grid": "Grid",
        "axes": "Assi",
        "section": "Sezione",
        "animation": "Animazione",
        "ghost": "Traiettoria futura",
        "studio": "Base render",
        "view": "Vista",
        "view_3d": "3D",
        "view_front": "Frontale",
        "view_side": "Laterale",
        "reset_view": "Reset vista",
        "hud_length": "Lunghezza",
        "hud_layer": "Strato",
        "hud_diameter": "Ø tubo",
    },
    "EN": {
        "title": "Coiling",
        "language": "🌍 Language",
        "bobina": "🟦 Coil",
        "tubo": "🟩 Tube",
        "avvolg": "🟧 Simulation",
        "diam_aspo": "Spool diameter (mm)",
        "spalla": "Width (mm)",
        "rame": "Copper size",
        "isolamento": "Foam thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm/rev)",
        "incremento": "Layer increment (mm)",
        "rit_min": "Bottom delay (°)",
        "rit_max": "Top delay (°)",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Max radial diameter",
        "metric5": "Max XY span",
        "metric6": "Wound length",
        "warning": "⚠️ Max XY span exceeds 750 mm.",
        "play": "Play",
        "pause": "Pause",
        "fullscreen": "Fullscreen",
        "exit": "Exit",
        "progress": "Progress",
        "speed": "Speed",
        "spool": "Spool",
        "visible": "Visible",
        "transparent": "Transparent",
        "hidden": "Hidden",
        "tube_color": "Tube",
        "gelwhite": "Gelwhite",
        "gelblack": "Gelblack",
        "grid": "Grid",
        "axes": "Axes",
        "section": "Section",
        "animation": "Animation",
        "ghost": "Future path",
        "studio": "Render base",
        "view": "View",
        "view_3d": "3D",
        "view_front": "Front",
        "view_side": "Side",
        "reset_view": "Reset view",
        "hud_length": "Length",
        "hud_layer": "Layer",
        "hud_diameter": "Tube Ø",
    },
}

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
gradi_start = 0.0
guide_offset_x = 555.0

# =========================
# PRESETS
# =========================

@st.cache_data
def load_presets(path="Presets.csv"):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    # Remove empty rows exported by Excel
    df = df.dropna(how="all")

    # Remove rows without product name
    df = df.dropna(subset=["Prodotto"])

    # Clean product names
    df["Prodotto"] = df["Prodotto"].astype(str).str.strip()

    return df


def format_preset_value(row, column):
    """Format CSV values for the preset cards."""
    if column not in row.index:
        return "-"

    value = row[column]

    if pd.isna(value):
        return "-"

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "-"

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)

        if abs(number - round(number)) < 1e-9:
            return f"{int(round(number))}"

        formatted = f"{number:.2f}".rstrip("0").rstrip(".")
        return formatted.replace(".", ",")

    return str(value).strip()


def split_label_and_unit(column_name):
    """Separate a CSV column label from the unit in parentheses, preserving CSV order."""
    label = str(column_name).strip()

    if "(" in label and ")" in label and label.rfind("(") < label.rfind(")"):
        start = label.rfind("(")
        end = label.rfind(")")
        unit = label[start + 1:end].strip()
        clean_label = (label[:start] + label[end + 1:]).strip()
        return clean_label, unit

    return label, ""


def preset_card_html(label, value, unit=""):
    unit_html = f'<span class="preset-unit">{unit}</span>' if unit else ""

    return f'''
    <div class="preset-card">
        <div class="preset-label">{label}</div>
        <div class="preset-value-row">
            <span class="preset-value">{value}</span>
            {unit_html}
        </div>
    </div>
    '''

# =========================
# LOGO
# =========================

def find_logo():
    candidates = [
        "New Logo PDM – rame.png",
        "New Logo PDM - rame.png",
        "new_logo_pdm_rame.png",
        "logo.png",
        "logo.svg",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
    ]

    for name in candidates:
        if os.path.exists(name):
            return name

    for pattern in ("*.png", "*.svg", "*.jpg", "*.jpeg", "*.webp"):
        files = glob.glob(pattern)
        if files:
            return files[0]

    return None


logo_path = find_logo()

# =========================
# HEADER
# =========================

top1, top2 = st.columns([1.0, 5.0])

with top1:
    if logo_path:
        st.image(logo_path, width=150)

with top2:
    st.markdown(f"## {TEXTS[st.session_state.lang]['title']}")
    lang_option = st.selectbox(
        TEXTS[st.session_state.lang]["language"],
        ["🇮🇹 Italiano", "🇺🇸 English (US)"],
        index=0 if st.session_state.lang == "IT" else 1,
        key="lang_selector_top",
    )

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
lang = st.session_state.lang
t = TEXTS[lang]

# =========================
# GEOMETRY HELPERS
# =========================

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def polyline_length(points: np.ndarray) -> float:
    if points is None or len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def deposit_point_world(radius: float, z: float) -> np.ndarray:
    return np.array([0.0, radius, z], dtype=float)


def world_to_spool_local(pt_world: np.ndarray, theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    x = pt_world[0] * c + pt_world[1] * s
    y = -pt_world[0] * s + pt_world[1] * c

    return np.array([x, y, pt_world[2]], dtype=float)

# =========================
# SIMULATION
# =========================

def simulate_winding_visual(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    lunghezza_m: float,
    gradi_start: float,
    deg_step: float = 2.0,
):
    max_len = lunghezza_m * 1000.0

    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla

    theta = np.deg2rad(gradi_start)
    z = Rt
    current_layer_radius = R + Rt

    first_contact_world = deposit_point_world(current_layer_radius, z)
    first_local = world_to_spool_local(first_contact_world, theta)

    contact_world = [first_contact_world]
    deposited_local = [first_local]
    theta_values = [theta]
    radius_values = [current_layer_radius]
    z_values = [z]
    mode_values = [0]
    layer_values = [0]
    length_values = [0.0]

    deposited_len = 0.0
    direction = 1
    mode = "axial"
    layer = 0

    transition_progress = 0.0
    transition_delay = 0.0
    transition_z = z
    transition_start_radius = current_layer_radius
    transition_end_radius = current_layer_radius

    for _ in range(1200000):
        next_theta = theta - np.deg2rad(deg_step)

        next_z = z
        next_direction = direction
        next_mode = mode
        next_radius = current_layer_radius
        next_layer = layer

        next_transition_progress = transition_progress
        next_transition_delay = transition_delay
        next_transition_z = transition_z
        next_transition_start_radius = transition_start_radius
        next_transition_end_radius = transition_end_radius

        if mode == "axial":
            next_z = z + direction * passo * (deg_step / 360.0)
            next_radius = current_layer_radius

            if next_z >= H - Rt:
                next_z = H - Rt

                next_transition_progress = 0.0
                next_transition_delay = max(rit_t, 0.0)
                next_transition_z = next_z
                next_transition_start_radius = current_layer_radius
                next_transition_end_radius = current_layer_radius + max(0.0, incremento)

                if next_transition_delay <= 0.0:
                    next_radius = next_transition_end_radius
                    current_layer_radius = next_transition_end_radius
                    next_mode = "axial"
                    next_direction = -direction
                    next_layer = layer + 1
                else:
                    next_mode = "transition"
                    next_radius = next_transition_start_radius

            elif next_z <= Rt:
                next_z = Rt

                next_transition_progress = 0.0
                next_transition_delay = max(rit_b, 0.0)
                next_transition_z = next_z
                next_transition_start_radius = current_layer_radius
                next_transition_end_radius = current_layer_radius + max(0.0, incremento)

                if next_transition_delay <= 0.0:
                    next_radius = next_transition_end_radius
                    current_layer_radius = next_transition_end_radius
                    next_mode = "axial"
                    next_direction = -direction
                    next_layer = layer + 1
                else:
                    next_mode = "transition"
                    next_radius = next_transition_start_radius

        else:
            next_z = transition_z
            next_transition_progress = transition_progress + deg_step

            if transition_delay <= 0.0:
                s = 1.0
            else:
                s = smoothstep(next_transition_progress / transition_delay)

            next_radius = transition_start_radius + s * (
                transition_end_radius - transition_start_radius
            )

            if next_transition_progress >= transition_delay:
                next_radius = transition_end_radius
                current_layer_radius = transition_end_radius
                next_mode = "axial"
                next_direction = -direction
                next_transition_progress = transition_delay
                next_layer = layer + 1

        new_contact_world = deposit_point_world(next_radius, next_z)
        new_local = world_to_spool_local(new_contact_world, next_theta)

        prev_local = deposited_local[-1]
        seg = float(np.linalg.norm(new_local - prev_local))

        if seg < max(0.25, Rt * 0.05):
            theta = next_theta
            z = next_z
            direction = next_direction
            mode = next_mode
            layer = next_layer

            transition_progress = next_transition_progress
            transition_delay = next_transition_delay
            transition_z = next_transition_z
            transition_start_radius = next_transition_start_radius
            transition_end_radius = next_transition_end_radius

            continue

        if deposited_len + seg >= max_len:
            remain = max_len - deposited_len

            if seg > EPS and remain > 0.0:
                a = remain / seg

                final_theta = theta + a * (next_theta - theta)
                final_z = z + a * (next_z - z)

                prev_r = radius_values[-1]
                final_r = prev_r + a * (next_radius - prev_r)

                final_contact_world = deposit_point_world(final_r, final_z)
                final_local = world_to_spool_local(final_contact_world, final_theta)

                contact_world.append(final_contact_world)
                deposited_local.append(final_local)
                theta_values.append(final_theta)
                radius_values.append(final_r)
                z_values.append(final_z)
                mode_values.append(1 if next_mode == "transition" else 0)
                layer_values.append(next_layer)

                deposited_len += float(np.linalg.norm(final_local - prev_local))
                length_values.append(deposited_len)

            break

        contact_world.append(new_contact_world)
        deposited_local.append(new_local)
        theta_values.append(next_theta)
        radius_values.append(next_radius)
        z_values.append(next_z)
        mode_values.append(1 if next_mode == "transition" else 0)
        layer_values.append(next_layer)

        deposited_len += seg
        length_values.append(deposited_len)

        theta = next_theta
        z = next_z
        direction = next_direction
        mode = next_mode
        layer = next_layer

        transition_progress = next_transition_progress
        transition_delay = next_transition_delay
        transition_z = next_transition_z
        transition_start_radius = next_transition_start_radius
        transition_end_radius = next_transition_end_radius

    return (
        np.array(contact_world, dtype=float),
        np.array(deposited_local, dtype=float),
        np.array(theta_values, dtype=float),
        np.array(radius_values, dtype=float),
        np.array(z_values, dtype=float),
        np.array(mode_values, dtype=int),
        np.array(layer_values, dtype=int),
        np.array(length_values, dtype=float),
        deposited_len,
    )

# =========================
# METRICS
# =========================

def compute_max_xy_span(points: np.ndarray, d_tubo: float) -> float:
    if points is None or len(points) < 2:
        return float(d_tubo)

    xy = points[:, :2]
    max_samples = 1200

    if len(xy) > max_samples:
        idx = np.linspace(0, len(xy) - 1, max_samples).astype(int)
        xy = xy[idx]

    diff = xy[:, None, :] - xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    max_centerline_span = float(np.sqrt(np.max(dist2)))

    return max_centerline_span + d_tubo


def compute_metrics(points: np.ndarray, d_tubo: float):
    if points is None or len(points) == 0:
        return {
            "diam_radiale": 0.0,
            "max_xy_span": 0.0,
            "wound_length_m": 0.0,
        }

    radial = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    max_centerline_r = float(np.max(radial))

    diam_radiale = 2.0 * (max_centerline_r + d_tubo / 2.0)
    max_xy_span = compute_max_xy_span(points, d_tubo)
    wound_length_m = polyline_length(points) / 1000.0

    return {
        "diam_radiale": diam_radiale,
        "max_xy_span": max_xy_span,
        "wound_length_m": wound_length_m,
    }

# =========================
# VIEWER
# =========================

def viewer(
    d_aspo,
    spalla,
    d_tubo,
    altezza,
    final_local_points,
    final_thetas,
    final_radii,
    final_zs,
    final_modes,
    final_layers,
    final_lengths,
    guide_offset_x,
    language,
):
    final_local_points_json = json.dumps(final_local_points)
    final_thetas_json = json.dumps(final_thetas)
    final_radii_json = json.dumps(final_radii)
    final_zs_json = json.dumps(final_zs)
    final_modes_json = json.dumps(final_modes)
    final_layers_json = json.dumps(final_layers)
    final_lengths_json = json.dumps(final_lengths)
    labels_json = json.dumps(TEXTS[language])

    return f"""
    <div id="viewer_root" style="
        width:100%;
        height:{altezza}px;
        background:#101216;
        border-radius:16px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,0.08);
        box-shadow:0 18px 42px rgba(0,0,0,0.28);
        position:relative;
    ">
        <div id="viewer_topbar" style="
            position:absolute;
            top:14px;
            left:14px;
            z-index:20;
            display:flex;
            align-items:center;
            gap:8px;
            padding:10px 12px;
            background:rgba(18,22,27,0.74);
            color:#f0f0f0;
            border:1px solid rgba(255,255,255,0.12);
            border-radius:14px;
            backdrop-filter: blur(10px);
            font-family:Arial, sans-serif;
            font-size:13px;
            user-select:none;
        ">
            <button id="play_pause_btn" class="viewer_btn">⏸</button>
            <button id="reset_view_btn" class="viewer_btn">↺</button>
            <button id="fullscreen_btn" class="viewer_btn">⛶</button>
            <span style="margin-left:6px;" id="progress_title"></span>
            <input id="progress_slider" type="range" min="0" max="1000" step="1" value="0" style="width:180px;" />
        </div>

        <div id="viewer_hud" style="
            position:absolute;
            left:14px;
            bottom:14px;
            z-index:20;
            display:grid;
            grid-template-columns:repeat(3, auto);
            gap:8px;
            font-family:Arial, sans-serif;
            color:#f2f2f2;
            user-select:none;
        ">
            <div class="hud_card"><div class="hud_label" id="hud_length_label"></div><div class="hud_value" id="hud_length_value">0.0 m</div></div>
            <div class="hud_card"><div class="hud_label" id="hud_layer_label"></div><div class="hud_value" id="hud_layer_value">1</div></div>
            <div class="hud_card"><div class="hud_label" id="hud_diameter_label"></div><div class="hud_value" id="hud_diameter_value"></div></div>
        </div>

        <div id="viewer_sidepanel" style="
            position:absolute;
            top:14px;
            right:14px;
            z-index:20;
            display:flex;
            flex-direction:column;
            gap:12px;
            width:238px;
            padding:14px;
            background:rgba(18,22,27,0.74);
            color:#f0f0f0;
            border:1px solid rgba(255,255,255,0.12);
            border-radius:14px;
            backdrop-filter: blur(10px);
            font-family:Arial, sans-serif;
            font-size:13px;
            user-select:none;
        ">
            <div>
                <div class="panel_label" id="animation_title"></div>
                <label class="panel_check">
                    <input type="checkbox" id="animation_check" checked />
                    <span id="animation_label_text"></span>
                </label>
            </div>

            <div>
                <div class="panel_label" id="speed_title"></div>
                <div class="btn_group_vertical" id="speed_group">
                    <button class="speed_btn viewer_btn_small" data-speed="0.1">x0.1</button>
                    <button class="speed_btn viewer_btn_small" data-speed="0.5">x0.5</button>
                    <button class="speed_btn viewer_btn_small active_speed" data-speed="1.0">x1</button>
                    <button class="speed_btn viewer_btn_small" data-speed="1.5">x1.5</button>
                    <button class="speed_btn viewer_btn_small" data-speed="2.0">x2</button>
                    <button class="speed_btn viewer_btn_small" data-speed="5.0">x5</button>
                </div>
            </div>

            <div>
                <div class="panel_label" id="view_title"></div>
                <div class="btn_group_vertical">
                    <button class="view_btn viewer_btn_small active_opt" data-view="3d" id="view_3d_btn"></button>
                    <button class="view_btn viewer_btn_small" data-view="front" id="view_front_btn"></button>
                    <button class="view_btn viewer_btn_small" data-view="side" id="view_side_btn"></button>
                </div>
            </div>

            <div>
                <div class="panel_label" id="spool_title"></div>
                <div class="btn_group_vertical">
                    <button class="spool_btn viewer_btn_small active_opt" data-spool="visible" id="spool_visible_btn"></button>
                    <button class="spool_btn viewer_btn_small" data-spool="transparent" id="spool_transparent_btn"></button>
                    <button class="spool_btn viewer_btn_small" data-spool="hidden" id="spool_hidden_btn"></button>
                </div>
            </div>

            <div>
                <div class="panel_label" id="tube_title"></div>
                <div class="btn_group_vertical">
                    <button class="tube_btn viewer_btn_small active_opt" data-tube="gelwhite" id="tube_gelwhite_btn"></button>
                    <button class="tube_btn viewer_btn_small" data-tube="gelblack" id="tube_gelblack_btn"></button>
                </div>
            </div>

            <div class="panel_checks_block">
                <label class="panel_check">
                    <input type="checkbox" id="studio_check" checked />
                    <span id="studio_title"></span>
                </label>

                <label class="panel_check">
                    <input type="checkbox" id="ghost_check" checked />
                    <span id="ghost_title"></span>
                </label>

                <label class="panel_check">
                    <input type="checkbox" id="grid_check" />
                    <span id="grid_title"></span>
                </label>

                <label class="panel_check">
                    <input type="checkbox" id="axes_check" />
                    <span id="axes_title"></span>
                </label>

                <label class="panel_check">
                    <input type="checkbox" id="section_check" />
                    <span id="section_title"></span>
                </label>
            </div>
        </div>
    </div>

    <style>
        .viewer_btn {{
            border:none;
            border-radius:9px;
            padding:7px 12px;
            background:#f4f4f4;
            color:#111;
            font-weight:700;
            cursor:pointer;
        }}

        .viewer_btn_small {{
            border:none;
            border-radius:9px;
            padding:7px 10px;
            background:rgba(235,235,235,0.88);
            color:#111;
            font-weight:600;
            cursor:pointer;
            text-align:left;
        }}

        .viewer_btn_small:hover,
        .viewer_btn:hover {{
            background:#ffffff;
        }}

        .active_speed,
        .active_opt {{
            outline:2px solid #ffffff;
            background:#ffffff;
        }}

        .panel_label {{
            font-size:11px;
            opacity:0.82;
            margin-bottom:6px;
            text-transform:uppercase;
            letter-spacing:0.06em;
        }}

        .btn_group_vertical {{
            display:flex;
            flex-direction:column;
            gap:6px;
        }}

        .panel_check {{
            display:flex;
            align-items:center;
            gap:8px;
        }}

        .panel_checks_block {{
            display:flex;
            flex-direction:column;
            gap:8px;
            padding-top:2px;
        }}

        .viewer_btn_disabled {{
            opacity:0.45;
            cursor:not-allowed;
        }}

        .hud_card {{
            min-width:86px;
            padding:10px 12px;
            background:rgba(18,22,27,0.56);
            border:1px solid rgba(255,255,255,0.12);
            border-radius:13px;
            backdrop-filter: blur(10px);
            box-shadow:0 10px 24px rgba(0,0,0,0.18);
        }}

        .hud_label {{
            font-size:10px;
            opacity:0.70;
            text-transform:uppercase;
            letter-spacing:0.06em;
            margin-bottom:4px;
        }}

        .hud_value {{
            font-size:15px;
            font-weight:700;
            white-space:nowrap;
        }}
    </style>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/TrackballControls.js"></script>

    <script>
    (() => {{
        const T = {labels_json};

        const host = document.getElementById("viewer_root");
        const playPauseBtn = document.getElementById("play_pause_btn");
        const resetViewBtn = document.getElementById("reset_view_btn");
        const fullscreenBtn = document.getElementById("fullscreen_btn");
        const progressSlider = document.getElementById("progress_slider");
        const animationCheck = document.getElementById("animation_check");

        const speedBtns = [...document.querySelectorAll(".speed_btn")];
        const spoolBtns = [...document.querySelectorAll(".spool_btn")];
        const tubeBtns = [...document.querySelectorAll(".tube_btn")];
        const viewBtns = [...document.querySelectorAll(".view_btn")];

        const studioCheck = document.getElementById("studio_check");
        const ghostCheck = document.getElementById("ghost_check");
        const gridCheck = document.getElementById("grid_check");
        const axesCheck = document.getElementById("axes_check");
        const sectionCheck = document.getElementById("section_check");

        document.getElementById("progress_title").textContent = T.progress;
        document.getElementById("speed_title").textContent = T.speed;
        document.getElementById("spool_title").textContent = T.spool;
        document.getElementById("tube_title").textContent = T.tube_color;
        document.getElementById("view_title").textContent = T.view;
        document.getElementById("grid_title").textContent = T.grid;
        document.getElementById("axes_title").textContent = T.axes;
        document.getElementById("section_title").textContent = T.section;
        document.getElementById("ghost_title").textContent = T.ghost;
        document.getElementById("studio_title").textContent = T.studio;
        document.getElementById("animation_title").textContent = T.animation;
        document.getElementById("animation_label_text").textContent = T.animation;
        document.getElementById("spool_visible_btn").textContent = T.visible;
        document.getElementById("spool_transparent_btn").textContent = T.transparent;
        document.getElementById("spool_hidden_btn").textContent = T.hidden;
        document.getElementById("tube_gelwhite_btn").textContent = T.gelwhite;
        document.getElementById("tube_gelblack_btn").textContent = T.gelblack;
        document.getElementById("view_3d_btn").textContent = T.view_3d;
        document.getElementById("view_front_btn").textContent = T.view_front;
        document.getElementById("view_side_btn").textContent = T.view_side;
        resetViewBtn.title = T.reset_view;

        document.getElementById("hud_length_label").textContent = T.hud_length;
        document.getElementById("hud_layer_label").textContent = T.hud_layer;
        document.getElementById("hud_diameter_label").textContent = T.hud_diameter;
        document.getElementById("hud_diameter_value").textContent = "{float(d_tubo):.2f} mm";

        const W = Math.max(host.clientWidth, 600);
        const Hview = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();

        const camera = new THREE.PerspectiveCamera(32, W / Hview, 0.1, 20000);
        camera.position.set(-950, -1500, 520);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({{
            antialias: true,
            powerPreference: "high-performance"
        }});

        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
        renderer.setSize(W, Hview);
        renderer.outputEncoding = THREE.sRGBEncoding;
        renderer.physicallyCorrectLights = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.04;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.localClippingEnabled = true;

        host.appendChild(renderer.domElement);

        const controls = new THREE.TrackballControls(camera, renderer.domElement);
        controls.rotateSpeed = 3.2;
        controls.zoomSpeed = 0.8;
        controls.panSpeed = 0.12;
        controls.dynamicDampingFactor = 0.18;
        controls.staticMoving = false;

        const R = {float(d_aspo)} / 2.0;
        const Rt = {float(d_tubo)} / 2.0;
        const Hs = {float(spalla)};
        const guideOffsetX = {float(guide_offset_x)};

        controls.target.set(0, 0, Hs * 0.52);
        camera.lookAt(0, 0, Hs * 0.52);

        const localRaw = {final_local_points_json};
        const thetaRaw = {final_thetas_json};
        const radiusRaw = {final_radii_json};
        const zRaw = {final_zs_json};
        const layerRaw = {final_layers_json};
        const lengthRaw = {final_lengths_json};

        const localPts = localRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

        let isPlaying = true;
        let animationEnabled = true;
        let speed = 1.0;
        let aspoMode = "visible";
        let tubeMode = "gelwhite";
        let currentView = "3d";
        let showStudio = true;
        let showGhost = true;
        let showGrid = false;
        let showAxes = false;
        let showSection = false;

        let clippingPlanes = [];
        let grid = null;
        let axes = null;
        let sectionPlaneHelper = null;
        let sectionFrame = null;
        let floor = null;
        let ghostLine = null;

        function getTheme() {{
            if (tubeMode === "gelblack") {{
                return {{
                    bg: 0xffffff,
                    floor: 0xf3f3f3,
                    tube: 0x343635,
                    freeTube: 0x3d403f,
                    activeTube: 0x252827,
                    ghost: 0x2c2c2c,
                    sectionFill: 0x111111,
                    sectionFrame: 0x111111,
                    gridMajor: 0x777777,
                    gridMinor: 0xd5d5d5,
                    gridOpacity: 0.38,
                    hemiSky: 0xffffff,
                    hemiGround: 0xd5d5d5,
                    ambient: 0.34,
                    key: 1.20,
                    fill: 0.66,
                    rim: 0.80,
                    exposure: 1.10
                }};
            }}

            return {{
                bg: 0x111419,
                floor: 0x1b1e23,
                tube: 0xd8d6cf,
                freeTube: 0xc6c3bb,
                activeTube: 0xf2efe6,
                ghost: 0xffffff,
                sectionFill: 0xffffff,
                sectionFrame: 0xffffff,
                gridMajor: 0x747474,
                gridMinor: 0x2d3035,
                gridOpacity: 0.30,
                hemiSky: 0xe2e8ef,
                hemiGround: 0x151719,
                ambient: 0.23,
                key: 1.30,
                fill: 0.52,
                rim: 0.82,
                exposure: 1.02
            }};
        }}

        function updatePlayBtn() {{
            playPauseBtn.textContent = isPlaying ? "⏸" : "▶";
            playPauseBtn.title = isPlaying ? T.pause : T.play;
        }}

        function updateAnimationUI() {{
            if (animationEnabled) {{
                playPauseBtn.classList.remove("viewer_btn_disabled");
                playPauseBtn.disabled = false;
            }} else {{
                playPauseBtn.classList.add("viewer_btn_disabled");
                playPauseBtn.disabled = true;
            }}
        }}

        function setActiveButton(group, value, attr, activeClass="active_opt") {{
            group.forEach(btn => {{
                btn.classList.toggle(activeClass, btn.getAttribute(attr) === value);
            }});
        }}

        function setCameraView(viewName) {{
            const target = new THREE.Vector3(0, 0, Hs * 0.52);

            if (viewName === "front") {{
                camera.position.set(0, -1900, Hs * 0.52);
            }} else if (viewName === "side") {{
                camera.position.set(-1900, 0, Hs * 0.52);
            }} else {{
                camera.position.set(-950, -1500, 520);
            }}

            camera.up.set(0, 0, 1);
            controls.target.copy(target);
            camera.lookAt(target);
            controls.update();
        }}

        speedBtns.forEach(btn => {{
            btn.addEventListener("click", () => {{
                speed = parseFloat(btn.dataset.speed);
                speedBtns.forEach(b => b.classList.remove("active_speed"));
                btn.classList.add("active_speed");
            }});
        }});

        spoolBtns.forEach(btn => {{
            btn.addEventListener("click", () => {{
                aspoMode = btn.dataset.spool;
                setActiveButton(spoolBtns, aspoMode, "data-spool");
                applyVisualState();
            }});
        }});

        tubeBtns.forEach(btn => {{
            btn.addEventListener("click", () => {{
                tubeMode = btn.dataset.tube;
                setActiveButton(tubeBtns, tubeMode, "data-tube");
                applyVisualState(true);
            }});
        }});

        viewBtns.forEach(btn => {{
            btn.addEventListener("click", () => {{
                currentView = btn.dataset.view;
                setActiveButton(viewBtns, currentView, "data-view");
                setCameraView(currentView);
            }});
        }});

        resetViewBtn.addEventListener("click", () => {{
            currentView = "3d";
            setActiveButton(viewBtns, currentView, "data-view");
            setCameraView("3d");
        }});

        studioCheck.addEventListener("change", () => {{
            showStudio = studioCheck.checked;
            applyVisualState();
        }});

        ghostCheck.addEventListener("change", () => {{
            showGhost = ghostCheck.checked;
            updateGhostLine();
        }});

        gridCheck.addEventListener("change", () => {{
            showGrid = gridCheck.checked;
            applyVisualState();
        }});

        axesCheck.addEventListener("change", () => {{
            showAxes = axesCheck.checked;
            applyVisualState();
        }});

        sectionCheck.addEventListener("change", () => {{
            showSection = sectionCheck.checked;
            applySectionState();
            rebuildDepositedMesh(Math.floor(drawPos), true);
            updateOverlayContinuous(true);
        }});

        animationCheck.addEventListener("change", () => {{
            animationEnabled = animationCheck.checked;

            if (!animationEnabled) {{
                isPlaying = false;
                drawPos = localPts.length - 1;
                rebuildDepositedMesh(Math.floor(drawPos), true);
                updateOverlayContinuous(true);
                progressSlider.value = 1000;
            }} else {{
                isPlaying = true;
            }}

            updatePlayBtn();
            updateAnimationUI();
        }});

        playPauseBtn.addEventListener("click", () => {{
            if (!animationEnabled) return;
            isPlaying = !isPlaying;
            updatePlayBtn();
        }});

        fullscreenBtn.addEventListener("click", async () => {{
            try {{
                if (!document.fullscreenElement) {{
                    await host.requestFullscreen();
                    fullscreenBtn.textContent = "🡼";
                    fullscreenBtn.title = T.exit;
                }} else {{
                    await document.exitFullscreen();
                    fullscreenBtn.textContent = "⛶";
                    fullscreenBtn.title = T.fullscreen;
                }}
            }} catch (err) {{
                console.error(err);
            }}
        }});

        fullscreenBtn.title = T.fullscreen;

        document.addEventListener("fullscreenchange", () => {{
            if (!document.fullscreenElement) {{
                fullscreenBtn.textContent = "⛶";
                fullscreenBtn.title = T.fullscreen;
            }}
            setTimeout(resizeViewer, 30);
        }});

        progressSlider.addEventListener("input", () => {{
            const maxPos = Math.max(1, localPts.length - 1);
            drawPos = (parseInt(progressSlider.value) / 1000.0) * maxPos;
            rebuildDepositedMesh(Math.floor(drawPos), true);
            updateOverlayContinuous(true);
        }});

        function resizeViewer() {{
            const nw = Math.max(host.clientWidth, 600);
            const nh = Math.max(host.clientHeight, 400);

            camera.aspect = nw / nh;
            camera.updateProjectionMatrix();

            renderer.setSize(nw, nh);
            controls.handleResize();
        }}

        // ==========================================
        // TEXTURES
        // ==========================================

        function makeSteelTexture(size = 256) {{
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;

            const ctx = canvas.getContext("2d");

            const grad = ctx.createLinearGradient(0, 0, size, 0);
            grad.addColorStop(0.0, "#565c64");
            grad.addColorStop(0.18, "#d9dee3");
            grad.addColorStop(0.36, "#747b84");
            grad.addColorStop(0.58, "#c2c8ce");
            grad.addColorStop(0.82, "#666d76");
            grad.addColorStop(1.0, "#e0e4e8");

            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, size, size);

            for (let y = 0; y < size; y += 2) {{
                const a = 0.035 + Math.random() * 0.04;
                ctx.fillStyle = `rgba(255,255,255,${{a}})`;
                ctx.fillRect(0, y, size, 1);
            }}

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(0.65, 0.65);
            tex.anisotropy = 8;

            return tex;
        }}

        function makeTubeTexture(size = 256, dark=false) {{
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;

            const ctx = canvas.getContext("2d");

            const base = dark ? 76 : 214;
            ctx.fillStyle = `rgb(${{base}}, ${{base}}, ${{base}})`;
            ctx.fillRect(0, 0, size, size);

            const img = ctx.getImageData(0, 0, size, size);
            const data = img.data;

            for (let y = 0; y < size; y++) {{
                for (let x = 0; x < size; x++) {{
                    const i = (y * size + x) * 4;

                    const grain = Math.random() * 18 - 9;
                    const microLine = Math.sin((x + y * 0.18) * 0.50) * 2.4;
                    const longLine = Math.sin(y * 0.13) * 2.0;

                    let v = base + grain + microLine + longLine;

                    if (dark) {{
                        v = Math.max(44, Math.min(112, v));
                    }} else {{
                        v = Math.max(154, Math.min(244, v));
                    }}

                    data[i] = v;
                    data[i + 1] = v;
                    data[i + 2] = v;
                    data[i + 3] = 255;
                }}
            }}

            ctx.putImageData(img, 0, 0);

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(2.0, 18.0);
            tex.anisotropy = 12;
            tex.needsUpdate = true;

            return tex;
        }}

        const steelTex = makeSteelTexture(256);
        const tubeWhiteTex = makeTubeTexture(256, false);
        const tubeBlackTex = makeTubeTexture(256, true);

        // ==========================================
        // MATERIALS
        // ==========================================

        function makeSteelMat(opacity=1.0, transparent=false) {{
            return new THREE.MeshStandardMaterial({{
                color: 0x6d7278,
                roughness: 0.58,
                metalness: 0.82,
                map: steelTex,
                transparent: transparent,
                opacity: opacity,
                depthWrite: !transparent
            }});
        }}

        function makeTubeMaterial(mode, active=false, free=false) {{
            const theme = getTheme();
            const chosen = active ? theme.activeTube : (free ? theme.freeTube : theme.tube);
            const tex = mode === "gelblack" ? tubeBlackTex : tubeWhiteTex;

            return new THREE.MeshStandardMaterial({{
                color: chosen,
                map: tex,
                roughness: active ? 0.82 : (free ? 0.94 : 0.90),
                metalness: 0.02,
                clippingPlanes: clippingPlanes,
                clipShadows: showSection
            }});
        }}

        let steelMat = makeSteelMat(1.0, false);
        let steelMatTransparent = makeSteelMat(0.18, true);

        let tubeMat = makeTubeMaterial(tubeMode, false, false);
        let activeTubeMat = makeTubeMaterial(tubeMode, true, false);
        let freeTubeMat = makeTubeMaterial(tubeMode, false, true);

        const markerStartMat = new THREE.MeshStandardMaterial({{
            color: 0x23a55a,
            roughness: 0.45,
            metalness: 0.02,
            emissive: 0x0b2013,
            emissiveIntensity: 0.12
        }});

        const markerEndMat = new THREE.MeshStandardMaterial({{
            color: 0xffb020,
            roughness: 0.40,
            metalness: 0.02,
            emissive: 0x2a1800,
            emissiveIntensity: 0.14
        }});

        // ==========================================
        // LIGHTING
        // ==========================================

        const ambient = new THREE.AmbientLight(0xffffff, 0.22);
        scene.add(ambient);

        const hemi = new THREE.HemisphereLight(0xd7dfe7, 0x1a1d20, 0.34);
        scene.add(hemi);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.30);
        keyLight.position.set(420, -520, 780);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 50;
        keyLight.shadow.camera.far = 3600;
        keyLight.shadow.camera.left = -1400;
        keyLight.shadow.camera.right = 1400;
        keyLight.shadow.camera.top = 1400;
        keyLight.shadow.camera.bottom = -1400;
        scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.52);
        fillLight.position.set(-700, 340, 360);
        scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 0.82);
        rimLight.position.set(-180, 760, 580);
        scene.add(rimLight);

        const softTopLight = new THREE.PointLight(0xffffff, 0.32, 2200);
        softTopLight.position.set(0, 0, 900);
        scene.add(softTopLight);

        // ==========================================
        // SCENE GROUPS
        // ==========================================

        const studioGroup = new THREE.Group();
        scene.add(studioGroup);

        const machine = new THREE.Group();
        scene.add(machine);

        const depositedGroup = new THREE.Group();
        machine.add(depositedGroup);

        const overlayGroup = new THREE.Group();
        scene.add(overlayGroup);

        const spoolParts = [];

        // ==========================================
        // STUDIO FLOOR ONLY
        // ==========================================

        function rebuildStudio() {{
            if (floor) {{
                studioGroup.remove(floor);
                floor.geometry.dispose();
                floor.material.dispose();
                floor = null;
            }}

            if (!showStudio) return;

            const theme = getTheme();

            const floorMat = new THREE.MeshStandardMaterial({{
                color: theme.floor,
                roughness: 0.88,
                metalness: 0.0
            }});

            floor = new THREE.Mesh(
                new THREE.PlaneGeometry(2600, 2600),
                floorMat
            );

            floor.position.set(0, 0, -38);
            floor.receiveShadow = true;
            studioGroup.add(floor);
        }}

        // ==========================================
        // SIMPLE ASPO
        // ==========================================

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, Hs, 128),
            steelMat
        );

        mandrel.rotation.x = Math.PI / 2;
        mandrel.position.z = Hs / 2.0;
        mandrel.castShadow = true;
        mandrel.receiveShadow = true;
        machine.add(mandrel);
        spoolParts.push(mandrel);

        const flangeR = R + 150.0;
        const flangeTh = 4.0;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 128),
            steelMat
        );

        base.rotation.x = Math.PI / 2;
        base.position.z = 0.0;
        base.castShadow = true;
        base.receiveShadow = true;
        machine.add(base);
        spoolParts.push(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 128),
            steelMat
        );

        top.rotation.x = Math.PI / 2;
        top.position.z = Hs;
        top.castShadow = true;
        top.receiveShadow = true;
        machine.add(top);
        spoolParts.push(top);

        // ==========================================
        // SIMPLE GUIDATUBO
        // ==========================================

        const nozzleDiameter = 55.0;
        const oldNozzleDiameter = Math.max(4.0, Rt * 0.56);
        const guideScale = (nozzleDiameter / oldNozzleDiameter) * 0.34;

        const guideGroup = new THREE.Group();
        scene.add(guideGroup);

        const guideBarrel = new THREE.Mesh(
            new THREE.CylinderGeometry(20 * guideScale, 20 * guideScale, 44 * guideScale, 40, 1, false),
            steelMat
        );

        guideBarrel.rotation.z = Math.PI / 2;
        guideBarrel.position.x = 0;
        guideBarrel.castShadow = true;
        guideBarrel.receiveShadow = true;
        guideGroup.add(guideBarrel);

        const guideShoulder = new THREE.Mesh(
            new THREE.CylinderGeometry(27 * guideScale, 20 * guideScale, 18 * guideScale, 40, 1, false),
            steelMat
        );

        guideShoulder.rotation.z = Math.PI / 2;
        guideShoulder.position.x = 22 * guideScale;
        guideShoulder.castShadow = true;
        guideShoulder.receiveShadow = true;
        guideGroup.add(guideShoulder);

        const guideTaper = new THREE.Mesh(
            new THREE.CylinderGeometry(12 * guideScale, 17 * guideScale, 22 * guideScale, 40, 1, false),
            steelMat
        );

        guideTaper.rotation.z = Math.PI / 2;
        guideTaper.position.x = 42 * guideScale;
        guideTaper.castShadow = true;
        guideTaper.receiveShadow = true;
        guideGroup.add(guideTaper);

        const guideNozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(nozzleDiameter / 2, nozzleDiameter / 2, 14 * guideScale, 48, 1, false),
            steelMat
        );

        guideNozzle.rotation.z = Math.PI / 2;
        guideNozzle.position.x = 58 * guideScale;
        guideNozzle.castShadow = true;
        guideNozzle.receiveShadow = true;
        guideGroup.add(guideNozzle);

        const guideBackCap = new THREE.Mesh(
            new THREE.CylinderGeometry(15 * guideScale, 15 * guideScale, 10 * guideScale, 36, 1, false),
            steelMat
        );

        guideBackCap.rotation.z = Math.PI / 2;
        guideBackCap.position.x = -28 * guideScale;
        guideBackCap.castShadow = true;
        guideBackCap.receiveShadow = true;
        guideGroup.add(guideBackCap);

        // ==========================================
        // VISUAL STATE
        // ==========================================

        function refreshThemeBackgroundAndLights() {{
            const theme = getTheme();

            scene.background = new THREE.Color(theme.bg);
            renderer.toneMappingExposure = theme.exposure;

            ambient.intensity = theme.ambient;
            hemi.color.setHex(theme.hemiSky);
            hemi.groundColor.setHex(theme.hemiGround);

            keyLight.intensity = theme.key;
            fillLight.intensity = theme.fill;
            rimLight.intensity = theme.rim;
        }}

        function applySectionState() {{
            clippingPlanes = [];

            if (sectionPlaneHelper) scene.remove(sectionPlaneHelper);
            if (sectionFrame) scene.remove(sectionFrame);

            sectionPlaneHelper = null;
            sectionFrame = null;

            if (showSection) {{
                const theme = getTheme();

                const cutPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
                clippingPlanes = [cutPlane];

                const sectionMat = new THREE.MeshBasicMaterial({{
                    color: theme.sectionFill,
                    transparent: true,
                    opacity: tubeMode === "gelwhite" ? 0.12 : 0.08,
                    side: THREE.DoubleSide,
                    depthWrite: false
                }});

                const sectionGeo = new THREE.PlaneGeometry(2 * (R + 320), Hs + 300);

                sectionPlaneHelper = new THREE.Mesh(sectionGeo, sectionMat);
                sectionPlaneHelper.position.set(0, 0, Hs * 0.5);
                sectionPlaneHelper.rotation.y = Math.PI / 2;
                scene.add(sectionPlaneHelper);

                const frameGeo = new THREE.EdgesGeometry(sectionGeo);

                const frameMat = new THREE.LineBasicMaterial({{
                    color: theme.sectionFrame,
                    transparent: true,
                    opacity: tubeMode === "gelwhite" ? 0.34 : 0.26
                }});

                sectionFrame = new THREE.LineSegments(frameGeo, frameMat);
                sectionFrame.position.copy(sectionPlaneHelper.position);
                sectionFrame.rotation.copy(sectionPlaneHelper.rotation);
                scene.add(sectionFrame);
            }}

            renderer.localClippingEnabled = showSection;

            tubeMat = makeTubeMaterial(tubeMode, false, false);
            activeTubeMat = makeTubeMaterial(tubeMode, true, false);
            freeTubeMat = makeTubeMaterial(tubeMode, false, true);
        }}

        function buildGridIfNeeded() {{
            if (grid) scene.remove(grid);
            grid = null;

            if (showGrid) {{
                const theme = getTheme();

                grid = new THREE.GridHelper(
                    2200,
                    22,
                    theme.gridMajor,
                    theme.gridMinor
                );

                grid.rotation.x = Math.PI / 2;
                grid.position.z = -36;
                grid.material.opacity = theme.gridOpacity;
                grid.material.transparent = true;

                scene.add(grid);
            }}
        }}

        function buildAxesIfNeeded() {{
            if (axes) scene.remove(axes);
            axes = null;

            if (showAxes) {{
                axes = new THREE.AxesHelper(380);
                scene.add(axes);
            }}
        }}

        function applySpoolMaterialState() {{
            const useMat = aspoMode === "transparent" ? steelMatTransparent : steelMat;

            spoolParts.forEach(part => {{
                part.visible = aspoMode !== "hidden";
                part.material = useMat;
            }});

            guideBarrel.material = useMat;
            guideShoulder.material = useMat;
            guideTaper.material = useMat;
            guideNozzle.material = useMat;
            guideBackCap.material = useMat;
        }}

        function applyVisualState(themeChanged=false) {{
            refreshThemeBackgroundAndLights();
            rebuildStudio();
            applySpoolMaterialState();
            buildGridIfNeeded();
            buildAxesIfNeeded();

            if (themeChanged) {{
                applySectionState();
            }}

            rebuildDepositedMesh(Math.floor(drawPos), true);
            updateOverlayContinuous(true);
            updateGhostLine();
        }}

        // ==========================================
        // HELPERS
        // ==========================================

        function guidePointWorld(radius, z) {{
            return new THREE.Vector3(
                -(radius + guideOffsetX),
                radius,
                z
            );
        }}

        function localPointToWorld(ptLocal, theta) {{
            return ptLocal.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), theta);
        }}

        function lerp(a, b, tt) {{
            return a + (b - a) * tt;
        }}

        function lerpVec3(a, b, tt) {{
            return new THREE.Vector3(
                lerp(a.x, b.x, tt),
                lerp(a.y, b.y, tt),
                lerp(a.z, b.z, tt)
            );
        }}

        class PolylineCurve3 extends THREE.Curve {{
            constructor(points) {{
                super();

                this.points = points || [];
                this.arc = [0];
                this.totalLength = 0;

                for (let i = 1; i < this.points.length; i++) {{
                    const seg = this.points[i].distanceTo(this.points[i - 1]);
                    this.totalLength += seg;
                    this.arc.push(this.totalLength);
                }}
            }}

            getPoint(tt) {{
                if (!this.points || this.points.length === 0) {{
                    return new THREE.Vector3(0, 0, 0);
                }}

                if (this.points.length === 1 || this.totalLength <= 1e-9) {{
                    return this.points[0].clone();
                }}

                const target = tt * this.totalLength;

                let i = 1;

                while (i < this.arc.length && this.arc[i] < target) {{
                    i++;
                }}

                if (i >= this.points.length) {{
                    return this.points[this.points.length - 1].clone();
                }}

                const l0 = this.arc[i - 1];
                const l1 = this.arc[i];

                const p0 = this.points[i - 1];
                const p1 = this.points[i];

                const denom = Math.max(1e-9, l1 - l0);
                const a = (target - l0) / denom;

                return new THREE.Vector3(
                    p0.x + a * (p1.x - p0.x),
                    p0.y + a * (p1.y - p0.y),
                    p0.z + a * (p1.z - p0.z)
                );
            }}
        }}

        function disposeMaterial(mat) {{
            if (!mat) return;

            if (Array.isArray(mat)) {{
                mat.forEach(m => m && m.dispose && m.dispose());
            }} else if (mat.dispose) {{
                mat.dispose();
            }}
        }}

        function disposeObj(obj, parentObj = scene) {{
            if (!obj) return;

            parentObj.remove(obj);

            if (obj.geometry) obj.geometry.dispose();

            disposeMaterial(obj.material);
        }}

        function makeTubeMeshFromPoints(points, radius, material) {{
            if (!points || points.length < 2) return null;

            let totalLen = 0;

            for (let i = 1; i < points.length; i++) {{
                totalLen += points[i].distanceTo(points[i - 1]);
            }}

            const curve = new PolylineCurve3(points);

            const tubularSegments = Math.max(
                24,
                Math.min(3200, Math.floor(totalLen / Math.max(1.10, radius * 0.40)))
            );

            const geo = new THREE.TubeGeometry(curve, tubularSegments, radius, 22, false);
            geo.computeVertexNormals();

            const mesh = new THREE.Mesh(geo, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            return mesh;
        }}

        function makeTubeSegment(p0, p1, radius, material) {{
            const dir = new THREE.Vector3().subVectors(p1, p0);
            const len = dir.length();

            if (len < 1e-6) return null;

            const geo = new THREE.CylinderGeometry(radius, radius, len, 22, 1, false);
            const mesh = new THREE.Mesh(geo, material);

            const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
            mesh.position.copy(mid);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.clone().normalize());

            mesh.setRotationFromQuaternion(quat);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            return mesh;
        }}

        function makeEndpointDisc(point, tangentDir, material, radiusScale = 0.92) {{
            const r = Math.max(7.0, Rt * radiusScale);
            const thickness = Math.max(2.0, Rt * 0.22);

            const geo = new THREE.CylinderGeometry(r, r * 0.95, thickness, 32);
            const mesh = new THREE.Mesh(geo, material);

            mesh.position.copy(point);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, tangentDir.clone().normalize());

            mesh.setRotationFromQuaternion(quat);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            return mesh;
        }}

        let depositedMesh = null;
        let freeMesh = null;
        let activeCoilMesh = null;
        let startMarker = null;
        let endMarker = null;

        let drawPos = 1.0;
        let lastRebuiltCompleted = -1;

        function rebuildDepositedMesh(completedIndex, force=false) {{
            if (completedIndex < 1) return;

            if (!force && completedIndex === lastRebuiltCompleted && depositedMesh) return;

            lastRebuiltCompleted = completedIndex;

            if (depositedMesh) {{
                disposeObj(depositedMesh, depositedGroup);
                depositedMesh = null;
            }}

            const pts = localPts.slice(0, completedIndex + 1);

            depositedMesh = makeTubeMeshFromPoints(pts, Rt, tubeMat);

            if (depositedMesh) {{
                depositedGroup.add(depositedMesh);
            }}
        }}

        function clearOverlay() {{
            if (freeMesh) {{
                disposeObj(freeMesh, overlayGroup);
                freeMesh = null;
            }}

            if (activeCoilMesh) {{
                disposeObj(activeCoilMesh, overlayGroup);
                activeCoilMesh = null;
            }}

            if (startMarker) {{
                disposeObj(startMarker, overlayGroup);
                startMarker = null;
            }}

            if (endMarker) {{
                disposeObj(endMarker, overlayGroup);
                endMarker = null;
            }}
        }}

        function updateHud(index) {{
            const i = Math.max(0, Math.min(index, lengthRaw.length - 1));

            const lengthM = (lengthRaw[i] || 0) / 1000.0;
            const layer = (layerRaw[i] || 0) + 1;

            document.getElementById("hud_length_value").textContent = `${{lengthM.toFixed(2)}} m`;
            document.getElementById("hud_layer_value").textContent = `${{layer}}`;
        }}

        function updateGhostLine() {{
            if (ghostLine) {{
                scene.remove(ghostLine);
                ghostLine.geometry.dispose();
                ghostLine.material.dispose();
                ghostLine = null;
            }}

            if (!showGhost || !animationEnabled || localPts.length < 3) return;

            const i0 = Math.floor(drawPos);
            const futureCount = 140;
            const end = Math.min(localPts.length - 1, i0 + futureCount);

            if (end <= i0 + 2) return;

            const theta = thetaRaw[Math.max(0, Math.min(i0, thetaRaw.length - 1))];

            const futurePts = [];

            for (let i = i0; i <= end; i++) {{
                futurePts.push(localPointToWorld(localPts[i], theta));
            }}

            const geo = new THREE.BufferGeometry().setFromPoints(futurePts);
            const theme = getTheme();

            const mat = new THREE.LineDashedMaterial({{
                color: theme.ghost,
                transparent: true,
                opacity: tubeMode === "gelblack" ? 0.24 : 0.18,
                dashSize: 18,
                gapSize: 10
            }});

            ghostLine = new THREE.Line(geo, mat);
            ghostLine.computeLineDistances();
            scene.add(ghostLine);
        }}

        function updateOverlayContinuous(force=false) {{
            clearOverlay();

            if (localPts.length < 2) return;

            const maxPos = localPts.length - 1;
            const clampedPos = Math.max(1.0, Math.min(drawPos, maxPos));

            const i0 = Math.floor(clampedPos);
            const i1 = Math.min(i0 + 1, localPts.length - 1);
            const frac = clampedPos - i0;

            const theta = lerp(thetaRaw[i0], thetaRaw[i1], frac);
            const radius = lerp(radiusRaw[i0], radiusRaw[i1], frac);
            const z = lerp(zRaw[i0], zRaw[i1], frac);

            machine.rotation.z = theta;

            const activeLocalStart = localPts[i0];
            const activeLocalEnd = lerpVec3(localPts[i0], localPts[i1], frac);

            const startWorld = localPointToWorld(localPts[0], theta);
            const endWorld = localPointToWorld(activeLocalEnd, theta);

            const startTangentLocal = localPts[Math.min(1, localPts.length - 1)].clone().sub(localPts[0]);
            const endTangentLocal = activeLocalEnd.clone().sub(activeLocalStart);

            const startTangentWorld = startTangentLocal.clone().applyAxisAngle(new THREE.Vector3(0,0,1), theta);
            const endTangentWorld = endTangentLocal.clone().applyAxisAngle(new THREE.Vector3(0,0,1), theta);

            startMarker = makeEndpointDisc(startWorld, startTangentWorld, markerStartMat, 0.70);

            endMarker = makeEndpointDisc(
                endWorld,
                endTangentWorld.length() > 1e-6 ? endTangentWorld : startTangentWorld,
                markerEndMat,
                0.82
            );

            overlayGroup.add(startMarker);
            overlayGroup.add(endMarker);

            if (animationEnabled) {{
                if (frac > 1e-6 && i1 > i0) {{
                    const activeStartWorld = localPointToWorld(activeLocalStart, theta);
                    activeCoilMesh = makeTubeSegment(activeStartWorld, endWorld, Rt, activeTubeMat);

                    if (activeCoilMesh) {{
                        overlayGroup.add(activeCoilMesh);
                    }}
                }}

                const guideWorld = guidePointWorld(radius, z);

                freeMesh = makeTubeSegment(guideWorld, endWorld, Rt, freeTubeMat);

                if (freeMesh) {{
                    overlayGroup.add(freeMesh);
                }}

                guideGroup.position.copy(guideWorld);
                guideGroup.visible = true;
            }} else {{
                guideGroup.visible = false;
            }}

            updateHud(i0);

            if (force || Math.random() < 0.08) {{
                updateGhostLine();
            }}
        }}

        applySectionState();
        applyVisualState(true);
        updateAnimationUI();
        updatePlayBtn();

        function animate() {{
            requestAnimationFrame(animate);

            if (animationEnabled && isPlaying && drawPos < localPts.length - 1) {{
                const advance = 0.08 + Math.pow(speed, 2.35) * 1.1;

                const oldCompleted = Math.floor(drawPos);

                drawPos = Math.min(localPts.length - 1, drawPos + advance);

                const newCompleted = Math.floor(drawPos);

                if (newCompleted > oldCompleted) {{
                    rebuildDepositedMesh(newCompleted);
                }}

                updateOverlayContinuous();

                progressSlider.value = Math.round(
                    (drawPos / Math.max(1, localPts.length - 1)) * 1000
                );
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        if (!animationEnabled) {{
            drawPos = localPts.length - 1;
            rebuildDepositedMesh(Math.floor(drawPos), true);
            updateOverlayContinuous(true);
            progressSlider.value = 1000;
        }} else {{
            rebuildDepositedMesh(1, true);
            updateOverlayContinuous(true);
        }}

        animate();

        window.addEventListener("resize", resizeViewer);
    }})();
    </script>
    """

# =========================
# UI
# =========================

tab_presets, tab_calculator = st.tabs([
    "📦 Presets",
    "🧮 Calcolatore / Render",
])

with tab_presets:
    st.markdown("### 📦 Presets prodotti")

    try:
        presets_df = load_presets("Presets.csv")

        st.caption(f"{len(presets_df)} preset caricati correttamente da Presets.csv")

        selected_product = st.selectbox(
            "Seleziona prodotto",
            presets_df["Prodotto"].tolist(),
        )

        selected_row = presets_df[presets_df["Prodotto"] == selected_product].iloc[0]

        st.markdown(
            f"""
            <div style="
                margin-top:12px;
                margin-bottom:18px;
                padding:22px 24px;
                border-radius:18px;
                background:linear-gradient(135deg, rgba(30,34,40,0.96), rgba(18,21,26,0.96));
                border:1px solid rgba(255,255,255,0.10);
                box-shadow:0 14px 34px rgba(0,0,0,0.22);
            ">
                <div style="font-size:13px; color:rgba(255,255,255,0.62); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                    Scheda preset
                </div>
                <div style="font-size:30px; font-weight:800; color:#ffffff; line-height:1.15;">
                    {selected_product}
                </div>
                <div style="font-size:14px; color:rgba(255,255,255,0.68); margin-top:8px;">
                    Configurazione tecnica prodotto · valori caricati da CSV
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Parametri CSV")

        st.markdown(
            """
            <style>
                .preset-card {
                    min-height: 112px;
                    padding: 16px 16px 14px 16px;
                    margin-bottom: 14px;
                    border-radius: 16px;
                    background: linear-gradient(180deg, rgba(255,255,255,0.070), rgba(255,255,255,0.035));
                    border: 1px solid rgba(255,255,255,0.105);
                    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
                }

                .preset-label {
                    min-height: 34px;
                    color: rgba(255,255,255,0.66);
                    font-size: 12px;
                    font-weight: 700;
                    line-height: 1.18;
                    text-transform: uppercase;
                    letter-spacing: 0.055em;
                    margin-bottom: 11px;
                }

                .preset-value-row {
                    display: flex;
                    align-items: baseline;
                    gap: 8px;
                    flex-wrap: wrap;
                }

                .preset-value {
                    color: #ffffff;
                    font-size: 25px;
                    font-weight: 800;
                    letter-spacing: -0.025em;
                    line-height: 1.05;
                }

                .preset-unit {
                    display: inline-flex;
                    align-items: center;
                    padding: 3px 7px;
                    border-radius: 999px;
                    background: rgba(255,255,255,0.10);
                    border: 1px solid rgba(255,255,255,0.12);
                    color: rgba(255,255,255,0.68);
                    font-size: 11px;
                    font-weight: 700;
                    line-height: 1;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Mostra tutte le colonne del CSV nello stesso ordine del file.
        # Ogni campo viene mostrato come card visuale, senza usare una tabella.
        columns_in_order = list(presets_df.columns)
        cards_per_row = 4

        for i in range(0, len(columns_in_order), cards_per_row):
            row_columns = columns_in_order[i:i + cards_per_row]
            metric_cols = st.columns(cards_per_row)

            for card, column_name in zip(metric_cols, row_columns):
                label, unit = split_label_and_unit(column_name)
                value = format_preset_value(selected_row, column_name)

                card.markdown(
                    preset_card_html(label, value, unit),
                    unsafe_allow_html=True,
                )

        st.info(
            "In questo passaggio i presets sono solo consultabili. "
            "Nel passaggio successivo aggiungeremo il pulsante per caricarli nel calcolatore."
        )

    except FileNotFoundError:
        st.error("File Presets.csv non trovato. Mettilo nella stessa cartella dell'app.")
    except Exception as e:
        st.error(f"Errore nel caricamento dei presets: {e}")


with tab_calculator:
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"#### {t['bobina']}")
        diametro_aspo = st.number_input(t["diam_aspo"], value=450.0, step=10.0)
        spalla = st.number_input(t["spalla"], value=95.0, step=1.0)

    with colB:
        st.markdown(f"#### {t['tubo']}")
        rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
        spessore = st.number_input(t["isolamento"], value=7.0, step=1.0)
        lunghezza = st.number_input(t["lunghezza"], value=50.0, step=5.0)
        d_rame = COPPER_SIZES_MM[rame]

    with colC:
        st.markdown(f"#### {t['avvolg']}")
        passo_visuale = st.number_input(t["passo_assiale"], value=20.0, step=0.5)
        incremento_visuale = st.number_input(t["incremento"], value=20.0, step=0.5)
        rit_b = st.number_input(t["rit_min"], value=360.0, step=1.0)
        rit_t = st.number_input(t["rit_max"], value=360.0, step=1.0)

    d_tubo = d_rame + 2.0 * spessore

    # =========================
    # BUILD
    # =========================

    (
        world_contacts,
        local_points,
        theta_values,
        radius_values,
        z_values,
        mode_values,
        layer_values,
        length_values,
        deposited_len_mm,
    ) = simulate_winding_visual(
        d_aspo=diametro_aspo,
        spalla=spalla,
        d_tubo=d_tubo,
        passo=passo_visuale,
        incremento=incremento_visuale,
        rit_b=rit_b,
        rit_t=rit_t,
        lunghezza_m=lunghezza,
        gradi_start=gradi_start,
        deg_step=2.0,
    )

    visual_metrics = compute_metrics(local_points, d_tubo)

    # =========================
    # VIEWER RENDER
    # =========================

    st.divider()

    components.html(
        viewer(
            diametro_aspo,
            spalla,
            d_tubo,
            820,
            local_points.tolist(),
            theta_values.tolist(),
            radius_values.tolist(),
            z_values.tolist(),
            mode_values.tolist(),
            layer_values.tolist(),
            length_values.tolist(),
            guide_offset_x,
            lang,
        ),
        height=820,
    )

    # =========================
    # METRICS
    # =========================

    st.divider()

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
    m2.metric(t["metric2"], f"{passo_visuale:.2f} mm")
    m3.metric(t["metric3"], f"{incremento_visuale:.2f} mm")
    m4.metric(t["metric4"], f"{visual_metrics['diam_radiale']:.1f} mm")
    m5.metric(t["metric5"], f"{visual_metrics['max_xy_span']:.1f} mm")
    m6.metric(t["metric6"], f"{visual_metrics['wound_length_m']:.3f} m")

    if visual_metrics["max_xy_span"] > 750:
        st.warning(t["warning"])
