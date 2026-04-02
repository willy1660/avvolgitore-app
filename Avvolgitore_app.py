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
    r_tubo = d_tubo / 2.0

    passo_assiale = max(float(passo_assiale), EPS)
    passo_radiale = max(float(passo_radiale), EPS)
    spalla_mm = max(float(spalla_mm), EPS)

    r0 = d_aspo_mm / 2.0 + r_tubo

    z_min = r_tubo
    z_max = spalla_mm - r_tubo

    z = z_min
    theta = 0.0
    r = r0
    direction = 1

    theta_step = np.deg2rad(4.0)
    dz_dtheta = passo_assiale / (2.0 * np.pi)

    points = []

    def add_point():
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])

    add_point()

    while True:

        if len(points) > 2 and polyline_length(np.array(points)) >= lunghezza_mm:
            break

        # HELIX
        while True:
            theta_prev = theta
            z_prev = z

            theta += theta_step
            z += direction * dz_dtheta * theta_step

            if direction == 1 and z >= z_max:
                frac = (z_max - z_prev) / (z - z_prev + EPS)
                theta = theta_prev + frac * (theta - theta_prev)
                z = z_max
                add_point()
                break

            if direction == -1 and z <= z_min:
                frac = (z_min - z_prev) / (z - z_prev + EPS)
                theta = theta_prev + frac * (theta - theta_prev)
                z = z_min
                add_point()
                break

            add_point()

        # RITARDO CORRECTE (clau)
        at_top = direction == 1
        ritardo_deg = ritardo_max_deg if at_top else ritardo_min_deg
        theta_dwell = np.deg2rad(ritardo_deg)

        if theta_dwell > EPS:
            dwell_steps = max(8, int(np.ceil(ritardo_deg / 4.0)))
            theta_step_dwell = theta_dwell / dwell_steps

            theta_start = theta
            r_start = r

            for _ in range(dwell_steps):
                theta += theta_step_dwell

                frac = (theta - theta_start) / theta_dwell
                frac = max(0.0, min(1.0, frac))

                r_curr = r_start + passo_radiale * frac

                x = r_curr * np.cos(theta)
                y = r_curr * np.sin(theta)
                points.append([x, y, z])

            r = r_start + passo_radiale
        else:
            r += passo_radiale
            add_point()

        direction *= -1

    path = np.array(points)
    path = trim_polyline(path, lunghezza_mm)

    r_path = np.sqrt(path[:, 0]**2 + path[:, 1]**2)
    r_max = float(np.max(r_path))
    diam_ext = 2.0 * (r_max + r_tubo)

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
# VIEWER + UI (igual que abans)
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=50.0)
    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0)
    incremento_strato = st.number_input(t["incremento"], value=20.0)
    ritardo_min = st.number_input(t["rit_min"], value=180.0)
    ritardo_max = st.number_input(t["rit_max"], value=180.0)

with colD:
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

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

st.write(meta)
