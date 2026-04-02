import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# 🌍 LANGUAGE (FLAGS)
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

col_lang1, col_lang2, _ = st.columns([1,1,6])

with col_lang1:
    if st.button("🇮🇹 IT"):
        st.session_state.lang = "IT"

with col_lang2:
    if st.button("🇺🇸 EN"):
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
        "rit_min": "Ritardo min (°)",
        "rit_max": "Ritardo max (°)",

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
        "rit_min": "Delay min (°)",
        "rit_max": "Delay max (°)",

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

col_logo, col_title = st.columns([1,7])

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
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    idx = max(0, min(idx, len(points)-2))

    p0, p1 = points[idx], points[idx+1]
    alpha = (target_length - cum[idx]) / np.linalg.norm(p1 - p0)

    return np.vstack([points[:idx+1], p0 + alpha*(p1-p0)])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2*np.pi))

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

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2*spessore_guaina_mm

    passo_assiale = max(passo_assiale, EPS)
    passo_radiale = max(passo_radiale, EPS)

    r0 = d_aspo_mm/2 + d_tubo/2
    r = r0

    z0, z1 = 0.0, spalla_mm
    theta = 0

    points = []

    base_transition_turn = 0.18

    while True:

        dz = z1 - z0
        turns = max(abs(dz)/passo_assiale, 0.1)
        dtheta = 2*np.pi*turns

        t = np.linspace(0, dtheta, int(turns*200)+80)

        theta_vals = theta + t
        z_vals = z0 + dz*(t/dtheta)

        x = r*np.cos(theta_vals)
        y = r*np.sin(theta_vals)

        layer = np.column_stack([x,y,z_vals])

        if len(points) > 0:
            layer = layer[1:]

        points.extend(layer.tolist())

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

        ritardo = np.random.uniform(ritardo_min_deg, ritardo_max_deg)
        extra_turn = ritardo / 360.0

        total_turn = base_transition_turn + extra_turn
        dtheta_trans = 2*np.pi*total_turn

        r_next = r + passo_radiale

        t_trans = np.linspace(0, dtheta_trans, int(total_turn*240)+60)

        theta_trans = theta + dtheta + t_trans

        s = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, len(t_trans)))
        r_trans = r + (r_next - r)*s

        z_trans = np.full_like(t_trans, z1)

        x = r_trans*np.cos(theta_trans)
        y = r_trans*np.sin(theta_trans)

        points.extend(np.column_stack([x,y,z_trans])[1:].tolist())

        theta += dtheta + dtheta_trans
        r = r_next
        z0, z1 = z1, z0

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

    path = trim_polyline(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    capes = int((r_max - r0)/passo_radiale) + 1
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
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"#### {t['bobina']}")
    diametro_aspo = st.number_input(t["diam_aspo"], 450.0)
    spalla = st.number_input(t["spalla"], 95.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], 7.0)
    lunghezza = st.number_input(t["lunghezza"], 50.0)

    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0)
    incremento_strato = st.number_input(t["incremento"], value=20.0)
    ritardo_min = st.number_input(t["rit_min"], 0.0, 720.0, 360.0)
    ritardo_max = st.number_input(t["rit_max"], 0.0, 720.0, 360.0)

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
)

# =========================
# METRICS
# =========================

st.divider()

m1,m2,m3,m4 = st.columns(4)

m1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
m2.metric(t["metric2"], f"{meta['PassoAssiale']:.2f} mm")
m3.metric(t["metric3"], f"{meta['IncrementoStrato']:.2f} mm")
m4.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")

if meta["DiametroEsterno"] > 750:
    st.warning(t["warning"])
