import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# 🌍 LANGUAGE
# =========================

lang = st.selectbox("Lingua / Language", ["IT", "EN"])

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
# UI COMPACTA EN COLUMNES
# =========================

colA, colB, colC, colD = st.columns(4)

# 🟦 BOBINA
with colA:
    st.markdown(f"#### {t['bobina']}")
    diametro_aspo = st.number_input(t["diam_aspo"], 450.0)
    spalla = st.number_input(t["spalla"], 95.0)

# 🟩 TUBO
with colB:
    st.markdown(f"#### {t['tubo']}")
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], 7.0)
    lunghezza = st.number_input(t["lunghezza"], 50.0)

    d_rame = COPPER_SIZES_MM[rame_label]
    d_tubo = d_rame + 2*spessore_guaina

# 🟧 AVVOLGIMENTO
with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo_assiale = st.number_input(t["passo_assiale"], value=float(d_tubo))
    incremento_strato = st.number_input(t["incremento"], value=float(d_tubo))

    ritardo_min = st.number_input(t["rit_min"], 0.0, 720.0, 360.0)
    ritardo_max = st.number_input(t["rit_max"], 0.0, 720.0, 360.0)

# ⚙️ VIEWER
with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# (la resta del teu codi NO CANVIA)
# =========================

# ... build_coil, viewer, etc EXACTAMENT IGUAL ...

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
