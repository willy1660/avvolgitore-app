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

col_logo, col_title = st.columns([1,6])

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

EPS = 1e-9

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
    alpha = (target - cum[idx]) / seg_len
    p_cut = p0 + alpha * (p1 - p0)

    return np.vstack([points[:idx + 1], p_cut])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    dtheta = np.diff(theta)
    return np.sum(np.abs(dtheta)) / (2*np.pi)

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
    ritardo_inv_max,
    ritardo_inv_min,
):

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2 * spessore_guaina_mm

    passo_radiale = d_tubo * (1 - compressione_pct / 100)
    passo_assiale = d_tubo + gap_axiale_mm

    r0 = d_aspo_mm/2 + d_tubo/2
    r = r0

    z0 = 0
    z1 = spalla_mm

    theta = 0
    points = []

    while True:

        dz = z1 - z0
        giri = max(abs(dz) / passo_assiale, 0.1)
        dtheta = 2 * math.pi * giri

        t = np.linspace(0, dtheta, max(100, int(giri*120)))

        theta_vals = theta + t
        z_vals = z0 + dz * t / dtheta

        x = r * np.cos(theta_vals)
        y = r * np.sin(theta_vals)

        layer = np.column_stack([x,y,z_vals])
        if len(points)>0:
            layer = layer[1:]

        points.extend(layer.tolist())
        theta += dtheta

        # 🔥 RITARDO
        ritardo = math.radians(ritardo_inv_max if z1 > z0 else ritardo_inv_min)

        if ritardo > 0:
            t_delay = np.linspace(0, ritardo, 30)
            theta_vals = theta + t_delay
            z_vals = np.full_like(theta_vals, z1)

            x = r * np.cos(theta_vals)
            y = r * np.sin(theta_vals)

            delay_pts = np.column_stack([x,y,z_vals])[1:]
            points.extend(delay_pts.tolist())

            theta += ritardo

        if polyline_length(np.array(points)) >= lunghezza_mm:
            break

        r += passo_radiale
        z0, z1 = z1, z0

    path = trim_polyline_to_length(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))

    return path, {
        "DiametroTubo": d_tubo,
        "PassoRadiale": passo_radiale,
        "PassoAssiale": passo_assiale,
        "DiametroEsterno": 2*(r_max + d_tubo/2),
        "LunghezzaM": polyline_length(path)/1000,
        "Capes": 1,
        "VolteTotali": compute_total_turns(path),
        "VoltePerCapa": compute_total_turns(path),
    }

# =========================================================
# UI
# =========================================================

c1,c2,c3,c4,c5=st.columns(5)

with c1:
    diametro_aspo=st.number_input("Diametro aspo (mm)",value=450.0)

with c2:
    spalla=st.number_input("Spalla (mm)",value=95.0)

with c3:
    lunghezza=st.number_input("Lunghezza (m)",value=50.0)

with c4:
    rame_label=st.selectbox("Diametro rame",list(COPPER_SIZES_MM.keys()))

with c5:
    spessore_guaina=st.number_input("Spessore guaina (mm)",value=7.0)

c6,c7,c8,c9,c10=st.columns(5)

with c6:
    compressione=st.slider("Compressione %",0.0,20.0,0.0)

with c7:
    gap=st.number_input("Gap axiale (mm)",value=0.0)

with c8:
    ritardo_inv_max=st.number_input("Ritardo inv max (°)",value=0.0)

with c9:
    ritardo_inv_min=st.number_input("Ritardo inv min (°)",value=0.0)

with c10:
    altezza=st.slider("Altezza viewer",400,900,700)

c11,c12=st.columns(2)

with c11:
    animazione=st.checkbox("Animazione avvolgimento",True)

with c12:
    velocita=st.slider("Velocità animazione",0.1,5.0,1.0)

d_rame=COPPER_SIZES_MM[rame_label]

path,meta=build_coil_centerline(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore_guaina,
    compressione,
    gap,
    ritardo_inv_max,
    ritardo_inv_min
)

components.html("<h3>Viewer OK</h3>",height=altezza)

st.write(meta)
