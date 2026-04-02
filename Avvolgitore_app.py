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

col_logo, col_title = st.columns([1,6])

with col_logo:
    logo_path = resource_path("New Logo PDM - rame.png")
    st.image(logo_path, width=120)

with col_title:
    st.title("Avvolgimento")

COPPER_SIZES_MM = {
    "1/4": 6.35,
    "3/8": 9.52,
    "1/2": 12.70,
    "5/8": 15.88,
    "3/4": 19.05,
    "7/8": 22.23,
}

def polyline_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points)>1 else 0.0

def trim_polyline_to_length(points, target):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= target:
        return points
    idx = np.searchsorted(cum, target)-1
    p0,p1 = points[idx],points[idx+1]
    alpha = (target-cum[idx]) / np.linalg.norm(p1-p0)
    return np.vstack([points[:idx+1], p0+alpha*(p1-p0)])

def compute_total_turns(points):
    theta = np.unwrap(np.arctan2(points[:,1], points[:,0]))
    return np.sum(np.abs(np.diff(theta))) / (2*np.pi)

def points_to_sldcrv(points):
    return "\n".join(f"{p[0]} {p[1]} {p[2]}" for p in points).encode()

def build_coil_centerline(
    d_aspo_mm, spalla_mm, lunghezza_m,
    d_rame_mm, spessore_guaina_mm,
    compressione_pct, gap_axiale_mm,
    ritardo_min_deg, ritardo_max_deg
):

    lunghezza_mm = lunghezza_m*1000
    d_tubo = d_rame_mm + 2*spessore_guaina_mm

    passo_radiale = d_tubo*(1-compressione_pct/100)
    passo_assiale = d_tubo + gap_axiale_mm

    r = d_aspo_mm/2 + d_tubo/2
    z0,z1 = 0,spalla_mm
    theta = 0
    points = []

    while True:

        dz = z1-z0
        giri = max(abs(dz)/passo_assiale,0.1)
        dtheta = 2*math.pi*giri

        t = np.linspace(0,dtheta,int(giri*120))
        x = r*np.cos(theta+t)
        y = r*np.sin(theta+t)
        z = z0 + dz*(t/dtheta)

        layer = np.column_stack([x,y,z])
        if points: layer = layer[1:]
        points.extend(layer.tolist())

        theta += dtheta

        if polyline_length(np.array(points))>=lunghezza_mm:
            break

        # ==============================
        # RITARDO CORREGIT (AMB RADIAL)
        # ==============================
        if ritardo_max_deg > 0:

            ritardo_deg = np.random.uniform(ritardo_min_deg, ritardo_max_deg)
            dtheta_delay = math.radians(ritardo_deg)

            r_next = r + passo_radiale

            t = np.linspace(0,dtheta_delay,40)
            s = np.linspace(0,1,40)

            r_vals = r + (r_next - r)*s
            theta_vals = theta + t
            z_vals = np.full_like(theta_vals,z1)

            x = r_vals*np.cos(theta_vals)
            y = r_vals*np.sin(theta_vals)

            delay = np.column_stack([x,y,z_vals])[1:]

            points.extend(delay.tolist())

            theta += dtheta_delay
            r = r_next   # 🔥 clau

        else:
            r = r + passo_radiale

        z0,z1 = z1,z0

    path = trim_polyline_to_length(np.array(points),lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    total_turns = compute_total_turns(path)

    meta = {
        "DiametroTubo":d_tubo,
        "PassoRadiale":passo_radiale,
        "PassoAssiale":passo_assiale,
        "DiametroEsterno":2*(r_max+d_tubo/2),
        "LunghezzaM":polyline_length(path)/1000,
        "Capes":int((r_max - (d_aspo_mm/2))/passo_radiale)+1,
        "VolteTotali":total_turns,
        "VoltePerCapa":total_turns,
    }

    return path,meta

# ================= UI =================

c1,c2,c3,c4,c5=st.columns(5)
diametro_aspo=c1.number_input("Diametro aspo",450.0)
spalla=c2.number_input("Spalla",95.0)
lunghezza=c3.number_input("Lunghezza",50.0)
rame_label=c4.selectbox("Rame",list(COPPER_SIZES_MM.keys()))
spessore_guaina=c5.number_input("Guaina",7.0)

c6,c7,c8,c9=st.columns(4)
compressione=c6.slider("Compressione",0.0,20.0,0.0)
gap=c7.number_input("Gap",0.0)
ritardo_min=c8.number_input("Ritardo min",0.0)
ritardo_max=c9.number_input("Ritardo max",0.0)

c10,c11,c12=st.columns(3)
animazione=c10.checkbox("Animazione",True)
velocita=c11.slider("Velocità",0.1,5.0,1.0)
altezza=c12.slider("Altezza",400,900,700)

path,meta=build_coil_centerline(
    diametro_aspo,spalla,lunghezza,
    COPPER_SIZES_MM[rame_label],
    spessore_guaina,compressione,
    gap,ritardo_min,ritardo_max
)

components.html(build_viewer_html(path,meta["DiametroTubo"],altezza,animazione,velocita),height=altezza)

st.write(meta)
