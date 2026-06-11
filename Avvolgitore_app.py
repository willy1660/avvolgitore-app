import os
import glob
import json
import html
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
        "language": "🌍 Lingua",
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
        "fullscreen": "Schermo intero",
        "exit": "Esci",
        "progress": "Progresso",
        "speed": "Velocità",
        "spool": "Aspo",
        "visible": "Visibile",
        "transparent": "Trasparente",
        "hidden": "Nascosto",
        "tube_color": "Tubo",
        "gelwhite": "Gelwhite",
        "gelblack": "Gelblack",
        "grid": "Griglia",
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
        "tab_presets": "📦 Preset",
        "tab_calculator": "🧮 Calcolatore / Render",
        "presets_title": "### 📦 Preset prodotto",
        "presets_loaded": "preset caricati correttamente da Presets.csv",
        "select_product": "Seleziona prodotto",
        "preset_sheet": "Scheda preset",
        "preset_subtitle": "Configurazione tecnica prodotto · valori caricati da CSV",
        "csv_params": "#### Parametri del preset",
        "presets_readonly": "In questo passaggio i preset sono solo consultabili. Nel passaggio successivo aggiungeremo il pulsante per caricarli nel calcolatore.",
        "presets_file_missing": "File Presets.csv non trovato. Mettilo nella stessa cartella dell'app.",
        "presets_load_error": "Errore nel caricamento dei preset",
        "preset_visual_title": "Anteprima tecnica",
        "load_to_calculator": "Carica nel calcolatore",
        "loaded_to_calculator": "Preset caricato nel calcolatore",
        "linked_params": "Parametri collegati al render",
        "non_render_params": "Parametri macchina consultivi",
        "calculator_loaded_from": "Valori caricati dal preset",
        "preset_render_note": "I parametri del preset sono stati caricati nel calcolatore. Puoi modificarli liberamente senza cambiare il CSV.",
        "preset_loaded_ok": "Preset {name} caricato correttamente.",
        "active_preset": "Preset attivo",
        "pallet_title": "Verifica pallet 750 × 750 mm",
        "pallet_subtitle": "Controllo visivo dell'ingombro del rotolo appoggiato in piano sul pallet.",
        "pallet_size": "Pallet",
        "coil_footprint": "Ingombro rotolo",
        "pallet_status_ok": "OK su pallet",
        "pallet_status_over": "Fuori sagoma",
        "pallet_overhang": "Sbalzo totale",
        "pallet_free_margin": "Margine residuo",
        "packaging_tab": "📦 Packaging",
        "render_tab": "🎥 Render",
        "packaging_title": "Packaging",
        "packaging_mode": "Tipo packaging",
        "packaging_box": "Scatola 750 × 750 × 1350 mm",
        "packaging_tower": "Torre su pallet",
        "roll_count": "Numero rotoli",
        "box_height": "Altezza scatola",
        "pallet_height": "Altezza pallet",
        "total_height": "Altezza totale con pallet",
        "roll_stack_height": "Altezza rotoli",
        "height_margin": "Margine altezza",
        "height_over": "Superamento altezza",
        "box_fit_ok": "Packaging OK",
        "box_fit_over": "Fuori limite",
        "box_fit_note": "Calcolo basato sui parametri attuali del render.",
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
        "tab_presets": "📦 Presets",
        "tab_calculator": "🧮 Calculator / Render",
        "presets_title": "### 📦 Product presets",
        "presets_loaded": "presets loaded correctly from Presets.csv",
        "select_product": "Select product",
        "preset_sheet": "Preset sheet",
        "preset_subtitle": "Product technical configuration · values loaded from CSV",
        "csv_params": "#### Preset parameters",
        "presets_readonly": "At this stage, presets are read-only. In the next step, we will add the button to load them into the calculator.",
        "presets_file_missing": "Presets.csv file not found. Put it in the same folder as the app.",
        "presets_load_error": "Error loading presets",
        "preset_visual_title": "Technical preview",
        "load_to_calculator": "Load into calculator",
        "loaded_to_calculator": "Preset loaded into calculator",
        "linked_params": "Parameters linked to the render",
        "non_render_params": "Consultative machine parameters",
        "calculator_loaded_from": "Values loaded from preset",
        "preset_render_note": "The preset parameters have been loaded into the calculator. You can edit them freely without changing the CSV.",
        "preset_loaded_ok": "Preset {name} loaded correctly.",
        "active_preset": "Active preset",
        "pallet_title": "750 × 750 mm pallet check",
        "pallet_subtitle": "Visual check of the coil footprint when laid flat on the pallet.",
        "pallet_size": "Pallet",
        "coil_footprint": "Coil footprint",
        "pallet_status_ok": "Fits on pallet",
        "pallet_status_over": "Out of bounds",
        "pallet_overhang": "Total overhang",
        "pallet_free_margin": "Remaining margin",
        "packaging_tab": "📦 Packaging",
        "render_tab": "🎥 Render",
        "packaging_title": "Packaging",
        "packaging_mode": "Packaging type",
        "packaging_box": "Box 750 × 750 × 1350 mm",
        "packaging_tower": "Tower on pallet",
        "roll_count": "Number of coils",
        "box_height": "Box height",
        "pallet_height": "Pallet height",
        "total_height": "Total height with pallet",
        "roll_stack_height": "Coil stack height",
        "height_margin": "Height margin",
        "height_over": "Height over limit",
        "box_fit_ok": "Packaging OK",
        "box_fit_over": "Out of bounds",
        "box_fit_note": "Calculation based on the current render parameters.",
    },
}

PARAM_LABELS = {
    "IT": {
        "Prodotto": "Prodotto",
        "Diametro Rame": "Diametro rame",
        "Spessore Guaina (mm)": "Spessore guaina (mm)",
        "Diametro esterno Guaina (mm)": "Diametro esterno guaina (mm)",
        "Lunghezza (m)": "Lunghezza (m)",
        "Velocita linea (m/min)": "Velocità linea (m/min)",
        "Boccole rulliera adrizzatubo": "Boccole rulliera adrizzatubo",
        "Boccola uscita rulliera": "Boccola uscita rulliera",
        "Rulliera adrizzatubo": "Rulliera adrizzatubo",
        "Boccola uscita traino": "Boccola uscita traino",
        "Rulli convogliatore (mm)": "Rulli convogliatore (mm)",
        "Rulli estrusore(mm)": "Rulli estrusore (mm)",
        "Ruote godronatore": "Ruote godronatore",
        "Soffiatori aria (mm)": "Soffiatori aria (mm)",
        "Rulli avvolgitore (mm)": "Rulli avvolgitore (mm)",
        "Paleta ferma coda (mm)": "Paletta ferma coda (mm)",
        "Guidatubo (mm)": "Guidatubo (mm)",
        "Spalla (mm)": "Spalla (mm)",
        "Diametro aspo (mm)": "Diametro aspo (mm)",
        "Nº Spire": "Nº spire",
        "Interasse regetta (mm)": "Interasse reggetta (mm)",
        "Velocita recupero (m/min)": "Velocità recupero (m/min)",
        "Quota start pinza (mm)": "Quota start pinza (mm)",
        "Quota coda tubo (mm)": "Quota coda tubo (mm)",
        "Quota chiusura morsa coda (mm)": "Quota chiusura morsa coda (mm)",
        "Ritardo invers max (º)": "Ritardo invers. max (º)",
        "Ritardo invers min (º)": "Ritardo invers. min (º)",
        "Quota massima (mm)": "Quota massima (mm)",
        "Quota minima (mm)": "Quota minima (mm)",
        "Passo (mm)": "Passo (mm)",
        "Incremento strato (mm)": "Incremento strato (mm)",
        "Coppia lavoro (%)": "Coppia lavoro (%)",
        "Riduzione coppia (%)": "Riduzione coppia (%)",
        "Coppia recupero (%)": "Coppia recupero (%)",
    },
    "EN": {
        "Prodotto": "Product",
        "Diametro Rame": "Copper diameter",
        "Spessore Guaina (mm)": "Foam thickness (mm)",
        "Diametro esterno Guaina (mm)": "Outer foam diameter (mm)",
        "Lunghezza (m)": "Length (m)",
        "Velocita linea (m/min)": "Line speed (m/min)",
        "Boccole rulliera adrizzatubo": "Straightener roller bushings",
        "Boccola uscita rulliera": "Roller outlet bushing",
        "Rulliera adrizzatubo": "Tube straightener rollers",
        "Boccola uscita traino": "Puller outlet bushing",
        "Rulli convogliatore (mm)": "Conveyor rollers (mm)",
        "Rulli estrusore(mm)": "Extruder rollers (mm)",
        "Ruote godronatore": "Knurling wheels",
        "Soffiatori aria (mm)": "Air blowers (mm)",
        "Rulli avvolgitore (mm)": "Coiler rollers (mm)",
        "Paleta ferma coda (mm)": "Tail stop paddle (mm)",
        "Guidatubo (mm)": "Tube guide (mm)",
        "Spalla (mm)": "Width (mm)",
        "Diametro aspo (mm)": "Spool diameter (mm)",
        "Nº Spire": "No. of turns",
        "Interasse regetta (mm)": "Strap spacing (mm)",
        "Velocita recupero (m/min)": "Recovery speed (m/min)",
        "Quota start pinza (mm)": "Clamp start position (mm)",
        "Quota coda tubo (mm)": "Tube tail position (mm)",
        "Quota chiusura morsa coda (mm)": "Tail clamp closing position (mm)",
        "Ritardo invers max (º)": "Max reverse delay (º)",
        "Ritardo invers min (º)": "Min reverse delay (º)",
        "Quota massima (mm)": "Maximum position (mm)",
        "Quota minima (mm)": "Minimum position (mm)",
        "Passo (mm)": "Pitch (mm)",
        "Incremento strato (mm)": "Layer increment (mm)",
        "Coppia lavoro (%)": "Working torque (%)",
        "Riduzione coppia (%)": "Torque reduction (%)",
        "Coppia recupero (%)": "Recovery torque (%)",
    },
}


def param_label(column_name, language):
    return PARAM_LABELS.get(language, {}).get(column_name, column_name)


def render_preset_param_cards(title, column_names, selected_row, language, cards_per_row=4):
    st.markdown(
        """
        <style>
        .preset-param-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 10px 24px rgba(0,0,0,0.10);
            margin-bottom: 10px;
        }
        .preset-param-label {
            font-size: 15px;
            line-height: 1.35;
            font-weight: 600;
            color: rgba(250,250,250,0.88);
            margin-bottom: 14px;
        }
        .preset-param-value {
            font-size: 28px;
            line-height: 1.08;
            font-weight: 800;
            color: #ffffff;
            word-break: break-word;
        }
        @media (max-width: 900px) {
            .preset-param-card {
                min-height: 104px;
                padding: 14px 16px;
            }
            .preset-param-label {
                font-size: 14px;
                margin-bottom: 10px;
            }
            .preset-param-value {
                font-size: 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"##### {title}")
    for i in range(0, len(column_names), cards_per_row):
        row_columns = column_names[i:i + cards_per_row]
        ui_cols = st.columns(cards_per_row)

        for ui_col, column_name in zip(ui_cols, row_columns):
            value = format_preset_value(selected_row[column_name]) if column_name in selected_row.index else "-"
            label = param_label(column_name, language)
            ui_col.markdown(
                f"""
                <div class="preset-param-card">
                    <div class="preset-param-label">{html.escape(str(label))}</div>
                    <div class="preset-param-value">{html.escape(str(value))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )




def make_pallet_visual(coil_diameter_mm, pallet_size_mm, language):
    labels = {
        "IT": {"title": "Vista dall'alto", "pallet": "Pallet", "coil": "Rotolo"},
        "EN": {"title": "Top view", "pallet": "Pallet", "coil": "Coil"},
    }[language]

    ratio = coil_diameter_mm / pallet_size_mm if pallet_size_mm > 0 else 1.0
    circle_r = max(12.0, 90.0 * ratio)
    coil_fill = "var(--warn-fill)" if ratio > 1.0 else "var(--coil-fill)"
    coil_stroke = "var(--warn-stroke)" if ratio > 1.0 else "var(--coil-stroke)"

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    :root {{
        --bg: transparent;
        --card-bg: rgba(255,255,255,0.04);
        --card-border: rgba(255,255,255,0.10);
        --text: #f8fafc;
        --muted: rgba(248,250,252,0.70);
        --pallet: rgba(194,154,106,0.38);
        --pallet-stroke: rgba(223,190,145,0.85);
        --coil-fill: rgba(96,165,250,0.18);
        --coil-stroke: rgba(191,219,254,0.96);
        --warn-fill: rgba(248,113,113,0.18);
        --warn-stroke: rgba(252,165,165,0.98);
    }}
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:var(--text); background:var(--bg); }}
    .wrap {{ border:1px solid var(--card-border); border-radius:18px; padding:18px; background:var(--card-bg); }}
    .title {{ font-size:13px; font-weight:800; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; }}
    .legend {{ display:flex; gap:16px; margin-top:10px; flex-wrap:wrap; font-size:12px; color:var(--muted); }}
    .legend-item {{ display:flex; align-items:center; gap:8px; }}
    .swatch {{ width:14px; height:14px; border-radius:4px; display:inline-block; }}
</style>
</head>
<body>
<div class="wrap">
    <div class="title">{labels['title']}</div>
    <svg viewBox="0 0 320 250" width="100%" height="auto" role="img" aria-label="Pallet preview">
        <rect x="70" y="35" width="180" height="180" rx="8" fill="var(--pallet)" stroke="var(--pallet-stroke)" stroke-width="2"/>
        <line x1="70" y1="95" x2="250" y2="95" stroke="var(--pallet-stroke)" opacity="0.35"/>
        <line x1="70" y1="155" x2="250" y2="155" stroke="var(--pallet-stroke)" opacity="0.35"/>
        <line x1="130" y1="35" x2="130" y2="215" stroke="var(--pallet-stroke)" opacity="0.35"/>
        <line x1="190" y1="35" x2="190" y2="215" stroke="var(--pallet-stroke)" opacity="0.35"/>
        <circle cx="160" cy="125" r="{circle_r:.2f}" fill="{coil_fill}" stroke="{coil_stroke}" stroke-width="3"/>
        <line x1="70" y1="230" x2="250" y2="230" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
        <line x1="70" y1="220" x2="70" y2="240" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
        <line x1="250" y1="220" x2="250" y2="240" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
        <text x="160" y="225" text-anchor="middle" fill="#f8fafc" font-size="14" font-weight="800">{pallet_size_mm:.0f} mm</text>
        <line x1="{160-circle_r:.2f}" y1="22" x2="{160+circle_r:.2f}" y2="22" stroke="#bfdbfe" stroke-width="3" stroke-linecap="round"/>
        <line x1="{160-circle_r:.2f}" y1="14" x2="{160-circle_r:.2f}" y2="30" stroke="#bfdbfe" stroke-width="3" stroke-linecap="round"/>
        <line x1="{160+circle_r:.2f}" y1="14" x2="{160+circle_r:.2f}" y2="30" stroke="#bfdbfe" stroke-width="3" stroke-linecap="round"/>
        <text x="160" y="16" text-anchor="middle" fill="#bfdbfe" font-size="14" font-weight="800">{coil_diameter_mm:.1f} mm</text>
    </svg>
    <div class="legend">
        <div class="legend-item"><span class="swatch" style="background:var(--pallet);"></span>{labels['pallet']}</div>
        <div class="legend-item"><span class="swatch" style="background:{coil_fill}; border:1px solid {coil_stroke};"></span>{labels['coil']}</div>
    </div>
</div>
</body>
</html>
"""



def make_packaging_visual(
    coil_diameter_mm,
    roll_height_mm,
    roll_count,
    pallet_size_mm,
    pallet_height_mm,
    box_height_mm,
    mode,
    language,
):
    labels = {
        "IT": {
            "side": "Vista laterale",
            "top": "Vista dall'alto",
            "box": "Scatola",
            "tower": "Torre",
            "pallet": "Pallet",
            "coil": "Rotolo",
            "total": "Totale",
            "height": "Altezza",
        },
        "EN": {
            "side": "Side view",
            "top": "Top view",
            "box": "Box",
            "tower": "Tower",
            "pallet": "Pallet",
            "coil": "Coil",
            "total": "Total",
            "height": "Height",
        },
    }[language]

    is_box = mode == "box"
    package_height_mm = box_height_mm if is_box else max(box_height_mm, roll_count * roll_height_mm)
    stack_height_mm = roll_count * roll_height_mm
    total_height_mm = pallet_height_mm + stack_height_mm

    # Scale dimensions to a stable schematic drawing.
    max_visual_height_mm = max(box_height_mm, stack_height_mm, 1)
    box_h_px = 150
    pallet_h_px = max(16, min(32, pallet_height_mm / max_visual_height_mm * box_h_px))
    roll_h_px = max(6, roll_height_mm / max_visual_height_mm * box_h_px)
    stack_h_px = roll_h_px * roll_count
    stack_h_px = min(stack_h_px, 188)

    box_top_y = 24
    box_bottom_y = box_top_y + box_h_px
    pallet_y = box_bottom_y + 10

    top_ratio = coil_diameter_mm / pallet_size_mm if pallet_size_mm > 0 else 1.0
    top_r = max(8.0, 58.0 * top_ratio)
    coil_fill = "var(--warn-fill)" if top_ratio > 1.0 else "var(--coil-fill)"
    coil_stroke = "var(--warn-stroke)" if top_ratio > 1.0 else "var(--coil-stroke)"

    roll_svgs = []
    base_y = box_bottom_y
    for i in range(int(roll_count)):
        cy = base_y - (i + 0.5) * roll_h_px
        if cy < 8:
            continue
        roll_svgs.append(
            f'<ellipse cx="118" cy="{cy:.2f}" rx="82" ry="{max(4, roll_h_px * 0.42):.2f}" fill="var(--coil-fill-2)" stroke="var(--coil-stroke)" stroke-width="2"/>'
        )
        roll_svgs.append(
            f'<ellipse cx="118" cy="{cy - max(1, roll_h_px * 0.12):.2f}" rx="78" ry="{max(3, roll_h_px * 0.30):.2f}" fill="var(--coil-highlight)" opacity="0.88"/>'
        )

    box_stroke = "var(--ok)" if stack_height_mm <= box_height_mm else "var(--warn-stroke)"
    title = labels["box"] if is_box else labels["tower"]

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    :root {{
        --bg: transparent;
        --card-bg: rgba(255,255,255,0.04);
        --card-border: rgba(255,255,255,0.10);
        --text: #f8fafc;
        --muted: rgba(248,250,252,0.70);
        --pallet: rgba(194,154,106,0.50);
        --pallet-stroke: rgba(223,190,145,0.92);
        --box-fill: rgba(148,163,184,0.08);
        --box-stroke: rgba(203,213,225,0.78);
        --coil-fill: rgba(96,165,250,0.18);
        --coil-fill-2: rgba(226,232,240,0.84);
        --coil-highlight: rgba(248,250,252,0.96);
        --coil-stroke: rgba(191,219,254,0.96);
        --warn-fill: rgba(248,113,113,0.18);
        --warn-stroke: rgba(252,165,165,0.98);
        --ok: rgba(74,222,128,0.98);
    }}
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:var(--text); background:var(--bg); }}
    .wrap {{ border:1px solid var(--card-border); border-radius:18px; padding:18px; background:var(--card-bg); }}
    .grid {{ display:grid; grid-template-columns: 1.25fr 0.75fr; gap:18px; align-items:center; }}
    .title {{ font-size:13px; font-weight:800; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px; }}
    .caption {{ font-size:12px; fill:var(--muted); font-weight:700; }}
</style>
</head>
<body>
<div class="wrap">
    <div class="title">{title}</div>
    <div class="grid">
        <svg viewBox="0 0 260 245" width="100%" height="auto" role="img" aria-label="Packaging side view">
            <text x="20" y="14" class="caption">{labels['side']}</text>
            <rect x="34" y="{box_top_y}" width="168" height="{box_h_px}" rx="8" fill="{'var(--box-fill)' if is_box else 'none'}" stroke="{box_stroke}" stroke-width="3" stroke-dasharray="{'0' if is_box else '8 7'}"/>
            {''.join(roll_svgs)}
            <rect x="18" y="{pallet_y}" width="200" height="{pallet_h_px}" rx="5" fill="var(--pallet)" stroke="var(--pallet-stroke)" stroke-width="2"/>
            <line x1="228" y1="{box_top_y}" x2="228" y2="{pallet_y + pallet_h_px}" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
            <line x1="218" y1="{box_top_y}" x2="238" y2="{box_top_y}" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
            <line x1="218" y1="{pallet_y + pallet_h_px}" x2="238" y2="{pallet_y + pallet_h_px}" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
            <text x="242" y="{(box_top_y + pallet_y + pallet_h_px)/2:.1f}" transform="rotate(90 242 {(box_top_y + pallet_y + pallet_h_px)/2:.1f})" fill="#f8fafc" font-size="13" font-weight="800" text-anchor="middle">{total_height_mm:.0f} mm</text>
            <text x="118" y="{pallet_y + pallet_h_px + 18}" fill="var(--muted)" font-size="12" font-weight="700" text-anchor="middle">{labels['pallet']} {pallet_height_mm:.0f} mm</text>
        </svg>

        <svg viewBox="0 0 220 220" width="100%" height="auto" role="img" aria-label="Packaging top view">
            <text x="22" y="18" class="caption">{labels['top']}</text>
            <rect x="40" y="38" width="140" height="140" rx="8" fill="var(--pallet)" stroke="var(--pallet-stroke)" stroke-width="2"/>
            <circle cx="110" cy="108" r="{top_r:.2f}" fill="{coil_fill}" stroke="{coil_stroke}" stroke-width="3"/>
            <line x1="40" y1="194" x2="180" y2="194" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
            <line x1="40" y1="184" x2="40" y2="204" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
            <line x1="180" y1="184" x2="180" y2="204" stroke="#f8fafc" stroke-width="3" stroke-linecap="round"/>
            <text x="110" y="190" fill="#f8fafc" font-size="13" font-weight="800" text-anchor="middle">{pallet_size_mm:.0f} mm</text>
            <text x="110" y="211" fill="var(--muted)" font-size="12" font-weight="700" text-anchor="middle">{labels['coil']} Ø {coil_diameter_mm:.0f} mm</text>
        </svg>
    </div>
</div>
</body>
</html>
"""

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


def safe_value(row, column, suffix=""):
    if column not in row.index:
        return "-"

    value = row[column]

    if pd.isna(value):
        return "-"

    return f"{value}{suffix}"


def parse_float_value(value, default=0.0):
    if pd.isna(value):
        return float(default)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", ".")

    # Keep the first number if a machine field contains a range like "75-80".
    if "-" in text and not text.startswith("-"):
        text = text.split("-")[0].strip()

    try:
        return float(text)
    except ValueError:
        return float(default)


def format_preset_value(value):
    if pd.isna(value):
        return "-"

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    return str(value)


def make_preset_visual(row, language):
    rame = safe_value(row, "Diametro Rame")
    d_rame = COPPER_SIZES_MM.get(str(rame), parse_float_value(row.get("Diametro Rame", 0.0), 0.0))
    spessore = parse_float_value(row.get("Spessore Guaina (mm)", 0.0), 0.0)
    d_tubo = parse_float_value(row.get("Diametro esterno Guaina (mm)", d_rame + 2.0 * spessore), d_rame + 2.0 * spessore)
    lunghezza = parse_float_value(row.get("Lunghezza (m)", 0.0), 0.0)
    aspo = parse_float_value(row.get("Diametro aspo (mm)", 0.0), 0.0)
    spalla = parse_float_value(row.get("Spalla (mm)", 0.0), 0.0)
    passo = parse_float_value(row.get("Passo (mm)", 0.0), 0.0)
    incremento = parse_float_value(row.get("Incremento strato (mm)", 0.0), 0.0)
    velocita_linea = parse_float_value(row.get("Velocita linea (m/min)", 0.0), 0.0)
    soffiatori = safe_value(row, "Soffiatori aria (mm)")
    rulli = safe_value(row, "Rulli avvolgitore (mm)")
    if rulli == "-":
        rulli = safe_value(row, "Rulli convogliatore (mm)")
    paletta = safe_value(row, "Paleta ferma coda (mm)")

    labels = {
        "IT": {
            "tube_section": "Anteprima tecnica · Sezione tubo",
            "coil": "Anteprima tecnica · Schema avvolgimento",
            "copper": "Rame",
            "foam": "Guaina",
            "outer": "Ø esterno",
            "length": "Lunghezza",
            "line_speed": "Velocità linea",
            "air": "Soffiatori aria",
            "spool": "Ø aspo",
            "width": "Spalla",
            "pitch": "Passo",
            "layer": "Incremento strato",
            "rollers": "Rulli",
            "tail": "Paletta ferma coda",
            "tube_note": "Sezione 2D semplificata del tubo isolato",
            "coil_note": "Vista laterale 2D del rotolo avvolto",
            "outer_dim": "Diametro esterno",
            "insulation": "Spessore guaina",
        },
        "EN": {
            "tube_section": "Technical preview · Tube section",
            "coil": "Technical preview · Coiling layout",
            "copper": "Copper",
            "foam": "Foam",
            "outer": "Outer Ø",
            "length": "Length",
            "line_speed": "Line speed",
            "air": "Air blowers",
            "spool": "Spool Ø",
            "width": "Width",
            "pitch": "Pitch",
            "layer": "Layer increment",
            "rollers": "Rollers",
            "tail": "Tail stop paddle",
            "tube_note": "Simplified 2D section of the insulated tube",
            "coil_note": "2D side view of the wound coil",
            "outer_dim": "Outer diameter",
            "insulation": "Foam thickness",
        },
    }[language]

    def v(value):
        return html.escape(format_preset_value(value))

    copper_label = html.escape(str(rame))
    line_speed_label = f"{v(velocita_linea)} m/min"
    soffiatori_label = html.escape(str(soffiatori))
    if soffiatori_label != "-" and "mm" not in soffiatori_label.lower():
        soffiatori_label += " mm"
    rulli_label = html.escape(str(rulli))
    if rulli_label != "-" and "mm" not in rulli_label.lower():
        rulli_label += " mm"
    paletta_label = html.escape(str(paletta))
    if paletta_label != "-" and "mm" not in paletta_label.lower():
        paletta_label += " mm"

    return f'''
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    :root {{
        --bg: transparent;
        --card-bg: rgba(255,255,255,0.72);
        --card-border: rgba(15,23,42,0.10);
        --text: #0f172a;
        --muted: rgba(15,23,42,0.64);
        --line: rgba(15,23,42,0.18);
        --shadow: 0 10px 24px rgba(15,23,42,0.08);
        --accent-soft: rgba(37,99,235,0.08);
        --copper: #b8602b;
        --copper-light: #e4a56f;
        --foam: #e7e2d8;
        --foam-stroke: #7e8691;
        --coil-top: #f3f5f7;
        --coil-mid: #d7dbe0;
        --coil-dark: #b9c0c8;
        --white-line: #f8fafc;
    }}
    html[data-theme="dark"] {{
        --card-bg: rgba(17,24,39,0.82);
        --card-border: rgba(255,255,255,0.10);
        --text: #f8fafc;
        --muted: rgba(248,250,252,0.70);
        --line: rgba(255,255,255,0.16);
        --shadow: 0 16px 34px rgba(0,0,0,0.24);
        --accent-soft: rgba(96,165,250,0.10);
        --foam: #d9d3c7;
        --foam-stroke: #8d96a0;
        --coil-top: #f1f3f5;
        --coil-mid: #cfd5dc;
        --coil-dark: #aeb6bf;
        --white-line: #f8fafc;
    }}
    html, body {{
        margin:0;
        padding:0;
        background:var(--bg);
        font-family: Arial, Helvetica, sans-serif;
        color:var(--text);
    }}
    .preset-preview {{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:18px;
        padding:2px;
    }}
    .card {{
        background:var(--card-bg);
        border:1px solid var(--card-border);
        box-shadow:var(--shadow);
        backdrop-filter: blur(8px);
        border-radius:18px;
        padding:18px;
        min-height:330px;
        box-sizing:border-box;
    }}
    .title {{
        font-size:13px;
        font-weight:800;
        letter-spacing:0.08em;
        text-transform:uppercase;
        color:var(--muted);
        margin-bottom:6px;
    }}
    .subtitle {{
        font-size:12px;
        color:var(--muted);
        margin-bottom:14px;
    }}
    .layout {{
        display:grid;
        grid-template-columns: 1.08fr 0.92fr;
        gap:18px;
        align-items:start;
    }}
    .drawing {{
        background:var(--accent-soft);
        border:1px solid var(--line);
        border-radius:16px;
        padding:12px;
    }}
    svg {{ display:block; width:100%; height:auto; }}
    .metrics {{ display:grid; gap:8px; }}
    .metric {{
        display:flex;
        justify-content:space-between;
        gap:12px;
        align-items:baseline;
        padding:10px 0;
        border-bottom:1px solid var(--line);
    }}
    .metric:last-child {{ border-bottom:none; }}
    .label {{ font-size:12px; color:var(--muted); font-weight:700; }}
    .value {{ font-size:18px; color:var(--text); font-weight:800; text-align:right; white-space:nowrap; }}
    .callout-box {{ fill:var(--card-bg); stroke:var(--line); stroke-width:1.2; rx:10; }}
    .d-label {{ fill:var(--muted); font-size:11px; font-weight:700; }}
    .d-value {{ fill:var(--text); font-size:15px; font-weight:800; }}
    .d-line {{ stroke:var(--white-line); stroke-width:3; stroke-linecap:round; stroke-linejoin:round; }}
    .d-guide {{ stroke:var(--white-line); stroke-width:2; stroke-linecap:round; stroke-dasharray:6 6; }}
    @media (max-width: 920px) {{
        .preset-preview {{ grid-template-columns:1fr; }}
        .layout {{ grid-template-columns:1fr; }}
    }}
</style>
</head>
<body>
<div class="preset-preview">
    <div class="card">
        <div class="title">{html.escape(labels['tube_section'])}</div>
        <div class="subtitle">{html.escape(labels['tube_note'])}</div>
        <div class="layout">
            <div class="drawing">
                <svg viewBox="0 0 340 240" role="img" aria-label="Tube section preview">
                    <circle cx="112" cy="100" r="74" fill="var(--foam)" stroke="var(--foam-stroke)" stroke-width="3"/>
                    <circle cx="112" cy="100" r="30" fill="var(--copper)" stroke="#9a4d1d" stroke-width="3"/>
                    <circle cx="112" cy="100" r="22" fill="var(--copper-light)" opacity="0.9"/>
                    <line x1="188" y1="26" x2="188" y2="174" class="d-line"/>
                    <line x1="112" y1="26" x2="188" y2="26" class="d-guide"/>
                    <line x1="112" y1="174" x2="188" y2="174" class="d-guide"/>
                    <polygon points="184,34 188,26 192,34" fill="var(--white-line)"/>
                    <polygon points="184,166 188,174 192,166" fill="var(--white-line)"/>
                    <rect x="208" y="54" width="106" height="48" class="callout-box"/>
                    <text x="261" y="73" text-anchor="middle" class="d-label">{html.escape(labels['outer_dim'])}</text>
                    <text x="261" y="92" text-anchor="middle" class="d-value">{v(d_tubo)} mm</text>
                    <line x1="138" y1="74" x2="218" y2="130" class="d-guide"/>
                    <rect x="214" y="120" width="100" height="44" class="callout-box"/>
                    <text x="264" y="139" text-anchor="middle" class="d-label">{html.escape(labels['insulation'])}</text>
                    <text x="264" y="157" text-anchor="middle" class="d-value">{v(spessore)} mm</text>
                </svg>
            </div>
            <div class="metrics">
                <div class="metric"><span class="label">{html.escape(labels['copper'])}</span><span class="value">{copper_label}</span></div>
                <div class="metric"><span class="label">{html.escape(labels['foam'])}</span><span class="value">{v(spessore)} mm</span></div>
                <div class="metric"><span class="label">{html.escape(labels['outer'])}</span><span class="value">{v(d_tubo)} mm</span></div>
                <div class="metric"><span class="label">{html.escape(labels['length'])}</span><span class="value">{v(lunghezza)} m</span></div>
                <div class="metric"><span class="label">{html.escape(labels['line_speed'])}</span><span class="value">{line_speed_label}</span></div>
                <div class="metric"><span class="label">{html.escape(labels['air'])}</span><span class="value">{soffiatori_label}</span></div>
            </div>
        </div>
    </div>

    <div class="card">
        <div class="title">{html.escape(labels['coil'])}</div>
        <div class="subtitle">{html.escape(labels['coil_note'])}</div>
        <div class="layout">
            <div class="drawing">
                <svg viewBox="0 0 420 220" role="img" aria-label="Coiling layout preview">
                    <g>
                        <ellipse cx="210" cy="70" rx="116" ry="9" fill="var(--coil-dark)" opacity="0.24"/>
                        <ellipse cx="210" cy="82" rx="108" ry="8.5" fill="var(--coil-top)"/>
                        <ellipse cx="210" cy="91" rx="110" ry="8.5" fill="var(--coil-mid)"/>
                        <ellipse cx="210" cy="100" rx="112" ry="8.5" fill="var(--coil-top)"/>
                        <ellipse cx="210" cy="109" rx="114" ry="8.5" fill="var(--coil-mid)"/>
                        <ellipse cx="210" cy="118" rx="116" ry="8.5" fill="var(--coil-top)"/>
                        <ellipse cx="210" cy="127" rx="114" ry="8.5" fill="var(--coil-mid)"/>
                        <ellipse cx="210" cy="136" rx="112" ry="8.5" fill="var(--coil-top)"/>
                        <ellipse cx="210" cy="145" rx="110" ry="8.5" fill="var(--coil-mid)"/>
                        <ellipse cx="210" cy="154" rx="108" ry="8.5" fill="var(--coil-top)"/>
                        <ellipse cx="210" cy="166" rx="118" ry="8" fill="var(--coil-dark)" opacity="0.18"/>
                    </g>

                    <line x1="102" y1="52" x2="318" y2="52" class="d-line"/>
                    <line x1="102" y1="40" x2="102" y2="64" class="d-line"/>
                    <line x1="318" y1="40" x2="318" y2="64" class="d-line"/>
                    <text x="210" y="46" text-anchor="middle" class="d-value">{v(aspo)} mm</text>
                    <text x="210" y="66" text-anchor="middle" class="d-label">{html.escape(labels['spool'])}</text>

                    <line x1="344" y1="82" x2="344" y2="154" class="d-line"/>
                    <line x1="332" y1="82" x2="356" y2="82" class="d-line"/>
                    <line x1="332" y1="154" x2="356" y2="154" class="d-line"/>
                    <text x="360" y="118" transform="rotate(90 360 118)" text-anchor="middle" class="d-value">{v(spalla)} mm</text>
                    <text x="344" y="118" transform="rotate(90 344 118)" text-anchor="middle" class="d-label">{html.escape(labels['width'])}</text>
                </svg>
            </div>
            <div class="metrics">
                <div class="metric"><span class="label">{html.escape(labels['spool'])}</span><span class="value">{v(aspo)} mm</span></div>
                <div class="metric"><span class="label">{html.escape(labels['width'])}</span><span class="value">{v(spalla)} mm</span></div>
                <div class="metric"><span class="label">{html.escape(labels['pitch'])}</span><span class="value">{v(passo)} mm</span></div>
                <div class="metric"><span class="label">{html.escape(labels['layer'])}</span><span class="value">{v(incremento)} mm</span></div>
                <div class="metric"><span class="label">{html.escape(labels['rollers'])}</span><span class="value">{rulli_label}</span></div>
                <div class="metric"><span class="label">{html.escape(labels['tail'])}</span><span class="value">{paletta_label}</span></div>
            </div>
        </div>
    </div>
</div>
<script>
(function() {{
    let dark = false;
    try {{
        const parentDoc = window.parent.document;
        const probe = parentDoc.querySelector('[data-testid="stAppViewContainer"]') || parentDoc.body;
        const bg = window.getComputedStyle(probe).backgroundColor;
        const nums = (bg || '').match(/\d+/g);
        if (nums && nums.length >= 3) {{
            const r = parseInt(nums[0], 10), g = parseInt(nums[1], 10), b = parseInt(nums[2], 10);
            const luminance = (0.2126*r + 0.7152*g + 0.0722*b) / 255;
            dark = luminance < 0.55;
        }} else if (window.matchMedia) {{
            dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        }}
    }} catch (e) {{
        if (window.matchMedia) {{
            dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        }}
    }}
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
}})();
</script>
</body>
</html>
'''

def current_calculator_snapshot():
    return {
        "calc_rame": str(st.session_state.get("calc_rame", "1/4")).strip(),
        "calc_spessore": float(st.session_state.get("calc_spessore", 7.0)),
        "calc_lunghezza": float(st.session_state.get("calc_lunghezza", 50.0)),
        "calc_diametro_aspo": float(st.session_state.get("calc_diametro_aspo", 450.0)),
        "calc_spalla": float(st.session_state.get("calc_spalla", 95.0)),
        "calc_passo_visuale": float(st.session_state.get("calc_passo_visuale", 20.0)),
        "calc_incremento_visuale": float(st.session_state.get("calc_incremento_visuale", 20.0)),
        "calc_rit_b": float(st.session_state.get("calc_rit_b", 360.0)),
        "calc_rit_t": float(st.session_state.get("calc_rit_t", 360.0)),
    }


def clear_active_preset_state():
    st.session_state.pop("loaded_preset_name", None)
    st.session_state.pop("loaded_preset_values", None)
    st.session_state["show_preset_loaded_success"] = False


def sync_active_preset_state():
    loaded_name = st.session_state.get("loaded_preset_name")
    loaded_values = st.session_state.get("loaded_preset_values")

    if not loaded_name or not loaded_values:
        return

    current = current_calculator_snapshot()

    for key, loaded_value in loaded_values.items():
        current_value = current.get(key)
        if isinstance(loaded_value, str):
            if str(current_value).strip() != str(loaded_value).strip():
                clear_active_preset_state()
                return
        else:
            if abs(float(current_value) - float(loaded_value)) > 1e-9:
                clear_active_preset_state()
                return


def apply_preset_to_calculator(row):
    rame = str(row.get("Diametro Rame", "1/4")).strip()
    if rame not in COPPER_SIZES_MM:
        rame = "1/4"

    st.session_state["calc_rame"] = rame
    st.session_state["calc_spessore"] = parse_float_value(row.get("Spessore Guaina (mm)", 7.0), 7.0)
    st.session_state["calc_lunghezza"] = parse_float_value(row.get("Lunghezza (m)", 50.0), 50.0)
    st.session_state["calc_diametro_aspo"] = parse_float_value(row.get("Diametro aspo (mm)", 450.0), 450.0)
    st.session_state["calc_spalla"] = parse_float_value(row.get("Spalla (mm)", 95.0), 95.0)
    st.session_state["calc_passo_visuale"] = parse_float_value(row.get("Passo (mm)", 20.0), 20.0)
    st.session_state["calc_incremento_visuale"] = parse_float_value(row.get("Incremento strato (mm)", 20.0), 20.0)

    # Mapping used by the current render: min delay -> base, max delay -> shoulder/top.
    st.session_state["calc_rit_b"] = parse_float_value(row.get("Ritardo invers min (º)", 360.0), 360.0)
    st.session_state["calc_rit_t"] = parse_float_value(row.get("Ritardo invers max (º)", 360.0), 360.0)
    st.session_state["loaded_preset_name"] = safe_value(row, "Prodotto")
    st.session_state["loaded_preset_values"] = current_calculator_snapshot()
    st.session_state["show_preset_loaded_success"] = True


def init_calculator_state():
    defaults = {
        "calc_diametro_aspo": 450.0,
        "calc_spalla": 95.0,
        "calc_rame": "1/4",
        "calc_spessore": 7.0,
        "calc_lunghezza": 50.0,
        "calc_passo_visuale": 20.0,
        "calc_incremento_visuale": 20.0,
        "calc_rit_b": 360.0,
        "calc_rit_t": 360.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
    title_placeholder = st.empty()
    current_lang = st.session_state.lang
    lang_option = st.selectbox(
        TEXTS[current_lang]["language"],
        ["🇮🇹 Italiano", "🇺🇸 English (US)"],
        index=0 if current_lang == "IT" else 1,
        key="lang_selector_top",
    )

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
lang = st.session_state.lang
t = TEXTS[lang]
title_placeholder.markdown(f"## {t['title']}")

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
    coil_footprint_mm=None,
):
    final_local_points_json = json.dumps(final_local_points)
    final_thetas_json = json.dumps(final_thetas)
    final_radii_json = json.dumps(final_radii)
    final_zs_json = json.dumps(final_zs)
    final_modes_json = json.dumps(final_modes)
    final_layers_json = json.dumps(final_layers)
    final_lengths_json = json.dumps(final_lengths)
    labels_json = json.dumps(TEXTS[language])
    if coil_footprint_mm is None:
        try:
            coil_footprint_mm = compute_max_xy_span(np.array(final_local_points, dtype=float), d_tubo)
        except Exception:
            coil_footprint_mm = d_aspo + 2.0 * d_tubo

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
                <div class="panel_label" id="scene_title"></div>
                <div class="btn_group_vertical">
                    <button class="scene_btn viewer_btn_small active_opt" data-scene="winding" id="scene_winding_btn">Avvolgimento</button>
                    <button class="scene_btn viewer_btn_small" data-scene="packaging" id="scene_packaging_btn">Packaging</button>
                </div>
            </div>

            <div id="packaging_controls" style="display:none;">
                <div class="panel_label" id="pack_roll_title"></div>
                <input id="pack_roll_count" type="number" min="1" max="50" step="1" value="5" style="
                    width:100%;
                    box-sizing:border-box;
                    border:none;
                    border-radius:9px;
                    padding:8px 10px;
                    font-weight:800;
                    font-size:15px;
                    background:rgba(255,255,255,0.92);
                    color:#111;
                    margin-bottom:10px;
                " />
                <div id="packaging_stats" class="packaging_stats"></div>
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

        .packaging_stats {{
            display:grid;
            gap:8px;
            margin-top:8px;
        }}

        .pack_stat {{
            padding:9px 10px;
            border-radius:11px;
            background:rgba(255,255,255,0.08);
            border:1px solid rgba(255,255,255,0.10);
        }}

        .pack_stat_label {{
            font-size:10px;
            opacity:0.72;
            text-transform:uppercase;
            letter-spacing:0.05em;
            margin-bottom:3px;
        }}

        .pack_stat_value {{
            font-size:16px;
            font-weight:800;
            line-height:1.1;
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
        const sceneBtns = [...document.querySelectorAll(".scene_btn")];
        const packagingControls = document.getElementById("packaging_controls");
        const packRollCountInput = document.getElementById("pack_roll_count");
        const packagingStats = document.getElementById("packaging_stats");
        const viewerHud = document.getElementById("viewer_hud");

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
        document.getElementById("scene_title").textContent = T.packaging_title || "Packaging";
        document.getElementById("scene_winding_btn").textContent = T.title || "Avvolgimento";
        document.getElementById("scene_packaging_btn").textContent = T.packaging_title || "Packaging";
        document.getElementById("pack_roll_title").textContent = T.roll_count || "Numero rotoli";
        document.getElementById("grid_title").textContent = T.grid;
        document.getElementById("axes_title").textContent = T.axes;
        document.getElementById("section_title").textContent = T.section;
        document.getElementById("ghost_title").textContent = T.ghost;
        const studioTitleEl = document.getElementById("studio_title");
        if (studioTitleEl) studioTitleEl.textContent = T.studio;
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
        const coilFootprint = {float(coil_footprint_mm):.6f};
        const palletSize = 750.0;
        const palletHeight = 130.0;
        const boxHeight = 1350.0;

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
        let sceneMode = "winding";
        let showStudio = false;
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

        if (studioCheck) {{
            studioCheck.addEventListener("change", () => {{
                showStudio = studioCheck.checked;
                applyVisualState();
            }});
        }}

        ghostCheck.addEventListener("change", () => {{
            showGhost = ghostCheck.checked;
            updateGhostLine();
            if (sceneMode === "packaging") updatePackagingScene();
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

        const packagingGroup = new THREE.Group();
        scene.add(packagingGroup);
        packagingGroup.visible = false;

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
        // PACKAGING SCENE
        // ==========================================

        function clearGroup(group) {{
            while (group.children.length) {{
                const obj = group.children.pop();
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) {{
                    if (Array.isArray(obj.material)) {{
                        obj.material.forEach(m => m.dispose && m.dispose());
                    }} else {{
                        obj.material.dispose && obj.material.dispose();
                    }}
                }}
                if (obj.children && obj.children.length) {{
                    obj.children.forEach(child => {{
                        if (child.geometry) child.geometry.dispose();
                        if (child.material) child.material.dispose && child.material.dispose();
                    }});
                }}
            }}
        }}

        function addBoxEdges(width, depth, height, zCenter, color=0xffffff, opacity=0.82) {{
            const geo = new THREE.BoxGeometry(width, depth, height);
            const edges = new THREE.EdgesGeometry(geo);
            const mat = new THREE.LineBasicMaterial({{
                color: color,
                transparent: true,
                opacity: opacity
            }});
            const line = new THREE.LineSegments(edges, mat);
            line.position.set(0, 0, zCenter);
            packagingGroup.add(line);
            geo.dispose();
            return line;
        }}

        function updatePackagingStats(rollCount) {{
            const stackHeight = rollCount * Hs;
            const totalHeight = palletHeight + stackHeight;
            const heightMargin = Math.max(0, boxHeight - stackHeight);
            const heightOver = Math.max(0, stackHeight - boxHeight);
            const footprintOver = Math.max(0, coilFootprint - palletSize);
            const ok = heightOver <= 0.001 && footprintOver <= 0.001;
            const statusText = ok ? (T.box_fit_ok || "OK") : (T.box_fit_over || "Fuori limite");

            packagingStats.innerHTML = `
                <div class="pack_stat">
                    <div class="pack_stat_label">Status</div>
                    <div class="pack_stat_value" style="color:${{ok ? "#4ade80" : "#fca5a5"}}">${{statusText}}</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.total_height || "Altezza totale"}}</div>
                    <div class="pack_stat_value">${{totalHeight.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.roll_stack_height || "Altezza rotoli"}}</div>
                    <div class="pack_stat_value">${{stackHeight.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.coil_footprint || "Ingombro"}}</div>
                    <div class="pack_stat_value">${{coilFootprint.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.height_margin || "Margine altezza"}}</div>
                    <div class="pack_stat_value">${{heightMargin.toFixed(1)}} mm</div>
                </div>
            `;
        }}

        function updatePackagingScene() {{
            if (!packagingGroup) return;

            clearGroup(packagingGroup);

            const rollCount = Math.max(1, Math.min(50, parseInt(packRollCountInput.value || "1", 10)));
            packRollCountInput.value = rollCount;

            const stackHeight = rollCount * Hs;
            const totalHeight = palletHeight + stackHeight;
            const footprintOk = coilFootprint <= palletSize + 0.001;
            const heightOk = stackHeight <= boxHeight + 0.001;
            const ok = footprintOk && heightOk;

            const palletMat = new THREE.MeshStandardMaterial({{
                color: 0xb9925a,
                roughness: 0.72,
                metalness: 0.02
            }});

            const pallet = new THREE.Mesh(
                new THREE.BoxGeometry(palletSize, palletSize, palletHeight),
                palletMat
            );
            pallet.position.set(0, 0, palletHeight / 2);
            pallet.castShadow = true;
            pallet.receiveShadow = true;
            packagingGroup.add(pallet);

            const boxMat = new THREE.MeshStandardMaterial({{
                color: ok ? 0x4ade80 : 0xf87171,
                transparent: true,
                opacity: 0.055,
                roughness: 0.70,
                metalness: 0.0,
                depthWrite: false
            }});
            const box = new THREE.Mesh(
                new THREE.BoxGeometry(palletSize, palletSize, boxHeight),
                boxMat
            );
            box.position.set(0, 0, palletHeight + boxHeight / 2);
            packagingGroup.add(box);
            addBoxEdges(palletSize, palletSize, boxHeight, palletHeight + boxHeight / 2, ok ? 0x4ade80 : 0xfca5a5, 0.95);

            const coilRadius = coilFootprint / 2.0;
            const rollMat = makeTubeMaterial(tubeMode, false, false);
            rollMat.roughness = 0.86;

            for (let i = 0; i < rollCount; i++) {{
                const zc = palletHeight + i * Hs + Hs / 2.0;
                const roll = new THREE.Mesh(
                    new THREE.CylinderGeometry(coilRadius, coilRadius, Hs, 128),
                    rollMat.clone()
                );
                roll.rotation.x = Math.PI / 2;
                roll.position.set(0, 0, zc);
                roll.castShadow = true;
                roll.receiveShadow = true;
                packagingGroup.add(roll);

                const innerRadius = Math.max(8, coilRadius * 0.56);
                const hole = new THREE.Mesh(
                    new THREE.CylinderGeometry(innerRadius, innerRadius, Hs + 1.5, 96),
                    new THREE.MeshStandardMaterial({{
                        color: tubeMode === "gelblack" ? 0xffffff : 0x111419,
                        roughness: 0.9,
                        metalness: 0.0
                    }})
                );
                hole.rotation.x = Math.PI / 2;
                hole.position.set(0, 0, zc + 0.5);
                // This is a visual inner disk, not boolean subtraction. It makes the roll readable.
                packagingGroup.add(hole);
            }}

            const limitLineMat = new THREE.LineBasicMaterial({{
                color: ok ? 0x4ade80 : 0xf87171,
                transparent: true,
                opacity: 0.95
            }});
            const heightPoints = [
                new THREE.Vector3(palletSize * 0.62, -palletSize * 0.62, 0),
                new THREE.Vector3(palletSize * 0.62, -palletSize * 0.62, totalHeight)
            ];
            const heightLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(heightPoints), limitLineMat);
            packagingGroup.add(heightLine);

            updatePackagingStats(rollCount);
        }}

        function applySceneMode() {{
            const packaging = sceneMode === "packaging";

            machine.visible = !packaging;
            guideGroup.visible = !packaging;
            overlayGroup.visible = !packaging;
            packagingGroup.visible = packaging;

            packagingControls.style.display = packaging ? "block" : "none";
            viewerHud.style.display = packaging ? "none" : "grid";
            progressSlider.disabled = packaging;
            playPauseBtn.disabled = packaging;
            animationCheck.disabled = packaging;

            if (packaging) {{
                updatePackagingScene();
                camera.position.set(-950, -1150, 980);
                controls.target.set(0, 0, 680);
                camera.lookAt(0, 0, 680);
                controls.update();
            }} else {{
                setCameraView(currentView);
            }}
        }}

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

init_calculator_state()

tab_presets, tab_calculator = st.tabs([
    t["tab_presets"],
    t["tab_calculator"],
])

with tab_presets:
    st.markdown(t["presets_title"])

    try:
        presets_df = load_presets("Presets.csv")

        st.caption(f"{len(presets_df)} {t['presets_loaded']}")

        selected_product = st.selectbox(
            t["select_product"],
            presets_df["Prodotto"].tolist(),
            key="selected_preset_product",
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
                    {t["preset_sheet"]}
                </div>
                <div style="font-size:30px; font-weight:800; color:#ffffff; line-height:1.15;">
                    {selected_product}
                </div>
                <div style="font-size:14px; color:rgba(255,255,255,0.68); margin-top:8px;">
                    {t["preset_subtitle"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"#### {t['preset_visual_title']}")
        components.html(make_preset_visual(selected_row, lang), height=400, scrolling=False)

        if st.button(t["load_to_calculator"], type="primary", use_container_width=True):
            apply_preset_to_calculator(selected_row)
            st.rerun()

        st.markdown(t["csv_params"])

        render_columns = {
            "Diametro Rame",
            "Spessore Guaina (mm)",
            "Diametro esterno Guaina (mm)",
            "Lunghezza (m)",
            "Guidatubo (mm)",
            "Spalla (mm)",
            "Diametro aspo (mm)",
            "Ritardo invers max (º)",
            "Ritardo invers min (º)",
            "Passo (mm)",
            "Incremento strato (mm)",
        }

        linked_cols = [c for c in presets_df.columns if c in render_columns]
        cards_per_row = 4
        render_preset_param_cards(t['linked_params'], linked_cols, selected_row, lang, cards_per_row=cards_per_row)

        consult_cols = [c for c in presets_df.columns if c not in render_columns]
        if consult_cols:
            render_preset_param_cards(t['non_render_params'], consult_cols, selected_row, lang, cards_per_row=cards_per_row)

        note_value = safe_value(selected_row, "Note")
        if note_value != "-":
            st.info(note_value)

    except FileNotFoundError:
        st.error(t["presets_file_missing"])
    except Exception as e:
        st.error(f"{t['presets_load_error']}: {e}")


with tab_calculator:
    colA, colB, colC = st.columns(3)

    sync_active_preset_state()
    loaded_preset_name = st.session_state.get("loaded_preset_name")
    show_loaded_success = st.session_state.get("show_preset_loaded_success", False)
    if loaded_preset_name:
        if show_loaded_success:
            st.success(t["preset_loaded_ok"].format(name=loaded_preset_name))
            st.session_state["show_preset_loaded_success"] = False
        st.caption(f"{t['active_preset']}: {loaded_preset_name}")

    with colA:
        st.markdown(f"#### {t['bobina']}")
        diametro_aspo = st.number_input(t["diam_aspo"], step=10.0, key="calc_diametro_aspo")
        spalla = st.number_input(t["spalla"], step=1.0, key="calc_spalla")

    with colB:
        st.markdown(f"#### {t['tubo']}")
        rame_options = list(COPPER_SIZES_MM.keys())
        if st.session_state.get("calc_rame") not in rame_options:
            st.session_state["calc_rame"] = "1/4"
        rame = st.selectbox(t["rame"], rame_options, key="calc_rame")
        spessore = st.number_input(t["isolamento"], step=1.0, key="calc_spessore")
        lunghezza = st.number_input(t["lunghezza"], step=5.0, key="calc_lunghezza")
        d_rame = COPPER_SIZES_MM[rame]

    with colC:
        st.markdown(f"#### {t['avvolg']}")
        passo_visuale = st.number_input(t["passo_assiale"], step=0.5, key="calc_passo_visuale")
        incremento_visuale = st.number_input(t["incremento"], step=0.5, key="calc_incremento_visuale")
        rit_b = st.number_input(t["rit_min"], step=1.0, key="calc_rit_b")
        rit_t = st.number_input(t["rit_max"], step=1.0, key="calc_rit_t")

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
            coil_footprint_mm=visual_metrics["max_xy_span"],
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

    pallet_size_mm = 750.0
    coil_footprint_mm = float(visual_metrics["max_xy_span"])

    if coil_footprint_mm > pallet_size_mm:
        st.warning(t["warning"])
