import os
import glob
import json
import html
import hashlib
import base64
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
except Exception:
    colors = None
    A4 = None
    mm = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None


st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

try:
    query_lang = st.query_params.get("lang", None)
    if query_lang in {"IT", "EN"} and query_lang != st.session_state.lang:
        st.session_state.lang = query_lang
except Exception:
    pass

# =========================
# TEXTS
# =========================

TEXTS = {
    "IT": {
        "title": "Avvolgimento",
        "language": "Lingua",
        "bobina": "Bobina",
        "tubo": "Tubo",
        "avvolg": "Simulazione",
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
        "results": "Risultati",
        "warning": "Ingombro max XY superiore a 750 mm.",
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
        "tab_presets": "Preset",
        "tab_calculator": "Calcolatore / Render",
        "presets_title": "### Preset prodotto",
        "presets_loaded": "preset caricati correttamente",
        "select_product": "Seleziona prodotto",
        "preset_sheet": "Scheda preset",
        "preset_subtitle": "Configurazione tecnica prodotto · valori caricati dal preset",
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
        "preset_render_note": "I parametri del preset sono stati caricati nel calcolatore. Puoi modificarli liberamente.",
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
        "packaging_tab": "Packaging",
        "render_tab": "Render",
        "packaging_title": "Packaging",
        "viewer_mode": "Visualizzazione",
        "scene_winding": "Avvolgimento",
        "scene_packaging": "Packaging",
        "packaging_controls_title": "Configurazione packaging",
        "packaging_results_title": "Risultati packaging",
        "packaging_mode": "Modalità packaging",
        "packaging_box": "Scatola",
        "packaging_tower": "Torretta",
        "packaging_box_desc": "Scatola 750 × 750 × 1030 mm",
        "packaging_tower_desc": "Torre su pallet, con limite container",
        "roll_count": "Numero rotoli",
        "box_height": "Altezza utile scatola",
        "pallet_height": "Altezza pallet",
        "total_height": "Altezza totale con pallet",
        "roll_stack_height": "Altezza rotoli",
        "width_margin": "Margine larghezza",
        "width_over": "Superamento larghezza",
        "height_margin": "Margine altezza",
        "height_over": "Superamento altezza",
        "height_limit": "Limite altezza",
        "no_height_limit": "Nessun limite",
        "container_type": "Container",
        "container_40hc": "40 HC",
        "container_20ft": "20 piedi",
        "container_40hc_desc": "Altezza max 2580 mm",
        "container_20ft_desc": "Altezza max 2280 mm",
        "box_fit_ok": "Packaging OK",
        "box_fit_over": "Fuori limite",
        "box_fit_note": "Calcolo basato sui parametri attuali del render.",
        "capture_render": "Salva immagine render",
        "print_sheet": "Scheda stampabile",
        "print_sheet_help": "Scarica una scheda HTML pronta da stampare.",
        "print_simulation": "PDF simulazione",
        "print_preset_csv": "PDF scheda preset",
    },
    "EN": {
        "title": "Coiling",
        "language": "Language",
        "bobina": "Coil",
        "tubo": "Tube",
        "avvolg": "Simulation",
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
        "results": "Results",
        "warning": "Max XY span exceeds 750 mm.",
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
        "tab_presets": "Presets",
        "tab_calculator": "Calculator / Render",
        "presets_title": "### Product presets",
        "presets_loaded": "presets loaded correctly",
        "select_product": "Select product",
        "preset_sheet": "Preset sheet",
        "preset_subtitle": "Product technical configuration · values loaded from the preset",
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
        "preset_render_note": "The preset parameters have been loaded into the calculator. You can edit them freely.",
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
        "packaging_tab": "Packaging",
        "render_tab": "Render",
        "packaging_title": "Packaging",
        "viewer_mode": "View mode",
        "scene_winding": "Winding",
        "scene_packaging": "Packaging",
        "packaging_controls_title": "Packaging configuration",
        "packaging_results_title": "Packaging results",
        "packaging_mode": "Packaging mode",
        "packaging_box": "Box",
        "packaging_tower": "Tower",
        "packaging_box_desc": "Box 750 × 750 × 1030 mm",
        "packaging_tower_desc": "Tower on pallet, with container limit",
        "roll_count": "Number of coils",
        "box_height": "Usable box height",
        "pallet_height": "Pallet height",
        "total_height": "Total height with pallet",
        "roll_stack_height": "Coil stack height",
        "width_margin": "Width margin",
        "width_over": "Width over limit",
        "height_margin": "Height margin",
        "height_over": "Height over limit",
        "height_limit": "Height limit",
        "no_height_limit": "No limit",
        "container_type": "Container",
        "container_40hc": "40 HC",
        "container_20ft": "20 ft",
        "container_40hc_desc": "Max height 2580 mm",
        "container_20ft_desc": "Max height 2280 mm",
        "box_fit_ok": "Packaging OK",
        "box_fit_over": "Out of bounds",
        "box_fit_note": "Calculation based on the current render parameters.",
        "capture_render": "Save render image",
        "print_sheet": "Printable sheet",
        "print_sheet_help": "Download a print-ready HTML sheet.",
        "print_simulation": "Simulation PDF",
        "print_preset_csv": "Preset sheet PDF",
    },
}

PARAM_LABELS = {
    "IT": {
        "Prodotto": "Prodotto",
        "Tipo tubo": "Tipo tubo",
        "Diametro rame inferiore": "Diametro rame inferiore",
        "Spessore guaina inferiore": "Spessore guaina inferiore",
        "Diametro rame superiore": "Diametro rame superiore",
        "Spessore guaina superiore": "Spessore guaina superiore",
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
        "Fattore passo effettivo": "Fattore passo effettivo",
        "Fattore compattazione radiale": "Fattore compattazione radiale",
        "Fattore compatazione radiale": "Fattore compattazione radiale",
        "Coppia lavoro (%)": "Coppia lavoro (%)",
        "Riduzione coppia (%)": "Riduzione coppia (%)",
        "Coppia recupero (%)": "Coppia recupero (%)",
    },
    "EN": {
        "Prodotto": "Product",
        "Tipo tubo": "Tube type",
        "Diametro rame inferiore": "Lower copper diameter",
        "Spessore guaina inferiore": "Lower foam thickness",
        "Diametro rame superiore": "Upper copper diameter",
        "Spessore guaina superiore": "Upper foam thickness",
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
        "Fattore passo effettivo": "Effective pitch factor",
        "Fattore compattazione radiale": "Radial compaction factor",
        "Fattore compatazione radiale": "Radial compaction factor",
        "Coppia lavoro (%)": "Working torque (%)",
        "Riduzione coppia (%)": "Torque reduction (%)",
        "Coppia recupero (%)": "Recovery torque (%)",
    },
}


def param_label(column_name, language):
    return PARAM_LABELS.get(language, {}).get(column_name, column_name)



def render_preset_param_cards(title, column_names, selected_row, language, cards_per_row=4):
    visible_column_names = []

    for column_name in column_names:
        if column_name not in selected_row.index:
            continue

        raw_value = selected_row[column_name]

        if pd.isna(raw_value):
            continue

        formatted_value = str(format_preset_value(raw_value)).strip()

        if formatted_value in {"", "-"}:
            continue

        visible_column_names.append(column_name)

    if not visible_column_names:
        return

    column_names = visible_column_names

    st.markdown(
        """
        <style>
        .preset-param-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 96%, var(--background-color))
            );
            border: 1px solid color-mix(in srgb, var(--text-color) 22%, transparent);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 126px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 10px 24px rgba(0,0,0,0.10);
            margin-bottom: 12px;
        }
        .preset-param-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: #C57E5A;
            opacity: 0.9;
        }
        .preset-param-label {
            font-size: 14px;
            line-height: 1.35;
            font-weight: 700;
            color: color-mix(in srgb, var(--text-color) 72%, transparent);
            margin-bottom: 14px;
            padding-left: 4px;
        }
        .preset-param-value {
            font-size: 31px;
            line-height: 1.08;
            font-weight: 800;
            color: var(--pdm-popup-text);
            word-break: break-word;
            padding-left: 4px;
        }
        @media (max-width: 900px) {
            .preset-param-card {
                min-height: 108px;
                padding: 14px 16px;
                margin-bottom: 10px;
            }
            .preset-param-card::before {
                width: 4px;
            }
            .preset-param-label {
                font-size: 13px;
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
    st.markdown("<div style=\"height:6px\"></div>", unsafe_allow_html=True)
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



def evaluate_packaging(coil_footprint_mm, roll_height_mm, roll_count, packaging_mode, container_mode, language, pallet_size_mm=750.0, pallet_height_mm=130.0, box_height_mm=1030.0):
    stack_height_mm = roll_count * roll_height_mm
    total_height_mm = pallet_height_mm + stack_height_mm

    if packaging_mode == "box":
        height_limit_mm = box_height_mm
        compared_height_mm = stack_height_mm
    else:
        height_limit_mm = 2580.0 if container_mode == "40hc" else 2280.0
        compared_height_mm = total_height_mm

    width_over_raw = max(0.0, coil_footprint_mm - pallet_size_mm)
    width_ok = width_over_raw <= 0.001
    width_warn = 0.001 < width_over_raw <= 20.001
    width_bad = width_over_raw > 20.001
    height_ok = compared_height_mm <= height_limit_mm
    ok = width_ok and height_ok
    warning = width_warn and height_ok

    if language == "IT":
        reasons = []
        if width_warn:
            reasons.append(f"Attenzione: ingombro {coil_footprint_mm:.1f} mm, limite pallet {pallet_size_mm:.0f} mm (+{width_over_raw:.1f} mm entro margine tollerato).")
        elif width_bad:
            reasons.append(f"Il rotolo è troppo largo: ingombro {coil_footprint_mm:.1f} mm, limite pallet {pallet_size_mm:.0f} mm (+{width_over_raw:.1f} mm).")
        if not height_ok and packaging_mode == "box":
            reasons.append(f"Il rotolo è troppo alto per la scatola: altezza rotoli {stack_height_mm:.1f} mm, limite utile {height_limit_mm:.0f} mm.")
        if not height_ok and packaging_mode != "box":
            label = "40 HC" if container_mode == "40hc" else "20 piedi"
            reasons.append(f"La torre è troppo alta per il container {label}: altezza totale con pallet {total_height_mm:.1f} mm, limite {height_limit_mm:.0f} mm.")
    else:
        reasons = []
        if width_warn:
            reasons.append(f"Attention: footprint {coil_footprint_mm:.1f} mm, pallet limit {pallet_size_mm:.0f} mm (+{width_over_raw:.1f} mm within tolerated margin).")
        elif width_bad:
            reasons.append(f"The coil is too wide: footprint {coil_footprint_mm:.1f} mm, pallet limit {pallet_size_mm:.0f} mm (+{width_over_raw:.1f} mm).")
        if not height_ok and packaging_mode == "box":
            reasons.append(f"The coil stack is too tall for the box: stack height {stack_height_mm:.1f} mm, usable limit {height_limit_mm:.0f} mm.")
        if not height_ok and packaging_mode != "box":
            label = "40 HC" if container_mode == "40hc" else "20 ft"
            reasons.append(f"The tower is too tall for container {label}: total height with pallet {total_height_mm:.1f} mm, limit {height_limit_mm:.0f} mm.")

    return {
        "ok": ok,
        "warning": warning,
        "width_ok": width_ok,
        "width_warn": width_warn,
        "width_bad": width_bad,
        "height_ok": height_ok,
        "reasons": reasons,
        "stack_height_mm": stack_height_mm,
        "total_height_mm": total_height_mm,
        "height_limit_mm": height_limit_mm,
        "height_margin_mm": max(0.0, height_limit_mm - compared_height_mm),
        "height_over_mm": max(0.0, compared_height_mm - height_limit_mm),
        "width_margin_mm": max(0.0, pallet_size_mm - coil_footprint_mm),
        "width_over_mm": max(0.0, coil_footprint_mm - pallet_size_mm),
        "coil_footprint_mm": coil_footprint_mm,
        "pallet_size_mm": pallet_size_mm,
        "pallet_height_mm": pallet_height_mm,
        "compared_height_mm": compared_height_mm,
    }



def render_summary_cards(title, items, cards_per_row=4):
    st.markdown(
        """
        <style>
        .summary-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 96%, var(--background-color))
            );
            border: 1px solid color-mix(in srgb, var(--text-color) 22%, transparent);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 126px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 10px 24px rgba(0,0,0,0.10);
            margin-bottom: 12px;
        }
        .summary-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: #C57E5A;
            opacity: 0.9;
        }
        .summary-card.status-ok {
            border-color: rgba(34,197,94,0.42);
            background: color-mix(in srgb, #22c55e 12%, var(--secondary-background-color));
        }
        .summary-card.status-ok::before {
            background: #22c55e;
        }
        .summary-card.status-bad {
            border-color: rgba(239,68,68,0.44);
            background: color-mix(in srgb, #ef4444 13%, var(--secondary-background-color));
        }
        .summary-card.status-bad::before {
            background: #ef4444;
        }
        .summary-card-label {
            font-size: 14px;
            line-height: 1.3;
            font-weight: 700;
            color: color-mix(in srgb, var(--text-color) 72%, transparent);
            margin-bottom: 12px;
            padding-left: 4px;
        }
        .summary-card-value {
            font-size: 34px;
            line-height: 1.08;
            font-weight: 800;
            color: var(--pdm-popup-text);
            word-break: break-word;
            padding-left: 4px;
        }
        .summary-card-note {
            font-size: 13px;
            color: color-mix(in srgb, var(--text-color) 64%, transparent);
            margin-top: 8px;
            line-height: 1.3;
            padding-left: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"##### {title}")
    st.markdown("<div style=\"height:6px\"></div>", unsafe_allow_html=True)
    for i in range(0, len(items), cards_per_row):
        chunk = items[i:i+cards_per_row]
        cols = st.columns(cards_per_row)
        for col, item in zip(cols, chunk):
            tone = item.get("tone", "")
            extra_class = f" status-{tone}" if tone in {"ok", "bad"} else ""
            note = item.get("note", "")
            note_html = f'<div class="summary-card-note">{html.escape(str(note))}</div>' if note else ""
            col.markdown(
                f"""
                <div class="summary-card{extra_class}">
                    <div class="summary-card-label">{html.escape(str(item['label']))}</div>
                    <div class="summary-card-value">{html.escape(str(item['value']))}</div>
                    {note_html}
                </div>
                """,
                unsafe_allow_html=True,
            )



def render_layer_diagnostics_panel(
    language,
    winding_diagnostics,
    passo_effettivo_assiale,
    incremento_effettivo_radiale,
    fattore_passo_effettivo,
    fattore_compattazione_radiale,
    use_correction_factors,
):
    """Native Streamlit layer diagnostics.

    Avoids raw HTML cards here because this block must be robust across
    desktop, tablet and mobile Streamlit renderers.
    """
    if language == "IT":
        title = "Diagnostica strati"
        subtitle = "Sintesi del risultato del render."
        layers_label = "Strati"
        side_label = "Lato finale"
        mode_label = "Modo"
        pitch_label = "Passo usato"
        increment_label = "Incremento usato"
        factors_label = "Fattori"
        mode_value = "Correzione" if use_correction_factors else "Ideale"
        factors_value = f"passo ×{fattore_passo_effettivo:.2f} · comp. ×{fattore_compattazione_radiale:.2f}" if use_correction_factors else "non applicati"
    else:
        title = "Layer diagnostics"
        subtitle = "Summary of the render result."
        layers_label = "Layers"
        side_label = "Final side"
        mode_label = "Mode"
        pitch_label = "Used pitch"
        increment_label = "Used increment"
        factors_label = "Factors"
        mode_value = "Correction" if use_correction_factors else "Ideal"
        factors_value = f"pitch ×{fattore_passo_effettivo:.2f} · comp. ×{fattore_compattazione_radiale:.2f}" if use_correction_factors else "not applied"

    layers_value = str(winding_diagnostics.get("strati_simulati", "-"))
    side_value = str(winding_diagnostics.get("lato_finale", "-"))
    pitch_value = f"{float(passo_effettivo_assiale):.2f} mm"
    increment_value = f"{float(incremento_effettivo_radiale):.2f} mm"

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            min-height: 104px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            font-weight: 850 !important;
            color: color-mix(in srgb, var(--text-color) 62%, transparent) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: clamp(1.35rem, 2.4vw, 2.25rem) !important;
            font-weight: 950 !important;
            letter-spacing: -0.025em !important;
        }
        .diagnostic-native-title {
            margin: 0 0 2px 0;
            font-size: 1.05rem;
            line-height: 1.12;
            font-weight: 950;
            letter-spacing: -0.015em;
        }
        .diagnostic-native-subtitle {
            margin: 0 0 10px 0;
            color: color-mix(in srgb, var(--text-color) 58%, transparent);
            font-size: 0.78rem;
            line-height: 1.25;
            font-weight: 650;
        }
        .diagnostic-native-detail {
            min-height: 104px;
            padding: 13px 14px;
            border-radius: 18px;
            border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            background: linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow: 0 4px 14px rgba(0,0,0,0.055);
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 7px;
        }
        .diagnostic-native-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: baseline;
        }
        .diagnostic-native-row span {
            font-size: 0.68rem;
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: 0.045em;
            text-transform: uppercase;
            color: color-mix(in srgb, var(--text-color) 55%, transparent);
        }
        .diagnostic-native-row strong {
            text-align: right;
            font-size: 0.92rem;
            line-height: 1.05;
            font-weight: 950;
            color: var(--text-color);
            overflow-wrap: anywhere;
        }
        @media (max-width: 820px) {
            div[data-testid="stMetric"] {
                min-height: 92px;
            }
            .diagnostic-native-detail {
                min-height: 92px;
                padding: 11px 12px;
                border-radius: 15px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(f"<div class='diagnostic-native-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='diagnostic-native-subtitle'>{html.escape(subtitle)}</div>", unsafe_allow_html=True)

        col_layers, col_side, col_mode, col_values = st.columns([0.85, 1.05, 0.90, 1.65], gap="small")

        with col_layers:
            st.metric(layers_label, layers_value)

        with col_side:
            st.metric(side_label, side_value)

        with col_mode:
            st.metric(mode_label, mode_value)

        with col_values:
            st.markdown(
                f"""
                <div class="diagnostic-native-detail">
                    <div class="diagnostic-native-row">
                        <span>{html.escape(pitch_label)}</span>
                        <strong>{html.escape(pitch_value)}</strong>
                    </div>
                    <div class="diagnostic-native-row">
                        <span>{html.escape(increment_label)}</span>
                        <strong>{html.escape(increment_value)}</strong>
                    </div>
                    <div class="diagnostic-native-row">
                        <span>{html.escape(factors_label)}</span>
                        <strong>{html.escape(factors_value)}</strong>
                    </div>
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
        --measure: #f8fafc;
        --coil-measure: #bfdbfe;
        --pallet: rgba(194,154,106,0.38);
        --pallet-stroke: rgba(223,190,145,0.85);
        --coil-fill: rgba(96,165,250,0.18);
        --coil-stroke: rgba(191,219,254,0.96);
        --warn-fill: rgba(248,113,113,0.18);
        --warn-stroke: rgba(252,165,165,0.98);
    }}
    html[data-theme="light"] {{
        --card-bg: rgba(255,255,255,0.78);
        --card-border: rgba(15,23,42,0.12);
        --text: #111827;
        --muted: rgba(75,85,99,0.72);
        --measure: #111827;
        --coil-measure: #2563eb;
        --pallet: rgba(194,154,106,0.28);
        --pallet-stroke: rgba(146,95,49,0.72);
        --coil-fill: rgba(37,99,235,0.12);
        --coil-stroke: rgba(37,99,235,0.80);
        --warn-fill: rgba(239,68,68,0.12);
        --warn-stroke: rgba(220,38,38,0.88);
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
        <line x1="70" y1="230" x2="250" y2="230" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
        <line x1="70" y1="220" x2="70" y2="240" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
        <line x1="250" y1="220" x2="250" y2="240" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
        <text x="160" y="225" text-anchor="middle" fill="var(--measure)" font-size="14" font-weight="800">{pallet_size_mm:.0f} mm</text>
        <line x1="{160-circle_r:.2f}" y1="22" x2="{160+circle_r:.2f}" y2="22" stroke="var(--coil-measure)" stroke-width="3" stroke-linecap="round"/>
        <line x1="{160-circle_r:.2f}" y1="14" x2="{160-circle_r:.2f}" y2="30" stroke="var(--coil-measure)" stroke-width="3" stroke-linecap="round"/>
        <line x1="{160+circle_r:.2f}" y1="14" x2="{160+circle_r:.2f}" y2="30" stroke="var(--coil-measure)" stroke-width="3" stroke-linecap="round"/>
        <text x="160" y="16" text-anchor="middle" fill="var(--coil-measure)" font-size="14" font-weight="800">{coil_diameter_mm:.1f} mm</text>
    </svg>
    <div class="legend">
        <div class="legend-item"><span class="swatch" style="background:var(--pallet);"></span>{labels['pallet']}</div>
        <div class="legend-item"><span class="swatch" style="background:{coil_fill}; border:1px solid {coil_stroke};"></span>{labels['coil']}</div>
    </div>
</div>

<script>
(() => {{
    function luminance(rgb) {{
        const parts = (rgb || "").match(/\\d+(\\.\\d+)?/g);
        if (!parts || parts.length < 3) return null;
        const vals = parts.slice(0,3).map(Number).map(v => {{
            v = v / 255;
            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        }});
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2];
    }}
    function applyTheme() {{
        try {{
            const parentDoc = window.parent && window.parent.document;
            const bg = parentDoc ? window.parent.getComputedStyle(parentDoc.body).backgroundColor : "";
            const lum = luminance(bg);
            if (lum !== null) {{
                document.documentElement.dataset.theme = lum > 0.55 ? "light" : "dark";
                return;
            }}
        }} catch (err) {{}}
        document.documentElement.dataset.theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }}
    applyTheme();
    setInterval(applyTheme, 1000);
}})();
</script>
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
        --measure: #f8fafc;
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
    html[data-theme="light"] {{
        --card-bg: rgba(255,255,255,0.78);
        --card-border: rgba(15,23,42,0.12);
        --text: #111827;
        --muted: rgba(75,85,99,0.72);
        --measure: #111827;
        --pallet: rgba(194,154,106,0.34);
        --pallet-stroke: rgba(146,95,49,0.72);
        --box-fill: rgba(15,23,42,0.035);
        --box-stroke: rgba(71,85,105,0.55);
        --coil-fill: rgba(37,99,235,0.12);
        --coil-fill-2: rgba(226,232,240,0.96);
        --coil-highlight: rgba(255,255,255,0.98);
        --coil-stroke: rgba(37,99,235,0.76);
        --warn-fill: rgba(239,68,68,0.12);
        --warn-stroke: rgba(220,38,38,0.88);
        --ok: rgba(22,163,74,0.96);
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
            <line x1="228" y1="{box_top_y}" x2="228" y2="{pallet_y + pallet_h_px}" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
            <line x1="218" y1="{box_top_y}" x2="238" y2="{box_top_y}" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
            <line x1="218" y1="{pallet_y + pallet_h_px}" x2="238" y2="{pallet_y + pallet_h_px}" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
            <text x="242" y="{(box_top_y + pallet_y + pallet_h_px)/2:.1f}" transform="rotate(90 242 {(box_top_y + pallet_y + pallet_h_px)/2:.1f})" fill="var(--measure)" font-size="13" font-weight="800" text-anchor="middle">{total_height_mm:.0f} mm</text>
            <text x="118" y="{pallet_y + pallet_h_px + 18}" fill="var(--muted)" font-size="12" font-weight="700" text-anchor="middle">{labels['pallet']} {pallet_height_mm:.0f} mm</text>
        </svg>

        <svg viewBox="0 0 220 220" width="100%" height="auto" role="img" aria-label="Packaging top view">
            <text x="22" y="18" class="caption">{labels['top']}</text>
            <rect x="40" y="38" width="140" height="140" rx="8" fill="var(--pallet)" stroke="var(--pallet-stroke)" stroke-width="2"/>
            <circle cx="110" cy="108" r="{top_r:.2f}" fill="{coil_fill}" stroke="{coil_stroke}" stroke-width="3"/>
            <line x1="40" y1="194" x2="180" y2="194" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
            <line x1="40" y1="184" x2="40" y2="204" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
            <line x1="180" y1="184" x2="180" y2="204" stroke="var(--measure)" stroke-width="3" stroke-linecap="round"/>
            <text x="110" y="190" fill="var(--measure)" font-size="13" font-weight="800" text-anchor="middle">{pallet_size_mm:.0f} mm</text>
            <text x="110" y="211" fill="var(--muted)" font-size="12" font-weight="700" text-anchor="middle">{labels['coil']} Ø {coil_diameter_mm:.0f} mm</text>
        </svg>
    </div>
</div>

<script>
(() => {{
    function luminance(rgb) {{
        const parts = (rgb || "").match(/\\d+(\\.\\d+)?/g);
        if (!parts || parts.length < 3) return null;
        const vals = parts.slice(0,3).map(Number).map(v => {{
            v = v / 255;
            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        }});
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2];
    }}
    function applyTheme() {{
        try {{
            const parentDoc = window.parent && window.parent.document;
            const bg = parentDoc ? window.parent.getComputedStyle(parentDoc.body).backgroundColor : "";
            const lum = luminance(bg);
            if (lum !== null) {{
                document.documentElement.dataset.theme = lum > 0.55 ? "light" : "dark";
                return;
            }}
        }} catch (err) {{}}
        document.documentElement.dataset.theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }}
    applyTheme();
    setInterval(applyTheme, 1000);
}})();
</script>
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

# Modello fisico semplificato:
# la macchina reale può arrivare a più strati perché il tubo si compatta radialmente
# mentre il guidatubo completa più corsa assiale a parità di metri lineari.
# Questi sono valori iniziali: l'operatore può regolarli nella UI.
DEFAULT_FATTORE_COMPATTAZIONE_RADIALE = 0.83
DEFAULT_FATTORE_PASSO_EFFETTIVO = 1.20

gradi_start = 0.0
guide_offset_x = 555.0

# =========================
# PRESETS
# =========================

@st.cache_data
def load_presets(path="Presets.csv"):
    # Excel sometimes saves preset files in Windows/Latin encoding instead of UTF-8.
    # Try UTF-8 first, then common Excel encodings.
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, sep=";", encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error

    # Clean column names exported by Excel
    df.columns = df.columns.astype(str).str.strip()

    # Normalize known calibration column aliases from Presets.csv.
    # Accept both the corrected Italian name and the common typo "compatazione".
    calibration_column_aliases = {
        "Fattore compatazione radiale": "Fattore compattazione radiale",
        "Fattore Compatazione Radiale": "Fattore compattazione radiale",
        "Fattore compatazione": "Fattore compattazione radiale",
        "Fattore Compatazione": "Fattore compattazione radiale",
        "Fattore modello compatazione": "Fattore compattazione radiale",
        "Fattore Modello Compatazione": "Fattore compattazione radiale",
        "Compatazione radiale fattore": "Fattore compattazione radiale",
        "Fattore Passo Effettivo": "Fattore passo effettivo",
        "Fattore modello passo": "Fattore passo effettivo",
        "Fattore Modello Passo": "Fattore passo effettivo",
        "Passo effettivo fattore": "Fattore passo effettivo",
    }
    rename_map = {}
    for alias, canonical in calibration_column_aliases.items():
        if alias in df.columns and canonical not in df.columns:
            rename_map[alias] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # Remove empty columns exported by Excel
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

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


def first_existing_value(row, columns, default=None):
    for column in columns:
        if column in row.index and not pd.isna(row[column]):
            return row[column]
    return default


def preset_cell_has_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text_value = str(value).strip()
    return text_value not in {"", "-", "nan", "NaN", "None"}


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


def tube_outer_diameter(rame, spessore):
    return COPPER_SIZES_MM.get(str(rame), 0.0) + 2.0 * float(spessore)



def make_preset_visual(row, language):
    def raw_value(*names, default="-"):
        for name in names:
            if name in row.index:
                value = safe_value(row, name)
                if value != "-":
                    return value
        return default

    def number(*names, default=0.0):
        value = raw_value(*names, default=default)
        return parse_float_value(value, default)

    def safe_text(value):
        return html.escape(str(value))

    def fmt(value):
        return html.escape(format_preset_value(value))

    def value_with_mm(value):
        text_value = str(value).strip()
        if text_value in {"", "-"}:
            return "-"
        if "mm" not in text_value.lower():
            text_value += " mm"
        return text_value

    def metric(label, value, suffix=""):
        value_text = format_preset_value(value).strip() if value is not None else "-"
        if value_text in {"", "-"}:
            return ""
        return f"""
        <div class="preview-metric">
            <div class="preview-metric-label">{html.escape(str(label))}</div>
            <div class="preview-metric-value">{html.escape(value_text + suffix)}</div>
        </div>
        """

    tipo_tubo = str(raw_value("Tipo tubo", default="Singolo")).strip().lower()
    is_doppio = tipo_tubo == "doppio"

    rame = raw_value("Diametro Rame")
    d_rame = COPPER_SIZES_MM.get(str(rame), parse_float_value(rame, 0.0))
    spessore = number("Spessore Guaina (mm)", default=0.0)
    d_tubo = number("Diametro esterno Guaina (mm)", default=d_rame + 2.0 * spessore)

    lunghezza = number("Lunghezza (m)", default=0.0)
    velocita_linea = number("Velocita linea (m/min)", default=0.0)
    soffiatori = raw_value("Soffiatori aria (mm)")

    labels = {
        "IT": {
            "title": "Sezione tubo",
            "single": "Singolo",
            "double": "Doppio",
            "upper": "Superiore",
            "lower": "Inferiore",
            "copper": "Rame",
            "foam": "Guaina",
            "outer": "Ø esterno",
            "foam_thickness": "Spessore guaina",
            "length": "Lunghezza",
            "line_speed": "Velocità linea",
            "air": "Soffiatori",
            "pair_height": "Altezza coppia",
        },
        "EN": {
            "title": "Tube section",
            "single": "Single",
            "double": "Double",
            "upper": "Upper",
            "lower": "Lower",
            "copper": "Copper",
            "foam": "Foam",
            "outer": "Outer Ø",
            "foam_thickness": "Foam thickness",
            "length": "Length",
            "line_speed": "Line speed",
            "air": "Air blowers",
            "pair_height": "Pair height",
        },
    }[language]

    if is_doppio:
        rame_inf = str(raw_value("Diametro rame inferiore", "Diametro Rame inferiore", default="3/8")).strip()
        rame_sup = str(raw_value("Diametro rame superiore", "Diametro Rame superiore", default="1/4")).strip()
        spessore_inf = number("Spessore guaina inferiore", "Spessore Guaina inferiore (mm)", default=0.0)
        spessore_sup = number("Spessore guaina superiore", "Spessore Guaina superiore (mm)", default=0.0)

        d_rame_inf = COPPER_SIZES_MM.get(str(rame_inf), parse_float_value(rame_inf, 0.0))
        d_rame_sup = COPPER_SIZES_MM.get(str(rame_sup), parse_float_value(rame_sup, 0.0))
        d_inf = d_rame_inf + 2.0 * spessore_inf
        d_sup = d_rame_sup + 2.0 * spessore_sup
        d_pair = d_inf + d_sup

        r_inf = 58.0
        r_sup = max(30.0, min(50.0, r_inf * d_sup / max(d_inf, 1e-9)))
        cx = 210.0
        cy_inf = 178.0
        cy_sup = cy_inf - r_inf - r_sup

        copper_r_inf = max(12.0, r_inf * d_rame_inf / max(d_inf, 1e-9))
        copper_r_sup = max(10.0, r_sup * d_rame_sup / max(d_sup, 1e-9))

        svg = f"""
        <svg viewBox="0 0 620 300" class="section-svg" role="img" aria-label="Double tube section">
            <line x1="{cx:.1f}" y1="34" x2="{cx:.1f}" y2="252" class="center-line"/>
            <line x1="96" y1="{cy_sup:.1f}" x2="324" y2="{cy_sup:.1f}" class="center-line"/>
            <line x1="96" y1="{cy_inf:.1f}" x2="324" y2="{cy_inf:.1f}" class="center-line"/>

            <circle cx="{cx:.1f}" cy="{cy_inf:.1f}" r="{r_inf:.1f}" class="foam"/>
            <circle cx="{cx:.1f}" cy="{cy_inf:.1f}" r="{copper_r_inf:.1f}" class="copper"/>
            <circle cx="{cx:.1f}" cy="{cy_inf:.1f}" r="{max(7, copper_r_inf * 0.55):.1f}" class="copper-hi"/>

            <circle cx="{cx:.1f}" cy="{cy_sup:.1f}" r="{r_sup:.1f}" class="foam"/>
            <circle cx="{cx:.1f}" cy="{cy_sup:.1f}" r="{copper_r_sup:.1f}" class="copper"/>
            <circle cx="{cx:.1f}" cy="{cy_sup:.1f}" r="{max(6, copper_r_sup * 0.55):.1f}" class="copper-hi"/>

            <line x1="338" y1="{cy_sup-r_sup:.1f}" x2="338" y2="{cy_inf+r_inf:.1f}" class="dim-line"/>
            <line x1="{cx+r_sup:.1f}" y1="{cy_sup-r_sup:.1f}" x2="350" y2="{cy_sup-r_sup:.1f}" class="dim-guide"/>
            <line x1="{cx+r_inf:.1f}" y1="{cy_inf+r_inf:.1f}" x2="350" y2="{cy_inf+r_inf:.1f}" class="dim-guide"/>
            <text x="364" y="{(cy_sup+cy_inf)/2:.1f}" class="dim-value vertical">{fmt(d_pair)} mm</text>
            <text x="388" y="{(cy_sup+cy_inf)/2:.1f}" class="dim-label vertical">{safe_text(labels['pair_height'])}</text>

            <line x1="300" y1="{cy_sup:.1f}" x2="434" y2="86" class="leader"/>
            <text x="450" y="78" class="call-label">{safe_text(labels['upper'])}</text>
            <text x="450" y="102" class="call-value">{safe_text(rame_sup)} · {fmt(d_sup)} mm</text>

            <line x1="300" y1="{cy_inf:.1f}" x2="434" y2="190" class="leader"/>
            <text x="450" y="182" class="call-label">{safe_text(labels['lower'])}</text>
            <text x="450" y="206" class="call-value">{safe_text(rame_inf)} · {fmt(d_inf)} mm</text>
        </svg>
        """

        tag = labels["double"]
        metrics = "".join([
            metric(labels["double"], f"{rame_sup}/{rame_inf}"),
            metric(labels["upper"], d_sup, " mm"),
            metric(labels["lower"], d_inf, " mm"),
            metric(labels["pair_height"], d_pair, " mm"),
            metric(labels["line_speed"], velocita_linea, " m/min"),
            metric(labels["air"], value_with_mm(soffiatori)),
        ])
    else:
        foam_r = 78.0
        copper_r = max(18.0, min(42.0, foam_r * d_rame / max(d_tubo, 1e-9)))
        cx = 210.0
        cy = 145.0

        svg = f"""
        <svg viewBox="0 0 620 300" class="section-svg" role="img" aria-label="Tube section">
            <line x1="98" y1="{cy:.1f}" x2="322" y2="{cy:.1f}" class="center-line"/>
            <line x1="{cx:.1f}" y1="34" x2="{cx:.1f}" y2="256" class="center-line"/>

            <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{foam_r:.1f}" class="foam"/>
            <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{copper_r:.1f}" class="copper"/>
            <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{max(8, copper_r * 0.55):.1f}" class="copper-hi"/>

            <line x1="{cx-foam_r:.1f}" y1="262" x2="{cx+foam_r:.1f}" y2="262" class="dim-line"/>
            <line x1="{cx-foam_r:.1f}" y1="246" x2="{cx-foam_r:.1f}" y2="278" class="dim-line"/>
            <line x1="{cx+foam_r:.1f}" y1="246" x2="{cx+foam_r:.1f}" y2="278" class="dim-line"/>
            <text x="{cx:.1f}" y="252" class="dim-value">{fmt(d_tubo)} mm</text>
            <text x="{cx:.1f}" y="286" class="dim-label">{safe_text(labels['outer'])}</text>

            <line x1="{cx+copper_r:.1f}" y1="{cy:.1f}" x2="{cx+foam_r:.1f}" y2="{cy:.1f}" class="dim-line"/>
            <text x="{cx+(copper_r+foam_r)/2:.1f}" y="{cy-16:.1f}" class="dim-label">{fmt(spessore)} mm</text>

            <line x1="{cx+copper_r:.1f}" y1="{cy-8:.1f}" x2="420" y2="82" class="leader"/>
            <text x="438" y="76" class="call-label">{safe_text(labels['copper'])}</text>
            <text x="438" y="102" class="call-value">{safe_text(rame)}</text>

            <line x1="{cx+foam_r:.1f}" y1="{cy:.1f}" x2="420" y2="166" class="leader"/>
            <text x="438" y="160" class="call-label">{safe_text(labels['foam_thickness'])}</text>
            <text x="438" y="186" class="call-value">{fmt(spessore)} mm</text>

            <text x="438" y="232" class="call-label">{safe_text(labels['outer'])}</text>
            <text x="438" y="258" class="call-value">{fmt(d_tubo)} mm</text>
        </svg>
        """

        tag = labels["single"]
        metrics = "".join([
            metric(labels["copper"], rame),
            metric(labels["foam"], spessore, " mm"),
            metric(labels["outer"], d_tubo, " mm"),
            metric(labels["length"], lunghezza, " m"),
            metric(labels["line_speed"], velocita_linea, " m/min"),
            metric(labels["air"], value_with_mm(soffiatori)),
        ])

    return f"""
    <style>
    :root {{
        color-scheme: light;
        --component-text: #111827;
        --component-muted: rgba(75,85,99,0.74);
        --component-muted-strong: rgba(31,41,55,0.82);
        --component-border: rgba(17,24,39,0.12);
        --component-border-soft: rgba(17,24,39,0.09);
        --component-drawing-bg: transparent;
        --component-surface-solid: rgba(17,24,39,0.035);
        --foam-fill: rgba(226,232,240,0.96);
        --foam-stroke: rgba(100,116,139,0.50);
        --center-line: rgba(51,65,85,0.24);
        --dim-line: rgba(15,23,42,0.78);
        --dim-guide: rgba(71,85,105,0.38);
        --dim-label: rgba(51,65,85,0.76);
        --copper-fill: #C57E5A;
        --copper-stroke: #7a4124;
        --copper-highlight: #E7B18F;
        --component-page-bg: #f8fafc;
    }}

    html[data-theme="dark"] {{
        color-scheme: dark;
        --component-text: #f8fafc;
        --component-muted: rgba(226,232,240,0.66);
        --component-muted-strong: rgba(226,232,240,0.82);
        --component-border: rgba(226,232,240,0.14);
        --component-border-soft: rgba(226,232,240,0.10);
        --component-drawing-bg: transparent;
        --component-surface-solid: rgba(226,232,240,0.055);
        --foam-fill: rgba(231,236,242,0.88);
        --foam-stroke: rgba(248,250,252,0.92);
        --center-line: rgba(226,232,240,0.24);
        --dim-line: rgba(235,241,248,0.92);
        --dim-guide: rgba(193,204,219,0.56);
        --dim-label: rgba(203,214,228,0.88);
        --component-page-bg: #020817;
    }}

    html, body {{
        margin:0;
        padding:0;
        width:100%;
        height:auto !important;
        min-height:0 !important;
        overflow:hidden;
        background:var(--component-page-bg) !important;
        color:var(--component-text);
        font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }}

    .preview-card {{
        width:100%;
        box-sizing:border-box;
        border:0 !important;
        outline:0 !important;
        box-shadow:none !important;
        background:var(--component-page-bg) !important;
        border-radius:0 !important;
        overflow:visible !important;
    }}

    .preview-head {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        padding:4px 8px 12px 8px;
        border-bottom:1px solid var(--component-border-soft);
        background:transparent !important;
    }}

    .preview-title {{
        font-size:20px;
        line-height:1.05;
        font-weight:780;
        letter-spacing:-0.035em;
        color:var(--component-text);
    }}

    .preview-tag {{
        flex:0 0 auto;
        min-height:30px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:999px;
        padding:0 12px;
        background:rgba(197,126,90,0.16);
        border:1px solid rgba(197,126,90,0.36);
        color:var(--component-text);
        font-size:12px;
        font-weight:900;
        letter-spacing:0.035em;
    }}

    .drawing-wrap {{
        margin:12px 8px;
        border-radius:18px;
        overflow:hidden;
        background:transparent;
        border:1px solid var(--component-border-soft);
        box-shadow:none !important;
    }}

    .section-svg {{
        width:100%;
        height:300px;
        display:block;
    }}

    .foam {{
        fill:var(--foam-fill);
        stroke:var(--foam-stroke);
        stroke-width:2.3;
    }}
    .copper {{
        fill:var(--copper-fill);
        stroke:var(--copper-stroke);
        stroke-width:2.0;
    }}
    .copper-hi {{
        fill:var(--copper-highlight);
        opacity:0.82;
    }}
    .center-line {{
        stroke:var(--center-line);
        stroke-width:1.15;
        stroke-dasharray:5 6;
    }}
    .dim-line {{
        stroke:var(--dim-line);
        stroke-width:1.9;
        stroke-linecap:round;
    }}
    .dim-guide,
    .leader {{
        stroke:var(--dim-guide);
        stroke-width:1.15;
        fill:none;
        stroke-dasharray:4 5;
    }}
    .dim-label {{
        fill:var(--dim-label);
        font-size:12px;
        font-weight:850;
        text-anchor:middle;
        letter-spacing:0.03em;
    }}
    .dim-value {{
        fill:var(--component-text);
        font-size:15px;
        font-weight:950;
        text-anchor:middle;
    }}
    .vertical {{
        transform-box:fill-box;
        transform-origin:center;
        transform:rotate(-90deg);
    }}
    .call-label {{
        fill:var(--dim-label);
        font-size:13px;
        font-weight:900;
        letter-spacing:0.055em;
        text-transform:uppercase;
        text-anchor:start;
    }}
    .call-value {{
        fill:var(--component-text);
        font-size:19px;
        font-weight:950;
        text-anchor:start;
    }}
    .metrics {{
        display:grid;
        grid-template-columns:repeat(3, minmax(0,1fr));
        gap:9px;
        padding:0 8px 0 8px;
    }}
    .preview-metric {{
        min-height:50px;
        border-radius:14px;
        padding:9px 10px;
        box-sizing:border-box;
        background:var(--component-surface-solid);
        border:1px solid var(--component-border-soft);
    }}
    .preview-metric-label {{
        font-size:10px;
        line-height:1.1;
        font-weight:900;
        letter-spacing:0.055em;
        text-transform:uppercase;
        color:var(--component-muted);
        margin-bottom:6px;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }}
    .preview-metric-value {{
        font-size:15px;
        line-height:1.06;
        font-weight:950;
        color:var(--component-text);
        overflow-wrap:anywhere;
    }}
    @media (max-width: 900px) {{
        .section-svg {{ height:300px; }}
        .metrics {{ grid-template-columns:1fr 1fr; }}
    }}

    /* Premium sweep per Anteprima tecnica */
    .drawing-wrap,
    .preview-metric,
    .preview-head,
    .preview-tag {{
        position: relative;
        overflow: hidden;
        isolation: isolate;
    }}

    .drawing-wrap > *,
    .preview-metric > *,
    .preview-head > *,
    .preview-tag > * {{
        position: relative;
        z-index: 2;
    }}

    .drawing-wrap::after,
    .preview-metric::after,
    .preview-head::after,
    .preview-tag::after {{
        content: "";
        position: absolute;
        top: -42%;
        bottom: -42%;
        left: -82%;
        width: 48%;
        pointer-events: none;
        border-radius: inherit;
        background: linear-gradient(
            105deg,
            transparent 0%,
            rgba(255,255,255,0.00) 25%,
            rgba(255,255,255,0.34) 48%,
            rgba(197,126,90,0.26) 56%,
            rgba(255,255,255,0.14) 64%,
            transparent 100%
        );
        transform: skewX(-16deg);
        opacity: 0;
        z-index: 4;
        mix-blend-mode: screen;
    }}

    .drawing-wrap:hover::after,
    .preview-metric:hover::after,
    .preview-head:hover::after,
    .preview-tag:hover::after {{
        opacity: 1;
        animation: pdmPreviewSweep 1.05s cubic-bezier(.2,.72,.22,1) both;
    }}

    @keyframes pdmPreviewSweep {{
        0% {{ left: -82%; opacity: 0; }}
        12% {{ opacity: 1; }}
        100% {{ left: 138%; opacity: 0; }}
    }}

    @media (hover: none) {{
        .drawing-wrap:active::after,
        .preview-metric:active::after,
        .preview-head:active::after,
        .preview-tag:active::after {{
            opacity: 1;
            animation: pdmPreviewSweep 1.05s cubic-bezier(.2,.72,.22,1) both;
        }}
    }}

    </style>

    <section class="preview-card">
        <div class="preview-head">
            <div class="preview-title">{safe_text(labels['title'])}</div>
            <div class="preview-tag">{safe_text(tag)}</div>
        </div>
        <div class="drawing-wrap">
            {svg}
        </div>
        <div class="metrics">{metrics}</div>
    </section>

    <script>
    (() => {{
        function parseRgb(value) {{
            if (!value) return null;
            value = String(value).trim();
            if (value.startsWith("#")) {{
                let hex = value.slice(1);
                if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
                const n = parseInt(hex, 16);
                if (Number.isNaN(n)) return null;
                return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
            }}
            const match = value.match(/rgba?\\(([^)]+)\\)/i);
            if (!match) return null;
            const parts = match[1].split(",").map(x => parseFloat(x.trim()));
            if (parts.length < 3) return null;
            return [parts[0], parts[1], parts[2]];
        }}

        function luminance(rgb) {{
            const mapped = rgb.map(v => {{
                v = Math.max(0, Math.min(255, v)) / 255;
                return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
            }});
            return 0.2126 * mapped[0] + 0.7152 * mapped[1] + 0.0722 * mapped[2];
        }}

        function applyTheme() {{
            try {{
                const parentDoc = window.parent && window.parent.document;
                if (parentDoc) {{
                    const rootStyle = window.parent.getComputedStyle(parentDoc.documentElement);
                    const bodyStyle = window.parent.getComputedStyle(parentDoc.body);
                    const candidates = [
                        rootStyle.getPropertyValue("--background-color"),
                        bodyStyle.backgroundColor,
                        rootStyle.getPropertyValue("--secondary-background-color")
                    ];
                    for (const candidate of candidates) {{
                        const rgb = parseRgb(candidate);
                        if (rgb) {{
                            document.documentElement.dataset.theme = luminance(rgb) < 0.35 ? "dark" : "light";
                            document.documentElement.style.setProperty("--component-page-bg", candidate.trim());
                            return;
                        }}
                    }}
                }}
            }} catch (err) {{}}
            document.documentElement.dataset.theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        }}
        applyTheme();
        setInterval(applyTheme, 1000);
    }})();
    </script>
    """

def current_calculator_snapshot():
    simulation_mode_raw = str(st.session_state.get("calc_simulation_mode", "Fattori correzione"))
    simulation_mode_normalized = "Fattori correzione" if simulation_mode_raw in {"Fattori correzione", "Correction factors"} else "Ideale macchina"

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
        "calc_simulation_mode": simulation_mode_normalized,
        "calc_fattore_passo_effettivo": float(st.session_state.get("calc_fattore_passo_effettivo", DEFAULT_FATTORE_PASSO_EFFETTIVO)),
        "calc_fattore_compattazione_radiale": float(st.session_state.get("calc_fattore_compattazione_radiale", DEFAULT_FATTORE_COMPATTAZIONE_RADIALE)),
        "calc_tube_layout": str(st.session_state.get("calc_tube_layout", "Singolo")),
        "calc_rame_inf": str(st.session_state.get("calc_rame_inf", "3/8")),
        "calc_spessore_inf": float(st.session_state.get("calc_spessore_inf", 7.0)),
        "calc_rame_sup": str(st.session_state.get("calc_rame_sup", "1/4")),
        "calc_spessore_sup": float(st.session_state.get("calc_spessore_sup", 7.0)),
    }


def clear_active_preset_state():
    st.session_state.pop("loaded_preset_name", None)
    st.session_state.pop("loaded_preset_values", None)
    st.session_state["show_preset_loaded_success"] = False


def sync_active_preset_state():
    loaded_name = st.session_state.get("loaded_preset_name")
    loaded_values = st.session_state.get("loaded_preset_values")

    if not loaded_name or not loaded_values:
        st.session_state["preset_values_modified"] = False
        st.session_state["modified_preset_fields"] = []
        return

    current = current_calculator_snapshot()
    modified_fields = []

    for key, loaded_value in loaded_values.items():
        current_value = current.get(key)
        if isinstance(loaded_value, str):
            if str(current_value).strip() != str(loaded_value).strip():
                modified_fields.append(key)
        else:
            try:
                if abs(float(current_value) - float(loaded_value)) > 1e-9:
                    modified_fields.append(key)
            except Exception:
                if str(current_value).strip() != str(loaded_value).strip():
                    modified_fields.append(key)

    st.session_state["preset_values_modified"] = bool(modified_fields)
    st.session_state["modified_preset_fields"] = modified_fields


def apply_preset_to_calculator(row):
    tipo_tubo = str(row.get("Tipo tubo", "Singolo")).strip().lower()

    if tipo_tubo == "doppio":
        rame_inf = str(first_existing_value(
            row,
            ["Diametro rame inferiore", "Diametro Rame inferiore"],
            "3/8",
        )).strip()

        rame_sup = str(first_existing_value(
            row,
            ["Diametro rame superiore", "Diametro Rame superiore"],
            "1/4",
        )).strip()

        if rame_inf not in COPPER_SIZES_MM:
            rame_inf = "3/8"
        if rame_sup not in COPPER_SIZES_MM:
            rame_sup = "1/4"

        spessore_inf = parse_float_value(first_existing_value(
            row,
            ["Spessore guaina inferiore", "Spessore Guaina inferiore (mm)"],
            7.0,
        ), 7.0)

        spessore_sup = parse_float_value(first_existing_value(
            row,
            ["Spessore guaina superiore", "Spessore Guaina superiore (mm)"],
            7.0,
        ), 7.0)

        st.session_state["calc_tube_layout"] = "Doppio"
        st.session_state["calc_rame_inf"] = rame_inf
        st.session_state["calc_spessore_inf"] = spessore_inf
        st.session_state["calc_rame_sup"] = rame_sup
        st.session_state["calc_spessore_sup"] = spessore_sup

        # Keep a coherent single-tube fallback value, although the render uses the doppio fields.
        st.session_state["calc_rame"] = rame_sup
        st.session_state["calc_spessore"] = spessore_sup

    else:
        rame = str(row.get("Diametro Rame", "1/4")).strip()
        if rame not in COPPER_SIZES_MM:
            rame = "1/4"

        st.session_state["calc_tube_layout"] = "Singolo"
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

    # Experimental model calibration values collected on the real product.
    # Add these columns to Presets.csv so each product/family can carry its own simulation correction.
    fattore_passo_raw = first_existing_value(
        row,
        [
            "Fattore passo effettivo",
            "Fattore Passo Effettivo",
            "Fattore modello passo",
            "Fattore Modello Passo",
            "Passo effettivo fattore",
        ],
        None,
    )
    fattore_compattazione_raw = first_existing_value(
        row,
        [
            "Fattore compattazione radiale",
            "Fattore Compattazione Radiale",
            "Fattore compatazione radiale",
            "Fattore Compatazione Radiale",
            "Fattore compatazione",
            "Fattore Compatazione",
            "Fattore modello compattazione",
            "Fattore Modello Compattazione",
            "Fattore modello compatazione",
            "Fattore Modello Compatazione",
            "Compattazione radiale fattore",
            "Compatazione radiale fattore",
        ],
        None,
    )

    has_passo_factor = preset_cell_has_value(fattore_passo_raw)
    has_compattazione_factor = preset_cell_has_value(fattore_compattazione_raw)
    has_complete_correction_factors = bool(has_passo_factor and has_compattazione_factor)

    if has_complete_correction_factors:
        st.session_state["calc_fattore_passo_effettivo"] = parse_float_value(fattore_passo_raw, 1.0)
        st.session_state["calc_fattore_compattazione_radiale"] = parse_float_value(fattore_compattazione_raw, 1.0)
        st.session_state["calc_simulation_mode"] = "Fattori correzione"
    else:
        # No complete experimental calibration in Presets.csv: the simulation must start in ideal mode.
        st.session_state["calc_fattore_passo_effettivo"] = parse_float_value(fattore_passo_raw, 1.0) if has_passo_factor else 1.0
        st.session_state["calc_fattore_compattazione_radiale"] = parse_float_value(fattore_compattazione_raw, 1.0) if has_compattazione_factor else 1.0
        st.session_state["calc_simulation_mode"] = "Ideale macchina"

    st.session_state["preset_has_correction_factors"] = has_complete_correction_factors
    st.session_state["loaded_preset_name"] = safe_value(row, "Prodotto")
    st.session_state["loaded_preset_values"] = current_calculator_snapshot()
    st.session_state["preset_values_modified"] = False
    st.session_state["modified_preset_fields"] = []
    st.session_state["changed_values_pulse"] = True
    st.session_state["show_preset_loaded_success"] = True


FIELD_LABELS_IT = {
    "calc_rame": "Rame",
    "calc_spessore": "Guaina",
    "calc_lunghezza": "Lunghezza",
    "calc_diametro_aspo": "Ø aspo",
    "calc_spalla": "Spalla",
    "calc_passo_visuale": "Passo",
    "calc_incremento_visuale": "Incremento",
    "calc_rit_b": "Ritardo base",
    "calc_rit_t": "Ritardo spalla",
    "calc_simulation_mode": "Modo simulazione",
    "calc_fattore_passo_effettivo": "Fattore passo effettivo",
    "calc_fattore_compattazione_radiale": "Fattore compattazione radiale",
    "calc_tube_layout": "Tipo tubo",
    "calc_rame_inf": "Rame inferiore",
    "calc_spessore_inf": "Guaina inferiore",
    "calc_rame_sup": "Rame superiore",
    "calc_spessore_sup": "Guaina superiore",
}

FIELD_LABELS_EN = {
    "calc_rame": "Copper",
    "calc_spessore": "Foam",
    "calc_lunghezza": "Length",
    "calc_diametro_aspo": "Spool Ø",
    "calc_spalla": "Width",
    "calc_passo_visuale": "Pitch",
    "calc_incremento_visuale": "Layer increment",
    "calc_rit_b": "Base delay",
    "calc_rit_t": "Shoulder delay",
    "calc_simulation_mode": "Simulation mode",
    "calc_fattore_passo_effettivo": "Effective pitch factor",
    "calc_fattore_compattazione_radiale": "Radial compaction factor",
    "calc_tube_layout": "Tube type",
    "calc_rame_inf": "Lower copper",
    "calc_spessore_inf": "Lower foam",
    "calc_rame_sup": "Upper copper",
    "calc_spessore_sup": "Upper foam",
}


def modified_field_labels(language):
    labels = FIELD_LABELS_IT if language == "IT" else FIELD_LABELS_EN
    fields = st.session_state.get("modified_preset_fields", [])
    return [labels.get(field, field) for field in fields]


def make_preset_export_html(product_name, selected_row, language, status_items=None):
    snapshot = current_calculator_snapshot()
    modified = bool(st.session_state.get("preset_values_modified", False))
    field_labels = FIELD_LABELS_IT if language == "IT" else FIELD_LABELS_EN

    title = "Scheda preset avvolgimento" if language == "IT" else "Winding preset sheet"
    calc_title = "Valori calcolatore" if language == "IT" else "Calculator values"
    csv_title = "Valori preset" if language == "IT" else "Preset values"
    print_label = "Stampa scheda" if language == "IT" else "Print sheet"
    modified_label = "Sì" if modified and language == "IT" else ("Yes" if modified else ("No" if language != "IT" else "No"))
    rows = []
    for key, label in field_labels.items():
        value = snapshot.get(key, "-")
        rows.append(f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(format_preset_value(value))}</td></tr>")

    source_rows = []
    for col in selected_row.index:
        val = safe_value(selected_row, col)
        if val != "-":
            source_rows.append(f"<tr><th>{html.escape(str(col))}</th><td>{html.escape(str(val))}</td></tr>")

    status_html = ""
    if status_items:
        cards = []
        for item in status_items:
            cards.append(
                f"<div class='status'><b>{html.escape(str(item.get('label','')))}</b>"
                f"<span>{html.escape(str(item.get('value','')))}</span>"
                f"<small>{html.escape(str(item.get('note','')))}</small></div>"
            )
        status_html = "<div class='statusgrid'>" + "".join(cards) + "</div>"

    return f"""<!doctype html>
<html lang="{html.escape(language.lower())}">
<head>
<meta charset="utf-8">
<title>{html.escape(str(product_name))} · {html.escape(title)}</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:32px;color:#111827;background:#f8fafc;}}
.header{{border-left:6px solid #C57E5A;padding:16px 20px;background:white;border-radius:16px;box-shadow:0 8px 20px rgba(0,0,0,.06);}}
h1{{margin:0;font-size:28px;}}
.subtitle{{margin-top:8px;color:#64748b;font-weight:700;}}
.badge{{display:inline-block;margin-top:12px;padding:7px 11px;border-radius:999px;background:#C57E5A;color:white;font-weight:800;font-size:12px;}}
.print-actions{{display:flex;justify-content:flex-end;margin:0 0 18px 0;}}
.print-btn{{border:none;border-radius:999px;padding:10px 16px;background:#C57E5A;color:white;font-weight:900;cursor:pointer;box-shadow:0 8px 18px rgba(197,126,90,.22);}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:24px;}}
.card{{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:18px;box-shadow:0 6px 16px rgba(0,0,0,.045);}}
h2{{margin:0 0 14px 0;font-size:18px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th{{text-align:left;color:#64748b;width:44%;padding:8px;border-bottom:1px solid #e5e7eb;}}
td{{font-weight:800;padding:8px;border-bottom:1px solid #e5e7eb;}}
.statusgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px;}}
.status{{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:14px;}}
.status b{{display:block;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.06em;}}
.status span{{display:block;font-size:22px;font-weight:900;margin-top:6px;}}
.status small{{display:block;color:#64748b;margin-top:6px;}}
@media print{{body{{background:white;margin:18px}}.card,.header,.status{{box-shadow:none}}.print-actions{{display:none}}}}
</style>
</head>
<body>
<div class="print-actions"><button class="print-btn" onclick="window.print()">{html.escape(print_label)}</button></div>
<div class="header">
<h1>{html.escape(str(product_name))}</h1>
<div class="subtitle">{html.escape(title)}</div>
<span class="badge">Preset modificato: {html.escape(modified_label)}</span>
</div>
{status_html}
<div class="grid">
<div class="card"><h2>{html.escape(calc_title)}</h2><table>{"".join(rows)}</table></div>
<div class="card"><h2>{html.escape(csv_title)}</h2><table>{"".join(source_rows)}</table></div>
</div>
</body>
</html>"""



def make_csv_preset_print_html(product_name, selected_row, language):
    title = "Scheda preset" if language == "IT" else "Preset sheet"
    subtitle = "Preset originale · valori ufficiali" if language == "IT" else "Original preset · official values"
    print_label = "Stampa scheda" if language == "IT" else "Print sheet"
    section_title = "Parametri preset" if language == "IT" else "Preset parameters"
    footer = "Preset originale · nessuna cattura render inclusa" if language == "IT" else "Original preset · no render capture included"

    source_rows = []
    for col in selected_row.index:
        val = safe_value(selected_row, col)
        if val != "-":
            label = param_label(col, language)
            source_rows.append((str(label), str(val)))

    rows_html = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in source_rows
    )

    # Print layout: stretch the Preset sheet vertically so the table uses the full A4 height.
    # The row height is calculated dynamically from the number of visible preset fields.
    row_count = max(1, len(source_rows))
    row_height_mm = max(5.8, min(8.6, 238.0 / row_count))

    return f"""<!doctype html>
<html lang="{html.escape(language.lower())}">
<head>
<meta charset="utf-8">
<title>{html.escape(str(product_name))} · {html.escape(title)}</title>
<style>
@page{{size:A4 portrait;margin:8mm;}}
*{{box-sizing:border-box;}}
:root{{--csv-row-height:{row_height_mm:.3f}mm;}}
html,body{{min-height:100%;}}
body{{font-family:Inter,Arial,sans-serif;margin:18px;color:#111827;background:#f8fafc;}}
.print-actions{{display:flex;justify-content:flex-end;margin:0 0 10px 0;}}
.print-btn{{border:none;border-radius:999px;padding:10px 17px;background:#C57E5A;color:white;font-weight:950;cursor:pointer;box-shadow:0 10px 22px rgba(197,126,90,.24);}}
.sheet{{min-height:calc(100vh - 46px);display:flex;flex-direction:column;background:white;border:1px solid #e5e7eb;border-radius:18px;padding:17px 18px;box-shadow:0 8px 20px rgba(0,0,0,.055);}}
.header{{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;border-left:6px solid #C57E5A;padding:8px 0 8px 15px;margin-bottom:12px;flex:0 0 auto;}}
h1{{margin:0;font-size:24px;line-height:1.02;letter-spacing:-.025em;}}
.subtitle{{margin-top:5px;color:#64748b;font-weight:750;font-size:12px;line-height:1.2;}}
.badge{{display:inline-flex;align-items:center;justify-content:center;padding:7px 11px;border-radius:999px;background:#f1f5f9;color:#334155;border:1px solid #e2e8f0;font-weight:900;font-size:11px;white-space:nowrap;}}
h2{{margin:0 0 8px 0;font-size:14px;line-height:1.1;flex:0 0 auto;}}
.param-table{{width:100%;height:100%;border-collapse:separate;border-spacing:0;border:1px solid #e5e7eb;border-radius:13px;overflow:hidden;background:#ffffff;font-size:10px;line-height:1.08;table-layout:fixed;flex:1 1 auto;}}
.param-table tr{{height:var(--csv-row-height);}}
th{{text-align:left;color:#475569;width:46%;padding:4.7px 7px;border-bottom:1px solid #e5e7eb;background:#f8fafc;font-weight:850;vertical-align:middle;}}
td{{font-weight:900;padding:4.7px 7px;border-bottom:1px solid #e5e7eb;word-break:break-word;color:#0f172a;vertical-align:middle;}}
tr:last-child th,tr:last-child td{{border-bottom:none;}}
.footer{{margin-top:8px;font-size:9.5px;color:#94a3b8;font-weight:750;flex:0 0 auto;}}
@media print{{
    html,body{{height:100%;}}
    body{{background:white;margin:0;}}
    .print-actions{{display:none;}}
    .sheet{{height:calc(297mm - 16mm);min-height:calc(297mm - 16mm);border:none;box-shadow:none;border-radius:0;padding:0;}}
    .header{{margin-bottom:7mm;padding:3mm 0 3mm 4mm;}}
    h1{{font-size:21px;}}
    .subtitle{{font-size:10.5px;margin-top:3px;}}
    .badge{{font-size:9.5px;padding:4px 8px;}}
    h2{{font-size:12px;margin-bottom:2.2mm;}}
    .param-table{{font-size:9px;line-height:1.04;}}
    .param-table tr{{height:var(--csv-row-height);}}
    th,td{{padding:1.2mm 2mm;}}
    .footer{{font-size:8.4px;margin-top:2mm;}}
}}
</style>
</head>
<body>
<div class="print-actions"><button class="print-btn" onclick="window.print()">{html.escape(print_label)}</button></div>
<main class="sheet">
    <div class="header">
        <div>
            <h1>{html.escape(str(product_name))}</h1>
            <div class="subtitle">{html.escape(subtitle)}</div>
        </div>
        <span class="badge">{html.escape(title)}</span>
    </div>
    <h2>{html.escape(section_title)}</h2>
    <table class="param-table">{rows_html}</table>
    <div class="footer">{html.escape(footer)}</div>
</main>
</body>
</html>"""


def make_csv_preset_pdf_bytes(product_name, selected_row, language):
    """Build a one-page portrait PDF for the original Preset."""
    if SimpleDocTemplate is None:
        return None

    title = "Scheda preset" if language == "IT" else "Preset sheet"
    subtitle = "Preset originale - valori ufficiali" if language == "IT" else "Original preset - official values"
    section_title = "Parametri preset" if language == "IT" else "Preset parameters"
    footer = "Preset originale da file preset - nessuna cattura render inclusa" if language == "IT" else "Original preset from Presets.csv - no render capture included"

    source_rows = []
    for col in selected_row.index:
        val = safe_value(selected_row, col)
        if val != "-":
            label = param_label(col, language)
            source_rows.append((str(label), str(val)))

    buffer = BytesIO()
    page_w, page_h = A4
    margin = 8 * mm
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=f"{product_name} - {title}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PDMTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=21, textColor=colors.HexColor("#111827"), spaceAfter=2)
    section_style = ParagraphStyle("PDMSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=12, textColor=colors.HexColor("#111827"), spaceAfter=4)
    label_style = ParagraphStyle("PDMLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.4, leading=8.2, textColor=colors.HexColor("#475569"))
    value_style = ParagraphStyle("PDMValue", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.6, leading=8.4, textColor=colors.HexColor("#0F172A"))
    footer_style = ParagraphStyle("PDMFooter", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.2, leading=8, textColor=colors.HexColor("#94A3B8"))

    story = []
    header_table = Table(
        [[
            Paragraph(f"<b>{html.escape(str(product_name))}</b><br/><font size='8' color='#64748B'>{html.escape(subtitle)}</font>", title_style),
            Paragraph(f"<b>{html.escape(title)}</b>", value_style),
        ]],
        colWidths=[(page_w - 2 * margin) * 0.72, (page_w - 2 * margin) * 0.28],
    )
    header_table.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, 0), 4, colors.HexColor("#C57E5A")),
        ("LEFTPADDING", (0, 0), (0, 0), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 5))
    story.append(Paragraph(section_title, section_style))

    data = [[Paragraph(label, label_style), Paragraph(value, value_style)] for label, value in source_rows]
    usable_h = page_h - 2 * margin
    header_h = 35 * mm
    footer_h = 7 * mm
    table_h = max(150 * mm, usable_h - header_h - footer_h)
    row_h = table_h / max(1, len(data))

    table = Table(
        data,
        colWidths=[(page_w - 2 * margin) * 0.46, (page_w - 2 * margin) * 0.54],
        rowHeights=[row_h] * len(data),
        repeatRows=0,
    )
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5E7EB")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(footer, footer_style))

    doc.build(story)
    return buffer.getvalue()

def build_simulation_print_payload(product_name, language, tube_diameter_label, lunghezza, diametro_aspo, spalla, passo_visuale, incremento_visuale, rit_b, rit_t, visual_metrics, status_items):
    if language == "IT":
        title = "Scheda simulazione"
        subtitle = "Proposta di simulazione · valori attuali del calcolatore"
        labels = {
            "product": "Prodotto", "tube": "Ø tubo", "length": "Lunghezza", "spool": "Ø aspo", "width": "Spalla",
            "pitch": "Passo", "increment": "Incremento strato", "delay_base": "Ritardo base", "delay_shoulder": "Ritardo spalla",
            "radial": "Diametro radiale max", "footprint": "Ingombro XY", "wound": "Lunghezza avvolta"
        }
        print_label = "Stampa scheda"
        capture_label = "Cattura render"
    else:
        title = "Simulation sheet"
        subtitle = "Simulation proposal · current calculator values"
        labels = {
            "product": "Product", "tube": "Tube Ø", "length": "Length", "spool": "Spool Ø", "width": "Width",
            "pitch": "Pitch", "increment": "Layer increment", "delay_base": "Base delay", "delay_shoulder": "Shoulder delay",
            "radial": "Max radial diameter", "footprint": "XY footprint", "wound": "Wound length"
        }
        print_label = "Print sheet"
        capture_label = "Render capture"

    rows = [
        [labels["product"], str(product_name)],
        [labels["tube"], str(tube_diameter_label)],
        [labels["length"], f"{float(lunghezza):.2f} m"],
        [labels["spool"], f"{float(diametro_aspo):.2f} mm"],
        [labels["width"], f"{float(spalla):.2f} mm"],
        [labels["pitch"], f"{float(passo_visuale):.2f} mm"],
        [labels["increment"], f"{float(incremento_visuale):.2f} mm"],
        [labels["delay_base"], f"{float(rit_b):.2f}°"],
        [labels["delay_shoulder"], f"{float(rit_t):.2f}°"],
        [labels["radial"], f"{float(visual_metrics.get('diam_radiale', 0)):.1f} mm"],
        [labels["footprint"], f"{float(visual_metrics.get('max_xy_span', 0)):.1f} mm"],
        [labels["wound"], f"{float(visual_metrics.get('wound_length_m', 0)):.3f} m"],
    ]

    return {
        "title": title,
        "subtitle": subtitle,
        "product": str(product_name),
        "print_label": print_label,
        "capture_label": capture_label,
        "rows": rows,
        "status_items": status_items or [],
    }

def render_preset_action_bar(selected_product, selected_row, language, modified, status_items=None):
    locked = bool(st.session_state.get("params_locked", False))
    field_list = modified_field_labels(language)
    modified_txt = "Modificato" if language == "IT" else "Modified"
    original_txt = "Originale" if language == "IT" else "Original"
    lock_txt = "Bloccato" if language == "IT" else "Locked"
    free_txt = "Editabile" if language == "IT" else "Editable"

    def gv(*names, default="-"):
        for name in names:
            if name in selected_row.index:
                value = safe_value(selected_row, name)
                if value != "-":
                    return value
        return default

    tipo = gv("Tipo tubo")
    lunghezza = gv("Lunghezza (m)")
    aspo = gv("Diametro aspo (mm)")
    spalla = gv("Spalla (mm)")

    status_badge = modified_txt if modified else original_txt
    lock_badge = lock_txt if locked else free_txt
    details = ", ".join(field_list[:4])
    if len(field_list) > 4:
        details += f" +{len(field_list) - 4}"
    if not details:
        details = "Nessuna modifica manuale" if language == "IT" else "No manual changes"

    chips = [
        ("Tubo" if language == "IT" else "Tube", tipo),
        ("Lunghezza" if language == "IT" else "Length", f"{lunghezza} m" if lunghezza != "-" else "-"),
        ("Aspo", f"Ø {aspo} mm" if aspo != "-" else "-"),
        ("Spalla", f"{spalla} mm" if spalla != "-" else "-"),
    ]
    chips_html = "".join(
        f"""
        <div class="preset-chip">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in chips
    )

    st.markdown(
        f"""
        <style>
        .preset-status-strip {{
            margin:12px 0 16px 0;
            padding:16px 18px;
            border-radius:18px;
            border:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 7px 18px rgba(0,0,0,0.055);
        }}
        .preset-status-top {{
            display:flex;
            align-items:flex-start;
            justify-content:space-between;
            gap:16px;
            flex-wrap:wrap;
        }}
        .preset-status-kicker {{
            font-size:11px;
            font-weight:900;
            letter-spacing:.06em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 58%, transparent);
            margin-bottom:5px;
        }}
        .preset-status-title {{
            font-size:24px;
            line-height:1.06;
            font-weight:950;
            letter-spacing:-.025em;
            color:var(--text-color);
            word-break:break-word;
        }}
        .preset-status-sub {{
            margin-top:6px;
            font-size:12px;
            line-height:1.25;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 62%, transparent);
        }}
        .preset-badges {{
            display:flex;
            gap:8px;
            flex-wrap:wrap;
            align-items:center;
            justify-content:flex-end;
        }}
        .preset-badge {{
            border-radius:999px;
            padding:8px 11px;
            font-size:11px;
            line-height:1;
            font-weight:900;
            letter-spacing:.045em;
            text-transform:uppercase;
            border:1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            background:color-mix(in srgb, var(--secondary-background-color) 82%, var(--background-color));
        }}
        .preset-badge.mod {{
            background:{'#f59e0b' if modified else '#C57E5A'};
            border-color:{'#f59e0b' if modified else '#C57E5A'};
            color:white;
        }}
        .preset-badge.lock {{
            background:{'#64748b' if locked else 'color-mix(in srgb, var(--secondary-background-color) 82%, var(--background-color))'};
            color:{'white' if locked else 'var(--text-color)'};
        }}
        .preset-chip-row {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin-top:14px;
        }}
        .preset-chip {{
            padding:10px 12px;
            border-radius:14px;
            border:1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            background:color-mix(in srgb, var(--text-color) 4%, transparent);
            min-height:58px;
        }}
        .preset-chip span {{
            display:block;
            font-size:10.5px;
            line-height:1.05;
            font-weight:900;
            letter-spacing:.05em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 58%, transparent);
            margin-bottom:7px;
        }}
        .preset-chip strong {{
            display:block;
            font-size:18px;
            line-height:1.06;
            font-weight:950;
            letter-spacing:-.018em;
            color:var(--text-color);
            word-break:break-word;
        }}
        @media(max-width:1000px){{
            .preset-chip-row {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            .preset-status-title {{ font-size:21px; }}
        }}
        </style>
        <div class="preset-status-strip">
            <div class="preset-status-top">
                <div>
                    <div class="preset-status-kicker">Preset attivo</div>
                    <div class="preset-status-title">{html.escape(str(selected_product))}</div>
                    <div class="preset-status-sub">{html.escape(details)}</div>
                </div>
                <div class="preset-badges">
                    <span class="preset-badge mod">{html.escape(status_badge)}</span>
                    <span class="preset-badge lock">{html.escape(lock_badge)}</span>
                </div>
            </div>
            <div class="preset-chip-row">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* Copper action button area: prevents the restore label from being cut */
        div[data-testid="stButton"] > button {
            min-height: 50px !important;
            height: auto !important;
            padding: 0.72rem 1.10rem !important;
            border-radius: 999px !important;
            background: #C57E5A !important;
            border: 1px solid color-mix(in srgb, #C57E5A 78%, var(--text-color)) !important;
            color: #FFFFFF !important;
            font-weight: 900 !important;
            white-space: normal !important;
            line-height: 1.15 !important;
            box-shadow: 0 8px 18px rgba(197,126,90,0.28) !important;
        }

        div[data-testid="stButton"] > button * {
            color: #FFFFFF !important;
            white-space: normal !important;
            line-height: 1.15 !important;
        }

        @media (max-width: 900px) {
            div[data-testid="stButton"] > button {
                min-height: 48px !important;
                font-size: 0.92rem !important;
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1.35, 1.0], gap="small")
    with b1:
        if st.button("Ripristina preset" if language == "IT" else "Restore preset", use_container_width=True, key="restore_preset_button"):
            st.session_state["restore_preset_request"] = str(selected_product)
            st.rerun()
    with b2:
        st.toggle("Blocca parametri" if language == "IT" else "Lock parameters", key="params_locked")


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
        "calc_simulation_mode": "Fattori correzione",
        "preset_has_correction_factors": False,
        "calc_fattore_passo_effettivo": DEFAULT_FATTORE_PASSO_EFFETTIVO,
        "calc_fattore_compattazione_radiale": DEFAULT_FATTORE_COMPATTAZIONE_RADIALE,
        "calc_tube_layout": "Singolo",
        "calc_rame_inf": "3/8",
        "calc_spessore_inf": 7.0,
        "calc_rame_sup": "1/4",
        "calc_spessore_sup": 7.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =========================
# LOGO
# =========================

def find_logo():
    # Use only the real PNG logo file. Do not generate a fake/fallback logo.
    candidates = [
        "New Logo PDM – rame.png",
        "New Logo PDM - rame.png",
        "New Logo PDM rame.png",
        "new_logo_pdm_rame.png",
        "logo_pdm.png",
        "pdm_logo.png",
        "logo.png",
    ]

    search_dirs = [
        ".",
        "assets",
        "asset",
        "static",
        "images",
        "img",
        "/mnt/data",
    ]

    for folder in search_dirs:
        for name in candidates:
            path = os.path.join(folder, name)
            if os.path.exists(path) and path.lower().endswith(".png"):
                return path

    patterns = []
    for folder in search_dirs:
        patterns.extend([
            os.path.join(folder, "*logo*.png"),
            os.path.join(folder, "*Logo*.png"),
            os.path.join(folder, "*PDM*.png"),
        ])

    for pattern in patterns:
        files = [f for f in glob.glob(pattern) if f.lower().endswith(".png")]
        if files:
            return files[0]

    return None


logo_path = find_logo()

# =========================
# HEADER
# =========================

# Header simple i estable:
# - cap st.columns al header, perquè en mòbil Streamlit les apila i crea espais gegants.
# - logo en un únic bloc normal.
# - selector idioma com a radio natiu, però posicionat a la dreta per CSS sense ocupar alçada.
current_lang = st.session_state.lang


def encode_logo_base64_clean(path):
    if not path:
        return ""

    try:
        raw = Path(path).read_bytes()
    except Exception:
        return ""

    try:
        from PIL import Image

        image = Image.open(BytesIO(raw)).convert("RGBA")
        bbox = image.getchannel("A").getbbox()
        if bbox:
            image = image.crop(bbox)
            pad_x = max(4, int(image.width * 0.025))
            pad_y = max(3, int(image.height * 0.025))
            canvas = Image.new(
                "RGBA",
                (image.width + 2 * pad_x, image.height + 2 * pad_y),
                (255, 255, 255, 0),
            )
            canvas.paste(image, (pad_x, pad_y), image)
            image = canvas

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()
    except Exception:
        pass

    return base64.b64encode(raw).decode("utf-8")


logo_b64 = encode_logo_base64_clean(logo_path)
logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="pdm-header-logo" alt="PDM logo" />' if logo_b64 else ""

st.markdown(
    f"""
    <style>
    :root {{
        --pdm-accent: #C57E5A;
    }}

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main,
    .main .block-container {{
        max-width: 100vw !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
    }}

    .main .block-container {{
        padding-top: 0.20rem !important;
    }}

    .pdm-header-shell {{
        position: relative !important;
        width: 100% !important;
        max-width: 100% !important;
        display: block !important;
        margin: 0 0 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        overflow: visible !important;
    }}

    .pdm-header-logo-wrap {{
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 0 0 !important;
        padding: 0 !important;
        line-height: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        overflow: visible !important;
        pointer-events: none;
    }}

    .pdm-lang-mini {{
        position: absolute !important;
        left: calc(50% + 150px) !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 50 !important;
        display: flex !important;
        align-items: center !important;
        gap: 7px !important;
        width: auto !important;
        max-width: calc(50vw - 18px) !important;
        overflow: visible !important;
        white-space: nowrap !important;
    }}

    .pdm-lang-mini a {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 45px !important;
        height: 32px !important;
        border-radius: 999px !important;
        text-decoration: none !important;
        color: var(--text-color) !important;
        background: color-mix(in srgb, var(--secondary-background-color) 94%, var(--background-color)) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent) !important;
        box-shadow: none !important;
        font-size: 0.70rem !important;
        line-height: 1 !important;
        font-weight: 900 !important;
        letter-spacing: 0.03em !important;
    }}

    .pdm-lang-mini a.is-active {{
        background: #C57E5A !important;
        border-color: #C57E5A !important;
        color: #ffffff !important;
        box-shadow: 0 6px 14px rgba(197,126,90,0.22) !important;
    }}

    .pdm-header-logo {{
        display: block !important;
        width: auto !important;
        height: clamp(150px, 10.2vw, 205px) !important;
        max-width: min(500px, 58vw) !important;
        object-fit: contain !important;
        filter: drop-shadow(0 10px 22px rgba(0,0,0,0.12));
    }}

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) {{
        position: absolute !important;
        top: 0.62rem !important;
        right: max(0.85rem, calc((100vw - 1800px) / 2 + 0.90rem)) !important;
        z-index: 100 !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) > label {{
        display: none !important;
    }}

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] {{
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        align-items: center !important;
        justify-content: flex-end !important;
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
    }}

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] label {{
        flex: 0 0 42px !important;
        width: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 999px !important;
        background: color-mix(in srgb, var(--secondary-background-color) 94%, var(--background-color)) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent) !important;
        box-shadow: none !important;
    }}

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] label:has(input:checked) {{
        background: #C57E5A !important;
        border-color: #C57E5A !important;
        box-shadow: 0 6px 14px rgba(197,126,90,0.22) !important;
    }}

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] label p {{
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: center !important;
        font-size: 0.70rem !important;
        line-height: 1 !important;
        font-weight: 900 !important;
        letter-spacing: 0.03em !important;
        white-space: nowrap !important;
    }}

    div[data-testid="stTabs"] {{
        margin-top: 0 !important;
        padding-top: 0 !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }}

    div[data-testid="stTabs"] [role="tablist"] {{
        margin-top: 0 !important;
        margin-bottom: 12px !important;
        padding-top: 0 !important;
        padding-bottom: 0.18rem !important;
        gap: clamp(16px, 1.8vw, 30px) !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
        scrollbar-width: none !important;
    }}

    div[data-testid="stTabs"] [role="tab"] {{
        min-height: 2.28rem !important;
        padding-top: 0.44rem !important;
        padding-bottom: 0.62rem !important;
    }}

    @media (max-width: 760px) {{
        .main .block-container {{
            padding-top: 0.08rem !important;
            padding-left: 1.00rem !important;
            padding-right: 1.00rem !important;
            width: 100vw !important;
            max-width: 100vw !important;
        }}

        .pdm-header-logo-wrap {{
            margin-bottom: 0.02rem !important;
        }}

        .pdm-header-logo {{
            height: clamp(76px, 22vw, 108px) !important;
            max-width: min(220px, 46vw) !important;
        }}

        div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) {{
            top: 0.34rem !important;
            right: 0.70rem !important;
        }}

        div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] {{
            gap: 3px !important;
        }}

        div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] label {{
            flex-basis: 28px !important;
            width: 28px !important;
            min-width: 28px !important;
            max-width: 28px !important;
            height: 26px !important;
            min-height: 26px !important;
        }}

        div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) div[role="radiogroup"] label p {{
            font-size: 0.58rem !important;
            letter-spacing: 0 !important;
        }}

        div[data-testid="stTabs"] [role="tablist"] {{
            display: flex !important;
            flex-wrap: nowrap !important;
            justify-content: stretch !important;
            gap: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0.42rem !important;
            padding: 0 !important;
            overflow-x: hidden !important;
            width: 100% !important;
        }}

        div[data-testid="stTabs"] [role="tab"] {{
            flex: 1 1 0 !important;
            width: 33.333% !important;
            max-width: 33.333% !important;
            min-width: 0 !important;
            min-height: 2.00rem !important;
            padding: 0.32rem 0.04rem 0.48rem 0.04rem !important;
            overflow: hidden !important;
        }}

        div[data-testid="stTabs"] [role="tab"] p {{
            font-size: clamp(0.64rem, 2.9vw, 0.80rem) !important;
            line-height: 1.04 !important;
            text-align: center !important;
            white-space: normal !important;
            word-break: normal !important;
            overflow-wrap: normal !important;
        }}

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {{
            left: 0.22rem !important;
            right: 0.22rem !important;
            bottom: 0.06rem !important;
            height: 2px !important;
        }}
    }}
    </style>
    <div class="pdm-header-shell">
        <div class="pdm-header-logo-wrap">{logo_html}</div>
        <nav class="pdm-lang-mini" aria-label="Language selector">
            <a class="{'is-active' if current_lang == 'IT' else ''}" href="?lang=IT" target="_self">IT</a>
            <a class="{'is-active' if current_lang == 'EN' else ''}" href="?lang=EN" target="_self">EN</a>
        </nav>
    </div>
    """,
    unsafe_allow_html=True,
)

lang = st.session_state.lang
t = TEXTS[lang]

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1800px;
        padding-top: 0.95rem;
        padding-bottom: 1.75rem;
        padding-left: 0.90rem;
        padding-right: 0.90rem;
    }

    [data-testid="stTabs"] {
        margin-top: 0 !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 32px;
        padding: 0;
        background: transparent;
        border-bottom: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
        margin: 0 0 18px 0;
        align-items: center;
    }

    [data-testid="stTabs"] button {
        font-weight: 780;
        min-height: 52px;
        padding: 0.50rem 0.18rem 0.64rem 0.18rem;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        border-bottom: 3px solid transparent !important;
        justify-content: center !important;
    }

    [data-testid="stTabs"] button p {
        text-align: center !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        background: transparent !important;
        box-shadow: none !important;
        border-bottom: 3px solid var(--pdm-accent) !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--pdm-accent) !important;
        border-bottom: 3px solid var(--pdm-accent) !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] p,
    [data-testid="stTabs"] button[aria-selected="true"] * {
        color: var(--pdm-accent) !important;
        font-weight: 900 !important;
    }

    [data-testid="stTabs"] button:hover p,
    [data-testid="stTabs"] button:hover * {
        color: color-mix(in srgb, var(--pdm-accent) 78%, var(--text-color)) !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px;
        min-height: 44px;
    }

    div[data-baseweb="select"] * {
        font-size: 0.98rem;
    }


    div[data-testid="stSelectbox"],
    div[data-testid="stNumberInput"],
    div[data-testid="stRadio"],
    div[data-testid="stMarkdownContainer"] {
        margin-top: 0 !important;
    }

    div[data-testid="stSelectbox"] > label,
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stRadio"] > label {
        margin-bottom: 0.28rem !important;
        font-weight: 760 !important;
    }

    /* v95 · refinement elegante mantenendo identità PDM */
    :root {
        --pdm-accent: #C57E5A;
        --pdm-radius: 18px;
        --pdm-radius-lg: 22px;
        --pdm-border: color-mix(in srgb, var(--text-color) 12%, transparent);
        --pdm-border-strong: color-mix(in srgb, var(--text-color) 18%, transparent);
        --pdm-surface: color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color));
        --pdm-surface-soft: color-mix(in srgb, var(--secondary-background-color) 74%, transparent);
        --pdm-shadow: 0 8px 22px rgba(0,0,0,0.075);
        --pdm-shadow-soft: 0 4px 14px rgba(0,0,0,0.055);
    }

    .main .block-container {
        letter-spacing: -0.006em;
    }

    div[data-testid="stMetric"] {
        border-radius: var(--pdm-radius);
        padding: 12px 14px;
        border: 1px solid var(--pdm-border);
        background: linear-gradient(180deg, var(--pdm-surface), color-mix(in srgb, var(--secondary-background-color) 96%, var(--background-color)));
        box-shadow: var(--pdm-shadow-soft);
    }

    div[data-testid="stMetric"] label {
        color: color-mix(in srgb, var(--text-color) 62%, transparent) !important;
        font-weight: 780 !important;
    }

    .stButton > button {
        border-radius: 14px;
        min-height: 46px;
        font-weight: 820;
        border-color: var(--pdm-border-strong);
        box-shadow: var(--pdm-shadow-soft);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--pdm-shadow);
    }

    hr {
        margin-top: 1.35rem;
        margin-bottom: 1.35rem;
        border-color: color-mix(in srgb, var(--text-color) 9%, transparent);
    }

    iframe {
        border-radius: 18px !important;
    }

    /* v110 · micro-interazioni eleganti */
    .summary-strip,
    .quick-card-v2,
    .semaphore-card,
    .tech-mini-card,
    .machine-card-native,
    .checklist-hero,
    .elegant-panel {
        transition:
            transform 0.16s ease,
            box-shadow 0.16s ease,
            border-color 0.16s ease,
            background 0.16s ease;
    }

    .summary-strip:hover,
    .quick-card-v2:hover,
    .semaphore-card:hover,
    .tech-mini-card:hover,
    .machine-card-native:hover,
    .elegant-panel:hover {
        transform: translateY(-1px);
        border-color: color-mix(in srgb, var(--pdm-accent) 28%, var(--text-color) 10%);
        box-shadow: 0 10px 24px rgba(0,0,0,0.10);
    }

    .machine-card-native:hover,
    .tech-mini-card:hover {
        background: linear-gradient(180deg,
            color-mix(in srgb, var(--secondary-background-color) 86%, var(--pdm-accent) 5%),
            color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
        );
    }

    .quick-card-v2:hover .quick-topline {
        opacity: 1;
    }

    [data-testid="stTabs"] button {
        transition:
            border-color 0.16s ease,
            color 0.16s ease,
            opacity 0.16s ease;
    }

    [data-testid="stTabs"] button:hover {
        opacity: 0.86;
        border-bottom-color: color-mix(in srgb, var(--pdm-accent) 45%, transparent) !important;
    }

    div[role="radiogroup"] {
        gap: 0.95rem;
        flex-wrap: wrap;
        align-items: stretch;
    }

    div[role="radiogroup"] label {
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
        border-radius: 999px;
        padding: 0.72rem 1.28rem;
        min-height: 50px;
        min-width: 120px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.075);
        transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, background 0.16s ease, filter 0.16s ease;
        box-sizing: border-box;
        position: relative;
    }

    div[role="radiogroup"] label input[type="radio"] {
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }

    div[role="radiogroup"] label > div {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        width:100%;
        height:100%;
        text-align:center;
        margin:0 !important;
        padding:0 !important;
    }

    div[role="radiogroup"] label p {
        width:100%;
        margin:0 !important;
        padding:0 !important;
        display:block !important;
        text-align:center !important;
        transform: translateY(0px);
    }

    /* Hide the default radio dot so the options look like real buttons */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[role="radiogroup"] label:hover {
        transform: translateY(-1px);
        border-color: color-mix(in srgb, var(--pdm-accent) 45%, var(--text-color) 12%);
        box-shadow: 0 9px 19px rgba(0,0,0,0.11);
        filter: brightness(1.035);
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: #C57E5A;
        border-color: #C57E5A;
        box-shadow: 0 0 0 2px rgba(197,126,90,0.30), 0 9px 19px rgba(0,0,0,0.16);
    }

    div[role="radiogroup"] label:has(input:checked):hover {
        filter: brightness(1.08);
        box-shadow: 0 0 0 2px rgba(197,126,90,0.38), 0 11px 22px rgba(0,0,0,0.18);
    }

    div[role="radiogroup"] label p {
        font-weight: 800;
        font-size: 0.99rem;
        line-height: 1.1;
        text-align: center;
        letter-spacing: -0.01em;
    }

    div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
    }

    /* Better tablet spacing and readability */

    /* v165 · selector compatti e touch-friendly: niente tastiera su iPad */
    div[data-testid="stRadio"]:has(input[value="IT"]),
    div[data-testid="stRadio"]:has(input[value="EN"]) {
        max-width: 180px;
        margin-left: auto;
    }

    div[data-testid="stRadio"]:has(input[value="IT"]) div[role="radiogroup"],
    div[data-testid="stRadio"]:has(input[value="EN"]) div[role="radiogroup"] {
        justify-content: flex-end;
        gap: 6px !important;
    }

    div[data-testid="stRadio"]:has(input[value="IT"]) div[role="radiogroup"] label,
    div[data-testid="stRadio"]:has(input[value="EN"]) div[role="radiogroup"] label {
        min-width: 58px !important;
        min-height: 42px !important;
        padding: 0.52rem 0.78rem !important;
    }

    /* Pills dei preset: più compatti e con wrap elegante */
    div[role="radiogroup"] {
        row-gap: 8px !important;
    }

    div[role="radiogroup"] label {
        cursor: pointer;
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            transition: none !important;
            animation: none !important;
        }
    }

    @media (max-width: 1280px) {
        .main .block-container {
            max-width: 100%;
            padding-left: 0.60rem;
            padding-right: 0.60rem;
        }

        [data-testid="stTabs"] button {
            min-height: 48px;
            font-size: 1rem;
        }

        div[role="radiogroup"] label {
            min-height: 50px;
            min-width: 112px;
            padding: 0.64rem 1.08rem;
        }

        div[role="radiogroup"] label p {
            font-size: 1.00rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            min-height: 46px;
        }

    }

    /* v135 · harmonia visual final: grids, cards, microtipografia */
    .summary-card,
    .preset-param-card,
    .quick-card-v2,
    .semaphore-card,
    .tech-mini-card,
    .machine-card-native,
    .preview-metric {
        border-radius: 18px !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent) !important;
        background: linear-gradient(180deg,
            color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
            color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
        ) !important;
        box-shadow: 0 7px 18px rgba(0,0,0,0.055) !important;
        position: relative;
        overflow: hidden;
    }

    .summary-card::after,
    .preset-param-card::after,
    .quick-card-v2::after,
    .semaphore-card::after,
    .tech-mini-card::after,
    .machine-card-native::after,
    .preview-metric::after,
    .summary-strip::after,
    .elegant-panel::after {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.11) 42%, transparent 68%);
        transform: translateX(-120%);
        opacity: 0;
        z-index: 1;
    }

    .summary-card:hover::after,
    .preset-param-card:hover::after,
    .quick-card-v2:hover::after,
    .semaphore-card:hover::after,
    .tech-mini-card:hover::after,
    .machine-card-native:hover::after,
    .preview-metric:hover::after,
    .summary-strip:hover::after,
    .elegant-panel:hover::after {
        opacity: 1;
        animation: pdmPresetCardShine 0.72s ease-out both;
    }

    .summary-card,
    .preset-param-card,
    .quick-card-v2,
    .semaphore-card,
    .tech-mini-card,
    .machine-card-native {
        min-height: 126px !important;
        padding: 16px 18px !important;
    }

    .tech-mini-card,
    .machine-card-native {
        min-height: 104px !important;
    }

    .summary-label,
    .preset-param-label,
    .quick-label-v2,
    .semaphore-label,
    .tech-mini-label,
    .machine-card-label-native,
    .preview-metric-label {
        font-size: 11px !important;
        line-height: 1.08 !important;
        font-weight: 900 !important;
        letter-spacing: 0.055em !important;
        text-transform: uppercase !important;
        color: color-mix(in srgb, var(--text-color) 60%, transparent) !important;
        margin-bottom: 7px !important;
    }

    .summary-value,
    .preset-param-value,
    .quick-value-v2,
    .semaphore-value,
    .tech-mini-value,
    .machine-card-value-native,
    .preview-metric-value {
        font-weight: 950 !important;
        line-height: 1.04 !important;
        letter-spacing: -0.022em !important;
        color: var(--text-color) !important;
    }

    .summary-value,
    .preset-param-value {
        font-size: clamp(22px, 1.75vw, 30px) !important;
    }

    .quick-value-v2,
    .semaphore-value,
    .tech-mini-value,
    .machine-card-value-native {
        font-size: clamp(19px, 1.32vw, 24px) !important;
    }

    .summary-note,
    .quick-note-v2,
    .semaphore-note {
        margin-top: 7px !important;
        font-size: 12px !important;
        line-height: 1.24 !important;
        font-weight: 650 !important;
        color: color-mix(in srgb, var(--text-color) 62%, transparent) !important;
    }

    .summary-card::before,
    .preset-param-card::before {
        width: 4px !important;
        background: var(--pdm-accent) !important;
        opacity: 0.88 !important;
    }

    .quick-grid-v2,
    .semaphore-grid,
    .tech-mini-grid {
        gap: 12px !important;
    }

    .pdm-action-bar,
    .summary-strip,
    .section-header,
    .workflow-step,
    .elegant-panel,
    .checklist-hero {
        border-radius: 18px !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent) !important;
        background: linear-gradient(180deg,
            color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
            color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
        ) !important;
        box-shadow: 0 7px 18px rgba(0,0,0,0.055) !important;
    }

    .pdm-action-bar,
    .summary-strip {
        padding: 16px 18px !important;
        margin: 12px 0 18px 0 !important;
    }

    .workflow-bar {
        gap: 12px !important;
        margin: 12px 0 20px 0 !important;
    }

    .workflow-step {
        min-height: 74px !important;
    }

    .workflow-title,
    .pdm-action-title,
    .section-title {
        letter-spacing: -0.018em !important;
    }

    .workflow-subtitle,
    .pdm-action-sub,
    .section-subtitle {
        font-size: 12.5px !important;
        line-height: 1.25 !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        min-height: 46px !important;
        border-color: color-mix(in srgb, var(--text-color) 16%, transparent) !important;
        background: color-mix(in srgb, var(--secondary-background-color) 92%, var(--background-color)) !important;
    }

    div[data-baseweb="input"] input {
        font-weight: 800 !important;
    }

    div[data-testid="stSelectbox"] > label,
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stRadio"] > label {
        font-size: 12px !important;
        line-height: 1.15 !important;
        font-weight: 850 !important;
        color: color-mix(in srgb, var(--text-color) 68%, transparent) !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button {
        border-radius: 999px !important;
        min-height: 46px !important;
        font-weight: 900 !important;
    }

    .section-badge,
    .workflow-num {
        width: 30px !important;
        height: 30px !important;
        min-width: 30px !important;
        font-size: 14px !important;
        box-shadow: 0 5px 13px rgba(197,126,90,0.24) !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    iframe {
        border-radius: 18px !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent) !important;
    }

    /* v160 · premium motion: subtil, industriale, no invadente */
    @keyframes pdmFadeUp {
        from { opacity: 0; transform: translateY(16px); filter: blur(2px); }
        to { opacity: 1; transform: translateY(0); filter: blur(0); }
    }

    @keyframes pdmPresetPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(197,126,90,0.00), 0 12px 30px rgba(0,0,0,0.075);
            border-color: color-mix(in srgb, var(--text-color) 13%, transparent);
        }
        42% {
            box-shadow: 0 0 0 7px rgba(197,126,90,0.24), 0 16px 34px rgba(0,0,0,0.11);
            border-color: rgba(197,126,90,0.66);
        }
        100% {
            box-shadow: 0 12px 30px rgba(0,0,0,0.075);
            border-color: color-mix(in srgb, var(--text-color) 13%, transparent);
        }
    }

    @keyframes pdmStatusIn {
        from { opacity: 0; transform: scale(0.986); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes pdmPillShine {
        from { transform: translateX(-125%); }
        to { transform: translateX(125%); }
    }

    .pdm-fade-up,
    .preset-hero,
    .elegant-panel {
        animation: pdmFadeUp 0.52s cubic-bezier(.16,1,.3,1) both;
    }

    .pdm-pulse {
        animation: pdmPresetPulse 1.05s cubic-bezier(.16,1,.3,1) both !important;
    }

    .pdm-status-animated {
        animation: pdmStatusIn 0.25s cubic-bezier(.2,.8,.2,1) both;
    }

    .preset-hero-chip,
    .summary-card,
    .preset-param-card,
    .quick-card-v2,
    .semaphore-card,
    .tech-mini-card,
    .machine-card-native,
    .preview-metric,
    .elegant-panel {
        will-change: transform, box-shadow, border-color;
    }

    div[role="radiogroup"] label {
        overflow: hidden;
        isolation: isolate;
    }

    div[role="radiogroup"] label:has(input:checked)::after {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.16) 46%, transparent 72%);
        animation: pdmPillShine 0.82s ease-out both;
    }

    div[role="radiogroup"] label p {
        position: relative;
        z-index: 1;
    }


    /* v165 · selector compatti e touch-friendly: niente tastiera su iPad */
    div[data-testid="stRadio"]:has(input[value="IT"]),
    div[data-testid="stRadio"]:has(input[value="EN"]) {
        max-width: 180px;
        margin-left: auto;
    }

    div[data-testid="stRadio"]:has(input[value="IT"]) div[role="radiogroup"],
    div[data-testid="stRadio"]:has(input[value="EN"]) div[role="radiogroup"] {
        justify-content: flex-end;
        gap: 6px !important;
    }

    div[data-testid="stRadio"]:has(input[value="IT"]) div[role="radiogroup"] label,
    div[data-testid="stRadio"]:has(input[value="EN"]) div[role="radiogroup"] label {
        min-width: 58px !important;
        min-height: 42px !important;
        padding: 0.52rem 0.78rem !important;
    }

    /* Pills dei preset: più compatti e con wrap elegante */
    div[role="radiogroup"] {
        row-gap: 8px !important;
    }

    div[role="radiogroup"] label {
        cursor: pointer;
    }

    @media (prefers-reduced-motion: reduce) {
        .pdm-fade-up,
        .pdm-pulse,
        .pdm-status-animated,
        .preset-hero,
        .elegant-panel,
        div[role="radiogroup"] label:has(input:checked)::after {
            animation: none !important;
        }
    }


    </style>
    """,
    unsafe_allow_html=True,
)



# =========================
# PREMIUM CARD SWEEP EFFECT
# =========================

def render_premium_card_sweep_effect():
    st.markdown(
        """
        <style>
        /*
        vPremium stable · sweep con ::after.
        No toca ::before, així es conserven els detalls verticals en coure.
        */
        .preset-hero,
        .preset-hero-chip,
        .preset-hero-badge,
        .preset-status-strip,
        .summary-card,
        .preset-param-card,
        .quick-card-v2,
        .semaphore-card,
        .tech-mini-card,
        .machine-card-native,
        .preview-metric,
        .summary-strip,
        .summary-strip-item,
        .section-header,
        .workflow-step,
        .elegant-panel,
        .checklist-hero,
        .pdm-action-bar,
        .pack_stat,
        .hud_card,
        .preset-chip,
        .preview-card,
        .preset-card,
        .machine-card,
        .operator-card,
        .tech-sheet-preset-card,
        .premium-sweep-card,
        div[data-testid="stMetric"] {
            position: relative !important;
            overflow: hidden !important;
            isolation: isolate !important;
        }

        .preset-hero > *,
        .preset-hero-chip > *,
        .preset-hero-badge > *,
        .preset-status-strip > *,
        .summary-card > *,
        .preset-param-card > *,
        .quick-card-v2 > *,
        .semaphore-card > *,
        .tech-mini-card > *,
        .machine-card-native > *,
        .preview-metric > *,
        .summary-strip > *,
        .summary-strip-item > *,
        .section-header > *,
        .workflow-step > *,
        .elegant-panel > *,
        .checklist-hero > *,
        .pdm-action-bar > *,
        .pack_stat > *,
        .hud_card > *,
        .preset-chip > *,
        .preview-card > *,
        .preset-card > *,
        .machine-card > *,
        .operator-card > *,
        .tech-sheet-preset-card > *,
        .premium-sweep-card > * {
            position: relative !important;
            z-index: 2 !important;
        }

        .preset-hero::after,
        .preset-hero-chip::after,
        .preset-hero-badge::after,
        .preset-status-strip::after,
        .summary-card::after,
        .preset-param-card::after,
        .quick-card-v2::after,
        .semaphore-card::after,
        .tech-mini-card::after,
        .machine-card-native::after,
        .preview-metric::after,
        .summary-strip::after,
        .summary-strip-item::after,
        .section-header::after,
        .workflow-step::after,
        .elegant-panel::after,
        .checklist-hero::after,
        .pdm-action-bar::after,
        .pack_stat::after,
        .hud_card::after,
        .preset-chip::after,
        .preview-card::after,
        .preset-card::after,
        .machine-card::after,
        .operator-card::after,
        .tech-sheet-preset-card::after,
        .premium-sweep-card::after,
        div[data-testid="stMetric"]::after {
            content: "" !important;
            position: absolute !important;
            top: -42% !important;
            bottom: -42% !important;
            left: -78% !important;
            width: 48% !important;
            pointer-events: none !important;
            border-radius: inherit !important;
            background: linear-gradient(
                105deg,
                transparent 0%,
                rgba(255,255,255,0.00) 25%,
                rgba(255,255,255,0.32) 48%,
                rgba(197,126,90,0.26) 56%,
                rgba(255,255,255,0.14) 64%,
                transparent 100%
            ) !important;
            transform: skewX(-16deg) !important;
            opacity: 0 !important;
            z-index: 4 !important;
            mix-blend-mode: screen !important;
        }

        .preset-hero:hover::after,
        .preset-hero-chip:hover::after,
        .preset-hero-badge:hover::after,
        .preset-status-strip:hover::after,
        .summary-card:hover::after,
        .preset-param-card:hover::after,
        .quick-card-v2:hover::after,
        .semaphore-card:hover::after,
        .tech-mini-card:hover::after,
        .machine-card-native:hover::after,
        .preview-metric:hover::after,
        .summary-strip:hover::after,
        .summary-strip-item:hover::after,
        .section-header:hover::after,
        .workflow-step:hover::after,
        .elegant-panel:hover::after,
        .checklist-hero:hover::after,
        .pdm-action-bar:hover::after,
        .pack_stat:hover::after,
        .hud_card:hover::after,
        .preset-chip:hover::after,
        .preview-card:hover::after,
        .preset-card:hover::after,
        .machine-card:hover::after,
        .operator-card:hover::after,
        .tech-sheet-preset-card:hover::after,
        .premium-sweep-card:hover::after,
        div[data-testid="stMetric"]:hover::after {
            opacity: 1 !important;
            animation: pdmCardSweepAfter 1.05s cubic-bezier(.2,.72,.22,1) both !important;
        }

        .pdm-pulse::after {
            opacity: 1 !important;
            animation: pdmCardSweepAfter 1.05s cubic-bezier(.2,.72,.22,1) both !important;
        }

        @keyframes pdmCardSweepAfter {
            0% {
                left: -82%;
                opacity: 0;
            }
            12% {
                opacity: 1;
            }
            100% {
                left: 138%;
                opacity: 0;
            }
        }

        @media (hover: none) {
            .preset-hero:active::after,
            .preset-hero-chip:active::after,
            .preset-hero-badge:active::after,
            .preset-status-strip:active::after,
            .summary-card:active::after,
            .preset-param-card:active::after,
            .quick-card-v2:active::after,
            .semaphore-card:active::after,
            .tech-mini-card:active::after,
            .machine-card-native:active::after,
            .preview-metric:active::after,
            .summary-strip:active::after,
            .summary-strip-item:active::after,
            .section-header:active::after,
            .workflow-step:active::after,
            .elegant-panel:active::after,
            .checklist-hero:active::after,
            .pdm-action-bar:active::after,
            .pack_stat:active::after,
            .hud_card:active::after,
            .preset-chip:active::after,
            .preview-card:active::after,
            .preset-card:active::after,
            .machine-card:active::after,
            .operator-card:active::after,
            .tech-sheet-preset-card:active::after,
            .premium-sweep-card:active::after,
            div[data-testid="stMetric"]:active::after {
                opacity: 1 !important;
                animation: pdmCardSweepAfter 1.05s cubic-bezier(.2,.72,.22,1) both !important;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .preset-hero::after,
            .preset-hero-chip::after,
            .preset-hero-badge::after,
            .preset-status-strip::after,
            .summary-card::after,
            .preset-param-card::after,
            .quick-card-v2::after,
            .semaphore-card::after,
            .tech-mini-card::after,
            .machine-card-native::after,
            .preview-metric::after,
            .summary-strip::after,
            .summary-strip-item::after,
            .section-header::after,
            .workflow-step::after,
            .elegant-panel::after,
            .checklist-hero::after,
            .pdm-action-bar::after,
            .pack_stat::after,
            .hud_card::after,
            .preset-chip::after,
            .preview-card::after,
            .preset-card::after,
            .machine-card::after,
            .operator-card::after,
            .tech-sheet-preset-card::after,
            .premium-sweep-card::after,
            div[data-testid="stMetric"]::after {
                animation: none !important;
                display: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# GLOBAL PREMIUM SWEEP ALL CARDS
# =========================

def render_global_all_cards_sweep_patch():
    st.markdown(
        """
        <style>
        .preset-hero,
        .preset-hero-chip,
        .preset-hero-badge,
        .preset-status-strip,
        .summary-card,
        .preset-param-card,
        .quick-card-v2,
        .semaphore-card,
        .tech-mini-card,
        .machine-card-native,
        .preview-metric,
        .summary-strip,
        .summary-strip-item,
        .section-header,
        .workflow-step,
        .elegant-panel,
        .checklist-hero,
        .pdm-action-bar,
        .pack_stat,
        .hud_card,
        .preset-chip,
        .preview-card,
        .preset-card,
        .machine-card,
        .operator-card,
        .tech-sheet-preset-card,
        .premium-sweep-card,
        .csv-print-row,
        div[data-testid="stMetric"],
        div[data-testid="stDownloadButton"] > button,
        .stButton > button {
            position: relative !important;
            overflow: hidden !important;
            isolation: isolate !important;
        }

        .preset-hero > *,
        .preset-hero-chip > *,
        .preset-hero-badge > *,
        .preset-status-strip > *,
        .summary-card > *,
        .preset-param-card > *,
        .quick-card-v2 > *,
        .semaphore-card > *,
        .tech-mini-card > *,
        .machine-card-native > *,
        .preview-metric > *,
        .summary-strip > *,
        .summary-strip-item > *,
        .section-header > *,
        .workflow-step > *,
        .elegant-panel > *,
        .checklist-hero > *,
        .pdm-action-bar > *,
        .pack_stat > *,
        .hud_card > *,
        .preset-chip > *,
        .preview-card > *,
        .preset-card > *,
        .machine-card > *,
        .operator-card > *,
        .tech-sheet-preset-card > *,
        .premium-sweep-card > *,
        .csv-print-row > * {
            position: relative !important;
            z-index: 2 !important;
        }

        .preset-hero::after,
        .preset-hero-chip::after,
        .preset-hero-badge::after,
        .preset-status-strip::after,
        .summary-card::after,
        .preset-param-card::after,
        .quick-card-v2::after,
        .semaphore-card::after,
        .tech-mini-card::after,
        .machine-card-native::after,
        .preview-metric::after,
        .summary-strip::after,
        .summary-strip-item::after,
        .section-header::after,
        .workflow-step::after,
        .elegant-panel::after,
        .checklist-hero::after,
        .pdm-action-bar::after,
        .pack_stat::after,
        .hud_card::after,
        .preset-chip::after,
        .preview-card::after,
        .preset-card::after,
        .machine-card::after,
        .operator-card::after,
        .tech-sheet-preset-card::after,
        .premium-sweep-card::after,
        .csv-print-row::after,
        div[data-testid="stMetric"]::after,
        div[data-testid="stDownloadButton"] > button::after,
        .stButton > button::after {
            content: "" !important;
            position: absolute !important;
            top: -42% !important;
            bottom: -42% !important;
            left: -82% !important;
            width: 48% !important;
            pointer-events: none !important;
            border-radius: inherit !important;
            background: linear-gradient(
                105deg,
                transparent 0%,
                rgba(255,255,255,0.00) 25%,
                rgba(255,255,255,0.34) 48%,
                rgba(197,126,90,0.26) 56%,
                rgba(255,255,255,0.14) 64%,
                transparent 100%
            ) !important;
            transform: skewX(-16deg) !important;
            opacity: 0 !important;
            z-index: 4 !important;
            mix-blend-mode: screen !important;
        }

        .preset-hero:hover::after,
        .preset-hero-chip:hover::after,
        .preset-hero-badge:hover::after,
        .preset-status-strip:hover::after,
        .summary-card:hover::after,
        .preset-param-card:hover::after,
        .quick-card-v2:hover::after,
        .semaphore-card:hover::after,
        .tech-mini-card:hover::after,
        .machine-card-native:hover::after,
        .preview-metric:hover::after,
        .summary-strip:hover::after,
        .summary-strip-item:hover::after,
        .section-header:hover::after,
        .workflow-step:hover::after,
        .elegant-panel:hover::after,
        .checklist-hero:hover::after,
        .pdm-action-bar:hover::after,
        .pack_stat:hover::after,
        .hud_card:hover::after,
        .preset-chip:hover::after,
        .preview-card:hover::after,
        .preset-card:hover::after,
        .machine-card:hover::after,
        .operator-card:hover::after,
        .tech-sheet-preset-card:hover::after,
        .premium-sweep-card:hover::after,
        .csv-print-row:hover::after,
        div[data-testid="stMetric"]:hover::after,
        div[data-testid="stDownloadButton"] > button:hover::after,
        .stButton > button:hover::after {
            opacity: 1 !important;
            animation: pdmGlobalCardSweep 1.05s cubic-bezier(.2,.72,.22,1) both !important;
        }

        .pdm-pulse::after {
            opacity: 1 !important;
            animation: pdmGlobalCardSweep 1.05s cubic-bezier(.2,.72,.22,1) both !important;
        }

        @keyframes pdmGlobalCardSweep {
            0% { left: -82%; opacity: 0; }
            12% { opacity: 1; }
            100% { left: 138%; opacity: 0; }
        }

        @media (hover: none) {
            .preset-hero:active::after,
            .preset-hero-chip:active::after,
            .preset-hero-badge:active::after,
            .preset-status-strip:active::after,
            .summary-card:active::after,
            .preset-param-card:active::after,
            .quick-card-v2:active::after,
            .semaphore-card:active::after,
            .tech-mini-card:active::after,
            .machine-card-native:active::after,
            .preview-metric:active::after,
            .summary-strip:active::after,
            .summary-strip-item:active::after,
            .section-header:active::after,
            .workflow-step:active::after,
            .elegant-panel:active::after,
            .checklist-hero:active::after,
            .pdm-action-bar:active::after,
            .pack_stat:active::after,
            .hud_card:active::after,
            .preset-chip:active::after,
            .preview-card:active::after,
            .preset-card:active::after,
            .machine-card:active::after,
            .operator-card:active::after,
            .tech-sheet-preset-card:active::after,
            .premium-sweep-card:active::after,
            .csv-print-row:active::after,
            div[data-testid="stMetric"]:active::after,
            div[data-testid="stDownloadButton"] > button:active::after,
            .stButton > button:active::after {
                opacity: 1 !important;
                animation: pdmGlobalCardSweep 1.05s cubic-bezier(.2,.72,.22,1) both !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# FORCE VISIBLE PREMIUM SWEEP PATCH
# =========================

def render_force_visible_premium_sweep_patch():
    st.markdown(
        """
        <style>
        /*
        Sweep realmente visibile per le card principali.
        Auto-run al caricamento + hover/touch.
        */
        .preset-hero,
        .tech-sheet-preset-card,
        .premium-sweep-card {
            position: relative !important;
            overflow: hidden !important;
            isolation: isolate !important;
        }

        .preset-hero > *,
        .tech-sheet-preset-card > *,
        .premium-sweep-card > * {
            position: relative !important;
            z-index: 3 !important;
        }

        .preset-hero::after,
        .tech-sheet-preset-card::after,
        .premium-sweep-card::after {
            content: "" !important;
            position: absolute !important;
            top: -45% !important;
            bottom: -45% !important;
            left: -70% !important;
            width: 34% !important;
            pointer-events: none !important;
            border-radius: inherit !important;
            background: linear-gradient(
                105deg,
                transparent 0%,
                rgba(255,255,255,0.00) 18%,
                rgba(255,255,255,0.42) 46%,
                rgba(197,126,90,0.34) 56%,
                rgba(255,255,255,0.22) 66%,
                transparent 100%
            ) !important;
            transform: skewX(-17deg) !important;
            opacity: 0 !important;
            z-index: 50 !important;
            mix-blend-mode: screen !important;
            animation: pdmForceSweepVisible 1.18s cubic-bezier(.18,.72,.22,1) 0.18s both !important;
        }

        .preset-hero:hover::after,
        .tech-sheet-preset-card:hover::after,
        .premium-sweep-card:hover::after,
        .preset-hero:active::after,
        .tech-sheet-preset-card:active::after,
        .premium-sweep-card:active::after {
            animation: pdmForceSweepVisible 1.18s cubic-bezier(.18,.72,.22,1) both !important;
        }

        @keyframes pdmForceSweepVisible {
            0% {
                left: -72%;
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            52% {
                opacity: 1;
            }
            100% {
                left: 135%;
                opacity: 0;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .preset-hero::after,
            .tech-sheet-preset-card::after,
            .premium-sweep-card::after {
                animation: none !important;
                display: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# SCHEDA PRESET REAL SWEEP CSS
# =========================

def render_scheda_preset_real_sweep_css():
    st.markdown(
        """
        <style>
        .tech-sheet-preset-card {
            position:relative !important;
            overflow:hidden !important;
            isolation:isolate !important;
        }
        .tech-sheet-preset-card > * {
            position:relative !important;
            z-index:3 !important;
        }
        .tech-sheet-preset-card .premium-sweep-layer {
            position:absolute !important;
            top:-45% !important;
            bottom:-45% !important;
            left:-72% !important;
            width:36% !important;
            pointer-events:none !important;
            border-radius:inherit !important;
            background:linear-gradient(105deg, transparent 0%, rgba(197,126,90,0.00) 18%, rgba(197,126,90,0.28) 42%, rgba(197,126,90,0.62) 56%, rgba(197,126,90,0.30) 70%, transparent 100%) !important;
            transform:skewX(-17deg) !important;
            opacity:0 !important;
            z-index:12 !important;
            mix-blend-mode:screen !important;
            filter:brightness(1.35) !important;
            animation:pdmRealSweepLayerScheda 1.25s cubic-bezier(.18,.72,.22,1) 0.25s both !important;
        }
        .tech-sheet-preset-card:hover .premium-sweep-layer,
        .tech-sheet-preset-card:active .premium-sweep-layer {
            animation:pdmRealSweepLayerScheda 1.25s cubic-bezier(.18,.72,.22,1) both !important;
        }
        @keyframes pdmRealSweepLayerScheda {
            0% { left:-72%; opacity:0; }
            10% { opacity:1; }
            52% { opacity:1; }
            100% { left:135%; opacity:0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    /*
    Main tabs · one single copper active line.
    The native Streamlit/BaseWeb indicators are hidden to avoid duplicate red lines.
    */
    div[data-testid="stTabs"] [role="tablist"] {
        position: relative !important;
        align-items: flex-start !important;
        padding-top: 0.12rem !important;
        padding-bottom: 0.42rem !important;
        min-height: 3.00rem !important;
        border-bottom: 0 !important;
        box-shadow: none !important;
    }

    div[data-testid="stTabs"] [role="tablist"] * {
        border-bottom-color: transparent !important;
        box-shadow: none !important;
    }

    div[data-testid="stTabs"] [role="tab"] {
        position: relative !important;
        min-height: 2.60rem !important;
        padding: 0.68rem 1.05rem 0.92rem 1.05rem !important;
        line-height: 1.15 !important;
        overflow: visible !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    div[data-testid="stTabs"] [role="tab"] p {
        margin: 0 !important;
        line-height: 1.15 !important;
        position: relative !important;
        z-index: 3 !important;
    }

    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--pdm-accent) !important;
    }

    div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
        content: "" !important;
        position: absolute !important;
        left: 0.92rem !important;
        right: 0.92rem !important;
        bottom: 0.22rem !important;
        height: 3px !important;
        border-radius: 999px !important;
        background: var(--pdm-accent) !important;
        z-index: 2 !important;
        pointer-events: none !important;
        opacity: 1 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stTabs"] [role="tab"]::before,
    div[data-testid="stTabs"] [role="tab"] > div::before,
    div[data-testid="stTabs"] [role="tab"] > div::after,
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-testid="stTabHighlight"],
    div[data-testid="stTabs"] [aria-hidden="true"] {
        display: none !important;
        opacity: 0 !important;
        background: transparent !important;
        border: 0 !important;
        height: 0 !important;
        box-shadow: none !important;
    }

    @media (max-width: 1180px) {
        div[data-testid="stTabs"] [role="tab"] {
            min-height: 2.70rem !important;
            padding-top: 0.72rem !important;
            padding-bottom: 1.00rem !important;
        }

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            bottom: 0.26rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



render_scheda_preset_real_sweep_css()
render_force_visible_premium_sweep_patch()
render_global_all_cards_sweep_patch()

st.markdown(
    """
    <style>
    /* Anteprima · elimina contorns/fons de l'iframe exterior */
    iframe {
        background: transparent !important;
    }

    div[data-testid="stIFrame"],
    div[data-testid="stIFrame"] iframe {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        outline: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

render_premium_card_sweep_effect()

# =========================
# CARD BACKGROUND PAGE-MATCH PATCH
# =========================

def render_card_background_page_match_patch():
    st.markdown(
        """
        <style>
        /*
        Les cards de paràmetres respiren amb el mateix fons de la pàgina.
        Funciona en mode clar/fosc perquè no fixa cap color sòlid de fons.
        */
        .machine-card-native,
        .tech-mini-card,
        .summary-strip-item {
            background: transparent !important;
        }

        .machine-card-native:hover,
        .tech-mini-card:hover,
        .summary-strip-item:hover {
            background: color-mix(in srgb, var(--pdm-accent) 3.5%, transparent) !important;
        }

        .machine-grid-native {
            background: transparent !important;
            box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# LIGHT DARK READABILITY PATCH
# =========================

def render_light_dark_readability_patch():
    st.markdown(
        """
        <style>
        .machine-card-native,
        .tech-mini-card,
        .summary-strip-item {
            background: transparent !important;
            color: var(--text-color) !important;
        }

        .machine-card-label-native,
        .tech-mini-label,
        .summary-strip-label {
            color: color-mix(in srgb, var(--text-color) 64%, transparent) !important;
        }

        .machine-card-value-native,
        .tech-mini-value,
        .summary-strip-value {
            color: var(--text-color) !important;
        }

        .machine-card-native:hover,
        .tech-mini-card:hover,
        .summary-strip-item:hover {
            background: color-mix(in srgb, var(--pdm-accent) 4%, transparent) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# ANTEPRIMA TECNICA BACKGROUND PATCH
# =========================

def render_preview_background_patch():
    st.markdown(
        """
        <style>
        /*
        Elimina il padding/fondo scuro attorno all'Anteprima tecnica.
        La sezione respira con il fondo pagina in light/dark mode.
        */
        .tube-section-card,
        .tube-preview-card,
        .tube-preview-wrapper,
        .tube-preview-shell,
        .tube-section,
        .preview-technical-card,
        .technical-preview-card,
        .section-tube-card {
            background: transparent !important;
            box-shadow: none !important;
        }

        .tube-section-card,
        .tube-preview-wrapper,
        .tube-preview-shell,
        .tube-section,
        .preview-technical-card,
        .technical-preview-card,
        .section-tube-card {
            padding-bottom: 0 !important;
            margin-bottom: 0 !important;
        }

        .tube-section-card::before,
        .tube-section-card::after,
        .tube-preview-card::before,
        .tube-preview-card::after,
        .tube-preview-wrapper::before,
        .tube-preview-wrapper::after,
        .tube-preview-shell::before,
        .tube-preview-shell::after,
        .tube-section::before,
        .tube-section::after,
        .preview-technical-card::before,
        .preview-technical-card::after,
        .technical-preview-card::before,
        .technical-preview-card::after,
        .section-tube-card::before,
        .section-tube-card::after {
            background: transparent !important;
        }

        /* Fallback per blocs markdown amb style inline fosc al voltant del preview */
        div[style*="background:#080b10"],
        div[style*="background: #080b10"],
        div[style*="background:#0b0f16"],
        div[style*="background: #0b0f16"],
        div[style*="background:#0f1117"],
        div[style*="background: #0f1117"],
        div[style*="background:rgb(8, 11, 16)"],
        div[style*="background: rgb(8, 11, 16)"] {
            background: transparent !important;
            box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# ANTEPRIMA CORNER CLEANUP PATCH
# =========================

def render_anteprima_corner_cleanup_patch():
    st.markdown(
        """
        <style>
        iframe {
            background: transparent !important;
        }

        div[data-testid="stIFrame"],
        div[data-testid="stIFrame"] iframe {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    z_min_center: float | None = None,
    z_max_center: float | None = None,
):
    max_len = lunghezza_m * 1000.0

    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla

    if z_min_center is None:
        z_min_center = Rt
    if z_max_center is None:
        z_max_center = H - Rt

    z_min_center = float(z_min_center)
    z_max_center = float(z_max_center)

    theta = np.deg2rad(gradi_start)
    z = z_min_center
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

            if next_z >= z_max_center:
                next_z = z_max_center

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

            elif next_z <= z_min_center:
                next_z = z_min_center

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


def compute_winding_diagnostics(layer_values, z_values, mode_values, z_min_center, z_max_center, language):
    """Small diagnostic block to compare the simulated winding cycle with the real machine."""
    if layer_values is None or len(layer_values) == 0 or z_values is None or len(z_values) == 0:
        return {
            "strati_simulati": 0,
            "strato_finale": 0,
            "inversioni": 0,
            "lato_finale": "-",
            "direzione_finale": "-",
            "quota_finale": 0.0,
        }

    layer_arr = np.asarray(layer_values, dtype=float)
    z_arr = np.asarray(z_values, dtype=float)
    mode_arr = np.asarray(mode_values, dtype=int) if mode_values is not None and len(mode_values) else np.zeros(len(z_arr), dtype=int)

    max_layer = int(np.nanmax(layer_arr)) if len(layer_arr) else 0
    final_layer = int(layer_arr[-1]) if len(layer_arr) else 0
    strati_simulati = max_layer + 1
    strato_finale = final_layer + 1
    inversioni = max_layer

    z_min = float(z_min_center)
    z_max = float(z_max_center)
    z_mid = (z_min + z_max) / 2.0
    quota_finale = float(z_arr[-1])

    if language == "IT":
        lato_finale = "Quota min" if quota_finale <= z_mid else "Quota max"
        direction_max = "verso quota max"
        direction_min = "verso quota min"
        direction_transition = "in inversione"
        direction_stable = "fermo"
    else:
        lato_finale = "Min side" if quota_finale <= z_mid else "Max side"
        direction_max = "towards max side"
        direction_min = "towards min side"
        direction_transition = "in transition"
        direction_stable = "steady"

    direzione_finale = direction_transition if int(mode_arr[-1]) == 1 else direction_stable
    if len(z_arr) >= 2:
        diffs = np.diff(z_arr)
        for dz in diffs[::-1]:
            if abs(float(dz)) > 0.05:
                direzione_finale = direction_max if dz > 0 else direction_min
                break

    return {
        "strati_simulati": strati_simulati,
        "strato_finale": strato_finale,
        "inversioni": inversioni,
        "lato_finale": lato_finale,
        "direzione_finale": direzione_finale,
        "quota_finale": quota_finale,
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
    initial_scene="winding",
    packaging_mode="box",
    container_mode="40hc",
    pack_roll_count=5,
    tube_mode_initial="gelwhite",
    tube_layout="single",
    d_tubo_lower=None,
    d_tubo_upper=None,
    tube_diameter_label=None,
    simulation_print_payload=None,
    active_product_name=None,
    active_product_kind="preset",
):
    final_local_points_json = json.dumps(final_local_points)
    final_thetas_json = json.dumps(final_thetas)
    final_radii_json = json.dumps(final_radii)
    final_zs_json = json.dumps(final_zs)
    final_modes_json = json.dumps(final_modes)
    final_layers_json = json.dumps(final_layers)
    final_lengths_json = json.dumps(final_lengths)
    labels_json = json.dumps(TEXTS[language])
    simulation_print_payload_json = json.dumps(simulation_print_payload or {})
    tube_layout = "double" if str(tube_layout).lower() in {"double", "doppio"} else "single"
    d_tubo_lower = float(d_tubo if d_tubo_lower is None else d_tubo_lower)
    d_tubo_upper = float(d_tubo if d_tubo_upper is None else d_tubo_upper)
    tube_diameter_label = tube_diameter_label or f"{float(d_tubo):.2f} mm"
    tube_diameter_label_json = json.dumps(str(tube_diameter_label))
    active_product_name = active_product_name or ("Preset attivo" if language == "IT" else "Active preset")
    active_product_name_json = json.dumps(str(active_product_name))
    active_product_kind_json = json.dumps(str(active_product_kind or "preset"))
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
        <div id="viewer_loading_overlay" class="viewer_loading_overlay">
            <div class="viewer_loading_card">
                <div class="viewer_loading_kicker">PDM</div>
                <div class="viewer_loading_title">Preparazione simulazione…</div>
                <div class="viewer_loading_bar"></div>
            </div>
        </div>

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
            line-height:1.25;
        ">
            <button id="play_pause_btn" class="viewer_btn viewer_icon_btn">⏸</button>
            <button id="reset_view_btn" class="viewer_btn viewer_icon_btn">↺</button>
            <button id="fullscreen_btn" class="viewer_btn viewer_icon_btn">⛶</button>
            <button id="capture_render_btn" class="viewer_btn viewer_icon_btn">📷</button>
            <button id="print_simulation_btn" class="viewer_btn viewer_print_btn">Stampa</button>
            <span style="margin-left:6px;" id="progress_title"></span>
            <input id="progress_slider" type="range" min="0" max="1000" step="1" value="0" style="width:180px;" />
        </div>

        <div id="active_preset_badge" style="
            position:absolute;
            top:22px;
            right:286px;
            left:auto;
            transform:none;
            z-index:21;
            display:flex;
            flex-direction:column;
            align-items:flex-start;
            justify-content:center;
            gap:0px;
            min-width:150px;
            max-width:180px;
            padding:9px 15px 9px 14px;
            background:rgba(18,22,27,0.80);
            color:#f8fafc;
            border:1px solid rgba(197,126,90,0.34);
            border-radius:16px;
            backdrop-filter:blur(10px);
            box-shadow:0 14px 30px rgba(0,0,0,0.24), inset 4px 0 0 #C57E5A;
            font-family:Arial, sans-serif;
            text-align:center;
            user-select:none;
            line-height:1.15;
        ">
            <div id="active_preset_badge_label" style="display:none;"></div>
            <div id="active_preset_badge_value" style="
                font-size:16px;
                font-weight:900;
                line-height:1.05;
                max-width:100%;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            "></div>
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

        <div id="packaging_status_badge" style="
            position:absolute;
            left:14px;
            bottom:14px;
            z-index:22;
            display:none;
            min-width:220px;
            padding:14px 16px;
            background:rgba(18,22,27,0.78);
            color:#f8fafc;
            border:1px solid rgba(255,255,255,0.12);
            border-radius:15px;
            backdrop-filter:blur(10px);
            font-family:Arial, sans-serif;
            box-shadow:0 14px 30px rgba(0,0,0,0.24);
        ">
            <div style="font-size:11px; opacity:0.72; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:5px;">Packaging</div>
            <div id="packaging_status_text" style="font-size:22px; font-weight:900; line-height:1.05;"></div>
            <div id="packaging_status_reason" style="font-size:12px; opacity:0.78; margin-top:6px; line-height:1.32;"></div>
        </div>

        <div id="viewer_sidepanel" style="
            position:absolute;
            top:14px;
            right:14px;
            z-index:20;
            display:flex;
            flex-direction:column;
            gap:10px;
            width:248px;
            max-width:calc(100% - 28px);
            max-height:calc(100% - 28px);
            overflow-y:auto;
            overscroll-behavior:contain;
            box-sizing:border-box;
            padding:12px;
            background:rgba(18,22,27,0.74);
            color:#f0f0f0;
            border:1px solid rgba(255,255,255,0.12);
            border-radius:14px;
            backdrop-filter: blur(10px);
            font-family:Arial, sans-serif;
            font-size:13px;
            user-select:none;
        ">
            <button id="viewer_sidepanel_toggle" class="viewer_sidepanel_toggle" title="Nascondi opzioni">❮</button>
            <div id="viewer_sidepanel_content">
            <div>
                <div class="panel_label" id="animation_title"></div>
                <label class="panel_check">
                    <input type="checkbox" id="animation_check" />
                    <span id="animation_label_text"></span>
                </label>
            </div>

            <div id="speed_block">
                <div class="panel_label" id="speed_title"></div>
                <div class="btn_group_vertical btn_grid_3" id="speed_group">
                    <button class="speed_btn viewer_btn_small" data-speed="0.5">x0.5</button>
                    <button class="speed_btn viewer_btn_small active_speed" data-speed="1.0">x1</button>
                    <button class="speed_btn viewer_btn_small" data-speed="2.0">x2</button>
                </div>
            </div>

            <div>
                <div class="panel_label" id="view_title"></div>
                <div class="btn_group_vertical btn_grid_3">
                    <button class="view_btn viewer_btn_small active_opt" data-view="3d" id="view_3d_btn"></button>
                    <button class="view_btn viewer_btn_small" data-view="front" id="view_front_btn"></button>
                    <button class="view_btn viewer_btn_small" data-view="side" id="view_side_btn"></button>
                </div>
            </div>

            <div id="scene_block" style="display:none;">
                <div class="panel_label" id="scene_title"></div>
                <div class="btn_group_vertical btn_grid_2">
                    <button class="scene_btn viewer_btn_small active_opt" data-scene="winding" id="scene_winding_btn">Avvolgimento</button>
                    <button class="scene_btn viewer_btn_small" data-scene="packaging" id="scene_packaging_btn">Packaging</button>
                </div>
            </div>

            <div id="packaging_controls" style="display:none;">
                <div class="panel_label" id="pack_roll_title"></div>
                <div class="pack_roll_inline">
                    <button id="pack_roll_minus" class="viewer_btn_small pack_roll_btn" type="button">−</button>
                    <input id="pack_roll_count" type="number" min="1" max="50" step="1" value="{int(pack_roll_count)}" />
                    <button id="pack_roll_plus" class="viewer_btn_small pack_roll_btn" type="button">+</button>
                </div>
                <div id="pack_roll_hint" class="pack_roll_hint">Modifica diretta nel render</div>
                <label class="panel_check" style="margin-top:9px;">
                    <input type="checkbox" id="pack_dimensions_check" checked />
                    <span id="pack_dimensions_label">Quote / linee verdi</span>
                </label>
            </div>

            <div id="spool_block">
                <div class="panel_label" id="spool_title"></div>
                <div class="btn_group_vertical btn_grid_3">
                    <button class="spool_btn viewer_btn_small" data-spool="visible" id="spool_visible_btn"></button>
                    <button class="spool_btn viewer_btn_small" data-spool="transparent" id="spool_transparent_btn"></button>
                    <button class="spool_btn viewer_btn_small active_opt" data-spool="hidden" id="spool_hidden_btn"></button>
                </div>
            </div>

            <div>
                <div class="panel_label" id="tube_title"></div>
                <div class="btn_group_vertical btn_grid_2">
                    <button class="tube_btn viewer_btn_small active_opt" data-tube="gelwhite" id="tube_gelwhite_btn"></button>
                    <button class="tube_btn viewer_btn_small" data-tube="gelblack" id="tube_gelblack_btn"></button>
                </div>
            </div>

            <div id="checks_block" class="panel_checks_block">
                <label class="panel_check">
                    <input type="checkbox" id="ghost_check" />
                    <span id="ghost_title"></span>
                </label>

                <label class="panel_check">
                    <input type="checkbox" id="grid_check" />
                    <span id="grid_title"></span>
                </label>

                <label class="panel_check" style="display:none;">
                    <input type="checkbox" id="axes_check" />
                    <span id="axes_title"></span>
                </label>

                <label class="panel_check" style="display:none;">
                    <input type="checkbox" id="section_check" />
                    <span id="section_title"></span>
                </label>
            </div>
            </div>
        </div>
    </div>

    <style>
        @keyframes viewerFadeUp {{
            from {{ opacity:0; transform:translateY(10px); }}
            to {{ opacity:1; transform:translateY(0); }}
        }}

        @keyframes viewerLoadingSweep {{
            from {{ transform:translateX(-125%); }}
            to {{ transform:translateX(260%); }}
        }}

        #viewer_root {{
            animation: viewerFadeUp 0.58s cubic-bezier(.16,1,.3,1) both;
        }}

        .viewer_loading_overlay {{
            position:absolute;
            inset:0;
            z-index:60;
            display:flex;
            align-items:center;
            justify-content:center;
            background:radial-gradient(circle at 50% 44%, rgba(197,126,90,0.16), transparent 38%), linear-gradient(180deg, rgba(16,18,22,0.94), rgba(16,18,22,0.78));
            backdrop-filter:blur(9px);
            transition:opacity 0.46s ease, visibility 0.46s ease;
        }}

        .viewer_loading_overlay.is-hidden {{
            opacity:0;
            visibility:hidden;
            pointer-events:none;
        }}

        .viewer_loading_card {{
            width:min(430px, calc(100% - 44px));
            padding:25px 27px;
            border-radius:22px;
            border:1px solid rgba(255,255,255,0.13);
            background:rgba(18,22,27,0.78);
            box-shadow:0 18px 42px rgba(0,0,0,0.30);
            color:#f8fafc;
            font-family:Arial, sans-serif;
        }}

        .viewer_loading_kicker {{
            font-size:11px;
            line-height:1;
            font-weight:900;
            letter-spacing:0.12em;
            text-transform:uppercase;
            color:#C57E5A;
            margin-bottom:9px;
        }}

        .viewer_loading_title {{
            font-size:18px;
            line-height:1.15;
            font-weight:900;
            margin-bottom:16px;
        }}

        .viewer_loading_bar {{
            position:relative;
            overflow:hidden;
            height:5px;
            border-radius:999px;
            background:rgba(255,255,255,0.12);
        }}

        .viewer_loading_bar::after {{
            content:"";
            position:absolute;
            inset:0;
            width:42%;
            border-radius:inherit;
            background:linear-gradient(90deg, transparent, #C57E5A, transparent);
            animation:viewerLoadingSweep 1.12s ease-in-out infinite;
        }}

        @media (prefers-reduced-motion: reduce) {{
            #viewer_root,
            .viewer_loading_bar::after {{
                animation:none !important;
            }}
        }}

        .viewer_btn {{
            border:none;
            border-radius:9px;
            padding:7px 12px;
            background:#f4f4f4;
            color:#111;
            font-weight:800;
            cursor:pointer;
        }}

        .viewer_icon_btn {{
            width:36px;
            min-width:36px;
            height:34px;
            padding:0;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            font-size:18px;
            line-height:1;
        }}

        .viewer_print_btn {{
            height:34px;
            min-width:118px;
            padding:0 16px;
            border-radius:999px;
            background:#C57E5A !important;
            color:#ffffff !important;
            font-size:12px;
            letter-spacing:0.02em;
            text-transform:uppercase;
            box-shadow:0 8px 18px rgba(197,126,90,0.28);
        }}

        html.pseudo_fullscreen_doc,
        body.pseudo_fullscreen_doc {{
            width:100vw !important;
            height:100vh !important;
            margin:0 !important;
            padding:0 !important;
            overflow:hidden !important;
            background:#101216 !important;
        }}

        #viewer_root.pseudo_fullscreen {{
            position:fixed !important;
            inset:0 !important;
            z-index:999999 !important;
            width:100vw !important;
            height:100vh !important;
            border-radius:0 !important;
            border:none !important;
        }}

        #viewer_root:fullscreen {{
            width:100vw !important;
            height:100vh !important;
            border-radius:0 !important;
            border:none !important;
        }}

        #viewer_root:-webkit-full-screen {{
            width:100vw !important;
            height:100vh !important;
            border-radius:0 !important;
            border:none !important;
        }}

        .viewer_btn_small {{
            border:none;
            border-radius:11px;
            padding:8px 9px;
            background:rgba(235,235,235,0.95);
            color:#111;
            font-weight:850;
            font-size:12px;
            cursor:pointer;
            text-align:center;
            white-space:normal;
            overflow-wrap:anywhere;
            line-height:1.12;
            min-height:42px;
            width:100%;
            box-sizing:border-box;
            display:flex;
            align-items:center;
            justify-content:center;
        }}

        #viewer_sidepanel {{
            transition: width 0.22s ease, padding 0.22s ease, opacity 0.22s ease;
        }}

        .viewer_sidepanel_toggle {{
            position:absolute;
            top:8px;
            right:8px;
            width:30px;
            height:30px;
            border:1px solid rgba(255,255,255,0.16);
            border-radius:10px;
            background:#C57E5A;
            color:#ffffff;
            font-size:16px;
            font-weight:900;
            cursor:pointer;
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow:0 8px 18px rgba(0,0,0,0.22);
            z-index:30;
        }}

        #viewer_sidepanel_content {{
            display:flex;
            flex-direction:column;
            gap:10px;
            margin-top:30px;
        }}

        #viewer_sidepanel.collapsed {{
            width:52px !important;
            min-width:52px !important;
            height:52px !important;
            min-height:52px !important;
            max-height:52px !important;
            padding:8px !important;
            overflow:visible !important;
        }}

        #viewer_sidepanel.collapsed #viewer_sidepanel_content {{
            display:none !important;
        }}

        #viewer_sidepanel.collapsed .viewer_sidepanel_toggle {{
            top:10px;
            right:10px;
            width:32px;
            height:32px;
            border-radius:999px;
        }}

        .viewer_btn_small,
        .viewer_btn {{
            transition: transform 0.14s ease, background 0.14s ease, box-shadow 0.14s ease, filter 0.14s ease;
        }}

        .viewer_btn_small:hover,
        .viewer_btn:hover {{
            background:#ffffff;
            transform:translateY(-1px);
            box-shadow:0 8px 16px rgba(0,0,0,0.16);
        }}

        .active_speed,
        .active_opt {{
            outline:2px solid #ffffff;
            background:#C57E5A !important;
            color:#ffffff !important;
            box-shadow:0 0 0 2px rgba(197,126,90,0.35), 0 8px 18px rgba(0,0,0,0.18);
        }}

        .active_speed:hover,
        .active_opt:hover {{
            filter:brightness(1.08);
            box-shadow:0 0 0 2px rgba(197,126,90,0.42), 0 10px 20px rgba(0,0,0,0.20);
        }}

        .panel_label {{
            font-size:11px;
            opacity:0.86;
            margin-bottom:7px;
            text-transform:uppercase;
            letter-spacing:0.06em;
            font-weight:800;
        }}

        .btn_group_vertical {{
            display:flex;
            flex-direction:column;
            gap:6px;
        }}

        .btn_grid_2 {{
            display:grid;
            grid-template-columns:repeat(2, minmax(0, 1fr));
            gap:8px;
        }}

        .btn_grid_3 {{
            display:grid;
            grid-template-columns:repeat(2, minmax(0, 1fr));
            gap:8px;
        }}

        #viewer_sidepanel::-webkit-scrollbar {{
            width:8px;
        }}

        #viewer_sidepanel::-webkit-scrollbar-thumb {{
            background:rgba(255,255,255,0.18);
            border-radius:999px;
        }}


        .panel_check {{
            display:flex;
            align-items:center;
            gap:8px;
            line-height:1.25;
            font-weight:650;
        }}

        .pack_roll_inline {{
            display:grid;
            grid-template-columns:42px 1fr 42px;
            gap:7px;
            align-items:center;
        }}

        .pack_roll_btn {{
            min-height:42px !important;
            font-size:20px !important;
            line-height:1 !important;
            padding:0 !important;
            display:flex !important;
            align-items:center !important;
            justify-content:center !important;
        }}

        #pack_roll_count {{
            width:100%;
            min-height:42px;
            box-sizing:border-box;
            border:none;
            border-radius:10px;
            padding:8px 10px;
            font-weight:900;
            font-size:18px;
            text-align:center;
            background:rgba(255,255,255,0.96);
            color:#111;
        }}

        .pack_roll_hint {{
            margin-top:7px;
            font-size:11px;
            opacity:0.72;
            line-height:1.25;
            font-weight:650;
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

        #active_preset_badge {{
            box-sizing:border-box !important;
        }}
        #active_preset_badge_label {{
            display:none !important;
        }}
        #active_preset_badge_value {{
            width:100% !important;
            text-align:left !important;
        }}

        /* Wide render: compact fixed badge on the top row, between toolbar and side panel. */
        @media (min-width: 1181px) {{
            #active_preset_badge {{
                top:22px !important;
                right:286px !important;
                left:auto !important;
                transform:none !important;
                min-width:150px !important;
                max-width:180px !important;
                padding:9px 15px 9px 14px !important;
                align-items:flex-start !important;
                text-align:left !important;
            }}
            #active_preset_badge_value {{
                font-size:16px !important;
            }}
        }}

        /* Medium width: still top row, but slightly narrower and further left. */
        @media (max-width: 1180px) and (min-width: 1001px) {{
            #active_preset_badge {{
                top:22px !important;
                right:272px !important;
                left:auto !important;
                transform:none !important;
                min-width:132px !important;
                max-width:150px !important;
                padding:8px 13px 8px 12px !important;
                align-items:flex-start !important;
                text-align:left !important;
            }}
            #active_preset_badge_value {{
                font-size:14px !important;
            }}
        }}

        /* iPad / narrow render: move below toolbar so it never covers the side panel. */
        @media (max-width: 1000px) and (min-width: 681px) {{
            #active_preset_badge {{
                top:112px !important;
                right:14px !important;
                left:14px !important;
                transform:none !important;
                min-width:0 !important;
                max-width:none !important;
                padding:8px 14px 9px 14px !important;
                align-items:flex-start !important;
                text-align:left !important;
            }}
            #active_preset_badge_value {{
                font-size:17px !important;
            }}
        }}

        @media (max-width: 680px) {{
            #active_preset_badge {{
                top:104px !important;
                left:14px !important;
                right:14px !important;
                transform:none !important;
                min-width:0 !important;
                max-width:none !important;
                padding:8px 14px 9px 14px !important;
                align-items:flex-start !important;
                text-align:left !important;
            }}
            #active_preset_badge_value {{
                font-size:16px !important;
                white-space:normal !important;
                overflow:visible !important;
                text-overflow:initial !important;
            }}
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

        .pack_mode_desc {{
            margin-top:2px;
            font-size:11px;
            line-height:1.25;
            opacity:0.72;
            font-weight:600;
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

        @media (max-width: 1180px) {{
            #viewer_sidepanel {{
                width:218px !important;
                top:10px !important;
                right:10px !important;
                padding:9px !important;
                gap:8px !important;
            }}

            #viewer_sidepanel.collapsed {{
                width:50px !important;
                min-width:50px !important;
                height:50px !important;
                min-height:50px !important;
                max-height:50px !important;
                padding:7px !important;
                overflow:visible !important;
            }}

            .viewer_sidepanel_toggle {{
                width:30px !important;
                height:30px !important;
                font-size:15px !important;
                top:10px !important;
                right:10px !important;
            }}

            #viewer_sidepanel_content {{
                margin-top:28px !important;
            }}

            .viewer_btn_small {{
                font-size:12px !important;
                min-height:38px !important;
                padding:7px 7px !important;
                border-radius:10px !important;
            }}

            .panel_label {{
                font-size:10px !important;
                margin-bottom:5px !important;
            }}

            .btn_grid_3,
            .btn_grid_2 {{
                gap:5px !important;
            }}

            #viewer_topbar {{
                top:10px !important;
                left:10px !important;
                padding:8px 9px !important;
                gap:6px !important;
                z-index:30 !important;
            }}

            #progress_slider {{
                width:120px !important;
            }}

            .viewer_btn {{
                padding:6px 9px !important;
                border-radius:8px !important;
            }}

            /* iPad/tablet: align bottom HUD with the render padding */
            #viewer_hud {{
                left:10px !important;
                right:10px !important;
                bottom:10px !important;
                width:calc(100% - 20px) !important;
                grid-template-columns:repeat(3, minmax(0, 1fr)) !important;
                gap:7px !important;
                box-sizing:border-box !important;
            }}

            .hud_card {{
                min-width:0 !important;
                width:100% !important;
                padding:9px 10px !important;
                box-sizing:border-box !important;
            }}

            .hud_value {{
                font-size:14px !important;
                white-space:nowrap !important;
                overflow:hidden !important;
                text-overflow:ellipsis !important;
            }}

            #packaging_status_badge {{
                left:10px !important;
                right:10px !important;
                bottom:10px !important;
                min-width:0 !important;
                width:auto !important;
                box-sizing:border-box !important;
            }}
        }}

        /* Final mobile render fix: avoid overlay conflicts and keep the viewer usable on phones. */
        @media (max-width: 680px) {{
            #viewer_root {{
                min-height: 960px !important;
            }}

            #viewer_topbar {{
                top: 8px !important;
                left: 8px !important;
                right: 8px !important;
                width: calc(100% - 16px) !important;
                max-width: calc(100% - 16px) !important;
                padding: 8px !important;
                gap: 6px !important;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                flex-wrap: nowrap !important;
                justify-content: flex-start !important;
                box-sizing: border-box !important;
                -webkit-overflow-scrolling: touch !important;
            }}

            #viewer_topbar::-webkit-scrollbar {{
                display: none !important;
            }}

            #progress_title {{
                display: none !important;
            }}

            #progress_slider {{
                width: 96px !important;
                min-width: 96px !important;
            }}

            .viewer_icon_btn {{
                width: 34px !important;
                min-width: 34px !important;
                height: 34px !important;
                font-size: 17px !important;
            }}

            .viewer_print_btn {{
                min-width: 132px !important;
                padding: 0 14px !important;
                font-size: 12px !important;
            }}

            #active_preset_badge {{
                display: none !important;
            }}

            #viewer_sidepanel {{
                top: 58px !important;
                right: 8px !important;
                z-index: 36 !important;
            }}

            #viewer_sidepanel.collapsed {{
                width: 50px !important;
                min-width: 50px !important;
                height: 50px !important;
                min-height: 50px !important;
                max-height: 50px !important;
                padding: 7px !important;
            }}

            #viewer_sidepanel:not(.collapsed) {{
                width: min(236px, calc(100% - 16px)) !important;
                max-width: min(236px, calc(100% - 16px)) !important;
                max-height: calc(100% - 74px) !important;
            }}

            #viewer_hud {{
                left: 8px !important;
                right: 8px !important;
                bottom: 8px !important;
                width: calc(100% - 16px) !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: 6px !important;
                box-sizing: border-box !important;
            }}

            .hud_card {{
                min-width: 0 !important;
                width: 100% !important;
                padding: 8px 9px !important;
                box-sizing: border-box !important;
            }}

            .hud_label {{
                font-size: 9px !important;
            }}

            .hud_value {{
                font-size: 13px !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }}

            #packaging_status_badge {{
                left: 8px !important;
                right: 8px !important;
                bottom: 8px !important;
                width: auto !important;
                min-width: 0 !important;
                box-sizing: border-box !important;
            }}
        }}
    </style>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/TrackballControls.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>

    <script>
    (() => {{
        const T = {labels_json};
        const SIM_PRINT = {simulation_print_payload_json};
        const ACTIVE_PRODUCT_NAME = {active_product_name_json};
        const ACTIVE_PRODUCT_KIND = {active_product_kind_json};

        const host = document.getElementById("viewer_root");
        const loadingOverlay = document.getElementById("viewer_loading_overlay");
        const playPauseBtn = document.getElementById("play_pause_btn");
        const resetViewBtn = document.getElementById("reset_view_btn");
        const fullscreenBtn = document.getElementById("fullscreen_btn");
        const captureRenderBtn = document.getElementById("capture_render_btn");
        const printSimulationBtn = document.getElementById("print_simulation_btn");
        const progressSlider = document.getElementById("progress_slider");
        const animationCheck = document.getElementById("animation_check");

        const speedBtns = [...document.querySelectorAll(".speed_btn")];
        const spoolBtns = [...document.querySelectorAll(".spool_btn")];
        const tubeBtns = [...document.querySelectorAll(".tube_btn")];
        const viewBtns = [...document.querySelectorAll(".view_btn")];
        const sceneBtns = [...document.querySelectorAll(".scene_btn")];
        const packagingControls = document.getElementById("packaging_controls");
        const animationBlock = document.getElementById("animation_block");
        const speedBlock = document.getElementById("speed_block");
        const spoolBlock = document.getElementById("spool_block");
        const checksBlock = document.getElementById("checks_block");
        const packModeBtns = [...document.querySelectorAll(".pack_mode_btn")];
        const containerBtns = [...document.querySelectorAll(".container_btn")];
        const packContainerBlock = document.getElementById("pack_container_block");
        const packRollCountInput = document.getElementById("pack_roll_count");
        const packRollMinusBtn = document.getElementById("pack_roll_minus");
        const packRollPlusBtn = document.getElementById("pack_roll_plus");
        const packDimensionsCheck = document.getElementById("pack_dimensions_check");
        const packDimensionsLabel = document.getElementById("pack_dimensions_label");
        const packagingStats = document.getElementById("packaging_stats");
        const viewerHud = document.getElementById("viewer_hud");
        const sidepanel = document.getElementById("viewer_sidepanel");
        const sidepanelContent = document.getElementById("viewer_sidepanel_content");
        const sidepanelToggle = document.getElementById("viewer_sidepanel_toggle");
        const packagingStatusBadge = document.getElementById("packaging_status_badge");
        const packagingStatusText = document.getElementById("packaging_status_text");
        const packagingStatusReason = document.getElementById("packaging_status_reason");
        const activePresetBadgeLabel = document.getElementById("active_preset_badge_label");
        const activePresetBadgeValue = document.getElementById("active_preset_badge_value");

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
        const packRollHint = document.getElementById("pack_roll_hint");
        if (packRollHint) packRollHint.textContent = (T.language === "Language") ? "Directly editable in render" : "Modifica diretta nel render";
        if (packDimensionsLabel) packDimensionsLabel.textContent = (T.language === "Language") ? "Dimensions / green lines" : "Quote / linee verdi";
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
        fullscreenBtn.title = T.fullscreen;
        captureRenderBtn.title = T.capture_render || "Save render image";
        if (printSimulationBtn) {{
            printSimulationBtn.textContent = "PDF render";
            printSimulationBtn.title = "Apri PDF render in una nuova scheda";
        }}

        if (activePresetBadgeLabel) {{
            const presetLabel = (ACTIVE_PRODUCT_KIND === "prototype")
                ? ((T.language === "Language") ? "Active prototype" : "Prototipo attivo")
                : ((T.language === "Language") ? "Active preset" : "Preset attivo");
            activePresetBadgeLabel.textContent = presetLabel;
        }}
        if (activePresetBadgeValue) {{
            activePresetBadgeValue.textContent = ACTIVE_PRODUCT_NAME || "";
            activePresetBadgeValue.title = ACTIVE_PRODUCT_NAME || "";
        }}

        function updateSidepanelToggle() {{
            const collapsed = sidepanel.classList.contains("collapsed");
            sidepanelToggle.textContent = collapsed ? "❯" : "❮";
            sidepanelToggle.title = collapsed ? "Mostra opzioni" : "Nascondi opzioni";
        }}

        const isNarrowMobile = window.innerWidth <= 680;
        if (isNarrowMobile && sidepanel) {{
            sidepanel.classList.add("collapsed");
        }}

        sidepanelToggle.addEventListener("click", () => {{
            sidepanel.classList.toggle("collapsed");
            updateSidepanelToggle();
            setTimeout(resizeViewer, 120);
        }});

        updateSidepanelToggle();

        document.getElementById("hud_length_label").textContent = T.hud_length;
        document.getElementById("hud_layer_label").textContent = T.hud_layer;
        document.getElementById("hud_diameter_label").textContent = T.hud_diameter;
        document.getElementById("hud_diameter_value").textContent = {tube_diameter_label_json};

        const W = Math.max(host.clientWidth, 600);
        const Hview = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();

        const camera = new THREE.PerspectiveCamera(32, W / Hview, 0.1, 20000);
        camera.position.set(-1350, -2150, 760);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({{
            antialias: true,
            preserveDrawingBuffer: true,
            powerPreference: "high-performance"
        }});

        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.15));
        renderer.setSize(W, Hview);
        renderer.outputEncoding = THREE.sRGBEncoding;
        renderer.physicallyCorrectLights = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.04;
        renderer.shadowMap.enabled = false;
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
        const tubeLayout = "{tube_layout}";
        const isDoubleTube = tubeLayout === "double";
        const RtLower = {d_tubo_lower:.6f} / 2.0;
        const RtUpper = {d_tubo_upper:.6f} / 2.0;
        const Hs = {float(spalla)};
        const guideOffsetX = {float(guide_offset_x)};
        const coilFootprint = {float(coil_footprint_mm):.6f};
        const palletSize = 750.0;
        const palletHeight = 130.0;
        const boxHeight = 1030.0;
        const initialPackRollCount = {int(pack_roll_count)};

        controls.target.set(0, 0, Hs * 0.52);
        camera.lookAt(0, 0, Hs * 0.52);

        const localRaw = {final_local_points_json};
        const thetaRaw = {final_thetas_json};
        const radiusRaw = {final_radii_json};
        const zRaw = {final_zs_json};
        const layerRaw = {final_layers_json};
        const lengthRaw = {final_lengths_json};

        const localPts = localRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));

        let isPlaying = false;
        let animationEnabled = false;
        let speed = 1.0;
        let aspoMode = "hidden";
        let tubeMode = "{tube_mode_initial}";
        let currentView = "3d";
        let sceneMode = "{initial_scene}";
        let packagingMode = "{packaging_mode}";
        let containerMode = "{container_mode}";
        let showPackagingDimensions = true;
        let showStudio = false;
        let showGhost = false;
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
                camera.position.set(0, -2600, Hs * 0.56);
            }} else if (viewName === "side") {{
                camera.position.set(-2600, 0, Hs * 0.56);
            }} else {{
                camera.position.set(-1350, -2150, 760);
            }}

            camera.up.set(0, 0, 1);
            controls.target.copy(target);
            camera.lookAt(target);
            controls.update();
        }}

        function setPackagingCamera() {{
            const totalHeight = packagingGroup.userData.totalHeight || 800;
            const sceneSpan = Math.max(palletSize * 2.05, totalHeight * 1.48);
            camera.position.set(-sceneSpan * 1.55, -sceneSpan * 1.78, Math.max(1380, totalHeight * 1.55));
            controls.target.set(0, 0, palletHeight + totalHeight * 0.40);
            camera.lookAt(0, 0, palletHeight + totalHeight * 0.40);
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

        sceneBtns.forEach(btn => {{
            btn.addEventListener("click", () => {{
                sceneMode = btn.dataset.scene;
                setActiveButton(sceneBtns, sceneMode, "data-scene");
                applySceneMode();
            }});
        }});

        function setPackRollCount(value) {{
            if (!packRollCountInput) return;
            const clamped = Math.max(1, Math.min(50, parseInt(value || "1", 10)));
            packRollCountInput.value = clamped;
            updatePackagingScene();

            if (sceneMode === "packaging") {{
                setPackagingCamera();
            }}
        }}

        if (packRollCountInput) {{
            packRollCountInput.addEventListener("input", () => {{
                setPackRollCount(packRollCountInput.value);
            }});
        }}

        if (packDimensionsCheck) {{
            packDimensionsCheck.addEventListener("change", () => {{
                showPackagingDimensions = packDimensionsCheck.checked;
                updatePackagingScene();
            }});
        }}

        if (packRollMinusBtn) {{
            packRollMinusBtn.addEventListener("click", () => {{
                setPackRollCount(parseInt(packRollCountInput.value || "1", 10) - 1);
            }});
        }}

        if (packRollPlusBtn) {{
            packRollPlusBtn.addEventListener("click", () => {{
                setPackRollCount(parseInt(packRollCountInput.value || "1", 10) + 1);
            }});
        }}

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

        let pseudoFullscreen = false;
        let savedFrameStyle = null;

        function setFrameFullscreen(active) {{
            try {{
                const frame = window.frameElement;
                if (!frame) return false;

                if (active) {{
                    savedFrameStyle = {{
                        position: frame.style.position,
                        inset: frame.style.inset,
                        top: frame.style.top,
                        left: frame.style.left,
                        width: frame.style.width,
                        height: frame.style.height,
                        zIndex: frame.style.zIndex,
                        border: frame.style.border,
                        borderRadius: frame.style.borderRadius,
                    }};

                    frame.style.position = "fixed";
                    frame.style.inset = "0";
                    frame.style.top = "0";
                    frame.style.left = "0";
                    frame.style.width = "100vw";
                    frame.style.height = "100vh";
                    frame.style.zIndex = "2147483647";
                    frame.style.border = "none";
                    frame.style.borderRadius = "0";
                }} else if (savedFrameStyle) {{
                    Object.assign(frame.style, savedFrameStyle);
                    savedFrameStyle = null;
                }}

                return true;
            }} catch (err) {{
                return false;
            }}
        }}

        function setPseudoFullscreen(active) {{
            pseudoFullscreen = active;
            host.classList.toggle("pseudo_fullscreen", active);
            document.documentElement.classList.toggle("pseudo_fullscreen_doc", active);
            document.body.classList.toggle("pseudo_fullscreen_doc", active);
            setFrameFullscreen(active);

            fullscreenBtn.textContent = active ? "×" : "⛶";
            fullscreenBtn.title = active ? T.exit : T.fullscreen;
            setTimeout(resizeViewer, 120);
        }}

        fullscreenBtn.addEventListener("click", async () => {{
            try {{
                if (document.fullscreenElement) {{
                    await document.exitFullscreen();
                    fullscreenBtn.textContent = "⛶";
                    fullscreenBtn.title = T.fullscreen;
                    setTimeout(resizeViewer, 80);
                    return;
                }}

                if (pseudoFullscreen) {{
                    setPseudoFullscreen(false);
                    return;
                }}

                if (host.requestFullscreen && document.fullscreenEnabled) {{
                    await host.requestFullscreen();
                    fullscreenBtn.textContent = "×";
                    fullscreenBtn.title = T.exit;
                    setTimeout(resizeViewer, 80);
                }} else {{
                    // Fallback for tablets/browsers where iframe fullscreen is blocked.
                    setPseudoFullscreen(true);
                }}
            }} catch (err) {{
                // Fallback for Streamlit iframe / iPad cases where Fullscreen API is blocked.
                setPseudoFullscreen(!pseudoFullscreen);
            }}
        }});

        fullscreenBtn.title = T.fullscreen;
        captureRenderBtn.title = T.capture_render || "Save render image";

        captureRenderBtn.addEventListener("click", () => {{
            try {{
                renderer.render(scene, camera);
                const link = document.createElement("a");
                const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
                link.download = `avvolgimento-render-${{stamp}}.png`;
                link.href = renderer.domElement.toDataURL("image/png");
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }} catch (err) {{
                console.warn("Render capture failed", err);
            }}
        }});

        function escapeHtmlForPrint(value) {{
            return String(value ?? "")
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;");
        }}

        function buildSimulationPrintHtml(imageDataUrl) {{
            const rows = (SIM_PRINT.rows || []).map(row => `
                <tr><th>${{escapeHtmlForPrint(row[0])}}</th><td>${{escapeHtmlForPrint(row[1])}}</td></tr>
            `).join("");
            const status = (SIM_PRINT.status_items || []).map(item => `
                <div class="status ${{escapeHtmlForPrint(item.tone || "")}}">
                    <b>${{escapeHtmlForPrint(item.label || "")}}</b>
                    <span>${{escapeHtmlForPrint(item.value || "")}}</span>
                    <small>${{escapeHtmlForPrint(item.note || "")}}</small>
                </div>
            `).join("");
            return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>${{escapeHtmlForPrint(SIM_PRINT.product || "Avvolgimento")}} · ${{escapeHtmlForPrint(SIM_PRINT.title || "Simulation")}}</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:32px;color:#111827;background:#f8fafc;}}
.print-actions{{display:flex;justify-content:flex-end;margin:0 0 18px 0;}}
.print-btn{{border:none;border-radius:999px;padding:11px 18px;background:#C57E5A;color:white;font-weight:950;cursor:pointer;box-shadow:0 10px 22px rgba(197,126,90,.24);}}
.header{{border-left:6px solid #C57E5A;padding:18px 22px;background:white;border-radius:18px;box-shadow:0 8px 20px rgba(0,0,0,.06);}}
h1{{margin:0;font-size:28px;}}.subtitle{{margin-top:8px;color:#64748b;font-weight:750;}}
.badge{{display:inline-block;margin-top:12px;padding:7px 11px;border-radius:999px;background:#C57E5A;color:white;font-weight:900;font-size:12px;}}
.grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:18px;margin-top:22px;}}.card{{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:18px;box-shadow:0 6px 16px rgba(0,0,0,.045);}}
h2{{margin:0 0 14px 0;font-size:18px;}}table{{width:100%;border-collapse:collapse;font-size:13px;}}th{{text-align:left;color:#64748b;width:44%;padding:8px;border-bottom:1px solid #e5e7eb;}}td{{font-weight:850;padding:8px;border-bottom:1px solid #e5e7eb;}}
.render-img{{width:100%;border-radius:14px;border:1px solid #e5e7eb;background:#111827;display:block;}}
.statusgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px;}}.status{{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:14px;}}.status b{{display:block;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.06em;}}.status span{{display:block;font-size:20px;font-weight:950;margin-top:6px;}}.status small{{display:block;color:#64748b;margin-top:6px;line-height:1.25;}}.status.ok{{border-color:#86efac}}.status.warn{{border-color:#fbbf24}}.status.bad{{border-color:#fca5a5}}
@media print{{body{{background:white;margin:18px}}.card,.header,.status{{box-shadow:none}}.print-actions{{display:none}}.grid{{grid-template-columns:1fr .85fr}}}}
</style>
</head>
<body>
<div class="print-actions"><button class="print-btn" onclick="window.print()">${{escapeHtmlForPrint(SIM_PRINT.print_label || "Print")}}</button></div>
<div class="header"><h1>${{escapeHtmlForPrint(SIM_PRINT.product || "")}}</h1><div class="subtitle">${{escapeHtmlForPrint(SIM_PRINT.subtitle || "")}}</div><span class="badge">${{escapeHtmlForPrint(SIM_PRINT.title || "")}}</span></div>
<div class="statusgrid">${{status}}</div>
<div class="grid"><div class="card"><h2>${{escapeHtmlForPrint(SIM_PRINT.capture_label || "Render")}}</h2><img class="render-img" src="${{imageDataUrl}}" /></div><div class="card"><h2>${{escapeHtmlForPrint(SIM_PRINT.title || "Simulation")}}</h2><table>${{rows}}</table></div></div>
</body>
</html>`;
        }}



        function downloadSimulationPdf(imageDataUrl) {{
            if (!window.jspdf || !window.jspdf.jsPDF) {{
                const printWindow = window.open("", "_blank");
                if (!printWindow) return;
                printWindow.document.open();
                printWindow.document.write(buildSimulationPrintHtml(imageDataUrl));
                printWindow.document.close();
                printWindow.focus();
                return;
            }}

            const {{ jsPDF }} = window.jspdf;
            const doc = new jsPDF({{ orientation: "landscape", unit: "mm", format: "a4" }});
            const pageW = doc.internal.pageSize.getWidth();
            const pageH = doc.internal.pageSize.getHeight();
            const accent = [197, 126, 90];
            const ink = [17, 24, 39];
            const muted = [100, 116, 139];
            const line = [229, 231, 235];
            const soft = [248, 250, 252];

            doc.setFillColor(255, 255, 255);
            doc.rect(0, 0, pageW, pageH, "F");
            doc.setFillColor(accent[0], accent[1], accent[2]);
            doc.roundedRect(12, 12, 3, 24, 1.2, 1.2, "F");
            doc.setTextColor(ink[0], ink[1], ink[2]);
            doc.setFont("helvetica", "bold");
            doc.setFontSize(19);
            doc.text(String(SIM_PRINT.product || ""), 20, 22);
            doc.setFontSize(9);
            doc.setTextColor(muted[0], muted[1], muted[2]);
            doc.text(String(SIM_PRINT.subtitle || ""), 20, 29);
            doc.setFillColor(accent[0], accent[1], accent[2]);
            doc.roundedRect(pageW - 62, 14, 47, 10, 5, 5, "F");
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(8);
            doc.text(String(SIM_PRINT.title || "Simulation"), pageW - 38.5, 20.5, {{ align: "center" }});

            const statuses = SIM_PRINT.status_items || [];
            const statusY = 42;
            const statusW = (pageW - 30 - Math.max(0, statuses.length - 1) * 5) / Math.max(1, statuses.length || 1);
            statuses.slice(0, 3).forEach((item, idx) => {{
                const x = 15 + idx * (statusW + 5);
                doc.setFillColor(soft[0], soft[1], soft[2]);
                doc.setDrawColor(line[0], line[1], line[2]);
                doc.roundedRect(x, statusY, statusW, 24, 3, 3, "FD");
                doc.setTextColor(muted[0], muted[1], muted[2]);
                doc.setFont("helvetica", "bold");
                doc.setFontSize(7);
                doc.text(String(item.label || "").toUpperCase(), x + 4, statusY + 7);
                doc.setTextColor(ink[0], ink[1], ink[2]);
                doc.setFontSize(13);
                doc.text(String(item.value || ""), x + 4, statusY + 15);
                doc.setTextColor(muted[0], muted[1], muted[2]);
                doc.setFontSize(7);
                const note = doc.splitTextToSize(String(item.note || ""), statusW - 8);
                doc.text(note.slice(0, 1), x + 4, statusY + 21);
            }});

            const imageX = 15, imageY = 76, imageW = 174, imageH = 104;
            doc.setDrawColor(line[0], line[1], line[2]);
            doc.roundedRect(imageX, imageY, imageW, imageH, 3, 3, "S");
            doc.addImage(imageDataUrl, "PNG", imageX + 2, imageY + 2, imageW - 4, imageH - 4, undefined, "FAST");

            const tableX = 198, tableY = 76, tableW = pageW - tableX - 15;
            doc.setFillColor(soft[0], soft[1], soft[2]);
            doc.setDrawColor(line[0], line[1], line[2]);
            doc.roundedRect(tableX, tableY, tableW, imageH, 3, 3, "FD");
            doc.setFont("helvetica", "bold");
            doc.setFontSize(11);
            doc.setTextColor(ink[0], ink[1], ink[2]);
            doc.text(String(SIM_PRINT.title || "Simulation"), tableX + 5, tableY + 9);

            const rows = SIM_PRINT.rows || [];
            const rowY0 = tableY + 16;
            const rowH = Math.min(7.2, (imageH - 21) / Math.max(1, rows.length));
            rows.forEach((row, i) => {{
                const y = rowY0 + i * rowH;
                if (i % 2 === 0) {{
                    doc.setFillColor(255, 255, 255);
                    doc.rect(tableX + 2, y - 4.6, tableW - 4, rowH, "F");
                }}
                doc.setDrawColor(line[0], line[1], line[2]);
                doc.line(tableX + 2, y + rowH - 4.6, tableX + tableW - 2, y + rowH - 4.6);
                doc.setFont("helvetica", "bold");
                doc.setTextColor(muted[0], muted[1], muted[2]);
                doc.setFontSize(7.5);
                doc.text(String(row[0] || ""), tableX + 5, y);
                doc.setTextColor(ink[0], ink[1], ink[2]);
                doc.setFontSize(8.2);
                const val = doc.splitTextToSize(String(row[1] || ""), tableW * 0.46);
                doc.text(val.slice(0, 1), tableX + tableW * 0.56, y);
            }});

            doc.setFont("helvetica", "bold");
            doc.setFontSize(7.2);
            doc.setTextColor(148, 163, 184);
            doc.text("PDF generato dalla simulazione - include cattura render", 15, pageH - 8);

            const cleanName = String(SIM_PRINT.product || "simulazione").replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "") || "simulazione";
            const fileName = "simulazione_" + cleanName + ".pdf";
            const pdfBlob = doc.output("blob");
            const pdfUrl = URL.createObjectURL(pdfBlob);
            const opened = window.open(pdfUrl, "_blank", "noopener");

            if (!opened) {{
                const a = document.createElement("a");
                a.href = pdfUrl;
                a.download = fileName;
                a.target = "_blank";
                document.body.appendChild(a);
                a.click();
                a.remove();
            }}

            setTimeout(() => {{
                URL.revokeObjectURL(pdfUrl);
            }}, 60000);
        }}


        if (printSimulationBtn) {{
            printSimulationBtn.addEventListener("click", () => {{
                try {{
                    renderer.render(scene, camera);
                    const imageDataUrl = renderer.domElement.toDataURL("image/png");
                    downloadSimulationPdf(imageDataUrl);
                }} catch (err) {{
                    console.warn("Simulation PDF failed", err);
                }}
            }});
        }}

        document.addEventListener("fullscreenchange", () => {{
            if (!document.fullscreenElement) {{
                fullscreenBtn.textContent = pseudoFullscreen ? "×" : "⛶";
                fullscreenBtn.title = pseudoFullscreen ? T.exit : T.fullscreen;
            }}
            setTimeout(resizeViewer, 120);
        }});

        window.addEventListener("keydown", (event) => {{
            if (event.key === "Escape" && pseudoFullscreen) {{
                setPseudoFullscreen(false);
            }}
        }});

        progressSlider.addEventListener("input", () => {{
            const maxPos = Math.max(1, localPts.length - 1);
            drawPos = (parseInt(progressSlider.value) / 1000.0) * maxPos;
            rebuildDepositedMesh(Math.floor(drawPos), true);
            updateOverlayContinuous(true);
        }});

        function resizeViewer() {{
            const nw = Math.max(host.clientWidth, 600);
            const nh = Math.max(host.clientHeight, pseudoFullscreen || document.fullscreenElement ? 520 : 360);

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

        function makeDimensionLabelSprite(text, colorHex="#ffffff") {{
            const canvas = document.createElement("canvas");
            canvas.width = 320;
            canvas.height = 100;
            const ctx = canvas.getContext("2d");

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // background
            ctx.fillStyle = "rgba(12, 17, 24, 0.88)";
            ctx.strokeStyle = "rgba(255,255,255,0.16)";
            ctx.lineWidth = 2;
            const x = 10, y = 10, w = 300, h = 80, r = 18;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = colorHex;
            ctx.font = "700 34px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(text, canvas.width / 2, canvas.height / 2);

            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;

            const material = new THREE.SpriteMaterial({{
                map: texture,
                transparent: true,
                depthTest: false,
                depthWrite: false
            }});

            const sprite = new THREE.Sprite(material);
            sprite.scale.set(180, 56, 1);
            return sprite;
        }}

        function makeTubeTexture(size = 256, dark=false) {{
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;

            const ctx = canvas.getContext("2d");
            const base = dark ? 72 : 232;

            ctx.fillStyle = `rgb(${{base}}, ${{base}}, ${{base}})`;
            ctx.fillRect(0, 0, size, size);

            const img = ctx.getImageData(0, 0, size, size);
            const data = img.data;

            for (let y = 0; y < size; y++) {{
                for (let x = 0; x < size; x++) {{
                    const i = (y * size + x) * 4;

                    const fineNoise = (Math.random() - 0.5) * (dark ? 4.0 : 3.0);
                    const softLongitudinal = Math.sin((x * 0.055) + (y * 0.035)) * (dark ? 1.2 : 1.8);
                    const subtleBand = Math.exp(-Math.pow((x - size * 0.43) / (size * 0.18), 2)) * (dark ? 5.0 : 8.0);
                    const faintSeam = Math.exp(-Math.pow((x - size * 0.76) / (size * 0.06), 2)) * (dark ? -1.4 : -2.0);

                    let v = base + fineNoise + softLongitudinal + subtleBand + faintSeam;

                    if (dark) {{
                        v = Math.max(45, Math.min(112, v));
                    }} else {{
                        v = Math.max(214, Math.min(252, v));
                    }}

                    data[i] = v;
                    data[i + 1] = v;
                    data[i + 2] = v;
                    data[i + 3] = 255;
                }}
            }}

            ctx.putImageData(img, 0, 0);

            const gloss = ctx.createLinearGradient(0, 0, size, 0);
            gloss.addColorStop(0.00, "rgba(255,255,255,0.00)");
            gloss.addColorStop(0.35, dark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.10)");
            gloss.addColorStop(0.54, dark ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.13)");
            gloss.addColorStop(0.82, "rgba(255,255,255,0.00)");
            ctx.fillStyle = gloss;
            ctx.fillRect(0, 0, size, size);

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(1.8, 18.0);
            tex.anisotropy = 12;
            tex.needsUpdate = true;

            return tex;
        }}

        function makeCopperTexture(size = 256) {{
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;

            const ctx = canvas.getContext("2d");
            const grad = ctx.createLinearGradient(0, 0, size, 0);
            grad.addColorStop(0.00, "#8c4f2b");
            grad.addColorStop(0.18, "#cb8c60");
            grad.addColorStop(0.34, "#f2bf96");
            grad.addColorStop(0.52, "#b96f41");
            grad.addColorStop(0.72, "#e7b188");
            grad.addColorStop(1.00, "#8a4a28");
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, size, size);

            const img = ctx.getImageData(0, 0, size, size);
            const data = img.data;

            for (let y = 0; y < size; y++) {{
                for (let x = 0; x < size; x++) {{
                    const i = (y * size + x) * 4;
                    const grain = Math.random() * 18 - 9;
                    const brushed = Math.sin((x * 0.48) + (y * 0.05)) * 10.0;
                    const warm = Math.sin(y * 0.16) * 4.0;

                    data[i] = Math.max(90, Math.min(255, data[i] + grain + brushed + warm));
                    data[i + 1] = Math.max(52, Math.min(225, data[i + 1] + grain * 0.65 + brushed * 0.55));
                    data[i + 2] = Math.max(28, Math.min(188, data[i + 2] + grain * 0.38 + brushed * 0.25));
                    data[i + 3] = 255;
                }}
            }}

            ctx.putImageData(img, 0, 0);

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(1.0, 8.0);
            tex.anisotropy = 12;
            tex.needsUpdate = true;

            return tex;
        }}

        const steelTex = makeSteelTexture(256);
        const tubeWhiteTex = makeTubeTexture(256, false);
        const tubeBlackTex = makeTubeTexture(256, true);
        const copperTex = makeCopperTexture(256);

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

        function makeCopperMat(bright=false) {{
            return new THREE.MeshPhysicalMaterial({{
                color: bright ? 0xe1a579 : 0xc47a4d,
                map: copperTex,
                roughness: bright ? 0.22 : 0.28,
                metalness: 0.86,
                clearcoat: 0.22,
                clearcoatRoughness: 0.18,
                reflectivity: 0.78,
                emissive: bright ? 0x201007 : 0x120804,
                emissiveIntensity: bright ? 0.06 : 0.03
            }});
        }}

        function makeTubeMaterial(mode, active=false, free=false) {{
            const theme = getTheme();
            const chosen = active ? theme.activeTube : (free ? theme.freeTube : theme.tube);
            const tex = mode === "gelblack" ? tubeBlackTex : tubeWhiteTex;

            return new THREE.MeshPhysicalMaterial({{
                color: chosen,
                map: tex,
                roughness: active ? 0.72 : (free ? 0.84 : 0.78),
                metalness: 0.01,
                clearcoat: mode === "gelblack" ? 0.10 : 0.18,
                clearcoatRoughness: mode === "gelblack" ? 0.20 : 0.14,
                reflectivity: mode === "gelblack" ? 0.22 : 0.34,
                clippingPlanes: clippingPlanes,
                clipShadows: showSection
            }});
        }}

        let steelMat = makeSteelMat(1.0, false);
        let steelMatTransparent = makeSteelMat(0.18, true);

        let tubeMat = makeTubeMaterial(tubeMode, false, false);
        let activeTubeMat = makeTubeMaterial(tubeMode, true, false);
        let freeTubeMat = makeTubeMaterial(tubeMode, false, true);

        let copperMat = makeCopperMat(false);
        let copperBrightMat = makeCopperMat(true);

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
            emissiveIntensity: 0.42
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
            const footprintOver = Math.max(0, coilFootprint - palletSize);
            const heightLimit = packagingMode === "box" ? boxHeight : (containerMode === "40hc" ? 2580.0 : 2280.0);
            const comparesTotal = packagingMode === "tower";
            const comparedHeight = comparesTotal ? totalHeight : stackHeight;
            const heightMargin = Math.max(0, heightLimit - comparedHeight);
            const heightOver = Math.max(0, comparedHeight - heightLimit);
            const widthWarn = footprintOver > 0.001 && footprintOver <= 20.001;
            const ok = footprintOver <= 0.001 && heightOver <= 0.001;
            const warn = widthWarn && heightOver <= 0.001;
            const toneColor = ok ? "#4ade80" : (warn ? "#f59e0b" : "#fb7185");
            const toneBorder = ok ? "rgba(74,222,128,0.35)" : (warn ? "rgba(251,191,36,0.42)" : "rgba(252,165,165,0.42)");
            const toneBg = ok ? "rgba(20,83,45,0.56)" : (warn ? "rgba(120,78,0,0.56)" : "rgba(127,29,29,0.56)");
            const statusText = ok ? (T.box_fit_ok || "OK") : (warn ? "Attenzione" : (T.box_fit_over || "Fuori limite"));
            const heightLimitText = `${{heightLimit.toFixed(0)}} mm`;
            const reasonText = ok
                ? (packagingMode === "box" ? (T.packaging_box_desc || "") : (containerMode === "40hc" ? (T.container_40hc_desc || "") : (T.container_20ft_desc || "")))
                : (warn
                    ? `${{T.coil_footprint || "Ingombro rotolo"}}: ${{coilFootprint.toFixed(1)}} mm (+${{footprintOver.toFixed(1)}} mm, margine tollerato)`
                    : (footprintOver > 20.001
                        ? `${{T.coil_footprint || "Ingombro rotolo"}}: ${{coilFootprint.toFixed(1)}} mm > ${{(palletSize + 20).toFixed(0)}} mm`
                        : `${{T.height_over || "Superamento altezza"}}: ${{heightOver.toFixed(1)}} mm`));

            if (packagingStatusBadge) {{
                packagingStatusBadge.style.display = sceneMode === "packaging" ? "block" : "none";
                packagingStatusBadge.style.borderColor = toneBorder;
                packagingStatusBadge.style.background = toneBg;
                packagingStatusText.textContent = statusText;
                packagingStatusText.style.color = toneColor;
                packagingStatusReason.textContent = reasonText || "";
            }}

            const widthMargin = Math.max(0, palletSize - coilFootprint);
            const widthOver = Math.max(0, coilFootprint - palletSize);

            if (packagingStats) packagingStats.innerHTML = `
                <div class="pack_stat">
                    <div class="pack_stat_label">Status</div>
                    <div class="pack_stat_value" style="color:${{toneColor}}">${{statusText}}</div>
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
                    <div class="pack_stat_label">${{T.coil_footprint || "Ingombro rotolo"}}</div>
                    <div class="pack_stat_value">${{coilFootprint.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.width_margin || "Margine larghezza"}}</div>
                    <div class="pack_stat_value">${{widthMargin.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.width_over || "Superamento larghezza"}}</div>
                    <div class="pack_stat_value">${{widthOver.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.height_limit || "Limite altezza"}}</div>
                    <div class="pack_stat_value">${{heightLimitText}}</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.height_margin || "Margine altezza"}}</div>
                    <div class="pack_stat_value">${{heightMargin.toFixed(1)}} mm</div>
                </div>
                <div class="pack_stat">
                    <div class="pack_stat_label">${{T.height_over || "Superamento altezza"}}</div>
                    <div class="pack_stat_value">${{heightOver.toFixed(1)}} mm</div>
                </div>
            `;
        }}

        function createPalletRealistic(width, depth, height) {{
            const group = new THREE.Group();
            const woodMat = new THREE.MeshStandardMaterial({{
                color: 0xc79a61,
                roughness: 0.88,
                metalness: 0.01
            }});
            const woodMatDark = new THREE.MeshStandardMaterial({{
                color: 0x916236,
                roughness: 0.92,
                metalness: 0.0
            }});

            const topDeckH = height * 0.11;
            const runnerH = height * 0.24;
            const bottomDeckH = height * 0.08;
            const totalGap = height - topDeckH - runnerH - bottomDeckH;
            const midGap = Math.max(6, totalGap);

            const topZ = height - topDeckH / 2.0;
            const runnerZ = bottomDeckH + midGap / 2.0 + runnerH / 2.0;
            const bottomZ = bottomDeckH / 2.0;

            const topBoardW = width / 7.2;
            for (let i = -3; i <= 3; i++) {{
                const board = new THREE.Mesh(new THREE.BoxGeometry(topBoardW * 0.84, depth, topDeckH), woodMat);
                board.position.set(i * topBoardW, 0, topZ);
                board.castShadow = true;
                board.receiveShadow = true;
                group.add(board);
            }}

            const runnerW = width * 0.18;
            for (const x of [-width * 0.33, 0, width * 0.33]) {{
                const runner = new THREE.Mesh(new THREE.BoxGeometry(runnerW, depth, runnerH), woodMatDark);
                runner.position.set(x, 0, runnerZ);
                runner.castShadow = true;
                runner.receiveShadow = true;
                group.add(runner);
            }}

            const bottomBoardD = depth / 4.8;
            for (const y of [-depth * 0.34, 0, depth * 0.34]) {{
                const board = new THREE.Mesh(new THREE.BoxGeometry(width * 0.92, bottomBoardD * 0.60, bottomDeckH), woodMatDark);
                board.position.set(0, y, bottomZ);
                board.castShadow = true;
                board.receiveShadow = true;
                group.add(board);
            }}
            return group;
        }}

        function createRollRealistic(outerRadius, innerRadius, height, tubeMode) {{
            const group = new THREE.Group();

            const packTubeMat = makeTubeMaterial(tubeMode, false, false);
            const pts = localPts.map(p => p.clone());
            const rollMesh = makeWoundTubeObject(pts, packTubeMat);

            if (!rollMesh) return group;

            const bbox = new THREE.Box3().setFromObject(rollMesh);
            const center = bbox.getCenter(new THREE.Vector3());
            const size = bbox.getSize(new THREE.Vector3());

            rollMesh.position.sub(center);
            group.add(rollMesh);

            if (pts.length >= 2) {{
                const startPoint = pts[0].clone();
                const endPoint = pts[pts.length - 1].clone();

                const startTangent = pts[Math.min(1, pts.length - 1)].clone().sub(pts[0]);
                let endTangent = pts[pts.length - 1].clone().sub(pts[Math.max(0, pts.length - 2)]);
                if (endTangent.length() < 1e-6) {{
                    endTangent = startTangent.clone();
                }}

                const startTip = makeRealisticTubeTip(startPoint, startTangent, packTubeMat, null, -1);
                const endTip = makeRealisticTubeTip(endPoint, endTangent, packTubeMat, null, 1);

                if (startTip) {{
                    startTip.position.sub(center);
                    group.add(startTip);
                }}
                if (endTip) {{
                    endTip.position.sub(center);
                    group.add(endTip);
                }}
            }}

            // Scale only in Z so the packaging height follows the actual spalla.
            if (size.z > 1e-6) {{
                group.scale.z = height / size.z;
            }}

            return group;
        }}

        function updatePackagingScene() {{
            if (!packagingGroup) return;
            clearGroup(packagingGroup);

            const rollCount = Math.max(1, Math.min(50, parseInt(packRollCountInput.value || "1", 10)));
            packRollCountInput.value = rollCount;

            const stackHeight = rollCount * Hs;
            const totalHeight = palletHeight + stackHeight;
            const heightLimit = packagingMode === "box" ? boxHeight : (containerMode === "40hc" ? 2580.0 : 2280.0);
            const comparedHeight = packagingMode === "box" ? stackHeight : totalHeight;
            const footprintOver = Math.max(0, coilFootprint - palletSize);
            const footprintOk = footprintOver <= 0.001;
            const footprintWarn = footprintOver > 0.001 && footprintOver <= 20.001;
            const heightOk = comparedHeight <= heightLimit + 0.001;
            const ok = footprintOk && heightOk;
            const warn = footprintWarn && heightOk;
            const limitColor = ok ? 0x4ade80 : (warn ? 0xf59e0b : 0xf87171);
            const limitSoftColor = ok ? 0x4ade80 : (warn ? 0xf59e0b : 0xfb7185);

            const ground = new THREE.Mesh(
                new THREE.PlaneGeometry(palletSize * 2.3, palletSize * 2.3),
                new THREE.ShadowMaterial({{ color: 0x000000, opacity: 0.18 }})
            );
            ground.rotation.x = -Math.PI / 2;
            ground.position.z = -0.5;
            ground.receiveShadow = true;
            packagingGroup.add(ground);

            const pallet = createPalletRealistic(palletSize, palletSize, palletHeight);
            packagingGroup.add(pallet);

            if (packagingMode === "box") {{
                const boxMat = new THREE.MeshStandardMaterial({{
                    color: limitColor,
                    transparent: true,
                    opacity: 0.04,
                    roughness: 0.70,
                    metalness: 0.0,
                    depthWrite: false
                }});
                const box = new THREE.Mesh(new THREE.BoxGeometry(palletSize, palletSize, boxHeight), boxMat);
                // Scatola appoggiata sul pallet: altezza utile sopra il pallet.
                box.position.set(0, 0, palletHeight + boxHeight / 2);
                packagingGroup.add(box);
                if (showPackagingDimensions) {{
                    addBoxEdges(palletSize, palletSize, boxHeight, palletHeight + boxHeight / 2, limitSoftColor, 0.95);
                }}
            }} else {{
                // Container height is total allowed height including pallet, so wireframe starts from ground.
                if (showPackagingDimensions) {{
                    addBoxEdges(palletSize, palletSize, heightLimit, heightLimit / 2, limitSoftColor, 0.45);
                }}
            }}

            const coilRadius = coilFootprint / 2.0;
            const innerRadius = Math.max(18, coilRadius * 0.56);

            // Il calcolo resta basato sulla spalla reale (Hs), ma visivamente
            // aggiungiamo una piccola sovrapposizione controllata per evitare
            // i falsi "spazi d'aria" dovuti alla geometria del tubo renderizzato.
            const visualContactOverlap = Math.min(10, Math.max(4, Hs * 0.07));
            const rollVisualHeight = Hs + visualContactOverlap;

            const baseRoll = createRollRealistic(coilRadius, innerRadius, rollVisualHeight, tubeMode);

            for (let i = 0; i < rollCount; i++) {{
                const zc = palletHeight + i * Hs + Hs / 2.0;
                const roll = i === 0 ? baseRoll : baseRoll.clone(true);
                roll.position.set(0, 0, zc);
                packagingGroup.add(roll);
            }}

            if (showPackagingDimensions) {{
            const lineColor = limitColor;
            const heightLineMat = new THREE.LineBasicMaterial({{ color: lineColor, transparent: true, opacity: 0.95 }});
            const xDim = palletSize * 0.72;
            const yDim = -palletSize * 0.70;
            const heightPoints = [new THREE.Vector3(xDim, yDim, 0), new THREE.Vector3(xDim, yDim, totalHeight)];
            packagingGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(heightPoints), heightLineMat));
            packagingGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xDim - 24, yDim, 0), new THREE.Vector3(xDim + 24, yDim, 0)]), heightLineMat));
            packagingGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xDim - 24, yDim, totalHeight), new THREE.Vector3(xDim + 24, yDim, totalHeight)]), heightLineMat));

            const heightLabel = makeDimensionLabelSprite(`${{totalHeight.toFixed(0)}} mm`, ok ? "#4ade80" : (warn ? "#f59e0b" : "#fb7185"));
            heightLabel.position.set(xDim + 95, yDim, totalHeight * 0.5);
            packagingGroup.add(heightLabel);

            // Cota del margine disponibile fino al limite massimo:
            // - Scatola: dal punto più alto dell'ultimo rotolo fino al tetto utile della scatola.
            // - Torretta: dal punto più alto dell'ultimo rotolo fino al limite del container.
            const maxAllowedZ = packagingMode === "box" ? (palletHeight + boxHeight) : heightLimit;
            const marginToLimit = maxAllowedZ - totalHeight;
            const absMarginToLimit = Math.abs(marginToLimit);
            const marginColor = marginToLimit >= -0.001 ? 0x4ade80 : 0xf87171;
            const marginTextColor = marginToLimit >= -0.001 ? "#4ade80" : "#fb7185";
            const marginMat = new THREE.LineBasicMaterial({{ color: marginColor, transparent: true, opacity: 0.98 }});
            const xMargin = xDim + 155;
            const yMargin = yDim;

            const zA = Math.min(totalHeight, maxAllowedZ);
            const zB = Math.max(totalHeight, maxAllowedZ);
            const minVisibleMargin = Math.max(18, totalHeight * 0.015);
            const visualZA = zA;
            const visualZB = Math.max(zB, zA + minVisibleMargin);

            const marginLine = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(xMargin, yMargin, visualZA),
                    new THREE.Vector3(xMargin, yMargin, visualZB)
                ]),
                marginMat
            );
            packagingGroup.add(marginLine);

            packagingGroup.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(xMargin - 20, yMargin, totalHeight),
                    new THREE.Vector3(xMargin + 20, yMargin, totalHeight)
                ]),
                marginMat
            ));

            packagingGroup.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(xMargin - 20, yMargin, maxAllowedZ),
                    new THREE.Vector3(xMargin + 20, yMargin, maxAllowedZ)
                ]),
                marginMat
            ));

            const marginLabelText = marginToLimit >= -0.001
                ? `Margine ${{absMarginToLimit.toFixed(0)}} mm`
                : `Supera ${{absMarginToLimit.toFixed(0)}} mm`;
            const marginLabel = makeDimensionLabelSprite(marginLabelText, marginTextColor);
            marginLabel.position.set(xMargin + 112, yMargin, (visualZA + visualZB) * 0.5);
            packagingGroup.add(marginLabel);

            }}

            packagingGroup.userData.totalHeight = totalHeight;
            updatePackagingStats(rollCount);
        }}

        function applySceneMode() {{
            const packaging = sceneMode === "packaging";
            machine.visible = !packaging;
            guideGroup.visible = !packaging;
            overlayGroup.visible = !packaging;
            packagingGroup.visible = packaging;
            if (packagingControls) {{
                packagingControls.style.display = packaging ? "block" : "none";
                packagingControls.style.marginTop = "2px";
            }}
            if (animationBlock) animationBlock.style.display = packaging ? "none" : "block";
            if (speedBlock) speedBlock.style.display = packaging ? "none" : "block";
            if (spoolBlock) spoolBlock.style.display = packaging ? "none" : "block";
            if (checksBlock) checksBlock.style.display = packaging ? "none" : "grid";
            viewerHud.style.display = packaging ? "none" : "grid";
            if (packagingStatusBadge) {{
                packagingStatusBadge.style.display = packaging ? "block" : "none";
            }}
            progressSlider.disabled = packaging;
            playPauseBtn.disabled = packaging;
            animationCheck.disabled = packaging;
            if (packaging) {{
                updatePackagingScene();
                setPackagingCamera();
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
            if (sceneMode === "packaging") {{
                updatePackagingScene();
            }}
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

        function makeTubeEndCap(point, tangentDir, radius, material) {{
            const thickness = Math.max(1.8, radius * 0.22);
            const geo = new THREE.CylinderGeometry(radius * 0.985, radius * 0.985, thickness, 16, 1, false);
            const mesh = new THREE.Mesh(geo, material);

            mesh.position.copy(point);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const dir = tangentDir.clone().normalize();
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir);
            mesh.setRotationFromQuaternion(quat);
            mesh.castShadow = false;
            mesh.receiveShadow = true;

            return mesh;
        }}

        function makeTubeMeshFromPoints(points, radius, material) {{
            if (!points || points.length < 2) return null;

            let totalLen = 0;

            for (let i = 1; i < points.length; i++) {{
                totalLen += points[i].distanceTo(points[i - 1]);
            }}

            const curve = new PolylineCurve3(points);

            const tubularSegments = Math.max(
                18,
                Math.min(1200, Math.floor(totalLen / Math.max(1.60, radius * 0.75)))
            );

            const geo = new THREE.TubeGeometry(curve, tubularSegments, radius, 14, false);
            geo.computeVertexNormals();

            const body = new THREE.Mesh(geo, material);
            body.castShadow = false;
            body.receiveShadow = true;

            const group = new THREE.Group();
            group.add(body);

            const startPoint = points[0];
            const endPoint = points[points.length - 1];
            const startDir = new THREE.Vector3().subVectors(points[1], points[0]);
            const endDir = new THREE.Vector3().subVectors(points[points.length - 1], points[points.length - 2]);

            if (startDir.length() > 1e-6) {{
                const startCap = makeTubeEndCap(startPoint, startDir, radius, material);
                group.add(startCap);
            }}

            if (endDir.length() > 1e-6) {{
                const endCap = makeTubeEndCap(endPoint, endDir, radius, material);
                group.add(endCap);
            }}

            return group;
        }}

        function offsetPointsVertical(points, offset) {{
            return points.map(p => new THREE.Vector3(p.x, p.y, p.z + offset));
        }}

        function offsetPointVertical(point, offset) {{
            return new THREE.Vector3(point.x, point.y, point.z + offset);
        }}

        function makeWoundTubeObject(points, material) {{
            if (!isDoubleTube) {{
                return makeTubeMeshFromPoints(points, Rt, material);
            }}

            // Doppio verticale:
            // - tubo inferiore = diametro maggiore, appoggiato "sotto"
            // - tubo superiore = diametro minore, posizionato "sopra"
            // L'offset è assiale/verticale (asse Z locale), non radiale.
            const group = new THREE.Group();

            const lowerMesh = makeTubeMeshFromPoints(points, RtLower, material);
            if (lowerMesh) group.add(lowerMesh);

            const verticalOffset = RtLower + RtUpper;
            const upperPts = offsetPointsVertical(points, verticalOffset);
            const upperMesh = makeTubeMeshFromPoints(upperPts, RtUpper, material);
            if (upperMesh) group.add(upperMesh);

            return group;
        }}

        function makeTubeSegment(p0, p1, radius, material) {{
            const dir = new THREE.Vector3().subVectors(p1, p0);
            const len = dir.length();

            if (len < 1e-6) return null;

            const geo = new THREE.CylinderGeometry(radius, radius, len, 14, 1, false);
            const mesh = new THREE.Mesh(geo, material);

            const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
            mesh.position.copy(mid);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.clone().normalize());

            mesh.setRotationFromQuaternion(quat);
            mesh.castShadow = false;
            mesh.receiveShadow = true;

            return mesh;
        }}

        function makeWoundTubeSegment(p0, p1, material) {{
            if (!isDoubleTube) {{
                return makeTubeSegment(p0, p1, Rt, material);
            }}

            const group = new THREE.Group();

            const lowerSeg = makeTubeSegment(p0, p1, RtLower, material);
            if (lowerSeg) group.add(lowerSeg);

            const verticalOffset = RtLower + RtUpper;
            const p0u = offsetPointVertical(p0, verticalOffset);
            const p1u = offsetPointVertical(p1, verticalOffset);
            const upperSeg = makeTubeSegment(p0u, p1u, RtUpper, material);
            if (upperSeg) group.add(upperSeg);

            return group;
        }}

        function makeEndpointDisc(point, tangentDir, material, radiusScale = 0.92) {{
            const r = Math.max(7.0, Rt * radiusScale);
            const thickness = Math.max(2.0, Rt * 0.22);

            const geo = new THREE.CylinderGeometry(r, r * 0.95, thickness, 18);
            const mesh = new THREE.Mesh(geo, material);

            mesh.position.copy(point);

            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, tangentDir.clone().normalize());

            mesh.setRotationFromQuaternion(quat);
            mesh.castShadow = false;
            mesh.receiveShadow = true;

            return mesh;
        }}

        function orientAlongDir(obj, dir) {{
            const yAxis = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir.clone().normalize());
            obj.setRotationFromQuaternion(quat);
        }}

        function computeTubeSurfaceNormal(point, tangentDir) {{
            const tangent = (tangentDir && tangentDir.length() > 1e-6)
                ? tangentDir.clone().normalize()
                : new THREE.Vector3(0, 1, 0);

            let radial = new THREE.Vector3(point.x, point.y, 0);
            if (radial.length() < 1e-6) radial = new THREE.Vector3(1, 0, 0);
            radial.normalize();

            // Remove any component along the tube axis so the result is really a surface normal.
            let normal = radial.clone().sub(tangent.clone().multiplyScalar(radial.dot(tangent)));

            if (normal.length() < 1e-6) {{
                normal = new THREE.Vector3(0, 0, 1).cross(tangent);
            }}
            if (normal.length() < 1e-6) {{
                normal = new THREE.Vector3(1, 0, 0).cross(tangent);
            }}
            if (normal.length() < 1e-6) {{
                normal = new THREE.Vector3(1, 0, 0);
            }}

            normal.normalize();

            // Keep the normal pointing outward from the coil center.
            const outwardRef = new THREE.Vector3(point.x, point.y, 0);
            if (outwardRef.length() > 1e-6 && normal.dot(outwardRef.normalize()) < 0) {{
                normal.multiplyScalar(-1);
            }}

            return normal;
        }}

        function makeSingleRealisticTubeTip(point, tangentDir, outerRadius, sleeveMaterial, markerMaterial, outwardSign=1) {{
            if (!tangentDir || tangentDir.length() < 1e-6) return null;

            // The exposed copper must keep the local direction of the spiral.
            const dir = tangentDir.clone().normalize().multiplyScalar(outwardSign >= 0 ? 1 : -1);
            const group = new THREE.Group();

            const copperLen = Math.max(16.0, outerRadius * 1.95);
            const copperRadius = Math.max(1.8, Math.min(outerRadius * 0.31, outerRadius - 1.1));

            const copperStart = point.clone().addScaledVector(dir, 0.9);
            const copperEnd = point.clone().addScaledVector(dir, copperLen);
            const copperSeg = makeTubeSegment(copperStart, copperEnd, copperRadius, copperBrightMat);
            if (copperSeg) {{
                copperSeg.castShadow = false;
                copperSeg.receiveShadow = true;
                group.add(copperSeg);
            }}

            return group;
        }}

        function makeRealisticTubeTip(point, tangentDir, sleeveMaterial, markerMaterial, outwardSign=1) {{
            if (!isDoubleTube) {{
                return makeSingleRealisticTubeTip(point, tangentDir, Rt, sleeveMaterial, markerMaterial, outwardSign);
            }}

            const group = new THREE.Group();

            const lower = makeSingleRealisticTubeTip(point, tangentDir, RtLower, sleeveMaterial, markerMaterial, outwardSign);
            if (lower) group.add(lower);

            const verticalOffset = RtLower + RtUpper;
            const upperPoint = offsetPointVertical(point, verticalOffset);
            const upper = makeSingleRealisticTubeTip(upperPoint, tangentDir, RtUpper, sleeveMaterial, markerMaterial, outwardSign);
            if (upper) group.add(upper);

            return group;
        }}

        let depositedMesh = null;
        let freeMesh = null;
        let activeCoilMesh = null;
        let startMarker = null;
        let endMarker = null;
        let endMarkerGlow = null;

        let drawPos = Math.max(1.0, localPts.length - 1.0);
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

            depositedMesh = makeWoundTubeObject(pts, tubeMat);

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

            if (endMarkerGlow) {{
                disposeObj(endMarkerGlow, overlayGroup);
                endMarkerGlow = null;
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
            const futureCount = 70;
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

            if (sceneMode === "packaging") {{
                guideGroup.visible = false;
                return;
            }}

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

            const startIndexA = 0;
            const startIndexB = Math.min(1, localPts.length - 1);
            const startTangentLocal = localPts[startIndexB].clone().sub(localPts[startIndexA]);

            let endTangentLocal;
            if (i0 >= localPts.length - 1) {{
                endTangentLocal = localPts[localPts.length - 1].clone().sub(localPts[Math.max(0, localPts.length - 2)]);
            }} else {{
                endTangentLocal = localPts[i1].clone().sub(localPts[i0]);
            }}

            if (endTangentLocal.length() < 1e-6 && i0 > 0) {{
                endTangentLocal = localPts[i0].clone().sub(localPts[i0 - 1]);
            }}

            const startTangentWorld = startTangentLocal.clone().applyAxisAngle(new THREE.Vector3(0,0,1), theta);
            const endTangentWorld = endTangentLocal.clone().applyAxisAngle(new THREE.Vector3(0,0,1), theta);

            startMarker = makeRealisticTubeTip(
                startWorld,
                startTangentWorld,
                tubeMat,
                null,
                -1
            );

            endMarker = makeRealisticTubeTip(
                endWorld,
                endTangentWorld.length() > 1e-6 ? endTangentWorld : startTangentWorld,
                tubeMat,
                null,
                1
            );

            endMarkerGlow = null;

            overlayGroup.add(startMarker);
            overlayGroup.add(endMarker);

            if (animationEnabled) {{
                if (frac > 1e-6 && i1 > i0) {{
                    const activeStartWorld = localPointToWorld(activeLocalStart, theta);
                    activeCoilMesh = makeWoundTubeSegment(activeStartWorld, endWorld, activeTubeMat);

                    if (activeCoilMesh) {{
                        overlayGroup.add(activeCoilMesh);
                    }}
                }}

                const guideWorld = guidePointWorld(radius, z);

                freeMesh = makeWoundTubeSegment(guideWorld, endWorld, freeTubeMat);

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
        applySceneMode();
        updateAnimationUI();
        updatePlayBtn();

        function animate() {{
            requestAnimationFrame(animate);

            if (sceneMode !== "packaging" && animationEnabled && isPlaying && drawPos < localPts.length - 1) {{
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

            if (ghostLine && ghostLine.material && typeof ghostLine.material.dashOffset === "number") {{
                ghostLine.material.dashOffset -= 0.35 * Math.max(0.5, speed);
            }}

            controls.update();
            renderer.render(scene, camera);

            if (loadingOverlay && !loadingOverlay.classList.contains("is-hidden")) {{
                window.setTimeout(() => loadingOverlay.classList.add("is-hidden"), 1050);
            }}
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


def render_touch_preset_selector(label, options, key):
    """Compact non-keyboard selector for tablets.

    It uses Streamlit pills when available and falls back to a horizontal radio.
    Both options avoid the searchable selectbox input that opens the iPad keyboard.
    """
    if not options:
        return None

    current = st.session_state.get(key, options[0])
    if current not in options:
        current = options[0]
        st.session_state[key] = current

    selected = current

    if hasattr(st, "pills"):
        try:
            value = st.pills(
                label,
                options,
                selection_mode="single",
                default=current,
                key=f"{key}_pills",
            )
            if isinstance(value, (list, tuple)):
                selected = value[0] if value else current
            elif value is not None:
                selected = value
        except TypeError:
            # Compatibility with Streamlit versions whose pills signature differs.
            selected = st.radio(
                label,
                options,
                index=options.index(current),
                horizontal=True,
                key=f"{key}_radio",
            )
    else:
        selected = st.radio(
            label,
            options,
            index=options.index(current),
            horizontal=True,
            key=f"{key}_radio",
        )

    if selected not in options:
        selected = current

    st.session_state[key] = selected
    return selected


def render_workflow_bar(language):
    if language == "IT":
        steps = [
            ("1", "Preset", "scegli prodotto"),
            ("2", "Parametri", "controlla o modifica"),
            ("3", "Render", "avvolgimento / packaging"),
            ("4", "Risultati", "verifica finale"),
        ]
    else:
        steps = [
            ("1", "Preset", "select product"),
            ("2", "Parameters", "check or edit"),
            ("3", "Render", "winding / packaging"),
            ("4", "Results", "final check"),
        ]

    items_html = "".join(
        f"""
        <div class="workflow-step">
            <div class="workflow-num">{num}</div>
            <div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-subtitle">{subtitle}</div>
            </div>
        </div>
        """
        for num, title, subtitle in steps
    )

    st.markdown(
        f"""
        <style>
        .workflow-bar {{
            display:grid;
            grid-template-columns:repeat(4, minmax(0, 1fr));
            gap:10px;
            margin:10px 0 18px 0;
        }}
        .workflow-step {{
            display:flex;
            align-items:center;
            gap:10px;
            padding:16px 18px;
            border-radius:16px;
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            box-shadow:0 8px 18px rgba(0,0,0,0.07);
        }}
        .workflow-num {{
            width:30px;
            height:30px;
            min-width:30px;
            border-radius:999px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#C57E5A;
            color:#fff;
            font-weight:900;
            font-size:14px;
        }}
        .workflow-title {{
            font-weight:900;
            line-height:1.05;
            font-size:15px;
        }}
        .workflow-subtitle {{
            margin-top:3px;
            font-size:12px;
            color:color-mix(in srgb, var(--text-color) 62%, transparent);
            font-weight:650;
            line-height:1.15;
        }}
        @media (max-width: 900px) {{
            .workflow-bar {{
                grid-template-columns:repeat(2, minmax(0, 1fr));
            }}
        }}
        </style>
        <div class="workflow-bar">{items_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_page_title(title):
    st.markdown(
        f"""
        <style>
        .page-title-shell {{
            margin:0 0 12px 0;
            padding:0;
        }}
        .page-title-text {{
            margin:0;
            padding:0;
            font-size:30px;
            line-height:1.04;
            font-weight:950;
            letter-spacing:-0.02em;
            color:var(--text-color);
        }}
        .page-title-subline {{
            margin-top:4px;
            height:1px;
            width:100%;
            background:linear-gradient(90deg,
                color-mix(in srgb, var(--pdm-accent) 42%, transparent) 0%,
                color-mix(in srgb, var(--text-color) 8%, transparent) 32%,
                transparent 100%
            );
        }}
        </style>
        <div class="page-title-shell">
            <div class="page-title-text">{html.escape(str(title))}</div>
            <div class="page-title-subline"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title, subtitle=None, icon=""):
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="section-subtitle">{html.escape(str(subtitle))}</div>'

    badge_html = ""
    if str(icon).strip():
        badge_html = f'<div class="section-badge">{html.escape(str(icon))}</div>'

    st.markdown(
        f"""
        <style>
        .section-header {{
            margin-top:14px;
            margin-bottom:12px;
            padding:16px 18px;
            border-radius:16px;
            border:1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 8px 18px rgba(0,0,0,0.07);
        }}
        .section-header-row {{
            display:flex;
            align-items:flex-start;
            gap:12px;
        }}
        .section-badge {{
            width:30px;
            height:30px;
            min-width:30px;
            border-radius:999px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#C57E5A;
            color:#ffffff;
            font-size:14px;
            font-weight:900;
            line-height:1;
            box-shadow:0 4px 10px rgba(197,126,90,0.22);
            margin-top:1px;
        }}
        .section-header-copy {{
            display:flex;
            flex-direction:column;
            justify-content:flex-start;
            min-width:0;
        }}
        .section-title {{
            font-size:18px;
            font-weight:900;
            line-height:1.1;
            margin:0;
            padding:0;
        }}
        .section-subtitle {{
            margin-top:5px;
            font-size:13px;
            line-height:1.28;
            color:color-mix(in srgb, var(--text-color) 66%, transparent);
            font-weight:650;
            margin-bottom:0;
        }}
        </style>
        <div class="section-header">
            <div class="section-header-row">
                {badge_html}
                <div class="section-header-copy">
                    <div class="section-title">{html.escape(str(title))}</div>
                    {subtitle_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )






def render_pdf_open_new_tab_link(pdf_bytes, file_name, label, helper_text=None):
    """Render an iPad-friendly PDF open button without exposing the base64/HTML in Streamlit."""
    if not pdf_bytes:
        return

    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    safe_file_name_js = json.dumps(str(file_name))
    safe_label = html.escape(str(label))
    safe_helper = html.escape(str(helper_text or ""))

    button_html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        html, body {{
            margin:0;
            padding:0;
            background:transparent;
            font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            overflow:hidden;
        }}
        .pdf-open-wrap {{
            width:100%;
            box-sizing:border-box;
        }}
        .pdf-open-button {{
            width:100%;
            min-height:42px;
            border-radius:999px;
            padding:0 18px;
            border:1px solid #C57E5A;
            background:linear-gradient(180deg, #D18A62, #B96F48);
            color:#ffffff;
            font-weight:950;
            font-size:13px;
            line-height:1;
            letter-spacing:0.01em;
            cursor:pointer;
            box-shadow:0 8px 18px rgba(197,126,90,0.12);
        }}
        .pdf-open-button:hover {{
            filter:brightness(1.06);
            transform:translateY(-1px);
        }}
        .pdf-open-helper {{
            margin-top:7px;
            font-size:11px;
            line-height:1.25;
            color:rgba(100,116,139,0.92);
            font-weight:650;
            text-align:center;
        }}
    </style>
    </head>
    <body>
        <div class="pdf-open-wrap">
            <button class="pdf-open-button" id="openPdfBtn">{safe_label}</button>
            {f'<div class="pdf-open-helper">{safe_helper}</div>' if safe_helper else ''}
        </div>

        <script>
        (function () {{
            const pdfBase64 = "{pdf_b64}";
            const fileName = {safe_file_name_js};

            function base64ToBlob(base64, mimeType) {{
                const byteCharacters = atob(base64);
                const byteArrays = [];
                const sliceSize = 1024;

                for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {{
                    const slice = byteCharacters.slice(offset, offset + sliceSize);
                    const byteNumbers = new Array(slice.length);
                    for (let i = 0; i < slice.length; i++) {{
                        byteNumbers[i] = slice.charCodeAt(i);
                    }}
                    byteArrays.push(new Uint8Array(byteNumbers));
                }}

                return new Blob(byteArrays, {{ type: mimeType }});
            }}

            document.getElementById("openPdfBtn").addEventListener("click", function () {{
                try {{
                    const blob = base64ToBlob(pdfBase64, "application/pdf");
                    const url = URL.createObjectURL(blob);
                    const opened = window.open(url, "_blank", "noopener");

                    if (!opened) {{
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = fileName;
                        a.target = "_blank";
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                    }}

                    setTimeout(function () {{
                        URL.revokeObjectURL(url);
                    }}, 60000);
                }} catch (err) {{
                    console.error(err);
                }}
            }});
        }})();
        </script>
    </body>
    </html>
    """

    components.html(button_html, height=52 if not helper_text else 78, scrolling=False)


def render_preset_reveal_overlay(product_name, language, source_mode="preset", reveal_key=None):
    """Premium reveal pop-up. It stays open until the operator presses OK."""
    if language == "IT":
        kicker = "Preset caricato" if source_mode != "prototype" else "Prototipo attivo"
        line_1 = "Parametri caricati"
        line_2 = "render pronto"
        line_3 = ""
        badge = "PRESET ATTIVO" if source_mode != "prototype" else "PROTOTIPO"
        ok_label = "OK"
    else:
        kicker = "Preset loaded" if source_mode != "prototype" else "Active prototype"
        line_1 = "Parameters loaded"
        line_2 = "render ready"
        line_3 = ""
        badge = "ACTIVE PRESET" if source_mode != "prototype" else "PROTOTYPE"
        ok_label = "OK"

    if reveal_key is None:
        reveal_key = f"{source_mode}::{product_name}"

    button_key = "pdm_reveal_ok_" + hashlib.md5(str(reveal_key).encode("utf-8")).hexdigest()[:10]

    def close_reveal():
        st.session_state["last_revealed_product_key"] = reveal_key
        st.session_state.pop("pending_reveal_key", None)
        st.session_state.pop("pending_reveal_product", None)
        st.session_state.pop("pending_reveal_source_mode", None)
        st.rerun()

    def render_reveal_content():
        st.markdown(
            f"""
            <style>
            div[data-testid="stDialog"] {{
                --pdm-popup-overlay: rgba(2, 6, 23, 0.78);
                --pdm-popup-surface-top: #121a26;
                --pdm-popup-surface-bottom: #0a111c;
                --pdm-popup-text: #f8fafc;
                --pdm-popup-muted: rgba(248, 250, 252, 0.74);
                --pdm-popup-badge-bg: rgba(197,126,90,0.18);
                --pdm-popup-badge-border: rgba(197,126,90,0.46);
                --pdm-popup-shadow: 0 28px 88px rgba(0,0,0,0.48);
                position: fixed !important;
                inset: 0 !important;
                z-index: 999999 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 24px !important;
                background: var(--pdm-popup-overlay) !important;
                backdrop-filter: blur(8px) !important;
            }}

            html[data-theme="light"] div[data-testid="stDialog"],
            body[data-theme="light"] div[data-testid="stDialog"] {{
                --pdm-popup-overlay: rgba(248, 250, 252, 0.72);
                --pdm-popup-surface-top: #ffffff;
                --pdm-popup-surface-bottom: #f3f4f6;
                --pdm-popup-text: #0f172a;
                --pdm-popup-muted: rgba(15, 23, 42, 0.72);
                --pdm-popup-badge-bg: rgba(197,126,90,0.14);
                --pdm-popup-badge-border: rgba(197,126,90,0.34);
                --pdm-popup-shadow: 0 24px 64px rgba(15,23,42,0.16);
            }}

            html[data-theme="dark"] div[data-testid="stDialog"],
            body[data-theme="dark"] div[data-testid="stDialog"] {{
                --pdm-popup-overlay: rgba(2, 6, 23, 0.78);
                --pdm-popup-surface-top: #121a26;
                --pdm-popup-surface-bottom: #0a111c;
                --pdm-popup-text: #f8fafc;
                --pdm-popup-muted: rgba(248, 250, 252, 0.74);
                --pdm-popup-badge-bg: rgba(197,126,90,0.18);
                --pdm-popup-badge-border: rgba(197,126,90,0.46);
                --pdm-popup-shadow: 0 28px 88px rgba(0,0,0,0.48);
            }}

            div[data-testid="stDialog"] > div {{
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }}

            div[data-testid="stDialog"] [role="dialog"] {{
                width: min(720px, calc(100vw - 44px)) !important;
                margin: 0 auto !important;
                border-radius: 30px !important;
                border: 1px solid rgba(197,126,90,0.36) !important;
                background:
                    radial-gradient(circle at 12% 0%, rgba(197,126,90,0.16), transparent 34%),
                    linear-gradient(
                        180deg,
                        var(--pdm-popup-surface-top),
                        var(--pdm-popup-surface-bottom)
                    ) !important;
                box-shadow:
                    var(--pdm-popup-shadow),
                    inset 6px 0 0 #C57E5A !important;
                overflow: hidden !important;
            }}

            div[data-testid="stDialog"] .stButton > button {{
                border-radius: 999px !important;
                min-height: 44px !important;
                padding: 0 24px !important;
                border: 1px solid #C57E5A !important;
                background: linear-gradient(180deg, #D18A62, #B96F48) !important;
                color: #ffffff !important;
                font-weight: 950 !important;
                letter-spacing: .075em !important;
                text-transform: uppercase !important;
                box-shadow: 0 12px 26px rgba(197,126,90,0.28) !important;
            }}

            div[data-testid="stDialog"] .stButton > button:hover {{
                transform: translateY(-1px) !important;
                filter: brightness(1.06) !important;
                box-shadow: 0 16px 30px rgba(197,126,90,0.34) !important;
            }}

            .pdm-reveal-card {{
                position: relative;
                overflow: hidden;
                padding: 6px 2px 4px 8px;
                color: var(--pdm-popup-text);
                min-height: 210px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}

            .pdm-reveal-card::after {{
                content: "";
                position: absolute;
                top: -48%;
                bottom: -48%;
                left: 50%;
                width: 38%;
                background: linear-gradient(
                    105deg,
                    transparent 0%,
                    rgba(197,126,90,0.00) 18%,
                    rgba(197,126,90,0.26) 42%,
                    rgba(255,255,255,0.38) 50%,
                    rgba(197,126,90,0.46) 59%,
                    rgba(197,126,90,0.20) 70%,
                    transparent 100%
                );
                transform: translate3d(-260%,0,0) skewX(-17deg);
                will-change: transform, opacity;
                backface-visibility: hidden;
                mix-blend-mode: screen;
                pointer-events: none;
                animation: pdmRevealSweep 1.05s cubic-bezier(.22,.72,.18,1) .08s both;
            }}

            .pdm-reveal-scanline {{
                position: absolute;
                left: -24px;
                right: -24px;
                top: 52%;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(197,126,90,0.95), rgba(255,255,255,0.72), transparent);
                box-shadow: 0 0 22px rgba(197,126,90,0.50);
                opacity: 0;
                pointer-events: none;
                will-change: transform, opacity;
                backface-visibility: hidden;
                transform: translate3d(0,-92px,0);
                animation: pdmRevealScan 1.12s cubic-bezier(.22,.72,.18,1) .08s both;
            }}

            .pdm-reveal-kicker {{
                position: relative;
                z-index: 2;
                font-size: 12px;
                line-height: 1;
                font-weight: 950;
                letter-spacing: .095em;
                text-transform: uppercase;
                color: var(--pdm-popup-muted);
                margin-bottom: 10px;
            }}

            .pdm-reveal-title {{
                position: relative;
                z-index: 2;
                font-size: clamp(34px, 5vw, 58px);
                line-height: .92;
                font-weight: 950;
                letter-spacing: -.055em;
                color: var(--pdm-popup-text);
                margin-bottom: 15px;
                padding-right: 150px;
            }}

            .pdm-reveal-meta {{
                position: relative;
                z-index: 2;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px 10px;
                font-size: 13px;
                line-height: 1.2;
                font-weight: 760;
                color: var(--pdm-popup-muted);
                margin-bottom: 22px;
            }}

            .pdm-reveal-dot {{
                width: 5px;
                height: 5px;
                border-radius: 999px;
                background: #C57E5A;
                box-shadow: 0 0 12px rgba(197,126,90,0.56);
            }}

            .pdm-reveal-badge {{
                position: absolute;
                z-index: 2;
                right: 4px;
                top: 2px;
                border-radius: 999px;
                padding: 8px 11px;
                background: var(--pdm-popup-badge-bg);
                border: 1px solid var(--pdm-popup-badge-border);
                color: var(--pdm-popup-text);
                font-size: 10.5px;
                line-height: 1;
                font-weight: 950;
                letter-spacing: .075em;
                text-transform: uppercase;
            }}

            @keyframes pdmRevealSweep {{
                0% {{ transform: translate3d(-260%,0,0) skewX(-17deg); opacity: 0; }}
                10% {{ opacity: .92; }}
                58% {{ opacity: .92; }}
                100% {{ transform: translate3d(260%,0,0) skewX(-17deg); opacity: 0; }}
            }}

            @keyframes pdmRevealScan {{
                0% {{ transform: translate3d(0,-92px,0); opacity: 0; }}
                12% {{ opacity: .92; }}
                72% {{ opacity: .86; }}
                100% {{ transform: translate3d(0,98px,0); opacity: 0; }}
            }}

            @media (max-width: 720px) {{
                .pdm-reveal-title {{
                    padding-right: 0;
                }}
                .pdm-reveal-badge {{
                    position: relative;
                    right: auto;
                    top: auto;
                    display: inline-flex;
                    margin-bottom: 14px;
                }}
            }}
            </style>

            <div class="pdm-reveal-card">
                <div class="pdm-reveal-scanline"></div>
                <div class="pdm-reveal-badge">{html.escape(badge)}</div>
                <div class="pdm-reveal-kicker">{html.escape(kicker)}</div>
                <div class="pdm-reveal-title">{html.escape(str(product_name))}</div>
                <div class="pdm-reveal-meta">
                    <span>{html.escape(line_1)}</span>
                    <span class="pdm-reveal-dot"></span>
                    <span>{html.escape(line_2)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(ok_label, key=button_key, use_container_width=False):
            close_reveal()

    if hasattr(st, "dialog"):
        @st.dialog(kicker, width="large")
        def _pdm_reveal_dialog():
            render_reveal_content()

        _pdm_reveal_dialog()
    else:
        # Fallback for older Streamlit versions.
        with st.container():
            render_reveal_content()


def render_active_preset_card(product_name, language, modified=False):
    if modified:
        title = "Parametri modificati" if language == "IT" else "Modified parameters"
        subtitle = "Base preset selezionato, ma valori cambiati manualmente" if language == "IT" else "Selected preset as base, but values changed manually"
        accent = "#f59e0b"
    else:
        title = "Preset attivo" if language == "IT" else "Active preset"
        subtitle = "Caricamento automatico nel render" if language == "IT" else "Auto-loaded into render"
        accent = "#C57E5A"

    st.markdown(
        f"""
        <div style="
            padding:14px 16px;
            border-radius:16px;
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 84%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            border:1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
            box-shadow:0 6px 16px rgba(0,0,0,0.055);
            border-left:4px solid {accent};
        ">
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:color-mix(in srgb, var(--text-color) 62%, transparent); font-weight:800; margin-bottom:5px;">
                {html.escape(title)}
            </div>
            <div style="font-size:21px; line-height:1.15; font-weight:900;">
                {html.escape(str(product_name))}
            </div>
            <div style="font-size:12px; color:color-mix(in srgb, var(--text-color) 62%, transparent); font-weight:650; margin-top:5px;">
                {html.escape(subtitle)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_preset_product_card(selected_product, selected_row, language, modified=False):
    """Hero card for the selected preset. Keeps the preset visible without duplicating the full technical sheet."""
    locked = bool(st.session_state.get("params_locked", False))
    pulse_class = " pdm-pulse" if bool(st.session_state.get("changed_values_pulse", False)) else ""
    if pulse_class:
        st.session_state["changed_values_pulse"] = False

    def gv(*names, default="-"):
        for name in names:
            if name in selected_row.index:
                value = safe_value(selected_row, name)
                if value != "-":
                    return format_preset_value(value)
        return default

    if language == "IT":
        kicker = "Preset selezionato"
        subtitle = "Base dati caricata automaticamente nel render · dettagli completi nella Scheda tecnica"
        original_txt = "Originale"
        modified_txt = "Modificato"
        locked_txt = "Bloccato"
        editable_txt = "Editabile"
        restore_label = "Ripristina preset"
        lock_label = "Blocca parametri"
        chip_defs = [
            ("Tipo tubo", gv("Tipo tubo")),
            ("Ø rame", gv("Diametro Rame", "Diametro rame superiore", default="-")),
            ("Guaina", f"{gv('Spessore Guaina (mm)', 'Spessore guaina superiore', default='-')} mm"),
            ("Lunghezza", f"{gv('Lunghezza (m)', default='-')} m"),
            ("Aspo", f"Ø {gv('Diametro aspo (mm)', default='-')} mm"),
            ("Spalla", f"{gv('Spalla (mm)', default='-')} mm"),
        ]
    else:
        kicker = "Selected preset"
        subtitle = "Data base automatically loaded into the render · full details in the Technical sheet"
        original_txt = "Original"
        modified_txt = "Modified"
        locked_txt = "Locked"
        editable_txt = "Editable"
        restore_label = "Restore preset"
        lock_label = "Lock parameters"
        chip_defs = [
            ("Tube type", gv("Tipo tubo")),
            ("Copper Ø", gv("Diametro Rame", "Diametro rame superiore", default="-")),
            ("Foam", f"{gv('Spessore Guaina (mm)', 'Spessore guaina superiore', default='-')} mm"),
            ("Length", f"{gv('Lunghezza (m)', default='-')} m"),
            ("Spool", f"Ø {gv('Diametro aspo (mm)', default='-')} mm"),
            ("Width", f"{gv('Spalla (mm)', default='-')} mm"),
        ]

    status_txt = modified_txt if modified else original_txt
    lock_txt = locked_txt if locked else editable_txt
    status_class = "modified" if modified else "original"
    lock_class = "locked" if locked else "editable"

    chips_html = "".join(
        f"""
        <div class="preset-hero-chip">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in chip_defs
    )

    st.markdown(
        f"""
        <style>
        .preset-hero {{
            position:relative;
            margin:10px 0 16px 0;
            border-radius:24px;
            overflow:hidden;
            border:1px solid color-mix(in srgb, var(--text-color) 13%, transparent);
            background:
                radial-gradient(circle at 6% 0%, rgba(197,126,90,0.18), transparent 30%),
                linear-gradient(180deg,
                    color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color)),
                    color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
                );
            box-shadow:0 12px 30px rgba(0,0,0,0.075);
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease,
                background 0.18s ease;
        }}
        .preset-hero::before {{
            content:"";
            position:absolute;
            inset:0 auto 0 0;
            width:6px;
            background:linear-gradient(180deg, #D18A62, #B96F48);
            opacity:0.95;
            box-shadow:0 0 24px rgba(197,126,90,0.32);
            z-index:1;
        }}
        .preset-hero::after {{
            content:"";
            position:absolute;
            top:-45%;
            bottom:-45%;
            left:-72%;
            width:36%;
            pointer-events:none;
            border-radius:inherit;
            background:linear-gradient(105deg, transparent 0%, rgba(197,126,90,0.00) 18%, rgba(197,126,90,0.26) 40%, rgba(197,126,90,0.58) 56%, rgba(197,126,90,0.28) 70%, transparent 100%);
            transform:translate3d(-260%,0,0) skewX(-17deg);
            opacity:0;
            z-index:4;
            mix-blend-mode:screen;
            will-change:transform, opacity;
            backface-visibility:hidden;
            animation:pdmPresetCardShine 1.05s cubic-bezier(.22,.72,.18,1) 0.12s both;
        }}
        .preset-hero:hover {{
            transform:translateY(-1px);
            border-color:color-mix(in srgb, var(--pdm-accent) 34%, var(--text-color) 10%);
            box-shadow:0 16px 34px rgba(0,0,0,0.105);
            background:
                radial-gradient(circle at 6% 0%, rgba(197,126,90,0.22), transparent 32%),
                linear-gradient(180deg,
                    color-mix(in srgb, var(--secondary-background-color) 86%, var(--pdm-accent) 4%),
                    color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
                );
        }}
        .preset-hero:hover::after,
        .preset-hero:active::after {{
            opacity:1;
            animation:pdmPresetCardShine 1.18s cubic-bezier(.18,.72,.22,1) both;
        }}
        @keyframes pdmPresetCardShine {{
            0% {{ transform:translate3d(-260%,0,0) skewX(-17deg); opacity:0; }}
            10% {{ opacity:.92; }}
            56% {{ opacity:.92; }}
            100% {{ transform:translate3d(260%,0,0) skewX(-17deg); opacity:0; }}
        }}

        .premium-sweep-layer {{
            position:absolute;
            top:-45%;
            bottom:-45%;
            left:-72%;
            width:36%;
            pointer-events:none;
            border-radius:inherit;
            background:linear-gradient(105deg, transparent 0%, rgba(197,126,90,0.00) 18%, rgba(197,126,90,0.28) 42%, rgba(197,126,90,0.62) 56%, rgba(197,126,90,0.30) 70%, transparent 100%);
            transform:translate3d(-260%,0,0) skewX(-17deg);
            opacity:0;
            z-index:12;
            mix-blend-mode:screen;
            will-change:transform, opacity;
            backface-visibility:hidden;
            animation:pdmRealSweepLayer 1.05s cubic-bezier(.22,.72,.18,1) 0.12s both;
        }}
        .preset-hero:hover .premium-sweep-layer,
        .preset-hero:active .premium-sweep-layer {{
            animation:pdmRealSweepLayer 1.25s cubic-bezier(.18,.72,.22,1) both;
        }}
        @keyframes pdmRealSweepLayer {{
            0% {{ transform:translate3d(-260%,0,0) skewX(-17deg); opacity:0; }}
            10% {{ opacity:.92; }}
            56% {{ opacity:.92; }}
            100% {{ transform:translate3d(260%,0,0) skewX(-17deg); opacity:0; }}
        }}

        .preset-hero-top {{
            position:relative;
            z-index:2;
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:18px;
            padding:20px 22px 16px 22px;
            border-bottom:1px solid color-mix(in srgb, var(--text-color) 9%, transparent);
        }}
        .preset-hero-kicker {{
            font-size:11px;
            line-height:1;
            font-weight:950;
            letter-spacing:.08em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 60%, transparent);
            margin-bottom:8px;
        }}
        .preset-hero-title {{
            font-size:clamp(25px, 2vw, 36px);
            line-height:1.02;
            font-weight:950;
            letter-spacing:-.04em;
            color:var(--text-color);
            word-break:break-word;
        }}
        .preset-hero-subtitle {{
            margin-top:8px;
            font-size:13px;
            line-height:1.28;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 63%, transparent);
        }}
        .preset-hero-badges {{
            display:flex;
            flex-wrap:wrap;
            justify-content:flex-end;
            gap:8px;
            min-width:max-content;
        }}
        .preset-hero-badge {{
            position:relative;
            overflow:hidden;
            isolation:isolate;
            border-radius:999px;
            padding:9px 12px;
            font-size:11px;
            line-height:1;
            font-weight:950;
            letter-spacing:.055em;
            text-transform:uppercase;
            border:1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            background:color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color));
            color:var(--text-color);
        }}
        .preset-hero-badge.original {{
            background:#C57E5A;
            border-color:#C57E5A;
            color:#fff;
        }}
        .preset-hero-badge.modified {{
            background:#f59e0b;
            border-color:#f59e0b;
            color:#fff;
        }}
        .preset-hero-badge.locked {{
            background:#64748b;
            border-color:#64748b;
            color:#fff;
        }}
        .preset-hero-badge.verified {{
            background:rgba(197,126,90,0.16);
            border-color:rgba(197,126,90,0.48);
            color:var(--text-color);
            padding-left:24px;
        }}
        .preset-hero-badge.verified::before {{
            content:"";
            position:absolute;
            left:10px;
            top:50%;
            width:7px;
            height:7px;
            border-radius:999px;
            background:#C57E5A;
            box-shadow:0 0 0 0 rgba(197,126,90,0.44);
            transform:translateY(-50%);
            animation:pdmCsvVerifiedPulse 1.8s ease-in-out infinite;
        }}
        @keyframes pdmCsvVerifiedPulse {{
            0% {{ box-shadow:0 0 0 0 rgba(197,126,90,0.42); }}
            70% {{ box-shadow:0 0 0 7px rgba(197,126,90,0.00); }}
            100% {{ box-shadow:0 0 0 0 rgba(197,126,90,0.00); }}
        }}
        .preset-hero-chips {{
            position:relative;
            z-index:2;
            display:grid;
            grid-template-columns:repeat(6, minmax(0, 1fr));
            gap:10px;
            padding:14px 18px 18px 18px;
        }}
        .preset-hero-chip {{
            min-height:62px;
            padding:11px 13px;
            border-radius:16px;
            border:1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            background:color-mix(in srgb, var(--text-color) 4%, transparent);
            position:relative;
            overflow:hidden;
            isolation:isolate;
        }}
        .preset-hero-chip > *,
        .preset-hero-badge > * {{
            position:relative;
            z-index:2;
        }}
        .preset-hero-chip::after,
        .preset-hero-badge::after {{
            content:"";
            position:absolute;
            top:-45%;
            bottom:-45%;
            left:-72%;
            width:36%;
            pointer-events:none;
            border-radius:inherit;
            background:linear-gradient(105deg, transparent 0%, rgba(197,126,90,0.00) 18%, rgba(197,126,90,0.26) 40%, rgba(197,126,90,0.58) 56%, rgba(197,126,90,0.28) 70%, transparent 100%);
            transform:skewX(-17deg);
            opacity:0;
            z-index:4;
            mix-blend-mode:screen;
        }}
        .preset-hero-chip:hover::after,
        .preset-hero-chip:active::after,
        .preset-hero-badge:hover::after,
        .preset-hero-badge:active::after {{
            animation:pdmPresetCardShine 1.18s cubic-bezier(.18,.72,.22,1) both;
        }}
        .preset-hero-chip span {{
            display:block;
            font-size:10.5px;
            line-height:1.05;
            font-weight:950;
            letter-spacing:.055em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 56%, transparent);
            margin-bottom:7px;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }}
        .preset-hero-chip strong {{
            display:block;
            font-size:18px;
            line-height:1.05;
            font-weight:950;
            letter-spacing:-.02em;
            color:var(--text-color);
            overflow-wrap:anywhere;
        }}
        @media(max-width:1200px) {{
            .preset-hero-chips {{ grid-template-columns:repeat(3, minmax(0,1fr)); }}
        }}
        @media(max-width:760px) {{
            .preset-hero-top {{ flex-direction:column; }}
            .preset-hero-badges {{ justify-content:flex-start; }}
            .preset-hero-chips {{ grid-template-columns:repeat(2, minmax(0,1fr)); }}
        }}
        </style>
        <div class="preset-hero pdm-pulse{pulse_class}">
            <div class="premium-sweep-layer"></div>
            <div class="preset-hero-top">
                <div>
                    <div class="preset-hero-kicker">{html.escape(kicker)}</div>
                    <div class="preset-hero-title">{html.escape(str(selected_product))}</div>
                    <div class="preset-hero-subtitle">{html.escape(subtitle)}</div>
                </div>
                <div class="preset-hero-badges">
                    <span class="preset-hero-badge {status_class}">{html.escape(status_txt)}</span>
                    <span class="preset-hero-badge verified">PRESET ATTIVO</span>
                    <span class="preset-hero-badge {lock_class}">{html.escape(lock_txt)}</span>
                </div>
            </div>
            <div class="preset-hero-chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )





def render_tech_sheet_preset_card(selected_product, selected_row, language):
    """Preset card in Scheda tecnica with the same style and dimensions as the Simulazione selected preset card."""
    def gv(*names, default="-"):
        for name in names:
            if name in selected_row.index:
                value = safe_value(selected_row, name)
                if value != "-":
                    return format_preset_value(value)
        return default

    if language == "IT":
        kicker = "Scheda preset"
        subtitle = "Configurazione tecnica prodotto · valori caricati da preset"
        status_txt = "Originale"
        lock_txt = "Scheda preset"
        status_class = "original"
        lock_class = "editable"
        chip_defs = [
            ("Tipo tubo", gv("Tipo tubo")),
            ("Ø rame", gv("Diametro Rame", "Diametro rame superiore", default="-")),
            ("Guaina", f"{gv('Spessore Guaina (mm)', 'Spessore guaina superiore', default='-')} mm"),
            ("Lunghezza", f"{gv('Lunghezza (m)', default='-')} m"),
            ("Aspo", f"Ø {gv('Diametro aspo (mm)', default='-')} mm"),
            ("Spalla", f"{gv('Spalla (mm)', default='-')} mm"),
        ]
    else:
        kicker = "Preset sheet"
        subtitle = "Product technical configuration · values loaded from preset"
        status_txt = "Original"
        lock_txt = "Preset sheet"
        status_class = "original"
        lock_class = "editable"
        chip_defs = [
            ("Tube type", gv("Tipo tubo")),
            ("Copper Ø", gv("Diametro Rame", "Diametro rame superiore", default="-")),
            ("Foam", f"{gv('Spessore Guaina (mm)', 'Spessore guaina superiore', default='-')} mm"),
            ("Length", f"{gv('Lunghezza (m)', default='-')} m"),
            ("Spool", f"Ø {gv('Diametro aspo (mm)', default='-')} mm"),
            ("Width", f"{gv('Spalla (mm)', default='-')} mm"),
        ]

    chips_html = "".join(
        f"""
        <div class="preset-hero-chip">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in chip_defs
    )

    st.markdown(
        f"""
        <div class="preset-hero">
            <div class="premium-sweep-layer"></div>
            <div class="preset-hero-top">
                <div>
                    <div class="preset-hero-kicker">{html.escape(kicker)}</div>
                    <div class="preset-hero-title">{html.escape(str(selected_product))}</div>
                    <div class="preset-hero-subtitle">{html.escape(subtitle)}</div>
                </div>
                <div class="preset-hero-badges">
                    <span class="preset-hero-badge {status_class}">{html.escape(status_txt)}</span>
                    <span class="preset-hero-badge verified">PRESET ATTIVO</span>
                    <span class="preset-hero-badge {lock_class}">{html.escape(lock_txt)}</span>
                </div>
            </div>
            <div class="preset-hero-chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prototype_product_card(prototype_name, language):
    """Hero card for a manual prototype. Makes clear it is not an official Preset."""
    locked = bool(st.session_state.get("params_locked", False))
    pulse_class = " pdm-pulse" if bool(st.session_state.get("changed_values_pulse", False)) else ""
    if pulse_class:
        st.session_state["changed_values_pulse"] = False

    tube_layout = str(st.session_state.get("calc_tube_layout", "Singolo"))
    length = format_preset_value(st.session_state.get("calc_lunghezza", "-"))
    aspo = format_preset_value(st.session_state.get("calc_diametro_aspo", "-"))
    spalla = format_preset_value(st.session_state.get("calc_spalla", "-"))

    if tube_layout == "Doppio":
        copper_value = f"{st.session_state.get('calc_rame_sup', '-')}/{st.session_state.get('calc_rame_inf', '-')}"
        foam_value = f"{format_preset_value(st.session_state.get('calc_spessore_sup', '-'))}/{format_preset_value(st.session_state.get('calc_spessore_inf', '-'))} mm"
    else:
        copper_value = st.session_state.get("calc_rame", "-")
        foam_value = f"{format_preset_value(st.session_state.get('calc_spessore', '-'))} mm"

    if language == "IT":
        kicker = "Prototipo prodotto"
        subtitle = "Configurazione manuale · non salvata nel preset · PDF scheda preset non disponibile"
        prototype_txt = "Prototipo"
        locked_txt = "Bloccato"
        editable_txt = "Editabile"
        chip_defs = [
            ("Tipo tubo", tube_layout),
            ("Ø rame", copper_value),
            ("Guaina", foam_value),
            ("Lunghezza", f"{length} m"),
            ("Aspo", f"Ø {aspo} mm"),
            ("Spalla", f"{spalla} mm"),
        ]
    else:
        kicker = "Product prototype"
        subtitle = "Manual configuration · not saved in the preset · Preset sheet PDF unavailable"
        prototype_txt = "Prototype"
        locked_txt = "Locked"
        editable_txt = "Editable"
        chip_defs = [
            ("Tube type", tube_layout),
            ("Copper Ø", copper_value),
            ("Foam", foam_value),
            ("Length", f"{length} m"),
            ("Spool", f"Ø {aspo} mm"),
            ("Width", f"{spalla} mm"),
        ]

    lock_txt = locked_txt if locked else editable_txt
    lock_class = "locked" if locked else "editable"
    chips_html = "".join(
        f"""
        <div class="preset-hero-chip">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in chip_defs
    )

    st.markdown(
        f"""
        <div class="preset-hero pdm-pulse{pulse_class}">
            <div class="premium-sweep-layer"></div>
            <div class="preset-hero-top">
                <div>
                    <div class="preset-hero-kicker">{html.escape(kicker)}</div>
                    <div class="preset-hero-title">{html.escape(str(prototype_name))}</div>
                    <div class="preset-hero-subtitle">{html.escape(subtitle)}</div>
                </div>
                <div class="preset-hero-badges">
                    <span class="preset-hero-badge modified">{html.escape(prototype_txt)}</span>
                    <span class="preset-hero-badge {lock_class}">{html.escape(lock_txt)}</span>
                </div>
            </div>
            <div class="preset-hero-chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_render_hero(language, selected_product, tube_diameter_label, lunghezza, view_mode_label, packaging_tone, packaging_value, coil_footprint_mm):
    if language == "IT":
        title = "Simulazione avvolgimento"
        subtitle = "Il render è il riferimento principale: controlla geometria, ingombro e packaging in tempo reale."
        chips = [
            ("Preset", selected_product),
            ("Ø tubo", tube_diameter_label),
            ("Lunghezza", f"{lunghezza:.1f} m"),
            ("Vista", view_mode_label),
            ("Packaging", packaging_value),
            ("Ingombro XY", f"{coil_footprint_mm:.1f} mm"),
        ]
    else:
        title = "Winding simulation"
        subtitle = "The render is the main reference: check geometry, footprint and packaging in real time."
        chips = [
            ("Preset", selected_product),
            ("Tube Ø", tube_diameter_label),
            ("Length", f"{lunghezza:.1f} m"),
            ("View", view_mode_label),
            ("Packaging", packaging_value),
            ("XY footprint", f"{coil_footprint_mm:.1f} mm"),
        ]

    chips_html = "".join(
        f"""
        <div class="render-hero-pill {html.escape(packaging_tone if label == 'Packaging' else 'neutral')}">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in chips
    )

    st.markdown(
        f"""
        <style>
        .render-hero {{
            margin:4px 0 14px 0;
            padding:18px 20px;
            border-radius:24px;
            border:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            background:
                radial-gradient(circle at 0% 0%, rgba(197,126,90,0.16), transparent 36%),
                linear-gradient(180deg,
                    color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                    color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
                );
            box-shadow:0 10px 26px rgba(0,0,0,0.07);
        }}
        .render-hero-head {{
            display:flex;
            justify-content:space-between;
            align-items:flex-end;
            gap:16px;
            margin-bottom:13px;
        }}
        .render-hero-title {{
            font-size:clamp(24px, 2vw, 34px);
            line-height:1.02;
            font-weight:950;
            letter-spacing:-.04em;
            color:var(--text-color);
        }}
        .render-hero-subtitle {{
            margin-top:6px;
            font-size:13px;
            line-height:1.28;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 63%, transparent);
        }}
        .render-hero-pills {{
            display:flex;
            flex-wrap:wrap;
            gap:9px;
        }}
        .render-hero-pill {{
            display:flex;
            align-items:center;
            gap:8px;
            min-height:38px;
            padding:8px 12px;
            border-radius:999px;
            border:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            background:color-mix(in srgb, var(--secondary-background-color) 78%, var(--background-color));
            box-shadow:0 5px 14px rgba(0,0,0,0.045);
        }}
        .render-hero-pill span {{
            font-size:10.5px;
            font-weight:950;
            letter-spacing:.055em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 58%, transparent);
        }}
        .render-hero-pill strong {{
            font-size:13px;
            font-weight:950;
            color:var(--text-color);
            max-width:260px;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
        }}
        .render-hero-pill.ok {{
            border-color:rgba(34,197,94,.34);
            background:color-mix(in srgb, #22c55e 14%, var(--secondary-background-color));
        }}
        .render-hero-pill.warn {{
            border-color:rgba(245,158,11,.40);
            background:color-mix(in srgb, #f59e0b 15%, var(--secondary-background-color));
        }}
        .render-hero-pill.bad {{
            border-color:rgba(239,68,68,.40);
            background:color-mix(in srgb, #ef4444 15%, var(--secondary-background-color));
        }}
        @media(max-width:900px) {{
            .render-hero-head {{ align-items:flex-start; flex-direction:column; }}
            .render-hero-pill strong {{ max-width:180px; }}
        }}
        </style>
        <div class="render-hero">
            <div class="render-hero-head">
                <div>
                    <div class="render-hero-title">{html.escape(title)}</div>
                    <div class="render-hero-subtitle">{html.escape(subtitle)}</div>
                </div>
            </div>
            <div class="render-hero-pills">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_panel(language, status_items, visual_metrics, coil_footprint_mm, pallet_size_mm=750.0):
    # Use the packaging status as the primary operational decision.
    packaging_item = next((item for item in status_items if str(item.get("label", "")).lower() in {"packaging"}), status_items[0])
    winding_item = status_items[0] if status_items else {}
    tone = packaging_item.get("tone", "neutral")
    value = packaging_item.get("value", "-")
    note = packaging_item.get("note", "")

    margin = pallet_size_mm - coil_footprint_mm
    if language == "IT":
        title = "Esito simulazione"
        decision_label = "Decisione rapida"
        metrics = [
            ("Ingombro XY", f"{coil_footprint_mm:.1f} mm"),
            ("Margine pallet", f"{margin:.1f} mm" if margin >= 0 else f"+{abs(margin):.1f} mm"),
            ("Lunghezza", f"{visual_metrics['wound_length_m']:.2f} m"),
            ("Avvolgimento", winding_item.get("value", "-")),
        ]
    else:
        title = "Simulation result"
        decision_label = "Quick decision"
        metrics = [
            ("XY footprint", f"{coil_footprint_mm:.1f} mm"),
            ("Pallet margin", f"{margin:.1f} mm" if margin >= 0 else f"+{abs(margin):.1f} mm"),
            ("Length", f"{visual_metrics['wound_length_m']:.2f} m"),
            ("Winding", winding_item.get("value", "-")),
        ]

    metrics_html = "".join(
        f"""
        <div class="decision-mini">
            <span>{html.escape(str(label))}</span>
            <strong>{html.escape(str(metric_value))}</strong>
        </div>
        """
        for label, metric_value in metrics
    )

    st.markdown(
        f"""
        <style>
        .decision-panel {{
            height:100%;
            min-height:660px;
            border-radius:24px;
            padding:18px;
            border:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 10px 26px rgba(0,0,0,0.07);
            display:flex;
            flex-direction:column;
            gap:14px;
        }}
        .decision-kicker {{
            font-size:11px;
            line-height:1;
            font-weight:950;
            letter-spacing:.075em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 60%, transparent);
        }}
        .decision-main {{
            border-radius:22px;
            padding:18px;
            border:1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            background:color-mix(in srgb, var(--text-color) 4%, transparent);
        }}
        .decision-main.ok {{
            border-color:rgba(34,197,94,.38);
            background:color-mix(in srgb, #22c55e 13%, var(--secondary-background-color));
        }}
        .decision-main.warn {{
            border-color:rgba(245,158,11,.42);
            background:color-mix(in srgb, #f59e0b 15%, var(--secondary-background-color));
        }}
        .decision-main.bad {{
            border-color:rgba(239,68,68,.42);
            background:color-mix(in srgb, #ef4444 15%, var(--secondary-background-color));
        }}
        .decision-dot {{
            width:18px;
            height:18px;
            border-radius:999px;
            background:#94a3b8;
            margin-bottom:14px;
            box-shadow:0 0 0 5px rgba(148,163,184,.14);
        }}
        .decision-main.ok .decision-dot {{ background:#22c55e; box-shadow:0 0 0 5px rgba(34,197,94,.18); }}
        .decision-main.warn .decision-dot {{ background:#f59e0b; box-shadow:0 0 0 5px rgba(245,158,11,.18); }}
        .decision-main.bad .decision-dot {{ background:#ef4444; box-shadow:0 0 0 5px rgba(239,68,68,.18); }}
        .decision-label {{
            font-size:11px;
            font-weight:950;
            letter-spacing:.07em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 58%, transparent);
            margin-bottom:8px;
        }}
        .decision-value {{
            font-size:clamp(28px, 2.4vw, 40px);
            line-height:.98;
            font-weight:950;
            letter-spacing:-.055em;
            color:var(--text-color);
        }}
        .decision-note {{
            margin-top:10px;
            font-size:13px;
            line-height:1.3;
            font-weight:700;
            color:color-mix(in srgb, var(--text-color) 66%, transparent);
        }}
        .decision-grid {{
            display:grid;
            grid-template-columns:1fr;
            gap:10px;
        }}
        .decision-mini {{
            min-height:70px;
            border-radius:16px;
            padding:12px 13px;
            border:1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            background:color-mix(in srgb, var(--text-color) 4%, transparent);
        }}
        .decision-mini span {{
            display:block;
            font-size:10.5px;
            font-weight:950;
            letter-spacing:.055em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 56%, transparent);
            margin-bottom:7px;
        }}
        .decision-mini strong {{
            display:block;
            font-size:21px;
            line-height:1.05;
            font-weight:950;
            letter-spacing:-.025em;
            color:var(--text-color);
            overflow-wrap:anywhere;
        }}
        @media(max-width:1100px) {{
            .decision-panel {{ min-height:auto; }}
            .decision-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
        }}
        </style>
        <div class="decision-panel">
            <div class="decision-kicker">{html.escape(title)}</div>
            <div class="decision-main {html.escape(tone)}">
                <div class="decision-dot"></div>
                <div class="decision-label">{html.escape(decision_label)}</div>
                <div class="decision-value">{html.escape(str(value))}</div>
                <div class="decision-note">{html.escape(str(note))}</div>
            </div>
            <div class="decision-grid">{metrics_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_quick_reading(language, tube_layout_code, tube_diameter_label, passo_visuale, incremento_visuale, visual_metrics, coil_footprint_mm, pallet_size_mm=750.0):
    if language == "IT":
        title = "Lettura rapida"
        tube_title = "Tubo"
        tube_note = "Doppio verticale" if tube_layout_code == "double" else "Singolo"
        wind_title = "Avvolgimento"
        wind_note = f"Passo {passo_visuale:.1f} mm · incremento {incremento_visuale:.1f} mm"
        size_title = "Ingombro XY"
        if coil_footprint_mm <= pallet_size_mm:
            ok_note = "OK su pallet 750 × 750"
            size_tone = "ok"
        elif coil_footprint_mm <= pallet_size_mm + 20.0:
            ok_note = f"Attenzione · +{coil_footprint_mm - pallet_size_mm:.1f} mm"
            size_tone = "warn"
        else:
            ok_note = "Fuori margine pallet"
            size_tone = "bad"
    else:
        title = "Quick reading"
        tube_title = "Tube"
        tube_note = "Vertical double" if tube_layout_code == "double" else "Single"
        wind_title = "Winding"
        wind_note = f"Pitch {passo_visuale:.1f} mm · layer {incremento_visuale:.1f} mm"
        size_title = "XY footprint"
        if coil_footprint_mm <= pallet_size_mm:
            ok_note = "OK on 750 × 750 pallet"
            size_tone = "ok"
        elif coil_footprint_mm <= pallet_size_mm + 20.0:
            ok_note = f"Attention · +{coil_footprint_mm - pallet_size_mm:.1f} mm"
            size_tone = "warn"
        else:
            ok_note = "Over pallet margin"
            size_tone = "bad"

    cards = [
        (tube_title, tube_diameter_label, tube_note, "neutral"),
        (wind_title, f"{visual_metrics['wound_length_m']:.2f}", "m · " + wind_note, "neutral"),
    ]

    cards_html = ""
    for label, value, note, tone in cards:
        cards_html += f"""
        <div class="quick-card-v2 {html.escape(tone)}">
            <div class="quick-topline"></div>
            <div class="quick-label-v2">{html.escape(str(label))}</div>
            <div class="quick-value-v2">{html.escape(str(value))}</div>
            <div class="quick-note-v2">{html.escape(str(note))}</div>
        </div>
        """

    st.markdown(
        f"""
        <style>
        .quick-title-v2 {{
            margin:0 0 10px 0;
            font-size:15px;
            font-weight:950;
            letter-spacing:0.06em;
            text-transform:uppercase;
            color:var(--text-color);
        }}
        .quick-grid-v2 {{
            display:grid;
            grid-template-columns:repeat(2, minmax(0, 1fr));
            gap:12px;
            margin:8px 0 16px 0;
        }}
        .quick-card-v2 {{
            position:relative;
            overflow:hidden;
            border-radius:20px;
            padding:18px 20px 16px 20px;
            min-height:126px;
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            border:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            box-shadow:0 7px 18px rgba(0,0,0,0.055);
        }}
        .quick-topline {{
            position:absolute;
            left:0;
            top:0;
            height:5px;
            width:100%;
            background:color-mix(in srgb, var(--text-color) 18%, transparent);
        }}
        .quick-card-v2.ok .quick-topline {{ background:#22c55e; }}
        .quick-card-v2.warn .quick-topline {{ background:#f59e0b; }}
        .quick-card-v2.bad .quick-topline {{ background:#fb7185; }}
        .quick-label-v2 {{
            font-size:12px;
            line-height:1.1;
            font-weight:900;
            letter-spacing:0.075em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 58%, transparent);
            margin-bottom:10px;
        }}
        .quick-value-v2 {{
            font-size:34px;
            line-height:0.95;
            font-weight:950;
            letter-spacing:-0.055em;
            color:var(--text-color);
            margin-bottom:10px;
        }}
        .quick-note-v2 {{
            font-size:12px;
            line-height:1.25;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 64%, transparent);
        }}
        @media (max-width: 900px) {{
            .quick-grid-v2 {{
                grid-template-columns:1fr;
            }}
        }}
        </style>
        <div class="quick-title-v2">{html.escape(title)}</div>
        <div class="quick-grid-v2">{cards_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_tech_snapshot_cards(selected_row, language):
    def gv(*names, default="-"):
        for name in names:
            if name in selected_row.index:
                value = safe_value(selected_row, name)
                if value != "-":
                    return value
        return default

    if language == "IT":
        items = [
            ("Tipo tubo", gv("Tipo tubo")),
            ("Lunghezza", f"{gv('Lunghezza (m)')} m" if gv("Lunghezza (m)") != "-" else "-"),
            ("Aspo", f"{gv('Diametro aspo (mm)')} mm" if gv("Diametro aspo (mm)") != "-" else "-"),
            ("Spalla", f"{gv('Spalla (mm)')} mm" if gv("Spalla (mm)") != "-" else "-"),
        ]
    else:
        items = [
            ("Tube type", gv("Tipo tubo")),
            ("Length", f"{gv('Lunghezza (m)')} m" if gv("Lunghezza (m)") != "-" else "-"),
            ("Spool", f"{gv('Diametro aspo (mm)')} mm" if gv("Diametro aspo (mm)") != "-" else "-"),
            ("Width", f"{gv('Spalla (mm)')} mm" if gv("Spalla (mm)") != "-" else "-"),
        ]

    cards_html = ""
    for label, value in items:
        cards_html += f"""
        <div class="tech-mini-card">
            <div class="tech-mini-label">{html.escape(str(label))}</div>
            <div class="tech-mini-value">{html.escape(str(value))}</div>
        </div>
        """

    st.markdown(
        f"""
        <style>
        .tech-mini-grid {{
            display:grid;
            grid-template-columns:repeat(4, minmax(0, 1fr));
            gap:12px;
            margin:0 0 16px 0;
        }}
        .tech-mini-card {{
            border-radius:16px;
            padding:16px 18px;
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            box-shadow:0 6px 16px rgba(0,0,0,0.055);
            border-left:5px solid #C57E5A;
        }}
        .tech-mini-label {{
            font-size:12px;
            text-transform:uppercase;
            letter-spacing:0.06em;
            font-weight:850;
            color:color-mix(in srgb, var(--text-color) 62%, transparent);
            margin-bottom:9px;
        }}
        .tech-mini-value {{
            font-size:22px;
            line-height:1.08;
            font-weight:900;
            word-break:break-word;
        }}
        @media (max-width:900px) {{
            .tech-mini-grid {{
                grid-template-columns:repeat(2, minmax(0, 1fr));
            }}
        }}
        </style>
        <div class="tech-mini-grid">{cards_html}</div>
        """,
        unsafe_allow_html=True,
    )












def build_operator_row_from_current_values(selected_row):
    """Return a preset row where the render-linked fields mirror the current calculator values."""
    row = selected_row.copy()

    def set_if_present(column, value):
        if column in row.index:
            row[column] = value

    tube_layout = str(st.session_state.get("calc_tube_layout", "Singolo"))
    set_if_present("Tipo tubo", tube_layout)
    set_if_present("Lunghezza (m)", st.session_state.get("calc_lunghezza", "-"))
    set_if_present("Diametro aspo (mm)", st.session_state.get("calc_diametro_aspo", "-"))
    set_if_present("Spalla (mm)", st.session_state.get("calc_spalla", "-"))
    set_if_present("Passo (mm)", st.session_state.get("calc_passo_visuale", "-"))
    set_if_present("Incremento strato (mm)", st.session_state.get("calc_incremento_visuale", "-"))
    set_if_present("Ritardo invers min (º)", st.session_state.get("calc_rit_b", "-"))
    set_if_present("Ritardo invers max (º)", st.session_state.get("calc_rit_t", "-"))

    if tube_layout == "Doppio":
        set_if_present("Diametro rame inferiore", st.session_state.get("calc_rame_inf", "-"))
        set_if_present("Spessore guaina inferiore", st.session_state.get("calc_spessore_inf", "-"))
        set_if_present("Diametro rame superiore", st.session_state.get("calc_rame_sup", "-"))
        set_if_present("Spessore guaina superiore", st.session_state.get("calc_spessore_sup", "-"))
    else:
        set_if_present("Diametro Rame", st.session_state.get("calc_rame", "-"))
        set_if_present("Spessore Guaina (mm)", st.session_state.get("calc_spessore", "-"))
        rame = str(st.session_state.get("calc_rame", "1/4"))
        spessore = parse_float_value(st.session_state.get("calc_spessore", 0), 0)
        d_tubo = COPPER_SIZES_MM.get(rame, parse_float_value(rame, 0.0)) + 2.0 * spessore
        set_if_present("Diametro esterno Guaina (mm)", d_tubo)

    return row


def render_machine_parameter_groups(selected_row, language, key_suffix=""):
    """Operator-facing parameter board: all values visible, grouped by real machine use."""
    group_defs = [
        (
            "Prodotto" if language == "IT" else "Product",
            "Identificazione rapida del prodotto da produrre." if language == "IT" else "Quick identification of the product to manufacture.",
            "primary",
            ["Prodotto", "Tipo tubo", "Diametro Rame", "Spessore Guaina (mm)", "Diametro esterno Guaina (mm)",
             "Diametro rame inferiore", "Spessore guaina inferiore", "Diametro rame superiore", "Spessore guaina superiore",
             "Lunghezza (m)", "Velocita linea (m/min)"],
        ),
        (
            "Macchina / Preparazione" if language == "IT" else "Machine / Preparation",
            "Attrezzaggio fisico della linea prima della produzione." if language == "IT" else "Physical line setup before production.",
            "compact",
            ["Boccole rulliera adrizzatubo", "Boccola uscita rulliera", "Rulliera adrizzatubo",
             "Boccola uscita traino", "Rulli convogliatore (mm)", "Rulli estrusore(mm)",
             "Ruote godronatore", "Soffiatori aria (mm)", "Rulli avvolgitore (mm)",
             "Paleta ferma coda (mm)", "Guidatubo (mm)"],
        ),
        (
            "Avvolgitore" if language == "IT" else "Coiler",
            "Parametri che guidano direttamente forma, spire e cambio strato." if language == "IT" else "Parameters directly driving coil shape, turns and layer change.",
            "primary",
            ["Spalla (mm)", "Diametro aspo (mm)", "Nº Spire", "Interasse regetta (mm)",
             "Velocita recupero (m/min)", "Quota start pinza (mm)", "Quota coda tubo (mm)",
             "Quota chiusura morsa coda (mm)", "Quota minima (mm)", "Quota massima (mm)",
             "Passo (mm)", "Incremento strato (mm)", "Ritardo invers min (º)", "Ritardo invers max (º)"],
        ),
        (
            "Coppie" if language == "IT" else "Torques",
            "Valori di coppia separati dalla geometria per lettura più rapida." if language == "IT" else "Torque values separated from geometry for faster reading.",
            "compact",
            ["Coppia lavoro (%)", "Riduzione coppia (%)", "Coppia recupero (%)"],
        ),
    ]

    search_label = "Cerca parametro" if language == "IT" else "Search parameter"
    search_placeholder = "Es. passo, quota, soffiatori, boccola..." if language == "IT" else "E.g. pitch, position, blowers, bushing..."
    query = st.text_input(
        search_label,
        value="",
        placeholder=search_placeholder,
        key=f"machine_param_search{key_suffix}",
    ).strip().lower()

    def visible_value(col):
        if col not in selected_row.index:
            return None
        raw = selected_row[col]
        if pd.isna(raw):
            return None
        formatted = str(format_preset_value(raw)).strip()
        if formatted in {"", "-"}:
            return None
        return formatted

    def value_with_unit(col, value):
        label_lower = str(col).lower()
        text = str(value)
        if text == "-":
            return text
        if "(mm)" in label_lower and "mm" not in text.lower():
            return f"{text} mm"
        if "(m/min)" in label_lower and "m/min" not in text.lower():
            return f"{text} m/min"
        if "(%)" in label_lower and "%" not in text:
            return f"{text} %"
        if "(º)" in label_lower or "(°)" in label_lower:
            return f"{text}°" if "°" not in text and "º" not in text else text
        if "lunghezza (m)" == label_lower and "m" not in text.lower():
            return f"{text} m"
        return text

    def matches_query(label, value):
        if not query:
            return True
        haystack = f"{label} {value}".lower()
        return query in haystack

    groups = []
    used = set()

    for group_title, group_subtitle, density, cols in group_defs:
        pairs = []
        for col in cols:
            value = visible_value(col)
            if value is None:
                continue
            label = param_label(col, language)
            formatted_value = value_with_unit(col, value)
            used.add(col)
            if matches_query(label, formatted_value):
                pairs.append((label, formatted_value))
        if pairs:
            groups.append((group_title, group_subtitle, density, pairs))

    extra_pairs = []
    for col in selected_row.index:
        if col in used or str(col).startswith("Unnamed") or col in {"Note"}:
            continue
        value = visible_value(col)
        if value is None:
            continue
        label = param_label(col, language)
        formatted_value = value_with_unit(col, value)
        if matches_query(label, formatted_value):
            extra_pairs.append((label, formatted_value))

    if extra_pairs:
        groups.append((
            "Altri parametri" if language == "IT" else "Other parameters",
            "Valori presenti nel preset non classificati nei gruppi principali." if language == "IT" else "Values present in the preset not classified in the main groups.",
            "compact",
            extra_pairs,
        ))

    if not groups:
        st.info("Nessun parametro trovato." if language == "IT" else "No parameter found.")
        return

    note = (
        "Tutti i parametri restano visibili per l'operatore; la ricerca serve solo a filtrare temporaneamente la vista."
        if language == "IT"
        else "All parameters remain visible for the operator; search only filters the view temporarily."
    )

    groups_html = []
    total_cards = 0
    for group_title, group_subtitle, density, pairs in groups:
        total_cards += len(pairs)
        cards_html = "".join(
            f"""
            <div class="machine-card-native {html.escape(str(density))}">
                <div class="machine-card-label-native">{html.escape(str(label))}</div>
                <div class="machine-card-value-native">{html.escape(str(value))}</div>
            </div>
            """
            for label, value in pairs
        )
        groups_html.append(
            f"""
            <section class="machine-section-native">
                <div class="machine-group-head-native">
                    <div>
                        <div class="machine-group-title-native">{html.escape(str(group_title))}</div>
                        <div class="machine-group-subtitle-native">{html.escape(str(group_subtitle))}</div>
                    </div>
                    <div class="machine-group-count-native">{len(pairs)}</div>
                </div>
                <div class="machine-grid-native">{cards_html}</div>
            </section>
            """
        )

    # iPad-friendly height: avoid internal iframe scrolling in "Scheda tecnica > Parametri macchina".
    # The HTML grid becomes 2 columns on iPad, so height is estimated using 2 columns.
    # This makes the Streamlit page itself scroll instead of the component.
    height_columns = 2
    estimated_height = 86  # operator note + breathing room
    for _, _, _, group_pairs in groups:
        row_count = max(1, (len(group_pairs) + height_columns - 1) // height_columns)
        estimated_height += 18 + 78 + (row_count * 124) + max(0, row_count - 1) * 12 + 18
    board_height = max(680, min(7200, estimated_height + 40))


    board_html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    :root {{
        color-scheme: light;
        --pdm-accent:#C57E5A;
        --bg: transparent;
        --text:#0f172a;
        --muted:#475569;
        --line:rgba(15,23,42,0.11);
        --card-bg:transparent;
        --header-bg:linear-gradient(90deg, rgba(197,126,90,0.10), transparent 72%);
        --hover-bg:rgba(197,126,90,0.045);
        --shadow:0 7px 18px rgba(15,23,42,0.045);
    }}
    body.dark-theme {{
        color-scheme: dark;
        --text:#f8fafc;
        --muted:rgba(248,250,252,0.68);
        --line:rgba(248,250,252,0.12);
        --card-bg:transparent;
        --header-bg:linear-gradient(90deg, rgba(197,126,90,0.18), transparent 70%);
        --hover-bg:rgba(197,126,90,0.055);
        --shadow:0 7px 18px rgba(0,0,0,0.14);
    }}
    *, *::before, *::after {{
        box-sizing:border-box;
    }}
    html, body {{
        margin:0;
        padding:0 2px 18px 0;
        background:var(--bg);
        color:var(--text);
        font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        overflow-x:hidden;
        overflow-y:hidden;
    }}
    .operator-board-note {{
        margin:4px 0 12px 0;
        padding:12px 14px;
        border-radius:16px;
        border:1px solid var(--line);
        background:var(--card-bg);
        font-size:12.5px;
        line-height:1.32;
        font-weight:700;
        color:var(--muted);
    }}
    .machine-section-native {{
        margin:18px 0 0 0;
    }}
    .machine-group-head-native {{
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:14px;
        padding:15px 18px;
        border-radius:18px 18px 0 0;
        background:var(--header-bg);
        border:1px solid var(--line);
        border-bottom:none;
        box-sizing:border-box;
    }}
    .machine-group-title-native {{
        font-size:15px;
        font-weight:950;
        letter-spacing:0.065em;
        text-transform:uppercase;
        color:var(--text);
    }}
    .machine-group-subtitle-native {{
        margin-top:4px;
        font-size:12px;
        line-height:1.25;
        font-weight:700;
        color:var(--muted);
        letter-spacing:0;
        text-transform:none;
    }}
    .machine-group-count-native {{
        min-width:30px;
        height:30px;
        padding:0 10px;
        border-radius:999px;
        background:var(--pdm-accent);
        color:#fff;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:13px;
        font-weight:950;
        box-shadow:0 6px 14px rgba(197,126,90,0.22);
        box-sizing:border-box;
    }}
    .machine-grid-native {{
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:12px;
        width:100%;
        border:0;
        border-radius:0 0 18px 18px;
        overflow:visible;
        background:var(--card-bg);
        box-shadow:none;
        box-sizing:border-box;
    }}
    .machine-card-native {{
        min-height:112px;
        padding:18px 20px;
        border:1px solid var(--line);
        border-radius:18px;
        background:transparent;
        display:flex;
        flex-direction:column;
        justify-content:center;
        gap:14px;
        box-sizing:border-box;
        min-width:0;
        position:relative;
        overflow:hidden;
        isolation:isolate;
    }}
    .machine-card-native:nth-child(4n) {{
        border-right:0;
    }}
    .machine-card-label-native {{
        font-size:11px;
        line-height:1.12;
        font-weight:950;
        text-transform:uppercase;
        letter-spacing:0.07em;
        color:var(--muted);
        word-break:break-word;
        min-height:24px;
        display:flex;
        align-items:flex-start;
        position:relative;
        z-index:2;
    }}
    .machine-card-value-native {{
        font-size:clamp(20px, 1.32vw, 26px);
        line-height:1.05;
        font-weight:950;
        color:var(--text);
        letter-spacing:-0.025em;
        word-break:break-word;
        min-height:32px;
        display:flex;
        align-items:flex-end;
        position:relative;
        z-index:2;
    }}
    .machine-card-native::after {{
        content:"";
        position:absolute;
        top:-42%;
        bottom:-42%;
        left:-78%;
        width:48%;
        pointer-events:none;
        border-radius:inherit;
        background:linear-gradient(105deg, transparent 0%, rgba(255,255,255,0.00) 25%, rgba(255,255,255,0.32) 48%, rgba(197,126,90,0.26) 56%, rgba(255,255,255,0.14) 64%, transparent 100%);
        transform:skewX(-16deg);
        opacity:0;
        z-index:4;
        mix-blend-mode:screen;
    }}
    .machine-card-native:hover {{
        background:var(--hover-bg);
    }}
    .machine-card-native:hover::after {{
        opacity:1;
        animation:pdmCardSweepAfter 1.05s cubic-bezier(.2,.72,.22,1) both;
    }}
    @keyframes pdmCardSweepAfter {{
        0% {{ left:-82%; opacity:0; }}
        12% {{ opacity:1; }}
        100% {{ left:138%; opacity:0; }}
    }}
    @media (max-width:1180px) {{
        .machine-grid-native {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }}
        .machine-card-native:nth-child(4n) {{ border-right:1px solid var(--line); }}
        .machine-card-native:nth-child(2n) {{ border-right:0; }}
    }}
    @media (max-width:720px) {{
        /* Manté 2 columnes també en mòbil per evitar tallar el contingut:
           l'altura del component està calculada sense scroll intern. */
        .machine-grid-native {{ grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; }}
        .machine-card-native,
        .machine-card-native:nth-child(2n),
        .machine-card-native:nth-child(4n) {{ border-right:1px solid var(--line); }}
        .machine-card-native {{ min-height:96px; padding:12px 11px; border-radius:14px; gap:9px; }}
        .machine-card-label-native {{ font-size:9px; min-height:20px; letter-spacing:0.045em; }}
        .machine-card-value-native {{ font-size:clamp(15px, 4.6vw, 20px); min-height:24px; }}
        .machine-group-head-native {{ padding:12px 13px; border-radius:15px 15px 0 0; }}
        .machine-group-title-native {{ font-size:12px; }}
        .machine-group-subtitle-native {{ font-size:10.5px; }}
    }}
    </style>
    </head>
    <body>
        <script>
        (function () {{
            function parseRgb(value) {{
                if (!value) return null;
                value = String(value).trim();
                if (value.startsWith("#")) {{
                    let hex = value.slice(1);
                    if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
                    const n = parseInt(hex, 16);
                    if (Number.isNaN(n)) return null;
                    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
                }}
                const match = value.match(/rgba?\(([^)]+)\)/i);
                if (!match) return null;
                const parts = match[1].split(",").map(x => parseFloat(x.trim()));
                if (parts.length < 3) return null;
                return [parts[0], parts[1], parts[2]];
            }}

            function luminance(rgb) {{
                const mapped = rgb.map(v => {{
                    v = Math.max(0, Math.min(255, v)) / 255;
                    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
                }});
                return 0.2126 * mapped[0] + 0.7152 * mapped[1] + 0.0722 * mapped[2];
            }}

            function rgbaFrom(rgb, alpha) {{
                if (!rgb) return "";
                return "rgba(" + Math.round(rgb[0]) + ", " + Math.round(rgb[1]) + ", " + Math.round(rgb[2]) + ", " + alpha + ")";
            }}

            function getThemeInfo() {{
                try {{
                    const parentDoc = window.parent && window.parent.document;
                    if (parentDoc) {{
                        const root = parentDoc.documentElement;
                        const body = parentDoc.body;
                        const appContainer =
                            parentDoc.querySelector('[data-testid="stAppViewContainer"]') ||
                            parentDoc.querySelector('.stApp') ||
                            body;

                        const rootStyle = window.parent.getComputedStyle(root);
                        const bodyStyle = window.parent.getComputedStyle(body);
                        const appStyle = window.parent.getComputedStyle(appContainer);

                        const textCandidates = [
                            rootStyle.getPropertyValue("--text-color"),
                            appStyle.color,
                            bodyStyle.color
                        ];
                        const bgCandidates = [
                            rootStyle.getPropertyValue("--background-color"),
                            rootStyle.getPropertyValue("--secondary-background-color"),
                            appStyle.backgroundColor,
                            bodyStyle.backgroundColor
                        ];

                        let textRgb = null;
                        let bgRgb = null;
                        let textValue = "";
                        let bgValue = "";

                        for (const candidate of textCandidates) {{
                            const rgb = parseRgb(candidate);
                            if (rgb) {{
                                textRgb = rgb;
                                textValue = String(candidate).trim();
                                break;
                            }}
                        }}

                        for (const candidate of bgCandidates) {{
                            const rgb = parseRgb(candidate);
                            if (rgb) {{
                                bgRgb = rgb;
                                bgValue = String(candidate).trim();
                                break;
                            }}
                        }}

                        let isDark = false;
                        if (textRgb) {{
                            // Light text implies dark theme.
                            isDark = luminance(textRgb) > 0.6;
                        }} else if (bgRgb) {{
                            isDark = luminance(bgRgb) < 0.42;
                        }} else {{
                            isDark = !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
                        }}

                        return {{
                            isDark,
                            textRgb,
                            bgRgb,
                            textValue,
                            bgValue
                        }};
                    }}
                }} catch (err) {{}}
                return {{
                    isDark: !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches),
                    textRgb: null,
                    bgRgb: null,
                    textValue: "",
                    bgValue: ""
                }};
            }}

            const theme = getThemeInfo();
            document.body.classList.toggle("dark-theme", theme.isDark);
            document.body.classList.toggle("light-theme", !theme.isDark);

            const fallbackText = theme.isDark ? "#f8fafc" : "#0f172a";
            const fallbackMuted = theme.isDark ? "rgba(248,250,252,0.70)" : "rgba(15,23,42,0.66)";
            const fallbackLine = theme.isDark ? "rgba(248,250,252,0.12)" : "rgba(15,23,42,0.11)";
            const fallbackHeader = theme.isDark
                ? "linear-gradient(90deg, rgba(197,126,90,0.18), transparent 70%)"
                : "linear-gradient(90deg, rgba(197,126,90,0.10), transparent 72%)";
            const fallbackHover = theme.isDark
                ? "rgba(197,126,90,0.055)"
                : "rgba(197,126,90,0.045)";

            const resolvedText = theme.textValue || fallbackText;
            const resolvedMuted = theme.textRgb ? rgbaFrom(theme.textRgb, theme.isDark ? 0.70 : 0.66) : fallbackMuted;
            const resolvedLine = theme.textRgb ? rgbaFrom(theme.textRgb, theme.isDark ? 0.12 : 0.11) : fallbackLine;

            document.body.style.setProperty("--text", resolvedText);
            document.body.style.setProperty("--muted", resolvedMuted);
            document.body.style.setProperty("--line", resolvedLine);
            document.body.style.setProperty("--header-bg", fallbackHeader);
            document.body.style.setProperty("--hover-bg", fallbackHover);
        }})();
        </script>
        <div class="operator-board-note">{html.escape(note)}</div>
        {''.join(groups_html)}
    </body>
    </html>
    """

    components.html(board_html, height=board_height, scrolling=False)

def render_preset_summary_strip(product_name, selected_row, language, modified=False):
    def gv(*names, default="-"):
        for name in names:
            if name in selected_row.index:
                value = safe_value(selected_row, name)
                if value != "-":
                    return value
        return default

    tipo = gv("Tipo tubo")
    lunghezza = gv("Lunghezza (m)")
    aspo = gv("Diametro aspo (mm)")
    spalla = gv("Spalla (mm)")

    if str(tipo).strip().lower() == "doppio":
        rame_inf = gv("Diametro rame inferiore", "Diametro Rame inferiore")
        rame_sup = gv("Diametro rame superiore", "Diametro Rame superiore")
        sp_inf = parse_float_value(gv("Spessore guaina inferiore", "Spessore Guaina inferiore (mm)", default=0), 0)
        sp_sup = parse_float_value(gv("Spessore guaina superiore", "Spessore Guaina superiore (mm)", default=0), 0)
        d_inf = COPPER_SIZES_MM.get(str(rame_inf), parse_float_value(rame_inf, 0.0)) + 2.0 * sp_inf
        d_sup = COPPER_SIZES_MM.get(str(rame_sup), parse_float_value(rame_sup, 0.0)) + 2.0 * sp_sup
        tubo_txt = f"{rame_sup}/{rame_inf} · {d_sup:.1f}/{d_inf:.1f} mm"
    else:
        rame = gv("Diametro Rame")
        esterno = gv("Diametro esterno Guaina (mm)")
        tubo_txt = f"{rame} · Ø {esterno} mm" if esterno != "-" else str(rame)

    if language == "IT":
        title = "Parametri attuali" if modified else "Preset attivo"
        fields = [
            ("Prodotto", product_name),
            ("Tubo", f"{tipo} · {tubo_txt}"),
            ("Lunghezza", f"{lunghezza} m" if lunghezza != "-" else "-"),
            ("Aspo", f"Ø {aspo} mm" if aspo != "-" else "-"),
            ("Spalla", f"{spalla} mm" if spalla != "-" else "-"),
        ]
    else:
        title = "Current parameters" if modified else "Active preset"
        fields = [
            ("Product", product_name),
            ("Tube", f"{tipo} · {tubo_txt}"),
            ("Length", f"{lunghezza} m" if lunghezza != "-" else "-"),
            ("Spool", f"Ø {aspo} mm" if aspo != "-" else "-"),
            ("Width", f"{spalla} mm" if spalla != "-" else "-"),
        ]

    items_html = "".join(
        f"""
        <div class="summary-strip-item">
            <div class="summary-strip-label">{html.escape(str(label))}</div>
            <div class="summary-strip-value">{html.escape(str(value))}</div>
        </div>
        """
        for label, value in fields
    )

    st.markdown(
        f"""
        <style>
        .summary-strip {{
            margin:8px 0 16px 0;
            border-radius:20px;
            overflow:hidden;
            border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 8px 20px rgba(0,0,0,0.065);
        }}
        .summary-strip-head {{
            padding:12px 16px;
            background:linear-gradient(90deg, rgba(197,126,90,0.20), transparent);
            border-bottom:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            font-size:13px;
            font-weight:950;
            letter-spacing:0.07em;
            text-transform:uppercase;
        }}
        .summary-strip-grid {{
            display:grid;
            grid-template-columns:repeat(5, minmax(0, 1fr));
            gap:16px;
            padding:16px 18px;
            align-items:start;
        }}
        .summary-strip-item {{
            min-width:0;
            display:flex;
            flex-direction:column;
            justify-content:flex-start;
            gap:6px;
            min-height:60px;
        }}
        .summary-strip-label {{
            font-size:11px;
            line-height:1.1;
            text-transform:uppercase;
            letter-spacing:0.06em;
            font-weight:850;
            color:color-mix(in srgb, var(--text-color) 58%, transparent);
            margin-bottom:6px;
        }}
        .summary-strip-value {{
            font-size:17px;
            line-height:1.14;
            font-weight:950;
            color:var(--text-color);
            word-break:break-word;
        }}
        @media (max-width:950px) {{
            .summary-strip-grid {{
                grid-template-columns:1fr 1fr;
            }}
        }}
        </style>
        <div class="summary-strip">
            <div class="summary-strip-head">{html.escape(title)}</div>
            <div class="summary-strip-grid">{items_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_semaphore(items, language):
    title = "Stato rapido" if language == "IT" else "Quick status"

    cards_html = ""
    for item in items:
        label = item.get("label", "")
        value = item.get("value", "")
        note = item.get("note", "")
        tone = item.get("tone", "neutral")
        cards_html += f"""
        <div class="semaphore-card pdm-status-animated {html.escape(tone)}">
            <div class="semaphore-dot"></div>
            <div>
                <div class="semaphore-label">{html.escape(str(label))}</div>
                <div class="semaphore-value">{html.escape(str(value))}</div>
                <div class="semaphore-note">{html.escape(str(note))}</div>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <style>
        .semaphore-title {{
            margin:4px 0 10px 0;
            font-size:15px;
            font-weight:950;
            letter-spacing:0.05em;
            text-transform:uppercase;
        }}
        .semaphore-grid {{
            display:grid;
            grid-template-columns:repeat(3, minmax(0, 1fr));
            gap:12px;
            margin:6px 0 16px 0;
        }}
        .semaphore-card {{
            display:grid;
            grid-template-columns:18px 1fr;
            gap:11px;
            align-items:start;
            min-height:88px;
            padding:15px 16px;
            border-radius:18px;
            border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 6px 16px rgba(0,0,0,0.055);
        }}
        .semaphore-dot {{
            width:15px;
            height:15px;
            border-radius:999px;
            margin-top:3px;
            background:#94a3b8;
            box-shadow:0 0 0 4px rgba(148,163,184,0.15);
        }}
        .semaphore-card.ok .semaphore-dot {{
            background:linear-gradient(90deg, #22c55e, #86efac);
            box-shadow:0 0 0 4px rgba(34,197,94,0.16);
        }}
        .semaphore-card.warn .semaphore-dot {{
            background:#f59e0b;
            box-shadow:0 0 0 4px rgba(245,158,11,0.16);
        }}
        .semaphore-card.bad .semaphore-dot {{
            background:#ef4444;
            box-shadow:0 0 0 4px rgba(239,68,68,0.16);
        }}
        .semaphore-label {{
            font-size:12px;
            text-transform:uppercase;
            letter-spacing:0.06em;
            font-weight:850;
            color:color-mix(in srgb, var(--text-color) 62%, transparent);
            margin-bottom:6px;
        }}
        .semaphore-value {{
            font-size:20px;
            line-height:1.08;
            font-weight:950;
        }}
        .semaphore-note {{
            margin-top:6px;
            font-size:12px;
            line-height:1.25;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 62%, transparent);
        }}
        @media (max-width:900px) {{
            .semaphore-grid {{
                grid-template-columns:1fr;
            }}
        }}
        </style>
        <div class="semaphore-title">{html.escape(title)}</div>
        <div class="semaphore-grid">{cards_html}</div>
        """,
        unsafe_allow_html=True,
    )



def render_startup_checklist(language):
    # Checklist cambio misura ordinata per area e tipo di intervento.
    # Stato salvato in session_state: resta spuntato durante la sessione.
    if language == "IT":
        title = "Checklist cambio misura"
        subtitle = "Sequenza operativa per cambio formato: linea + avvolgitore."
        reset_label = "Azzera checklist"
        progress_label = "Avanzamento"
        complete_label = "Cambio misura completato"
        pending_label = "Cambio misura in corso"
        groups = [
            (
                "Linea",
                [
                    (
                        "Comprovazioni",
                        [
                            "Comprovare rame corretto",
                            "Comprovare isolamento corretto",
                            "Comprovare materiale corretto di estrussore",
                            "Comprovare stampante e marcatura",
                        ],
                    ),
                    (
                        "Cambio attrezzatura",
                        [
                            "Cambiare boccole di entrate e uscite del rame",
                            "Cambiare adrizzatubi rame",
                            "Cambiare rulli convogliatore ed estrussore",
                            "Cambiare soffiatori",
                        ],
                    ),
                    (
                        "Regolazioni",
                        [
                            "Regolare rulli di guida di tutta la maquina",
                            "Regolare traino",
                            "Impostare lunghezza taglio e velocita",
                            "Regolare godronatore",
                        ],
                    ),
                ],
            ),
            (
                "Avvolgitore",
                [
                    (
                        "Cambio attrezzatura",
                        [
                            "Cambiare rulli avvolgitore",
                            "Cambiare paletta ferma coda avvolgitore",
                            "Cambiare guidatubo avvolgitore",
                        ],
                    ),
                    (
                        "Regolazioni",
                        [
                            "Regolare spalla avvolgitore",
                            "Regolare interasse regetta avvolgitore",
                            "Regolare diametro aspo avvolgitore",
                            "Impostare/caricare parametri avvolgitore",
                        ],
                    ),
                    (
                        "Comprovazioni",
                        [
                            "Simulare parametri di avvolgimento",
                        ],
                    ),
                ],
            ),
        ]
    else:
        title = "Size change checklist"
        subtitle = "Operating sequence for format change: line + coiler."
        reset_label = "Reset checklist"
        progress_label = "Progress"
        complete_label = "Size change completed"
        pending_label = "Size change in progress"
        groups = [
            (
                "Line",
                [
                    (
                        "Checks",
                        [
                            "Check correct copper",
                            "Check correct insulation",
                            "Check correct extruder material",
                            "Check printer and marking",
                        ],
                    ),
                    (
                        "Tooling change",
                        [
                            "Change copper inlet and outlet bushings",
                            "Change copper straightener",
                            "Change conveyor and extruder rollers",
                            "Change air blowers",
                        ],
                    ),
                    (
                        "Adjustments",
                        [
                            "Adjust all machine guide rollers",
                            "Adjust puller",
                            "Set cutting length and speed",
                            "Adjust knurling unit",
                        ],
                    ),
                ],
            ),
            (
                "Coiler",
                [
                    (
                        "Tooling change",
                        [
                            "Change coiler rollers",
                            "Change coiler tail stop paddle",
                            "Change coiler tube guide",
                        ],
                    ),
                    (
                        "Adjustments",
                        [
                            "Adjust coiler width",
                            "Adjust coiler strap distance",
                            "Adjust coiler spool diameter",
                            "Set/load coiler parameters",
                        ],
                    ),
                    (
                        "Checks",
                        [
                            "Simulate winding parameters",
                        ],
                    ),
                ],
            ),
        ]

    # Flatten checklist for progress and reset.
    flat_items = []
    for area_name, sections in groups:
        for section_name, items in sections:
            for item in items:
                key_base = f"{area_name}_{section_name}_{item}".lower()
                key = "check_cambio_" + "".join(ch if ch.isalnum() else "_" for ch in key_base)
                flat_items.append((key, item))

    if st.button(reset_label, use_container_width=False):
        for key, _ in flat_items:
            st.session_state[key] = False
        st.rerun()

    completed = sum(1 for key, _ in flat_items if st.session_state.get(key, False))
    total = len(flat_items)
    progress = completed / total if total else 0.0
    checklist_done = total > 0 and completed == total
    status_label = complete_label if checklist_done else pending_label
    status_class = "is-complete" if checklist_done else "is-pending"

    st.markdown(
        f"""
        <style>
        .stButton > button {{
            border-radius:999px !important;
            min-height:46px;
            padding:0.72rem 1.35rem !important;
            border:1px solid #C57E5A !important;
            background:#C57E5A !important;
            color:#ffffff !important;
            font-weight:850 !important;
            letter-spacing:0.01em;
            box-shadow:0 8px 18px rgba(197,126,90,0.22), 0 3px 8px rgba(0,0,0,0.10) !important;
            transition:transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease, background 0.16s ease !important;
        }}
        .stButton > button:hover {{
            transform:translateY(-1px);
            filter:brightness(1.05);
            box-shadow:0 10px 22px rgba(197,126,90,0.28), 0 5px 12px rgba(0,0,0,0.12) !important;
            border-color:#C57E5A !important;
            background:#C57E5A !important;
            color:#ffffff !important;
        }}
        .stButton > button:focus,
        .stButton > button:focus-visible {{
            outline:none !important;
            box-shadow:0 0 0 2px rgba(255,255,255,0.88), 0 0 0 5px rgba(197,126,90,0.34), 0 10px 22px rgba(197,126,90,0.24) !important;
            border-color:#C57E5A !important;
        }}
        .stButton > button:active {{
            transform:translateY(0);
            filter:brightness(0.98);
        }}

        .checklist-hero {{
            margin:8px 0 14px 0;
            border-radius:24px;
            overflow:hidden;
            border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 8px 20px rgba(0,0,0,0.065);
        }}
        .checklist-hero-head {{
            padding:15px 18px;
            background:linear-gradient(90deg, rgba(197,126,90,0.22), transparent);
            border-bottom:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        }}
        .checklist-title {{
            font-size:22px;
            font-weight:950;
            line-height:1.1;
            letter-spacing:-0.02em;
        }}
        .checklist-subtitle {{
            margin-top:5px;
            font-size:13px;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 64%, transparent);
        }}
        .checklist-status-pill {{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            margin-top:10px;
            padding:7px 11px;
            border-radius:999px;
            font-size:11px;
            line-height:1;
            font-weight:950;
            letter-spacing:0.06em;
            text-transform:uppercase;
            border:1px solid color-mix(in srgb, var(--text-color) 13%, transparent);
        }}
        .checklist-status-pill.is-pending {{
            color:var(--pdm-accent);
            background:rgba(197,126,90,0.12);
            border-color:rgba(197,126,90,0.32);
        }}
        .checklist-status-pill.is-complete {{
            color:#ffffff;
            background:#22c55e;
            border-color:#22c55e;
        }}
        .checklist-progress {{
            padding:14px 18px 16px 18px;
        }}
        .checklist-progress-row {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:12px;
            margin-bottom:8px;
            font-size:13px;
            font-weight:850;
            color:color-mix(in srgb, var(--text-color) 68%, transparent);
        }}
        .checklist-bar {{
            width:100%;
            height:13px;
            border-radius:999px;
            background:color-mix(in srgb, var(--text-color) 9%, transparent);
            overflow:hidden;
        }}
        .checklist-bar-fill {{
            height:100%;
            width:{progress * 100:.1f}%;
            background:#22c55e;
            border-radius:999px;
            transition:width .2s ease;
        }}
        .checklist-area-title {{
            margin:20px 0 10px 0;
            font-size:18px;
            font-weight:950;
            letter-spacing:0.035em;
            text-transform:uppercase;
            color:var(--text-color);
        }}
        .checklist-section-title {{
            margin:2px 0 8px 0;
            font-size:13px;
            font-weight:950;
            letter-spacing:0.06em;
            text-transform:uppercase;
            color:color-mix(in srgb, var(--text-color) 72%, transparent);
        }}
        div[data-testid="stCheckbox"] label {{
            transition: background 0.14s ease, border-color 0.14s ease, transform 0.14s ease;
            border-radius:13px;
            padding:8px 8px;
            min-height:42px;
        }}
        div[data-testid="stCheckbox"] label:hover {{
            background:color-mix(in srgb, var(--pdm-accent) 8%, transparent);
            transform:translateX(1px);
        }}
        div[data-testid="stCheckbox"] label p {{
            font-size:0.98rem !important;
            line-height:1.22 !important;
            font-weight:720 !important;
        }}
        </style>
        <div class="checklist-hero">
            <div class="checklist-hero-head">
                <div class="checklist-title">{html.escape(title)}</div>
                <div class="checklist-subtitle">{html.escape(subtitle)}</div>
                <div class="checklist-status-pill {status_class}">{html.escape(status_label)}</div>
            </div>
            <div class="checklist-progress">
                <div class="checklist-progress-row">
                    <span>{html.escape(progress_label)}</span>
                    <span>{completed}/{total}</span>
                </div>
                <div class="checklist-bar"><div class="checklist-bar-fill"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step_idx = 1
    for area_name, sections in groups:
        st.markdown(f'<div class="checklist-area-title">{html.escape(area_name)}</div>', unsafe_allow_html=True)
        area_cols = st.columns(len(sections), gap="large")
        for col_ui, (section_name, items) in zip(area_cols, sections):
            with col_ui:
                st.markdown(
                    f'<div class="checklist-section-title">{step_idx}. {html.escape(section_name)}</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    for item in items:
                        key_base = f"{area_name}_{section_name}_{item}".lower()
                        key = "check_cambio_" + "".join(ch if ch.isalnum() else "_" for ch in key_base)
                        st.checkbox(item, key=key)
            step_idx += 1









def render_elegant_panel_open(title=None, subtitle=None, tag=None):
    # Deprecated raw-HTML wrapper removed. Streamlit can render closing </div> text
    # when open/close tags are split across different st.markdown calls.
    return


def render_elegant_panel_close():
    # Deprecated raw-HTML wrapper removed.
    return


def render_app_footer():
    st.markdown(
        """
        <style>
        .pdm-app-footer {
            margin: 38px 0 18px 0;
            padding: 14px 0 6px 0;
            border-top: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            color: color-mix(in srgb, var(--text-color) 48%, transparent);
            font-size: 11px;
            line-height: 1.2;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            text-align: center;
            user-select: none;
        }
        </style>
        <div class="pdm-app-footer">Avvolgimento · PDM · versione stabile v1.0</div>
        """,
        unsafe_allow_html=True,
    )




def render_operator_field_label(label):
    st.markdown(
        f"<div class='operator-field-label'>{html.escape(str(label))}</div>",
        unsafe_allow_html=True,
    )



st.markdown(
    """
    <style>
    /*
    Configurazione · soft card feel without altering native inputs.
    */
    .operator-field-label {
        margin: 0.58rem 0 0.32rem 0;
        color: var(--text-color);
        font-size: 0.80rem;
        line-height: 1.15;
        font-weight: 850;
        letter-spacing: 0.01em;
        opacity: 0.92;
    }

    .calibration-note-mini {
        margin: 5px 0 5px 0;
        padding: 6px 8px;
        border-radius: 10px;
        border: 1px solid color-mix(in srgb, var(--pdm-accent) 16%, var(--text-color) 7%);
        background: color-mix(in srgb, var(--pdm-accent) 4.5%, transparent);
        color: color-mix(in srgb, var(--text-color) 66%, transparent);
        font-size: 0.68rem;
        line-height: 1.22;
        font-weight: 620;
    }

    .calibration-compact-caption {
        margin: 2px 0 7px 0;
        color: color-mix(in srgb, var(--text-color) 56%, transparent);
        font-size: 0.70rem;
        line-height: 1.18;
        font-weight: 650;
    }

    .calibration-mode-label {
        margin-top: 0.34rem !important;
        margin-bottom: 0.20rem !important;
    }

    .calibration-horizontal-help {
        margin: 2px 0 10px 0;
        color: color-mix(in srgb, var(--text-color) 62%, transparent);
        font-size: 0.74rem;
        line-height: 1.28;
        font-weight: 650;
    }

    .calibration-horizontal-note {
        min-height: 44px;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, var(--pdm-accent) 16%, var(--text-color) 7%);
        background: color-mix(in srgb, var(--pdm-accent) 4.5%, transparent);
        color: color-mix(in srgb, var(--text-color) 68%, transparent);
        font-size: 0.72rem;
        line-height: 1.22;
        font-weight: 680;
        display: flex;
        align-items: center;
    }

    .calibration-warning-mini {
        margin: 9px 0 0 0;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, #f59e0b 42%, transparent);
        background: color-mix(in srgb, #f59e0b 10%, var(--secondary-background-color));
        color: color-mix(in srgb, var(--text-color) 82%, transparent);
        font-size: 0.73rem;
        line-height: 1.24;
        font-weight: 760;
    }

    .calibration-warning-mini strong {
        font-weight: 950;
        color: var(--text-color);
    }

    .calibration-factor-pill {
        min-height: 44px;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, var(--text-color) 11%, transparent);
        background: color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color));
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 2px;
    }

    .calibration-factor-pill span {
        font-size: 0.66rem;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: color-mix(in srgb, var(--text-color) 58%, transparent);
    }

    .calibration-factor-pill strong {
        font-size: 1.02rem;
        line-height: 1;
        font-weight: 950;
        color: var(--text-color);
    }

    .preset-live-status {
        margin: 10px 0 8px 0;
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        background: linear-gradient(180deg,
            color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
            color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
        );
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.045);
    }

    .preset-live-status.modified {
        border-color: color-mix(in srgb, #f59e0b 42%, transparent);
        background: color-mix(in srgb, #f59e0b 8%, var(--secondary-background-color));
    }

    .preset-live-status.original {
        border-color: color-mix(in srgb, #22c55e 32%, transparent);
    }

    .preset-live-status.prototype {
        border-color: color-mix(in srgb, var(--pdm-accent) 32%, transparent);
    }

    .preset-live-status strong {
        display: block;
        font-size: 0.86rem;
        line-height: 1.12;
        font-weight: 950;
        color: var(--text-color);
    }

    .preset-live-status span {
        display: block;
        margin-top: 2px;
        font-size: 0.72rem;
        line-height: 1.16;
        font-weight: 650;
        color: color-mix(in srgb, var(--text-color) 60%, transparent);
    }

    .preset-live-status em {
        flex: 0 0 auto;
        border-radius: 999px;
        padding: 6px 9px;
        background: var(--pdm-accent);
        color: #fff;
        font-size: 0.67rem;
        line-height: 1;
        font-style: normal;
        font-weight: 950;
        letter-spacing: 0.055em;
    }

    .preset-live-status.modified em {
        background: #f59e0b;
    }

    .preset-live-status.original em {
        background: #22c55e;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: color-mix(in srgb, var(--text-color) 12%, transparent) !important;
        background: linear-gradient(180deg,
            color-mix(in srgb, var(--secondary-background-color) 82%, var(--background-color)),
            color-mix(in srgb, var(--secondary-background-color) 96%, var(--background-color))
        ) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.055) !important;
        border-radius: 18px !important;
    }

    @media (max-width: 1180px) {
        .operator-field-label {
            margin-top: 0.54rem;
            margin-bottom: 0.30rem;
            font-size: 0.78rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /*
    Native Streamlit number input · visible +/- icons.
    Minimal patch: keeps the native control, only paints symbols inside the existing buttons.
    */
    div[data-testid="stNumberInput"] button {
        color: var(--text-color) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    div[data-testid="stNumberInput"] button svg {
        display: none !important;
    }

    div[data-testid="stNumberInput"] button::before {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
        color: var(--text-color) !important;
        font-weight: 950 !important;
        line-height: 1 !important;
        opacity: 0.96 !important;
        pointer-events: none !important;
    }

    div[data-testid="stNumberInput"] button:first-of-type::before {
        content: "−" !important;
        font-size: 1.20rem !important;
        transform: translateY(-1px);
    }

    div[data-testid="stNumberInput"] button:last-of-type::before {
        content: "+" !important;
        font-size: 1.06rem !important;
        transform: translateY(-1px);
    }

    div[data-testid="stNumberInput"] button:hover::before {
        color: var(--pdm-accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <style>
    /*
    Render badge · keep preset title below the progress controls.
    This is intentionally global because wide iPad iframes can bypass tablet media queries.
    */
    iframe {
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <style>
    /*
    Motion cleanup · reduce decorative shine. Keep hover feedback, avoid excessive sweep effects.
    */
    .summary-card::after,
    .preset-param-card::after,
    .quick-card-v2::after,
    .semaphore-card::after,
    .tech-mini-card::after,
    .machine-card-native::after,
    .preview-metric::after,
    .summary-strip::after,
    .summary-strip-item::after,
    .section-header::after,
    .workflow-step::after,
    .elegant-panel::after,
    .checklist-hero::after,
    .pdm-action-bar::after,
    .pack_stat::after {
        opacity: 0 !important;
        animation: none !important;
    }

    .summary-card:hover::after,
    .preset-param-card:hover::after,
    .quick-card-v2:hover::after,
    .semaphore-card:hover::after,
    .tech-mini-card:hover::after,
    .machine-card-native:hover::after,
    .preview-metric:hover::after,
    .summary-strip:hover::after,
    .summary-strip-item:hover::after,
    .section-header:hover::after,
    .workflow-step:hover::after,
    .elegant-panel:hover::after,
    .checklist-hero:hover::after,
    .pdm-action-bar:hover::after,
    .pack_stat:hover::after {
        opacity: 0 !important;
        animation: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <style>
    /*
    Responsive layout pass · desktop / iPad / mobile.
    Manté l'estètica actual, però evita columnes massa estretes i tabs tallats.
    */

    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
    }

    .main .block-container {
        width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="column"] {
        min-width: 0 !important;
    }

    div[data-testid="stIFrame"],
    div[data-testid="stIFrame"] iframe,
    iframe {
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .summary-card-value,
    .preset-param-value,
    .machine-card-value-native,
    .tech-mini-value,
    .summary-strip-value,
    .preview-metric-value,
    .preset-chip strong {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    /* TABLET · iPad horitzontal/vertical */
    @media (max-width: 1180px) {
        .main .block-container {
            max-width: 100% !important;
            padding-left: 0.72rem !important;
            padding-right: 0.72rem !important;
            padding-top: 0.70rem !important;
        }

        [data-testid="stTabs"] {
            margin-top: 0 !important;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 18px !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
            scrollbar-width: thin;
            padding-bottom: 2px !important;
            margin-bottom: 14px !important;
        }

        [data-testid="stTabs"] [role="tab"] {
            flex: 0 0 auto !important;
            min-width: max-content !important;
            min-height: 46px !important;
            padding-left: 0.28rem !important;
            padding-right: 0.28rem !important;
        }

        [data-testid="stTabs"] [role="tab"] p {
            white-space: nowrap !important;
            font-size: 0.98rem !important;
        }

        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.78rem !important;
        }

        div[data-testid="column"] {
            flex: 1 1 calc(50% - 0.78rem) !important;
            width: calc(50% - 0.78rem) !important;
            min-width: 300px !important;
        }

        .page-title-text {
            font-size: clamp(24px, 3.2vw, 30px) !important;
        }

        .section-header {
            padding: 14px 15px !important;
            margin-top: 12px !important;
            margin-bottom: 10px !important;
        }

        .section-title {
            font-size: clamp(17px, 2.25vw, 21px) !important;
        }

        .section-subtitle {
            font-size: 0.88rem !important;
            line-height: 1.35 !important;
        }

        .summary-card,
        .preset-param-card,
        .quick-card-v2,
        .semaphore-card,
        .tech-mini-card,
        .preview-metric,
        .preset-chip,
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
        }

        .summary-card {
            min-height: 112px !important;
            padding: 14px 15px !important;
        }

        .summary-card-value {
            font-size: clamp(24px, 3.2vw, 32px) !important;
        }

        .summary-card-label,
        .summary-card-note {
            font-size: 12.5px !important;
        }

        .preset-chip-row,
        .preset-hero-chips {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-baseweb="select"] * {
            font-size: 0.98rem !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] button {
            min-height: 44px !important;
        }

        .operator-field-label {
            font-size: 0.78rem !important;
            margin-top: 0.48rem !important;
        }
    }

    /* TABLET PETITA / MÒBIL GRAN */
    @media (max-width: 820px) {
        .main .block-container {
            padding-left: 0.58rem !important;
            padding-right: 0.58rem !important;
        }

        div[data-testid="column"] {
            flex: 1 1 calc(50% - 0.78rem) !important;
            width: calc(50% - 0.78rem) !important;
            min-width: 280px !important;
        }

        .preset-status-top,
        .preset-hero-top,
        .section-header-row {
            gap: 10px !important;
        }

        .preset-badges,
        .preset-hero-badges {
            justify-content: flex-start !important;
        }

        .preset-chip-row,
        .preset-hero-chips,
        .statusgrid {
            grid-template-columns: 1fr !important;
        }

        .summary-card {
            min-height: 96px !important;
        }

        .summary-card-value {
            font-size: clamp(22px, 6.2vw, 28px) !important;
        }

        .preset-param-value {
            font-size: clamp(22px, 6.2vw, 29px) !important;
        }

        .pdm-app-footer {
            margin-top: 26px !important;
            font-size: 10px !important;
            letter-spacing: 0.055em !important;
        }
    }

    /* MÒBIL */
    @media (max-width: 560px) {
        .main .block-container {
            padding-left: 0.42rem !important;
            padding-right: 0.42rem !important;
            padding-bottom: 1.10rem !important;
        }

        div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 12px !important;
            margin-bottom: 10px !important;
        }

        [data-testid="stTabs"] [role="tab"] {
            min-height: 42px !important;
            padding-top: 0.42rem !important;
            padding-bottom: 0.56rem !important;
        }

        [data-testid="stTabs"] [role="tab"] p {
            font-size: 0.90rem !important;
        }

        .page-title-shell {
            margin-bottom: 9px !important;
        }

        .page-title-text {
            font-size: 23px !important;
            line-height: 1.08 !important;
        }

        .section-header {
            padding: 12px 13px !important;
            border-radius: 14px !important;
        }

        .section-header-row {
            align-items: flex-start !important;
        }

        .section-badge {
            width: 27px !important;
            height: 27px !important;
            min-width: 27px !important;
            font-size: 12px !important;
        }

        .section-title {
            font-size: 17px !important;
        }

        .section-subtitle {
            font-size: 0.82rem !important;
        }

        .summary-card,
        .preset-param-card,
        .quick-card-v2,
        .semaphore-card,
        .tech-mini-card,
        .preview-metric,
        .preset-chip,
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px !important;
        }

        .summary-card {
            padding: 12px 13px !important;
            min-height: 88px !important;
            margin-bottom: 8px !important;
        }

        .summary-card::before,
        .preset-param-card::before {
            width: 4px !important;
        }

        .summary-card-label {
            font-size: 11.5px !important;
            margin-bottom: 8px !important;
        }

        .summary-card-note {
            font-size: 11.5px !important;
            margin-top: 6px !important;
        }

        .summary-card-value {
            font-size: 23px !important;
            line-height: 1.05 !important;
        }

        .preset-status-strip,
        .preset-hero,
        .tech-sheet-preset-card {
            padding: 13px 14px !important;
            border-radius: 15px !important;
        }

        .preset-status-title,
        .preset-hero-title {
            font-size: 20px !important;
            line-height: 1.08 !important;
        }

        .preset-chip {
            min-height: 52px !important;
            padding: 9px 10px !important;
        }

        .preset-chip strong {
            font-size: 16px !important;
        }

        div[role="radiogroup"] {
            gap: 7px !important;
            row-gap: 7px !important;
        }

        div[role="radiogroup"] label {
            min-height: 44px !important;
            min-width: 0 !important;
            padding: 0.50rem 0.70rem !important;
        }

        div[role="radiogroup"] label p {
            font-size: 0.88rem !important;
        }

        .stButton > button,
        div[data-testid="stDownloadButton"] > button {
            min-height: 44px !important;
            border-radius: 12px !important;
            font-size: 0.92rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            min-height: 42px !important;
            border-radius: 11px !important;
        }

        iframe {
            border-radius: 14px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <style>
    @media (max-width: 1180px) {
        .calibration-note-mini {
            padding: 5px 7px !important;
            font-size: 0.64rem !important;
            line-height: 1.18 !important;
        }
        .calibration-compact-caption {
            font-size: 0.66rem !important;
            margin-bottom: 5px !important;
        }
        .calibration-horizontal-help {
            font-size: 0.68rem !important;
            margin-bottom: 7px !important;
        }
        .calibration-horizontal-note,
        .calibration-factor-pill {
            min-height: 40px !important;
            padding: 6px 8px !important;
            font-size: 0.66rem !important;
        }
        .calibration-warning-mini {
            padding: 6px 8px !important;
            font-size: 0.66rem !important;
            line-height: 1.18 !important;
        }
        .preset-live-status {
            padding: 8px 10px !important;
            margin-top: 8px !important;
        }
        .operator-field-label {
            margin-top: 0.38rem !important;
            margin-bottom: 0.20rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding-top: 0.72rem !important;
            padding-bottom: 0.72rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)




st.markdown(
    """
    <style>
    /* Title-only header: reduce the blank gap between title and main tabs */
    div[data-testid="stTabs"] {
        margin-top: 0 !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        margin-top: 0 !important;
        margin-bottom: 12px !important;
    }

    @media (max-width: 1180px) {
        div[data-testid="stTabs"] {
            margin-top: -260px !important;
        }
    }

    @media (max-width: 900px) {
        div[data-testid="stTabs"] {
            margin-top: -430px !important;
        }
    }

    @media (max-width: 520px) {
        div[data-testid="stTabs"] {
            margin-top: -470px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# FINAL MOBILE HEADER / NO HORIZONTAL SCROLL PATCH
# =========================

st.markdown(
    """
    <style>
    /*
    v6 · Fix final:
    - elimina el scroll lateral en mòbil
    - anul·la els marges negatius antics de les pestanyes
    - deixa les pestanyes enganxades al header sense tocar la mida global dels botons
    */

    :root {
        --pdm-accent: #C57E5A;
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main,
    .main .block-container {
        max-width: 100% !important;
        overflow-x: clip !important;
        box-sizing: border-box !important;
    }

    @supports not (overflow-x: clip) {
        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main,
        .main .block-container {
            overflow-x: hidden !important;
        }
    }

    .main .block-container {
        padding-top: 0.18rem !important;
    }

    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"],
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stElementContainer"],
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stSelectbox"],
    div[data-testid="stRadio"],
    div[data-testid="stNumberInput"],
    div[data-testid="stTextInput"],
    div[data-testid="stButton"],
    div[data-testid="stDownloadButton"],
    div[data-baseweb="select"],
    div[data-baseweb="input"] {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="column"] {
        min-width: 0 !important;
    }

    img,
    svg,
    canvas,
    iframe,
    video {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Header estable: logo centrat + idioma a la dreta */
    div[data-testid="stHorizontalBlock"]:has(.pdm-header-logo-wrap) {
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
        gap: 0.25rem !important;
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 0 0.02rem 0 !important;
        padding: 0 !important;
        overflow: visible !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.pdm-header-logo-wrap) > div[data-testid="column"] {
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }

    .pdm-header-logo-wrap {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 0 !important;
    }

    .pdm-header-logo {
        height: clamp(170px, 10.6vw, 215px) !important;
        max-width: min(500px, 52vw) !important;
    }

    div[data-testid="column"]:has(.pdm-lang-slot) {
        padding-top: clamp(14px, 1.35vw, 22px) !important;
    }

    /* Tabs: sense marge fantasma ni scroll lateral */
    div[data-testid="stTabs"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
        max-width: 100% !important;
        overflow-x: clip !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        margin-top: 0 !important;
        margin-bottom: 10px !important;
        padding-top: 0 !important;
        padding-bottom: 0.20rem !important;
        max-width: 100% !important;
        overflow-x: clip !important;
        overflow-y: visible !important;
        scrollbar-width: none !important;
    }

    div[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
        display: none !important;
    }

    div[data-testid="stTabs"] [role="tab"] {
        min-width: 0 !important;
        box-sizing: border-box !important;
    }

    @supports not (overflow-x: clip) {
        div[data-testid="stTabs"],
        div[data-testid="stTabs"] [role="tablist"] {
            overflow-x: hidden !important;
        }
    }

    @media (max-width: 760px) {
        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main,
        .main .block-container {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        .main .block-container {
            padding-left: 0.42rem !important;
            padding-right: 0.42rem !important;
            padding-top: 0.08rem !important;
        }

        /* Header en una sola línia, però sense sumar més de 100vw */
        div[data-testid="stHorizontalBlock"]:has(.pdm-header-logo-wrap) {
            flex-wrap: nowrap !important;
            gap: 0 !important;
            margin-bottom: 0 !important;
            overflow: hidden !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.pdm-header-logo-wrap) > div[data-testid="column"]:nth-child(1) {
            flex: 0 0 20px !important;
            width: 20px !important;
            max-width: 20px !important;
            min-width: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.pdm-header-logo-wrap) > div[data-testid="column"]:nth-child(2) {
            flex: 1 1 auto !important;
            width: auto !important;
            max-width: calc(100% - 76px) !important;
            min-width: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.pdm-header-logo-wrap) > div[data-testid="column"]:nth-child(3) {
            flex: 0 0 56px !important;
            width: 56px !important;
            max-width: 56px !important;
            min-width: 0 !important;
        }

        .pdm-header-logo {
            height: clamp(92px, 26vw, 122px) !important;
            max-width: 100% !important;
        }

        div[data-testid="column"]:has(.pdm-lang-slot) {
            padding-top: 6px !important;
            align-items: flex-end !important;
        }

        div[data-testid="column"]:has(.pdm-lang-slot) div[data-testid="stRadio"] {
            width: 52px !important;
            max-width: 52px !important;
            overflow: hidden !important;
        }

        div[data-testid="column"]:has(.pdm-lang-slot) div[role="radiogroup"] {
            width: 52px !important;
            max-width: 52px !important;
            gap: 2px !important;
            overflow: hidden !important;
            flex-wrap: nowrap !important;
        }

        div[data-testid="column"]:has(.pdm-lang-slot) div[role="radiogroup"] label {
            flex: 0 0 25px !important;
            width: 25px !important;
            min-width: 25px !important;
            max-width: 25px !important;
            height: 26px !important;
            min-height: 26px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="column"]:has(.pdm-lang-slot) div[role="radiogroup"] label p {
            font-size: 0.58rem !important;
            line-height: 1 !important;
            letter-spacing: 0 !important;
        }

        /* La resta de blocs passen a una columna real: no poden crear amplada lateral */
        div[data-testid="stHorizontalBlock"]:not(:has(.pdm-header-logo-wrap)) {
            flex-wrap: wrap !important;
            width: 100% !important;
            max-width: 100% !important;
            gap: 0.58rem !important;
            overflow-x: hidden !important;
        }

        div[data-testid="stHorizontalBlock"]:not(:has(.pdm-header-logo-wrap)) > div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }

        /* Pestanyes encaixades a l'ample del mòbil, sense scroll horitzontal */
        div[data-testid="stTabs"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
            overflow-x: hidden !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            gap: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            margin-top: -0.10rem !important;
            margin-bottom: 7px !important;
            padding-bottom: 0.12rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stTabs"] [role="tab"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            min-height: 1.96rem !important;
            padding: 0.30rem 0.03rem 0.47rem 0.03rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stTabs"] [role="tab"] p {
            font-size: clamp(0.62rem, 2.85vw, 0.78rem) !important;
            line-height: 1.05 !important;
            text-align: center !important;
            white-space: normal !important;
            overflow-wrap: normal !important;
            word-break: normal !important;
        }

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            left: 0.18rem !important;
            right: 0.18rem !important;
            bottom: 0.05rem !important;
            height: 2px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# UI
# =========================

init_calculator_state()

production_label = "Simulazione" if lang == "IT" else "Simulation"
tech_sheet_label = "Scheda tecnica" if lang == "IT" else "Technical sheet"

try:
    presets_df = load_presets("Presets.csv")
    presets_load_exception = None
except Exception as e:
    presets_df = None
    presets_load_exception = e

checklist_label = "Cambio misura" if lang == "IT" else "Size change"


# =========================
# FINAL HEADER / MOBILE PATCH
# =========================

st.markdown(
    """
    <style>
    /* Últim override: només header, pestanyes i overflow mòbil. */
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main,
    .main .block-container {
        overflow-x: hidden !important;
        max-width: 100vw !important;
        box-sizing: border-box !important;
    }

    .pdm-header-logo-wrap {
        margin: 0 0 0.08rem 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 0 !important;
    }

    .pdm-header-logo {
        height: clamp(150px, 10.2vw, 205px) !important;
        max-width: min(500px, 58vw) !important;
    }

    div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) {
        position: absolute !important;
        top: 0.62rem !important;
        right: max(0.85rem, calc((100vw - 1800px) / 2 + 0.90rem)) !important;
        z-index: 100 !important;
        width: auto !important;
        max-width: none !important;
        min-width: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stTabs"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
        overflow-x: hidden !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 12px !important;
        overflow-x: hidden !important;
    }

    @media (max-width: 760px) {
        .main .block-container {
            padding-top: 0.08rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            width: 100vw !important;
            max-width: 100vw !important;
        }

        .pdm-header-logo-wrap {
            margin-bottom: 0 !important;
        }

        .pdm-header-logo {
            height: clamp(76px, 22vw, 108px) !important;
            max-width: min(220px, 46vw) !important;
        }

        div[data-testid="stRadio"]:has(input[value="IT"]):has(input[value="EN"]) {
            top: 0.34rem !important;
            right: 0.70rem !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            gap: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
            margin-top: 0 !important;
            margin-bottom: 0.42rem !important;
            padding: 0 !important;
        }

        div[data-testid="stTabs"] [role="tab"] {
            flex: 1 1 0 !important;
            width: 33.333% !important;
            max-width: 33.333% !important;
            min-width: 0 !important;
            min-height: 2.00rem !important;
            padding: 0.32rem 0.04rem 0.48rem 0.04rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stTabs"] [role="tab"] p {
            font-size: clamp(0.64rem, 2.9vw, 0.80rem) !important;
            line-height: 1.04 !important;
            text-align: center !important;
            white-space: normal !important;
        }

        div[data-testid="stHorizontalBlock"],
        div[data-testid="column"],
        div[data-testid="stVerticalBlock"],
        div[data-testid="stElementContainer"] {
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# FINAL HEADER V8 PATCH - no language radio, compact mobile header
st.markdown(
    """
    <style>
    .pdm-header-shell,
    .pdm-header-logo-wrap {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        height: auto !important;
        min-height: 0 !important;
    }

    .pdm-header-logo {
        height: clamp(150px, 10.2vw, 205px) !important;
        max-width: min(500px, 58vw) !important;
    }

    div[data-testid="stTabs"] {
        margin-top: -0.12rem !important;
        padding-top: 0 !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0.52rem !important;
    }

    @media (max-width: 760px) {
        .main .block-container {
            padding-top: 0.02rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .pdm-header-logo {
            height: clamp(112px, 31vw, 142px) !important;
            max-width: min(260px, 50vw) !important;
        }

        .pdm-lang-mini {
            left: calc(50% + 92px) !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            gap: 5px !important;
            max-width: calc(50vw - 10px) !important;
        }

        .pdm-lang-mini a {
            width: 34px !important;
            height: 30px !important;
            font-size: 0.62rem !important;
            letter-spacing: 0.02em !important;
        }

        div[data-testid="stTabs"] {
            margin-top: -0.28rem !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            margin-top: 0 !important;
            margin-bottom: 0.30rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        div[data-testid="stTabs"] [role="tab"] {
            min-height: 1.86rem !important;
            padding-top: 0.22rem !important;
            padding-bottom: 0.38rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# FINAL HEADER V9 PATCH - hard mobile layout: fixed header height, language at right, tabs attached
st.markdown(
    """
    <style>
    /* V9: override final només per arreglar header/tabs en desktop i mòbil */
    div[data-testid="stElementContainer"]:has(.pdm-header-shell) {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        overflow: visible !important;
    }

    .pdm-header-shell {
        position: relative !important;
        width: 100% !important;
        max-width: 100% !important;
        height: clamp(142px, 10.8vw, 210px) !important;
        min-height: clamp(142px, 10.8vw, 210px) !important;
        max-height: clamp(142px, 10.8vw, 210px) !important;
        margin: 0 0 -0.18rem 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }

    .pdm-header-logo-wrap {
        position: static !important;
        width: auto !important;
        max-width: none !important;
        height: auto !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 0 !important;
    }

    .pdm-header-logo {
        height: clamp(138px, 10.2vw, 198px) !important;
        max-height: 100% !important;
        max-width: min(520px, 50vw) !important;
        width: auto !important;
        object-fit: contain !important;
    }

    .pdm-lang-mini {
        position: absolute !important;
        right: max(0.90rem, calc((100vw - 1800px) / 2 + 0.90rem)) !important;
        left: auto !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 80 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        gap: 6px !important;
        width: auto !important;
        max-width: 140px !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        white-space: nowrap !important;
    }

    .pdm-lang-mini a {
        width: 38px !important;
        min-width: 38px !important;
        height: 30px !important;
        min-height: 30px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 999px !important;
        font-size: 0.66rem !important;
        line-height: 1 !important;
        font-weight: 900 !important;
        letter-spacing: 0.015em !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stTabs"] {
        margin-top: -0.55rem !important;
        padding-top: 0 !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        margin-top: 0 !important;
        margin-bottom: 0.48rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        min-height: 2.10rem !important;
        gap: clamp(14px, 1.45vw, 26px) !important;
        overflow-x: hidden !important;
    }

    div[data-testid="stTabs"] [role="tab"] {
        min-height: 2.05rem !important;
        padding-top: 0.26rem !important;
        padding-bottom: 0.44rem !important;
    }

    @media (max-width: 760px) {
        div[data-testid="stElementContainer"]:has(.pdm-header-shell) {
            height: 132px !important;
            min-height: 132px !important;
            max-height: 132px !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
        }

        .pdm-header-shell {
            height: 132px !important;
            min-height: 132px !important;
            max-height: 132px !important;
            margin: 0 0 -1.10rem 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: visible !important;
        }

        .pdm-header-logo {
            height: 122px !important;
            max-height: 122px !important;
            max-width: 52vw !important;
            width: auto !important;
        }

        .pdm-lang-mini {
            right: 0.10rem !important;
            left: auto !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            gap: 4px !important;
            max-width: 84px !important;
        }

        .pdm-lang-mini a {
            width: 31px !important;
            min-width: 31px !important;
            height: 28px !important;
            min-height: 28px !important;
            font-size: 0.58rem !important;
            box-shadow: none !important;
        }

        .main .block-container {
            padding-top: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }

        div[data-testid="stTabs"] {
            margin-top: -1.18rem !important;
            padding-top: 0 !important;
            overflow-x: hidden !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            width: 100% !important;
            max-width: 100% !important;
            min-height: 1.78rem !important;
            gap: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0.22rem !important;
            padding: 0 !important;
            overflow: hidden !important;
        }

        div[data-testid="stTabs"] [role="tab"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            min-height: 1.72rem !important;
            padding: 0.16rem 0.02rem 0.30rem 0.02rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stTabs"] [role="tab"] p {
            font-size: clamp(0.60rem, 2.75vw, 0.76rem) !important;
            line-height: 1.02 !important;
            text-align: center !important;
            white-space: normal !important;
            overflow-wrap: normal !important;
        }

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            left: 0.10rem !important;
            right: 0.10rem !important;
            bottom: 0 !important;
            height: 2px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_production, tab_tech_sheet, tab_checklist = st.tabs([
    production_label,
    tech_sheet_label,
    checklist_label,
])

# =========================
# GUARANTEED HEADER/TABS GAP FIX
# =========================
# CSS alone was not reliable enough on Streamlit mobile because Streamlit can wrap
# markdown blocks with dynamic heights. This tiny same-origin component measures the
# real rendered distance between the logo and the tabs, then moves ONLY the tabs up
# until the gap is compact. It does not touch buttons, cards, inputs or the rest of
# the layout.
components.html(
    """
    <script>
    (function () {
        const DESKTOP_GAP = 14;
        const MOBILE_GAP = 10;

        function getDoc() {
            try {
                return window.parent && window.parent.document ? window.parent.document : document;
            } catch (error) {
                return document;
            }
        }

        function compactHeaderGap() {
            const doc = getDoc();
            const logo = doc.querySelector('.pdm-header-logo');
            const tabs = doc.querySelector('div[data-testid=\"stTabs\"]');

            if (!logo || !tabs) return;

            const viewportWidth = doc.documentElement.clientWidth || window.innerWidth || 1024;
            const desiredGap = viewportWidth <= 760 ? MOBILE_GAP : DESKTOP_GAP;

            // Reset before measuring, otherwise repeated runs would accumulate.
            tabs.style.setProperty('margin-top', '0px', 'important');

            const logoRect = logo.getBoundingClientRect();
            const tabsRect = tabs.getBoundingClientRect();
            const currentGap = tabsRect.top - logoRect.bottom;

            if (currentGap > desiredGap) {
                const pullUp = Math.round(currentGap - desiredGap);
                tabs.style.setProperty('margin-top', '-' + pullUp + 'px', 'important');
            } else {
                tabs.style.setProperty('margin-top', '0px', 'important');
            }

            tabs.style.setProperty('padding-top', '0px', 'important');
            tabs.style.setProperty('overflow-x', 'hidden', 'important');

            const tablist = tabs.querySelector('[role=\"tablist\"]');
            if (tablist) {
                tablist.style.setProperty('margin-top', '0px', 'important');
                tablist.style.setProperty('padding-top', '0px', 'important');
                tablist.style.setProperty('overflow-x', 'hidden', 'important');
                if (viewportWidth <= 760) {
                    tablist.style.setProperty('margin-bottom', '6px', 'important');
                    tablist.style.setProperty('display', 'grid', 'important');
                    tablist.style.setProperty('grid-template-columns', 'repeat(3, minmax(0, 1fr))', 'important');
                    tablist.style.setProperty('gap', '0px', 'important');
                    tablist.style.setProperty('width', '100%', 'important');
                }
            }
        }

        function schedule() {
            compactHeaderGap();
            window.requestAnimationFrame(compactHeaderGap);
            setTimeout(compactHeaderGap, 120);
            setTimeout(compactHeaderGap, 450);
            setTimeout(compactHeaderGap, 900);
        }

        schedule();
        window.addEventListener('resize', schedule, { passive: true });

        try {
            const doc = getDoc();
            const observer = new MutationObserver(schedule);
            observer.observe(doc.body, { childList: true, subtree: true });
            setTimeout(function () { observer.disconnect(); }, 5000);
        } catch (error) {}
    })();
    </script>
    """,
    height=0,
)

if presets_df is None:
    if isinstance(presets_load_exception, FileNotFoundError):
        st.error(t["presets_file_missing"])
    else:
        st.error(f"{t['presets_load_error']}: {presets_load_exception}")
    st.stop()

# Shared selected preset
preset_names = presets_df["Prodotto"].tolist()

if "selected_preset_product" not in st.session_state or st.session_state["selected_preset_product"] not in preset_names:
    st.session_state["selected_preset_product"] = preset_names[0]

with tab_production:
    render_page_title(production_label)

    render_section_header(
        "Selezione prodotto" if lang == "IT" else "Product selection",
        "Scegli un preset: i valori si caricano automaticamente nel render." if lang == "IT" else "Choose a preset: values are automatically loaded into the render.",
        "1",
    )

    source_options = ["Preset", "Prototipo"] if lang == "IT" else ["Preset", "Prototype"]
    default_source = st.session_state.get("product_source_mode", "preset")
    source_index = 1 if default_source == "prototype" else 0

    source_col, selector_col, spacer_col = st.columns([0.70, 2.05, 1.25], gap="large")
    with source_col:
        source_label = "Origine dati" if lang == "IT" else "Data source"
        selected_source_label = st.radio(
            source_label,
            source_options,
            index=source_index,
            horizontal=True,
            key="product_source_radio",
        )

    is_prototype = selected_source_label == source_options[1]
    st.session_state["product_source_mode"] = "prototype" if is_prototype else "preset"

    with selector_col:
        st.markdown(
            f"""
            <div class="preset-selector-card">
                <div class="preset-selector-title">{"Preset prodotto" if lang == "IT" else "Product preset"}</div>
                <div class="preset-selector-subtitle">{"Seleziona il prodotto da simulare." if lang == "IT" else "Select the product to simulate."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if is_prototype:
            prototype_default = st.session_state.get("prototype_name", "Nuovo prototipo" if lang == "IT" else "New prototype")
            selected_product = st.text_input(
                "Nome prototipo" if lang == "IT" else "Prototype name",
                value=prototype_default,
                key="prototype_name_input",
            ).strip() or ("Nuovo prototipo" if lang == "IT" else "New prototype")
            st.session_state["prototype_name"] = selected_product
        else:
            selected_product = st.selectbox(
                t["select_product"],
                preset_names,
                index=preset_names.index(st.session_state["selected_preset_product"]),
                key="selected_preset_product_selectbox",
            )
            st.session_state["selected_preset_product"] = selected_product

    selected_row = None if is_prototype else presets_df[presets_df["Prodotto"] == selected_product].iloc[0]

    # Prototype reset must be applied before any calc_* widget is instantiated.
    if st.session_state.get("reset_prototype_request"):
        st.session_state.pop("reset_prototype_request", None)
        prototype_defaults = {
            "calc_diametro_aspo": 450.0,
            "calc_spalla": 95.0,
            "calc_rame": "1/4",
            "calc_spessore": 7.0,
            "calc_lunghezza": 50.0,
            "calc_passo_visuale": 20.0,
            "calc_incremento_visuale": 20.0,
            "calc_rit_b": 360.0,
            "calc_rit_t": 360.0,
            "calc_tube_layout": "Singolo",
            "calc_rame_inf": "3/8",
            "calc_spessore_inf": 7.0,
            "calc_rame_sup": "1/4",
            "calc_spessore_sup": 7.0,
        }
        for key, value in prototype_defaults.items():
            st.session_state[key] = value
        st.session_state["changed_values_pulse"] = True

    if is_prototype:
        # Keep prototype clearly independent from official Presets.
        st.session_state["last_auto_loaded_preset"] = None
        st.session_state.pop("loaded_preset_name", None)
        st.session_state.pop("loaded_preset_values", None)
        st.session_state["preset_values_modified"] = False
        st.session_state["modified_preset_fields"] = []
        preset_modified = False
    else:
        # Pending restore request must be applied before any calc_* widget is instantiated.
        if st.session_state.get("restore_preset_request") == selected_product:
            st.session_state.pop("restore_preset_request", None)
            apply_preset_to_calculator(selected_row)

        # Auto-load preset when product changes. This removes the old "select + load + switch tab" flow.
        last_auto_loaded = st.session_state.get("last_auto_loaded_preset")
        if last_auto_loaded != selected_product:
            apply_preset_to_calculator(selected_row)
            st.session_state["last_auto_loaded_preset"] = selected_product
            st.session_state["loaded_preset_name"] = selected_product
            st.session_state["show_preset_loaded_success"] = False

        # If the user changes any calculator value manually, keep the preset as base and mark it as modified.
        sync_active_preset_state()
        preset_modified = bool(st.session_state.get("preset_values_modified", False))

    params_locked = bool(st.session_state.get("params_locked", False))

    reveal_source_mode = "prototype" if is_prototype else "preset"
    reveal_key = f"{reveal_source_mode}::{selected_product}"

    # First app load: mark current preset as already seen, without opening the pop-up.
    if "last_revealed_product_key" not in st.session_state:
        st.session_state["last_revealed_product_key"] = reveal_key
        st.session_state.pop("pending_reveal_key", None)
        st.session_state.pop("pending_reveal_product", None)
        st.session_state.pop("pending_reveal_source_mode", None)
    elif st.session_state.get("last_revealed_product_key") != reveal_key:
        st.session_state["pending_reveal_key"] = reveal_key
        st.session_state["pending_reveal_product"] = selected_product
        st.session_state["pending_reveal_source_mode"] = reveal_source_mode

    pending_reveal_key = st.session_state.get("pending_reveal_key")
    if pending_reveal_key == reveal_key:
        render_preset_reveal_overlay(
            st.session_state.get("pending_reveal_product", selected_product),
            lang,
            st.session_state.get("pending_reveal_source_mode", reveal_source_mode),
            reveal_key,
        )

    if is_prototype:
        render_prototype_product_card(selected_product, lang)
    else:
        render_preset_product_card(selected_product, selected_row, lang, preset_modified)

    render_section_header(
        "Configurazione" if lang == "IT" else "Configuration",
        "Imposta solo i valori collegati al render: la scheda tecnica resta il riferimento completo per l’operatore." if lang == "IT" else "Set only the values linked to the render: the technical sheet remains the complete operator reference.",
        "2",
    )

    rame_options = list(COPPER_SIZES_MM.keys())
    colA, colB, colC = st.columns([1, 1, 1], gap="large")

    with colA:

        with st.container(border=True):
            st.markdown("**Tubo**")
            tube_layout_label = st.radio(
                "Tipo tubo",
                ["Singolo", "Doppio"],
                horizontal=True,
                key="calc_tube_layout",
                disabled=params_locked,
            )

            if tube_layout_label == "Singolo":
                if st.session_state.get("calc_rame") not in rame_options:
                    st.session_state["calc_rame"] = "1/4"
                render_operator_field_label(t["rame"])
                rame = st.selectbox(t["rame"], rame_options, key="calc_rame", disabled=params_locked, label_visibility="collapsed")
                render_operator_field_label(t["isolamento"])
                spessore = st.number_input(t["isolamento"], step=1.0, key="calc_spessore", disabled=params_locked, label_visibility="collapsed")
                render_operator_field_label(t["lunghezza"])
                lunghezza = st.number_input(t["lunghezza"], step=5.0, key="calc_lunghezza", disabled=params_locked, label_visibility="collapsed")

                d_rame = COPPER_SIZES_MM[rame]
                d_tubo = d_rame + 2.0 * spessore
                d_tubo_lower = d_tubo
                d_tubo_upper = d_tubo
                d_tubo_sim = d_tubo
                d_tubo_footprint = d_tubo
                tube_layout_code = "single"
                tube_diameter_label = f"{d_tubo:.2f} mm"
                passo_consigliato = d_tubo
                incremento_consigliato = d_tubo
            else:
                st.caption("Doppio verticale: diametro grande sotto, diametro piccolo sopra")
                c_inf, c_sup = st.columns(2)
                with c_inf:
                    if st.session_state.get("calc_rame_inf") not in rame_options:
                        st.session_state["calc_rame_inf"] = "3/8"
                    render_operator_field_label("Rame inferiore")
                    rame_inf = st.selectbox("Rame inferiore", rame_options, key="calc_rame_inf", disabled=params_locked, label_visibility="collapsed")
                    render_operator_field_label("Guaina inferiore (mm)")
                    spessore_inf = st.number_input("Guaina inferiore (mm)", step=1.0, key="calc_spessore_inf", disabled=params_locked, label_visibility="collapsed")
                with c_sup:
                    if st.session_state.get("calc_rame_sup") not in rame_options:
                        st.session_state["calc_rame_sup"] = "1/4"
                    render_operator_field_label("Rame superiore")
                    rame_sup = st.selectbox("Rame superiore", rame_options, key="calc_rame_sup", disabled=params_locked, label_visibility="collapsed")
                    render_operator_field_label("Guaina superiore (mm)")
                    spessore_sup = st.number_input("Guaina superiore (mm)", step=1.0, key="calc_spessore_sup", disabled=params_locked, label_visibility="collapsed")
                render_operator_field_label(t["lunghezza"])
                lunghezza = st.number_input(t["lunghezza"], step=5.0, key="calc_lunghezza", disabled=params_locked, label_visibility="collapsed")

                d_tubo_lower = COPPER_SIZES_MM[rame_inf] + 2.0 * spessore_inf
                d_tubo_upper = COPPER_SIZES_MM[rame_sup] + 2.0 * spessore_sup
                d_tubo_sim = max(d_tubo_lower, d_tubo_upper)
                d_tubo = d_tubo_sim
                d_tubo_footprint = d_tubo_sim
                tube_layout_code = "double"
                tube_diameter_label = f"Inferiore {d_tubo_lower:.2f} / Superiore {d_tubo_upper:.2f} mm"
                passo_consigliato = d_tubo_lower + d_tubo_upper
                incremento_consigliato = max(d_tubo_lower, d_tubo_upper)

    with colB:

        with st.container(border=True):
            st.markdown("**Avvolgitore**")
            render_operator_field_label(t["diam_aspo"])
            diametro_aspo = st.number_input(t["diam_aspo"], step=10.0, key="calc_diametro_aspo", disabled=params_locked, label_visibility="collapsed")
            render_operator_field_label(t["spalla"])
            spalla = st.number_input(t["spalla"], step=1.0, key="calc_spalla", disabled=params_locked, label_visibility="collapsed")
            render_operator_field_label(t["passo_assiale"])
            passo_visuale = st.number_input(t["passo_assiale"], step=0.5, key="calc_passo_visuale", disabled=params_locked, label_visibility="collapsed")
            render_operator_field_label(t["incremento"])
            incremento_visuale = st.number_input(t["incremento"], step=0.5, key="calc_incremento_visuale", disabled=params_locked, label_visibility="collapsed")

    with colC:

        with st.container(border=True):
            st.markdown("**Ritardi e regolazione**" if lang == "IT" else "Delays and setup")
            render_operator_field_label(t["rit_min"])
            rit_b = st.number_input(t["rit_min"], step=1.0, key="calc_rit_b", disabled=params_locked, label_visibility="collapsed")
            render_operator_field_label(t["rit_max"])
            rit_t = st.number_input(t["rit_max"], step=1.0, key="calc_rit_t", disabled=params_locked, label_visibility="collapsed")

            restore_label_inline = "Ripristina preset" if lang == "IT" else "Restore preset"
            lock_label_inline = "Blocca parametri" if lang == "IT" else "Lock parameters"

            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
            action_a, action_b = st.columns([1, 1], gap="small")

            with action_a:
                if is_prototype:
                    reset_label_inline = "Reset prototipo" if lang == "IT" else "Reset prototype"
                    if st.button(reset_label_inline, use_container_width=True, key="reset_prototype_inline"):
                        st.session_state["reset_prototype_request"] = True
                        st.rerun()
                else:
                    if st.button(restore_label_inline, use_container_width=True, key="restore_preset_inline"):
                        st.session_state["restore_preset_request"] = str(selected_product)
                        st.rerun()

            with action_b:
                st.toggle(lock_label_inline, key="params_locked")


    # Horizontal simulation calibration section, separate from the "Ritardi e regolazione" card.
    correction_label = "Fattori correzione" if lang == "IT" else "Correction factors"
    ideal_label = "Ideale macchina" if lang == "IT" else "Machine ideal"
    mode_raw = str(st.session_state.get("calc_simulation_mode", "Fattori correzione"))
    if mode_raw in {"Fattori correzione", "Correction factors"}:
        st.session_state["calc_simulation_mode"] = correction_label
    elif mode_raw in {"Ideale macchina", "Machine ideal"}:
        st.session_state["calc_simulation_mode"] = ideal_label
    else:
        st.session_state["calc_simulation_mode"] = correction_label

    correction_factors_enabled = st.session_state.get("calc_simulation_mode") == correction_label
    preset_has_correction_factors = bool(st.session_state.get("preset_has_correction_factors", False))

    with st.container(border=True):
        st.markdown("**Calibrazione simulazione**" if lang == "IT" else "**Simulation calibration**")
        explanation_text = (
            "1,00 = nessuna correzione. Il fattore passo moltiplica il Passo usato nel render; "
            "il fattore compattazione moltiplica l’Incremento strato. Sono valori sperimentali salvati nel Presets.csv."
            if lang == "IT"
            else
            "1.00 = no correction. The pitch factor multiplies the Pitch used in the render; "
            "the compaction factor multiplies the Layer increment. They are experimental values stored in Presets.csv."
        )
        st.markdown(
            f"<div class='calibration-horizontal-help'>{html.escape(explanation_text)}</div>",
            unsafe_allow_html=True,
        )

        cal_mode_col, cal_status_col, cal_passo_col, cal_comp_col = st.columns([1.15, 1.45, 0.95, 0.95], gap="small")

        with cal_mode_col:
            render_operator_field_label("Modo simulazione" if lang == "IT" else "Simulation mode")
            simulation_mode = st.radio(
                "Modo simulazione" if lang == "IT" else "Simulation mode",
                [correction_label, ideal_label],
                horizontal=True,
                key="calc_simulation_mode",
                disabled=params_locked,
                label_visibility="collapsed",
            )

        correction_factors_enabled = st.session_state.get("calc_simulation_mode") == correction_label

        with cal_status_col:
            if not preset_has_correction_factors and not is_prototype:
                mode_note = "Nessun fattore completo nel preset: avvio automatico in ideale." if lang == "IT" else "No complete factors in preset: starts in ideal mode."
            elif correction_factors_enabled:
                mode_note = "Correzione attiva: usa i fattori sperimentali del preset." if lang == "IT" else "Correction active: uses experimental preset factors."
            else:
                mode_note = "Ideale attivo: i fattori non vengono applicati." if lang == "IT" else "Ideal active: factors are not applied."
            st.markdown(
                f"<div class='calibration-horizontal-note'>{html.escape(mode_note)}</div>",
                unsafe_allow_html=True,
            )

        with cal_passo_col:
            if correction_factors_enabled:
                render_operator_field_label("Fattore passo" if lang == "IT" else "Pitch factor")
                fattore_passo_effettivo = st.number_input(
                    "Fattore passo effettivo" if lang == "IT" else "Effective pitch factor",
                    min_value=0.50,
                    max_value=1.80,
                    step=0.01,
                    format="%.2f",
                    key="calc_fattore_passo_effettivo",
                    disabled=params_locked,
                    label_visibility="collapsed",
                )
            else:
                st.markdown(
                    "<div class='calibration-factor-pill'><span>Passo</span><strong>×1.00</strong></div>",
                    unsafe_allow_html=True,
                )

        with cal_comp_col:
            if correction_factors_enabled:
                render_operator_field_label("Fattore compattazione" if lang == "IT" else "Compaction factor")
                fattore_compattazione_radiale = st.number_input(
                    "Fattore compattazione radiale" if lang == "IT" else "Radial compaction factor",
                    min_value=0.50,
                    max_value=3.00,
                    step=0.01,
                    format="%.2f",
                    key="calc_fattore_compattazione_radiale",
                    disabled=params_locked,
                    label_visibility="collapsed",
                )
            else:
                st.markdown(
                    "<div class='calibration-factor-pill'><span>Comp.</span><strong>×1.00</strong></div>",
                    unsafe_allow_html=True,
                )

        if not preset_has_correction_factors and not is_prototype:
            warning_text = (
                "<strong>Fattori correzione non consolidati.</strong> Questo preset non ha ancora entrambi i fattori salvati nel Presets.csv; il render parte in Ideale macchina finché non vengono raccolti e inseriti."
                if lang == "IT"
                else
                "<strong>Correction factors not consolidated.</strong> This preset does not yet have both factors saved in Presets.csv; the render starts in Machine ideal until they are collected and entered."
            )
            st.markdown(
                f"<div class='calibration-warning-mini'>{warning_text}</div>",
                unsafe_allow_html=True,
            )
        elif is_prototype and not correction_factors_enabled:
            warning_text = (
                "<strong>Prototipo senza fattori consolidati.</strong> Usa il modo ideale o inserisci fattori provvisori solo per stimare l’ingombro."
                if lang == "IT"
                else
                "<strong>Prototype without consolidated factors.</strong> Use ideal mode or enter temporary factors only to estimate the footprint."
            )
            st.markdown(
                f"<div class='calibration-warning-mini'>{warning_text}</div>",
                unsafe_allow_html=True,
            )

    if is_prototype:
        status_title = "Prototipo" if lang == "IT" else "Prototype"
        status_detail = "Valori liberi: non collegati a un preset ufficiale." if lang == "IT" else "Free values: not linked to an official preset."
        status_badge = "PROTOTIPO" if lang == "IT" else "PROTOTYPE"
        status_class = "prototype"
    else:
        status_title = "Preset modificato" if preset_modified and lang == "IT" else ("Modified preset" if preset_modified else ("Preset originale" if lang == "IT" else "Original preset"))
        field_list = modified_field_labels(lang)
        if preset_modified and field_list:
            status_detail = ("Campi modificati: " if lang == "IT" else "Modified fields: ") + ", ".join(field_list[:5])
            if len(field_list) > 5:
                status_detail += f" +{len(field_list) - 5}"
        else:
            status_detail = "Valori originali caricati da Presets.csv." if lang == "IT" else "Original values loaded from Presets.csv."

        if not bool(st.session_state.get("preset_has_correction_factors", False)):
            status_detail += " · Fattori correzione non consolidati." if lang == "IT" else " · Correction factors not consolidated."

        status_badge = "MODIFICATO" if preset_modified and lang == "IT" else ("MODIFIED" if preset_modified else ("ORIGINALE" if lang == "IT" else "ORIGINAL"))
        status_class = "modified" if preset_modified else "original"

    st.markdown(
        f"""
        <div class="preset-live-status {status_class}">
            <div>
                <strong>{html.escape(status_title)}</strong>
                <span>{html.escape(status_detail)}</span>
            </div>
            <em>{html.escape(status_badge)}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


    z_min_center = None
    z_max_center = None

    if tube_layout_code == "double":
        RtLower = d_tubo_lower / 2.0
        RtUpper = d_tubo_upper / 2.0

        candidate_min = RtLower
        candidate_max = spalla - (RtLower + 2.0 * RtUpper)

        if candidate_max > candidate_min:
            z_min_center = candidate_min
            z_max_center = candidate_max
        else:
            st.warning(
                "Attenzione: la configurazione doppio non entra interamente nella spalla "
                f"(spalla {spalla:.2f} mm, altezza coppia {d_tubo_lower + d_tubo_upper:.2f} mm)."
            )

    simulation_mode_raw = str(st.session_state.get("calc_simulation_mode", "Fattori correzione"))
    use_correction_factors = simulation_mode_raw in {"Fattori correzione", "Correction factors"}
    simulation_mode = "Fattori correzione" if use_correction_factors else "Ideale macchina"

    fattore_passo_preset = float(st.session_state.get("calc_fattore_passo_effettivo", DEFAULT_FATTORE_PASSO_EFFETTIVO))
    fattore_compattazione_preset = float(st.session_state.get("calc_fattore_compattazione_radiale", DEFAULT_FATTORE_COMPATTAZIONE_RADIALE))

    fattore_passo_effettivo = fattore_passo_preset if use_correction_factors else 1.0
    fattore_compattazione_radiale = fattore_compattazione_preset if use_correction_factors else 1.0

    passo_effettivo_assiale = max(0.0, float(passo_visuale) * fattore_passo_effettivo)
    incremento_effettivo_radiale = max(0.0, float(incremento_visuale) * fattore_compattazione_radiale)

    z_min_used = float(z_min_center) if z_min_center is not None else float(d_tubo_sim / 2.0)
    z_max_used = float(z_max_center) if z_max_center is not None else float(spalla - d_tubo_sim / 2.0)

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
        d_tubo=d_tubo_sim,
        passo=passo_effettivo_assiale,
        incremento=incremento_effettivo_radiale,
        rit_b=rit_b,
        rit_t=rit_t,
        lunghezza_m=lunghezza,
        gradi_start=gradi_start,
        deg_step=4.0,
        z_min_center=z_min_center,
        z_max_center=z_max_center,
    )

    winding_diagnostics = compute_winding_diagnostics(
        layer_values,
        z_values,
        mode_values,
        z_min_used,
        z_max_used,
        lang,
    )

    # Reference simulation without calibration factors.
    # This is especially useful for prototypes, where the real compaction is not validated yet.
    (
        reference_world_contacts,
        reference_local_points,
        reference_theta_values,
        reference_radius_values,
        reference_z_values,
        reference_mode_values,
        reference_layer_values,
        reference_length_values,
        reference_deposited_len_mm,
    ) = simulate_winding_visual(
        d_aspo=diametro_aspo,
        spalla=spalla,
        d_tubo=d_tubo_sim,
        passo=passo_visuale,
        incremento=incremento_visuale,
        rit_b=rit_b,
        rit_t=rit_t,
        lunghezza_m=lunghezza,
        gradi_start=gradi_start,
        deg_step=4.0,
        z_min_center=z_min_center,
        z_max_center=z_max_center,
    )

    reference_metrics = compute_metrics(reference_local_points, d_tubo_footprint)
    reference_diagnostics = compute_winding_diagnostics(
        reference_layer_values,
        reference_z_values,
        reference_mode_values,
        z_min_used,
        z_max_used,
        lang,
    )

    visual_metrics = compute_metrics(local_points, d_tubo_footprint)

    calibrated_footprint_mm = float(visual_metrics["max_xy_span"])
    reference_footprint_mm = float(reference_metrics["max_xy_span"])
    prototype_safe_footprint_mm = max(calibrated_footprint_mm, reference_footprint_mm)
    coil_footprint_for_status = prototype_safe_footprint_mm if is_prototype else calibrated_footprint_mm
    winding_ok = bool(local_points is not None and len(local_points) > 1 and visual_metrics["wound_length_m"] > 0)
    packaging_width_over = max(0.0, coil_footprint_for_status - 750.0)
    if packaging_width_over <= 0.001:
        packaging_tone = "ok"
        packaging_value_it = "OK pallet"
        packaging_value_en = "Pallet OK"
    elif packaging_width_over <= 20.001:
        packaging_tone = "warn"
        packaging_value_it = "Attenzione"
        packaging_value_en = "Attention"
    else:
        packaging_tone = "bad"
        packaging_value_it = "Fuori sagoma"
        packaging_value_en = "Over footprint"

    machine_complete = all([
        st.session_state.get("calc_diametro_aspo", 0) not in [None, 0],
        st.session_state.get("calc_spalla", 0) not in [None, 0],
        st.session_state.get("calc_lunghezza", 0) not in [None, 0],
        st.session_state.get("calc_passo_visuale", 0) not in [None, 0],
    ])

    if lang == "IT":
        status_items = [
            {
                "label": "Avvolgimento",
                "value": "OK" if winding_ok else "Da verificare",
                "note": f"Lunghezza simulata {visual_metrics['wound_length_m']:.2f} m",
                "tone": "ok" if winding_ok else "warn",
            },
            {
                "label": "Packaging",
                "value": packaging_value_it,
                "note": f"Ingombro XY {coil_footprint_for_status:.1f} mm / limite 750 mm",
                "tone": packaging_tone,
            },
            {
                "label": "Dati macchina",
                "value": "Completi" if machine_complete else "Da completare",
                "note": "Prototipo manuale" if is_prototype else "Preset caricato e parametri principali presenti",
                "tone": "ok" if machine_complete else "warn",
            },
        ]
    else:
        status_items = [
            {
                "label": "Winding",
                "value": "OK" if winding_ok else "Check",
                "note": f"Simulated length {visual_metrics['wound_length_m']:.2f} m",
                "tone": "ok" if winding_ok else "warn",
            },
            {
                "label": "Packaging",
                "value": packaging_value_en,
                "note": f"XY footprint {coil_footprint_for_status:.1f} mm / limit 750 mm",
                "tone": packaging_tone,
            },
            {
                "label": "Machine data",
                "value": "Complete" if machine_complete else "Incomplete",
                "note": "Manual prototype" if is_prototype else "Preset loaded and main parameters present",
                "tone": "ok" if machine_complete else "warn",
            },
        ]

    st.markdown(
        f"""
        <div class="section-header pdm-fade-up">
            <div class="section-header-row">
                <div class="section-badge">3</div>
                <div class="section-header-copy">
                    <div class="section-title">{"Render 3D" if lang == "IT" else "3D render"}</div>
                    <div class="section-subtitle">{"Anteprima generata dai parametri attuali." if lang == "IT" else "Preview generated from current parameters."}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    view_mode = st.radio(
        t["viewer_mode"],
        [t["scene_winding"], t["scene_packaging"]],
        horizontal=True,
        key="viewer_mode_selector",
    )

    packaging_mode_selected = "box"
    container_mode_selected = "40hc"
    pack_roll_count = int(st.session_state.get("pack_roll_count_external", 5))

    if view_mode == t["scene_packaging"]:
        st.markdown(f"#### {t['packaging_controls_title']}")
        pc1, pc2 = st.columns([1.0, 1.0], gap="large")
        with pc1:
            packaging_mode_label = st.radio(
                t["packaging_mode"],
                [t["packaging_box"], t["packaging_tower"]],
                horizontal=True,
                key="packaging_mode_external",
            )
            packaging_mode_selected = "box" if packaging_mode_label == t["packaging_box"] else "tower"
        with pc2:
            if packaging_mode_selected == "tower":
                container_label = st.radio(
                    t["container_type"],
                    [t["container_40hc"], t["container_20ft"]],
                    horizontal=True,
                    key="container_mode_external",
                )
                container_mode_selected = "40hc" if container_label == t["container_40hc"] else "20ft"
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.caption(t["packaging_box_desc"])

    simulation_print_payload = build_simulation_print_payload(
        selected_product,
        lang,
        tube_diameter_label,
        lunghezza,
        diametro_aspo,
        spalla,
        passo_visuale,
        incremento_visuale,
        rit_b,
        rit_t,
        visual_metrics,
        status_items,
    )

    components.html(
        viewer(
            diametro_aspo,
            spalla,
            d_tubo,
            720,
            local_points.tolist(),
            theta_values.tolist(),
            radius_values.tolist(),
            z_values.tolist(),
            mode_values.tolist(),
            layer_values.tolist(),
            length_values.tolist(),
            guide_offset_x,
            lang,
            coil_footprint_mm=coil_footprint_for_status,
            initial_scene="packaging" if view_mode == t["scene_packaging"] else "winding",
            packaging_mode=packaging_mode_selected,
            container_mode=container_mode_selected,
            pack_roll_count=pack_roll_count,
            tube_layout=tube_layout_code,
            d_tubo_lower=d_tubo_lower,
            d_tubo_upper=d_tubo_upper,
            tube_diameter_label=tube_diameter_label,
            simulation_print_payload=simulation_print_payload,
            active_product_name=selected_product,
            active_product_kind="prototype" if is_prototype else "preset",
        ),
        height=720,
    )

    render_status_semaphore(status_items, lang)

    st.divider()

    render_section_header(
        "Risultati" if lang == "IT" else "Results",
        "I risultati principali sono visibili direttamente qui sotto, senza passaggi extra." if lang == "IT" else "The main results are shown directly below, without extra steps.",
        "4",
    )

    pallet_size_mm = 750.0
    coil_footprint_mm = float(coil_footprint_for_status)

    result_cards = [
        {"label": "Modo" if lang == "IT" else "Mode", "value": ("Correzione" if use_correction_factors and lang == "IT" else ("Correction" if use_correction_factors else ("Ideale" if lang == "IT" else "Ideal")))},
        {"label": "Ø esterno tubo" if lang == "IT" else "Tube outer Ø", "value": str(tube_diameter_label)},
        {"label": t["metric2"], "value": f"{passo_visuale:.2f} mm"},
        {"label": t["metric3"], "value": f"{incremento_visuale:.2f} mm"},
        {"label": t["metric4"], "value": f"{visual_metrics['diam_radiale']:.1f} mm"},
        {"label": t["metric5"], "value": f"{coil_footprint_for_status:.1f} mm", "note": ("prudenziale prototipo" if is_prototype and lang == "IT" else ("prototype safe" if is_prototype else ""))},
    ]

    render_summary_cards(
        t["results"],
        result_cards,
        cards_per_row=3,
    )

    if is_prototype:
        if lang == "IT":
            prototype_occupation_cards = [
                {"label": "Ingombro stimato", "value": f"{calibrated_footprint_mm:.1f} mm", "note": "con calibrazione fisica"},
                {"label": "Ingombro nominale", "value": f"{reference_footprint_mm:.1f} mm", "note": "senza calibrazione"},
                {"label": "Ingombro prudenziale", "value": f"{prototype_safe_footprint_mm:.1f} mm", "note": "usa questo per pallet"},
                {"label": "Delta stima", "value": f"{calibrated_footprint_mm - reference_footprint_mm:+.1f} mm"},
                {"label": "Strati stimati", "value": str(winding_diagnostics["strati_simulati"]), "note": "con fattori attuali"},
                {"label": "Strati nominali", "value": str(reference_diagnostics["strati_simulati"]), "note": "senza fattori"},
            ]
            prototype_note = (
                "Per un prototipo non validato non esiste ancora un ingombro reale certo. "
                "La vista mostra l'ingombro stimato con i fattori attuali, l'ingombro nominale senza calibrazione "
                "e un ingombro prudenziale da usare per il pallet."
            )
        else:
            prototype_occupation_cards = [
                {"label": "Estimated footprint", "value": f"{calibrated_footprint_mm:.1f} mm", "note": "with physical calibration"},
                {"label": "Nominal footprint", "value": f"{reference_footprint_mm:.1f} mm", "note": "without calibration"},
                {"label": "Safe footprint", "value": f"{prototype_safe_footprint_mm:.1f} mm", "note": "use this for pallet"},
                {"label": "Estimate delta", "value": f"{calibrated_footprint_mm - reference_footprint_mm:+.1f} mm"},
                {"label": "Estimated layers", "value": str(winding_diagnostics["strati_simulati"]), "note": "with current factors"},
                {"label": "Nominal layers", "value": str(reference_diagnostics["strati_simulati"]), "note": "without factors"},
            ]
            prototype_note = (
                "For an unvalidated prototype there is not yet a certain real footprint. "
                "The view shows the estimated footprint with the current factors, the nominal footprint without calibration, "
                "and a safe footprint to use for pallet checks."
            )

        render_summary_cards(
            "Ingombro prototipo" if lang == "IT" else "Prototype footprint",
            prototype_occupation_cards,
            cards_per_row=3,
        )
        st.info(prototype_note)

    render_layer_diagnostics_panel(
        lang,
        winding_diagnostics,
        passo_effettivo_assiale,
        incremento_effettivo_radiale,
        fattore_passo_effettivo,
        fattore_compattazione_radiale,
        use_correction_factors,
    )

    if coil_footprint_mm > pallet_size_mm:
        st.warning(t["warning"])

with tab_tech_sheet:
    render_page_title(tech_sheet_label)

    render_section_header(
        "Consultazione preset" if lang == "IT" else "Preset reference",
        "Qui trovi anteprima, lettura rapida e parametri completi del preset in una vista più ordinata." if lang == "IT" else "Here you find preview, quick reading and full Preset parameters in a cleaner layout.",
        "i",
    )

    if st.session_state.get("product_source_mode") == "prototype":
        prototype_name = st.session_state.get("prototype_name", "Nuovo prototipo" if lang == "IT" else "New prototype")
        st.markdown(
            f"""
            <div class="preset-hero">
                <div class="preset-hero-top">
                    <div>
                        <div class="preset-hero-kicker">{"Prototipo prodotto" if lang == "IT" else "Product prototype"}</div>
                        <div class="preset-hero-title">{html.escape(str(prototype_name))}</div>
                        <div class="preset-hero-subtitle">{"La scheda preset non è disponibile perché questo prodotto non esiste ancora nei preset ufficiali." if lang == "IT" else "The Preset sheet is unavailable because this product does not yet exist in the official presets."}</div>
                    </div>
                    <div class="preset-hero-badges">
                        <span class="preset-hero-badge modified">{"PROTOTIPO" if lang == "IT" else "PROTOTYPE"}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Per i prototipi usa il PDF di simulazione nella scheda Simulazione. La scheda preset resta disponibile solo per preset ufficiali." if lang == "IT" else "For prototypes, use the simulation PDF in the Simulation tab. The Preset sheet remains available only for official presets."
        )
    else:
        selected_product = st.session_state.get("selected_preset_product", preset_names[0])
        selected_row = presets_df[presets_df["Prodotto"] == selected_product].iloc[0]
        
        render_tech_sheet_preset_card(selected_product, selected_row, lang)

        safe_product_filename = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(selected_product)).strip("_") or "preset"
        csv_print_pdf = make_csv_preset_pdf_bytes(selected_product, selected_row, lang)

        st.markdown(
            """
            <style>
            div[data-testid="stDownloadButton"] > button {
                border-radius:999px !important;
                min-height:42px !important;
                background:#C57E5A !important;
                color:#ffffff !important;
                border:1px solid #C57E5A !important;
                font-weight:950 !important;
                letter-spacing:0.01em !important;
                box-shadow:0 9px 20px rgba(197,126,90,0.24) !important;
                padding-left:18px !important;
                padding-right:18px !important;
            }
            div[data-testid="stDownloadButton"] > button:hover {
                filter:brightness(1.06) !important;
                transform:translateY(-1px) !important;
                box-shadow:0 12px 24px rgba(197,126,90,0.30) !important;
            }
            .csv-print-row {
                display:flex;
                align-items:center;
                justify-content:space-between;
                gap:12px;
                margin:-4px 0 16px 0;
                padding:12px 14px;
                border-radius:16px;
                border:1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
                background:color-mix(in srgb, var(--secondary-background-color) 72%, transparent);
            }
            .csv-print-copy {
                font-size:12px;
                line-height:1.25;
                font-weight:700;
                color:color-mix(in srgb, var(--text-color) 62%, transparent);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        csv_copy = "Scarica il PDF del preset selezionato, senza cattura render." if lang == "IT" else "Download the selected preset PDF, without render capture."
        csv_note_col, csv_button_col = st.columns([0.78, 0.22], gap="small")
        with csv_note_col:
            st.markdown(f'<div class="csv-print-copy">{html.escape(csv_copy)}</div>', unsafe_allow_html=True)
        with csv_button_col:
            if csv_print_pdf is not None:
                render_pdf_open_new_tab_link(
                    csv_print_pdf,
                    f"scheda_preset_{safe_product_filename}.pdf",
                    "PDF scheda",
                    None,
                )
            else:
                st.warning("Per scaricare il PDF aggiungi `reportlab` a requirements.txt." if lang == "IT" else "To download the PDF, add `reportlab` to requirements.txt.")

        render_tech_snapshot_cards(selected_row, lang)

        # Small spacer before the internal tabs.
        INNER_TABS_SPACER = 14
        st.markdown(f"<div style='height: {INNER_TABS_SPACER}px;'></div>", unsafe_allow_html=True)

        overview_tab, machine_sheet_tab = st.tabs([
            "Anteprima" if lang == "IT" else "Overview",
            "Parametri macchina" if lang == "IT" else "Machine parameters",
        ])

        render_columns = {
            "Tipo tubo",
            "Diametro rame inferiore",
            "Spessore guaina inferiore",
            "Diametro rame superiore",
            "Spessore guaina superiore",
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
        consult_cols = [c for c in presets_df.columns if c not in render_columns]

        with overview_tab:
            render_section_header(
                "Anteprima tecnica" if lang == "IT" else "Technical preview",
                "Disegno del tubo e schema sintetico del preset selezionato." if lang == "IT" else "Tube drawing and compact scheme of the selected preset.",
                "A",
            )
            components.html(make_preset_visual(selected_row, lang), height=570, scrolling=False)

        with machine_sheet_tab:
            render_section_header(
                "Scheda parametri macchina" if lang == "IT" else "Machine parameter sheet",
                "Vista unica ordinata per introdurre tutti i valori in macchina. Usa la ricerca per trovare subito il parametro che ti serve." if lang == "IT" else "Single grouped view to enter all machine values. Use search to find the parameter you need instantly.",
                "B",
            )

            render_machine_parameter_groups(selected_row, lang, key_suffix="_tech")
with tab_checklist:
    render_startup_checklist(lang)

render_app_footer()
