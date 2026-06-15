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
            padding: 18px 20px;
            min-height: 128px;
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
            color: var(--text-color);
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
            padding: 18px 20px;
            min-height: 128px;
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
            color: var(--text-color);
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
    # Excel sometimes saves CSV files in Windows/Latin encoding instead of UTF-8.
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

    extra_labels = {
        "IT": {
            "upper": "Superiore",
            "lower": "Inferiore",
            "double_note": "Doppio verticale · grande sotto, piccolo sopra",
            "type": "Tipo tubo",
            "double": "Doppio",
        },
        "EN": {
            "upper": "Upper",
            "lower": "Lower",
            "double_note": "Vertical double · large below, small above",
            "type": "Tube type",
            "double": "Double",
        },
    }[language]

    def v(value):
        return html.escape(format_preset_value(value))

    def metric_html(label, value, suffix=""):
        formatted = format_preset_value(value).strip() if value is not None else "-"
        if formatted in {"", "-"}:
            return ""
        return f'<div class="metric"><span class="label">{html.escape(str(label))}</span><span class="value">{html.escape(formatted + suffix)}</span></div>'

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


    tipo_tubo = str(row.get("Tipo tubo", "Singolo")).strip().lower()
    is_doppio_preview = tipo_tubo == "doppio"

    if is_doppio_preview:
        rame_inf = str(first_existing_value(row, ["Diametro rame inferiore", "Diametro Rame inferiore"], "3/8")).strip()
        rame_sup = str(first_existing_value(row, ["Diametro rame superiore", "Diametro Rame superiore"], "1/4")).strip()

        spessore_inf = parse_float_value(first_existing_value(row, ["Spessore guaina inferiore", "Spessore Guaina inferiore (mm)"], 0.0), 0.0)
        spessore_sup = parse_float_value(first_existing_value(row, ["Spessore guaina superiore", "Spessore Guaina superiore (mm)"], 0.0), 0.0)

        d_inf = COPPER_SIZES_MM.get(str(rame_inf), parse_float_value(rame_inf, 0.0)) + 2.0 * spessore_inf
        d_sup = COPPER_SIZES_MM.get(str(rame_sup), parse_float_value(rame_sup, 0.0)) + 2.0 * spessore_sup
        d_coppia = d_inf + d_sup

        # SVG scale: keep the preview readable independently from real dimensions.
        r_inf_svg = 50.0
        r_sup_svg = max(22.0, min(46.0, r_inf_svg * (d_sup / max(d_inf, 1e-9))))
        cy_inf = 135.0
        cy_sup = cy_inf - r_inf_svg - r_sup_svg
        cx_pair = 122.0

        tube_section_html = f"""
            <div class="drawing">
                <svg viewBox="0 0 340 260" role="img" aria-label="Double tube section preview">
                    <line x1="32" y1="{cy_sup:.1f}" x2="200" y2="{cy_sup:.1f}" class="d-center"/>
                    <line x1="32" y1="{cy_inf:.1f}" x2="200" y2="{cy_inf:.1f}" class="d-center"/>
                    <line x1="{cx_pair:.1f}" y1="20" x2="{cx_pair:.1f}" y2="206" class="d-center"/>

                    <circle cx="{cx_pair}" cy="{cy_inf}" r="{r_inf_svg}" fill="var(--foam)" stroke="var(--foam-stroke)" stroke-width="2.8"/>
                    <circle cx="{cx_pair}" cy="{cy_inf}" r="{max(16, r_inf_svg * 0.36):.1f}" fill="var(--copper)" stroke="#8f4a1f" stroke-width="2.4"/>
                    <circle cx="{cx_pair}" cy="{cy_inf}" r="{max(10, r_inf_svg * 0.24):.1f}" fill="var(--copper-light)" opacity="0.95"/>

                    <circle cx="{cx_pair}" cy="{cy_sup}" r="{r_sup_svg}" fill="var(--foam)" stroke="var(--foam-stroke)" stroke-width="2.8"/>
                    <circle cx="{cx_pair}" cy="{cy_sup}" r="{max(13, r_sup_svg * 0.36):.1f}" fill="var(--copper)" stroke="#8f4a1f" stroke-width="2.4"/>
                    <circle cx="{cx_pair}" cy="{cy_sup}" r="{max(8, r_sup_svg * 0.24):.1f}" fill="var(--copper-light)" opacity="0.95"/>

                    <line x1="197" y1="{cy_sup-r_sup_svg:.1f}" x2="197" y2="{cy_inf+r_inf_svg:.1f}" class="d-line"/>
                    <line x1="{cx_pair+r_sup_svg:.1f}" y1="{cy_sup-r_sup_svg:.1f}" x2="197" y2="{cy_sup-r_sup_svg:.1f}" class="d-guide"/>
                    <line x1="{cx_pair+r_inf_svg:.1f}" y1="{cy_inf+r_inf_svg:.1f}" x2="197" y2="{cy_inf+r_inf_svg:.1f}" class="d-guide"/>
                    <text x="207" y="{(cy_sup + cy_inf) / 2:.1f}" transform="rotate(90 207 {(cy_sup + cy_inf) / 2:.1f})" text-anchor="middle" class="d-value">{format_preset_value(d_coppia)} mm</text>

                    <rect x="222" y="42" width="102" height="44" class="callout-box"/>
                    <text x="273" y="60" text-anchor="middle" class="d-label">{html.escape(extra_labels['upper'])}</text>
                    <text x="273" y="79" text-anchor="middle" class="d-value">{html.escape(str(rame_sup))} · {format_preset_value(d_sup)} mm</text>

                    <rect x="222" y="144" width="102" height="44" class="callout-box"/>
                    <text x="273" y="162" text-anchor="middle" class="d-label">{html.escape(extra_labels['lower'])}</text>
                    <text x="273" y="181" text-anchor="middle" class="d-value">{html.escape(str(rame_inf))} · {format_preset_value(d_inf)} mm</text>

                    <text x="122" y="248" text-anchor="middle" class="d-caption">{html.escape(extra_labels['double_note'])}</text>
                </svg>
            </div>
        """

        tube_metrics_rows = "".join([
            metric_html("Tipo tubo", "Doppio"),
            metric_html("Inferiore", f"{rame_inf} · {format_preset_value(d_inf)} mm"),
            metric_html("Superiore", f"{rame_sup} · {format_preset_value(d_sup)} mm"),
            metric_html("Altezza coppia", d_coppia, " mm"),
            metric_html(labels['length'], lunghezza, " m"),
            metric_html(labels['line_speed'], velocita_linea, " m/min"),
        ])

        tube_metrics_html = f"""
            <div class="metrics">
                {tube_metrics_rows}
            </div>
        """
    else:
        tube_section_html = f"""
            <div class="drawing">
                <svg viewBox="0 0 340 248" role="img" aria-label="Tube section preview">
                    <line x1="28" y1="110" x2="210" y2="110" class="d-center"/>
                    <line x1="119" y1="20" x2="119" y2="200" class="d-center"/>

                    <circle cx="119" cy="110" r="78" fill="var(--foam)" stroke="var(--foam-stroke)" stroke-width="2.8"/>
                    <circle cx="119" cy="110" r="33" fill="var(--copper)" stroke="#8f4a1f" stroke-width="2.4"/>
                    <circle cx="119" cy="110" r="24" fill="var(--copper-light)" opacity="0.95"/>

                    <line x1="41" y1="214" x2="197" y2="214" class="d-line"/>
                    <line x1="41" y1="199" x2="41" y2="229" class="d-line"/>
                    <line x1="197" y1="199" x2="197" y2="229" class="d-line"/>
                    <text x="119" y="207" text-anchor="middle" class="d-value">{v(d_tubo)} mm</text>
                    <text x="119" y="232" text-anchor="middle" class="d-label">Ø esterno</text>

                    <line x1="152" y1="110" x2="197" y2="110" class="d-thin"/>
                    <polygon points="152,110 160,106 160,114" fill="var(--white-line)"/>
                    <polygon points="197,110 189,106 189,114" fill="var(--white-line)"/>
                    <text x="174" y="102" text-anchor="middle" class="d-label">{v(spessore)} mm</text>

                    <rect x="218" y="44" width="104" height="42" class="callout-box"/>
                    <text x="270" y="62" text-anchor="middle" class="d-label">{html.escape(labels['outer_dim'])}</text>
                    <text x="270" y="80" text-anchor="middle" class="d-value">{v(d_tubo)} mm</text>

                    <rect x="218" y="103" width="104" height="42" class="callout-box"/>
                    <text x="270" y="121" text-anchor="middle" class="d-label">{html.escape(labels['insulation'])}</text>
                    <text x="270" y="139" text-anchor="middle" class="d-value">{v(spessore)} mm</text>

                    <rect x="218" y="162" width="104" height="42" class="callout-box"/>
                    <text x="270" y="180" text-anchor="middle" class="d-label">{html.escape(labels['copper'])}</text>
                    <text x="270" y="198" text-anchor="middle" class="d-value">{copper_label}</text>
                </svg>
            </div>
        """

        tube_metrics_rows = "".join([
            metric_html(labels['copper'], rame),
            metric_html(labels['foam'], spessore, " mm"),
            metric_html(labels['outer'], d_tubo, " mm"),
            metric_html(labels['length'], lunghezza, " m"),
            metric_html(labels['line_speed'], velocita_linea, " m/min"),
            metric_html(labels['air'], soffiatori_label),
        ])

        tube_metrics_html = f"""
            <div class="metrics">
                {tube_metrics_rows}
            </div>
        """

    coil_preview_html = f"""
            <div class="drawing">
                <svg viewBox="0 0 420 230" role="img" aria-label="Coiling layout preview">
                    <defs>
                        <linearGradient id="coilBand" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0%" stop-color="var(--coil-top)"/>
                            <stop offset="55%" stop-color="var(--coil-mid)"/>
                            <stop offset="100%" stop-color="var(--coil-dark)"/>
                        </linearGradient>
                        <linearGradient id="coreBody" x1="0" x2="1" y1="0" y2="0">
                            <stop offset="0%" stop-color="#4b5563"/>
                            <stop offset="50%" stop-color="#1f2937"/>
                            <stop offset="100%" stop-color="#111827"/>
                        </linearGradient>
                    </defs>

                    <line x1="38" y1="114" x2="382" y2="114" class="d-center"/>

                    <rect x="118" y="82" width="164" height="64" rx="18" fill="url(#coreBody)" stroke="rgba(255,255,255,0.12)" stroke-width="1.2"/>
                    <rect x="106" y="72" width="12" height="84" rx="4" fill="#2f3743" stroke="rgba(255,255,255,0.18)" stroke-width="1.1"/>
                    <rect x="282" y="72" width="12" height="84" rx="4" fill="#2f3743" stroke="rgba(255,255,255,0.18)" stroke-width="1.1"/>

                    <rect x="98" y="70" width="204" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.20)" stroke-width="1"/>
                    <rect x="96" y="82" width="208" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
                    <rect x="94" y="94" width="212" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
                    <rect x="92" y="106" width="216" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
                    <rect x="94" y="118" width="212" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
                    <rect x="96" y="130" width="208" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
                    <rect x="98" y="142" width="204" height="16" rx="8" fill="url(#coilBand)" stroke="rgba(255,255,255,0.20)" stroke-width="1"/>

                    <line x1="106" y1="44" x2="294" y2="44" class="d-line"/>
                    <line x1="106" y1="32" x2="106" y2="56" class="d-line"/>
                    <line x1="294" y1="32" x2="294" y2="56" class="d-line"/>
                    <text x="200" y="38" text-anchor="middle" class="d-value">{v(aspo)} mm</text>
                    <text x="200" y="58" text-anchor="middle" class="d-label">{html.escape(labels['spool'])}</text>

                    <line x1="52" y1="82" x2="52" y2="146" class="d-line"/>
                    <line x1="40" y1="82" x2="64" y2="82" class="d-line"/>
                    <line x1="40" y1="146" x2="64" y2="146" class="d-line"/>
                    <text x="69" y="116" transform="rotate(90 69 116)" text-anchor="middle" class="d-value">{v(spalla)} mm</text>
                    <text x="52" y="116" transform="rotate(90 52 116)" text-anchor="middle" class="d-label">{html.escape(labels['width'])}</text>

                    <line x1="324" y1="82" x2="354" y2="82" class="d-guide"/>
                    <line x1="324" y1="94" x2="354" y2="94" class="d-guide"/>
                    <line x1="354" y1="82" x2="354" y2="94" class="d-line"/>
                    <polygon points="350,82 354,78 358,82" fill="var(--white-line)"/>
                    <polygon points="350,94 354,98 358,94" fill="var(--white-line)"/>
                    <text x="364" y="90" class="d-label">{html.escape(labels['pitch'])}</text>
                    <text x="364" y="104" class="d-value">{v(passo)} mm</text>

                </svg>
            </div>
    """

    return f'''
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    :root {{
        --bg: transparent;
        --card-bg: rgba(255,255,255,0.78);
        --card-border: rgba(15,23,42,0.12);
        --text: #0f172a;
        --muted: rgba(15,23,42,0.64);
        --line: rgba(15,23,42,0.18);
        --shadow: 0 16px 38px rgba(15,23,42,0.12);
        --accent: #C57E5A;
        --accent-soft: rgba(255,75,75,0.075);
        --drawing-bg: linear-gradient(180deg, rgba(248,250,252,0.92), rgba(241,245,249,0.76));
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
        --shadow: 0 20px 42px rgba(0,0,0,0.32);
        --accent-soft: rgba(197,126,90,0.10);
        --drawing-bg: linear-gradient(180deg, rgba(30,41,59,0.72), rgba(15,23,42,0.58));
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
        padding:4px;
    }}
    .card {{
        position:relative;
        overflow:hidden;
        background:linear-gradient(180deg, var(--card-bg), rgba(255,255,255,0.58));
        border:1px solid var(--card-border);
        border-left:4px solid var(--accent);
        box-shadow:var(--shadow);
        backdrop-filter: blur(10px);
        border-radius:20px;
        padding:20px;
        min-height:365px;
        box-sizing:border-box;
    }}
    html[data-theme="dark"] .card {{
        background:linear-gradient(180deg, rgba(17,24,39,0.88), rgba(15,23,42,0.74));
    }}
    .card::before {{
        content:"";
        position:absolute;
        right:-90px;
        top:-90px;
        width:190px;
        height:190px;
        border-radius:999px;
        background:var(--accent-soft);
        pointer-events:none;
    }}
    .title {{
        font-size:14px;
        font-weight:900;
        letter-spacing:0.075em;
        text-transform:uppercase;
        color:var(--text);
        margin-bottom:6px;
    }}
    .subtitle {{
        font-size:12px;
        color:var(--muted);
        margin-bottom:16px;
        font-weight:650;
    }}
    .layout {{
        display:grid;
        grid-template-columns: 1.08fr 0.92fr;
        gap:18px;
        align-items:start;
    }}
    .drawing {{
        background:
            repeating-linear-gradient(0deg, rgba(148,163,184,0.08) 0 1px, transparent 1px 28px),
            repeating-linear-gradient(90deg, rgba(148,163,184,0.08) 0 1px, transparent 1px 28px),
            var(--drawing-bg);
        border:1px solid var(--line);
        border-radius:18px;
        padding:12px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
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
        position:relative;
    }}
    .metric:last-child {{ border-bottom:none; }}
    .label {{ font-size:12px; color:var(--muted); font-weight:700; }}
    .value {{ font-size:18px; color:var(--text); font-weight:900; text-align:right; white-space:nowrap; }}
    .callout-box {{ fill:var(--card-bg); stroke:rgba(255,75,75,0.28); stroke-width:1.25; rx:10; }}
    .d-label {{ fill:var(--muted); font-size:11px; font-weight:800; letter-spacing:0.02em; }}
    .d-value {{ fill:var(--text); font-size:15px; font-weight:900; }}
    .d-line {{ stroke:var(--white-line); stroke-width:3.2; stroke-linecap:round; stroke-linejoin:round; }}
    .d-guide {{ stroke:var(--white-line); stroke-width:2.1; stroke-linecap:round; stroke-dasharray:6 6; }}
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
            {tube_section_html}
            {tube_metrics_html}
        </div>
    </div>

    <div class="card">
        <div class="title">{html.escape(labels['coil'])}</div>
        <div class="subtitle">{html.escape(labels['coil_note'])}</div>
        <div class="layout">
{coil_preview_html}
            <div class="metrics">
                {''.join([
                    metric_html(labels['spool'], aspo, " mm"),
                    metric_html(labels['width'], spalla, " mm"),
                    metric_html(labels['pitch'], passo, " mm"),
                    metric_html(labels['layer'], incremento, " mm"),
                    metric_html(labels['rollers'], rulli_label),
                    metric_html(labels['tail'], paletta_label),
                ])}
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
    candidates = [
        "New Logo PDM – rame.png",
        "New Logo PDM - rame.png",
        "New Logo PDM rame.png",
        "new_logo_pdm_rame.png",
        "logo_pdm.png",
        "pdm_logo.png",
        "logo.png",
        "logo.svg",
        "logo.jpg",
        "logo.jpeg",
        "logo.webp",
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
            if os.path.exists(path):
                return path

    patterns = []
    for folder in search_dirs:
        patterns.extend([
            os.path.join(folder, "*logo*.png"),
            os.path.join(folder, "*Logo*.png"),
            os.path.join(folder, "*PDM*.png"),
            os.path.join(folder, "*.svg"),
            os.path.join(folder, "*.jpg"),
            os.path.join(folder, "*.jpeg"),
            os.path.join(folder, "*.webp"),
        ])

    for pattern in patterns:
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
    else:
        st.markdown(
            """
            <div style="font-size:24px;font-weight:950;letter-spacing:-0.04em;line-height:1;">
                PDM
            </div>
            <div style="font-size:10px;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;color:color-mix(in srgb, var(--text-color) 58%, transparent);">
                avvolgimento
            </div>
            """,
            unsafe_allow_html=True,
        )

with top2:
    title_placeholder = st.empty()
    current_lang = st.session_state.lang
    lang_option = st.selectbox(
        TEXTS[current_lang]["language"],
        ["Italiano", "English (US)"],
        index=0 if current_lang == "IT" else 1,
        key="lang_selector_top",
    )

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
lang = st.session_state.lang
t = TEXTS[lang]

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1800px;
        padding-top: 0.8rem;
        padding-bottom: 1.6rem;
        padding-left: 0.55rem;
        padding-right: 0.55rem;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 24px;
        padding: 0 8px;
        background: transparent;
        border-bottom: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
        margin-bottom: 20px;
        align-items: center;
    }

    [data-testid="stTabs"] button {
        font-weight: 780;
        min-height: 48px;
        padding: 0.45rem 0.15rem 0.60rem 0.15rem;
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

    [data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--text-color) !important;
        font-weight: 900 !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px;
        min-height: 44px;
    }

    div[data-baseweb="select"] * {
        font-size: 0.98rem;
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

    div[role="radiogroup"] {
        gap: 0.85rem;
        flex-wrap: wrap;
        align-items: stretch;
    }

    div[role="radiogroup"] label {
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
        border-radius: 999px;
        padding: 0.62rem 1.15rem;
        min-height: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        transition: transform 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
        box-sizing: border-box;
    }

    div[role="radiogroup"] label > div {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        width:100%;
        height:100%;
        text-align:center;
    }

    /* Hide the default radio dot so the options look like real buttons */
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[role="radiogroup"] label:hover {
        transform: translateY(-1px);
        border-color: color-mix(in srgb, var(--text-color) 34%, transparent);
        box-shadow: 0 10px 18px rgba(0,0,0,0.10);
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: #C57E5A;
        border-color: #C57E5A;
        box-shadow: 0 0 0 2px rgba(197,126,90,0.34), 0 10px 20px rgba(0,0,0,0.18);
    }

    div[role="radiogroup"] label p {
        font-weight: 800;
        font-size: 0.99rem;
        line-height: 1.15;
        text-align: center;
    }

    div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
    }

    /* Better tablet spacing and readability */
    @media (max-width: 1280px) {
        .main .block-container {
            max-width: 100%;
            padding-left: 0.35rem;
            padding-right: 0.35rem;
        }

        [data-testid="stTabs"] button {
            min-height: 48px;
            font-size: 1rem;
        }

        div[role="radiogroup"] label {
            min-height: 48px;
            padding: 0.62rem 1rem;
        }

        div[role="radiogroup"] label p {
            font-size: 1.00rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            min-height: 46px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
):
    final_local_points_json = json.dumps(final_local_points)
    final_thetas_json = json.dumps(final_thetas)
    final_radii_json = json.dumps(final_radii)
    final_zs_json = json.dumps(final_zs)
    final_modes_json = json.dumps(final_modes)
    final_layers_json = json.dumps(final_layers)
    final_lengths_json = json.dumps(final_lengths)
    labels_json = json.dumps(TEXTS[language])
    tube_layout = "double" if str(tube_layout).lower() in {"double", "doppio"} else "single"
    d_tubo_lower = float(d_tubo if d_tubo_lower is None else d_tubo_lower)
    d_tubo_upper = float(d_tubo if d_tubo_upper is None else d_tubo_upper)
    tube_diameter_label = tube_diameter_label or f"{float(d_tubo):.2f} mm"
    tube_diameter_label_json = json.dumps(str(tube_diameter_label))
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
            line-height:1.25;
        ">
            <button id="play_pause_btn" class="viewer_btn viewer_icon_btn">⏸</button>
            <button id="reset_view_btn" class="viewer_btn viewer_icon_btn">↺</button>
            <button id="fullscreen_btn" class="viewer_btn viewer_icon_btn">⛶</button>
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
                    <input type="checkbox" id="animation_check" checked />
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
            </div>

            <div id="spool_block">
                <div class="panel_label" id="spool_title"></div>
                <div class="btn_group_vertical btn_grid_3">
                    <button class="spool_btn viewer_btn_small active_opt" data-spool="visible" id="spool_visible_btn"></button>
                    <button class="spool_btn viewer_btn_small" data-spool="transparent" id="spool_transparent_btn"></button>
                    <button class="spool_btn viewer_btn_small" data-spool="hidden" id="spool_hidden_btn"></button>
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
                    <input type="checkbox" id="ghost_check" checked />
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
            border-radius:10px;
            padding:7px 8px;
            background:rgba(235,235,235,0.95);
            color:#111;
            font-weight:800;
            font-size:12px;
            cursor:pointer;
            text-align:center;
            white-space:normal;
            overflow-wrap:anywhere;
            line-height:1.12;
            min-height:38px;
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

        .viewer_btn_small:hover,
        .viewer_btn:hover {{
            background:#ffffff;
        }}

        .active_speed,
        .active_opt {{
            outline:2px solid #ffffff;
            background:#C57E5A !important;
            color:#ffffff !important;
            box-shadow:0 0 0 2px rgba(197,126,90,0.35), 0 8px 18px rgba(0,0,0,0.18);
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
            gap:6px;
        }}

        .btn_grid_3 {{
            display:grid;
            grid-template-columns:repeat(3, minmax(0, 1fr));
            gap:7px;
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
            min-height:40px !important;
            font-size:20px !important;
            line-height:1 !important;
            padding:0 !important;
        }}

        #pack_roll_count {{
            width:100%;
            min-height:40px;
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
            }}

            #progress_slider {{
                width:120px !important;
            }}

            .viewer_btn {{
                padding:6px 9px !important;
                border-radius:8px !important;
            }}
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
        const packagingStats = document.getElementById("packaging_stats");
        const viewerHud = document.getElementById("viewer_hud");
        const sidepanel = document.getElementById("viewer_sidepanel");
        const sidepanelContent = document.getElementById("viewer_sidepanel_content");
        const sidepanelToggle = document.getElementById("viewer_sidepanel_toggle");
        const packagingStatusBadge = document.getElementById("packaging_status_badge");
        const packagingStatusText = document.getElementById("packaging_status_text");
        const packagingStatusReason = document.getElementById("packaging_status_reason");

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

        function updateSidepanelToggle() {{
            const collapsed = sidepanel.classList.contains("collapsed");
            sidepanelToggle.textContent = collapsed ? "❯" : "❮";
            sidepanelToggle.title = collapsed ? "Mostra opzioni" : "Nascondi opzioni";
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

        let isPlaying = true;
        let animationEnabled = true;
        let speed = 1.0;
        let aspoMode = "visible";
        let tubeMode = "{tube_mode_initial}";
        let currentView = "3d";
        let sceneMode = "{initial_scene}";
        let packagingMode = "{packaging_mode}";
        let containerMode = "{container_mode}";
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
                const totalHeight = packagingGroup.userData.totalHeight || 800;
                const sceneSpan = Math.max(palletSize * 1.45, totalHeight * 1.10);
                camera.position.set(-sceneSpan * 1.15, -sceneSpan * 1.25, Math.max(980, totalHeight * 1.15));
                controls.target.set(0, 0, palletHeight + totalHeight * 0.42);
                camera.lookAt(0, 0, palletHeight + totalHeight * 0.42);
                controls.update();
            }}
        }}

        if (packRollCountInput) {{
            packRollCountInput.addEventListener("input", () => {{
                setPackRollCount(packRollCountInput.value);
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
                addBoxEdges(palletSize, palletSize, boxHeight, palletHeight + boxHeight / 2, limitSoftColor, 0.95);
            }} else {{
                // Container height is total allowed height including pallet, so wireframe starts from ground.
                addBoxEdges(palletSize, palletSize, heightLimit, heightLimit / 2, limitSoftColor, 0.45);
            }}

            const coilRadius = coilFootprint / 2.0;
            const innerRadius = Math.max(18, coilRadius * 0.56);
            const visualGap = Math.min(10, Math.max(4, Hs * 0.07));
            const rollVisualHeight = Math.max(Hs - visualGap, Hs * 0.90);

            const baseRoll = createRollRealistic(coilRadius, innerRadius, rollVisualHeight, tubeMode);

            for (let i = 0; i < rollCount; i++) {{
                const zc = palletHeight + i * Hs + Hs / 2.0;
                const roll = i === 0 ? baseRoll : baseRoll.clone(true);
                roll.position.set(0, 0, zc);
                packagingGroup.add(roll);

                // Separatore visuale rimosso: i rotoli restano impilati senza disco bianco intermedio.
                // Il piccolo gap è solo grafico e non entra nel calcolo dell'altezza.
            }}

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
                const totalHeight = packagingGroup.userData.totalHeight || 800;
                const sceneSpan = Math.max(palletSize * 1.45, totalHeight * 1.10);
                camera.position.set(-sceneSpan * 1.15, -sceneSpan * 1.25, Math.max(980, totalHeight * 1.15));
                controls.target.set(0, 0, palletHeight + totalHeight * 0.42);
                camera.lookAt(0, 0, palletHeight + totalHeight * 0.42);
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
            padding:12px 14px;
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


def render_section_header(title, subtitle=None, icon=""):
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="section-subtitle">{html.escape(str(subtitle))}</div>'

    st.markdown(
        f"""
        <style>
        .section-header {{
            margin-top:12px;
            margin-bottom:10px;
            padding:14px 16px;
            border-radius:16px;
            border:1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 8px 18px rgba(0,0,0,0.07);
        }}
        .section-title {{
            font-size:18px;
            font-weight:900;
            line-height:1.1;
            display:flex;
            align-items:center;
            gap:8px;
        }}
        .section-subtitle {{
            margin-top:5px;
            font-size:13px;
            line-height:1.28;
            color:color-mix(in srgb, var(--text-color) 66%, transparent);
            font-weight:650;
        }}
        </style>
        <div class="section-header">
            <div class="section-title">{html.escape(icon)} {html.escape(str(title))}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        (size_title, f"{coil_footprint_mm:.1f}", "mm · " + ok_note, size_tone),
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
            grid-template-columns:repeat(3, minmax(0, 1fr));
            gap:12px;
            margin:8px 0 16px 0;
        }}
        .quick-card-v2 {{
            position:relative;
            overflow:hidden;
            border-radius:20px;
            padding:18px 20px 16px 20px;
            min-height:128px;
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
            height:4px;
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











def render_machine_parameter_groups(selected_row, language):
    group_defs = [
        (
            "Tubo e prodotto" if language == "IT" else "Tube and product",
            ["Tipo tubo", "Diametro Rame", "Spessore Guaina (mm)", "Diametro esterno Guaina (mm)",
             "Diametro rame inferiore", "Spessore guaina inferiore", "Diametro rame superiore", "Spessore guaina superiore",
             "Lunghezza (m)"],
        ),
        (
            "Avvolgimento" if language == "IT" else "Winding",
            ["Diametro aspo (mm)", "Spalla (mm)", "Nº Spire", "Passo (mm)", "Incremento strato (mm)",
             "Ritardo invers min (º)", "Ritardo invers max (º)", "Quota massima (mm)", "Quota minima (mm)",
             "Quota start pinza (mm)", "Quota coda tubo (mm)", "Quota chiusura morsa coda (mm)",
             "Interasse regetta (mm)"],
        ),
        (
            "Attrezzaggio linea" if language == "IT" else "Line setup",
            ["Boccole rulliera adrizzatubo", "Boccola uscita rulliera", "Rulliera adrizzatubo",
             "Boccola uscita traino", "Rulli convogliatore (mm)", "Rulli estrusore(mm)",
             "Ruote godronatore", "Soffiatori aria (mm)", "Rulli avvolgitore (mm)",
             "Paleta ferma coda (mm)", "Guidatubo (mm)"],
        ),
        (
            "Velocità e coppie" if language == "IT" else "Speed and torque",
            ["Velocita linea (m/min)", "Velocita recupero (m/min)", "Coppia lavoro (%)",
             "Riduzione coppia (%)", "Coppia recupero (%)"],
        ),
    ]

    search_label = "Cerca parametro" if language == "IT" else "Search parameter"
    search_placeholder = "Es. passo, quota, soffiatori, boccola..." if language == "IT" else "E.g. pitch, quota, blowers, bushing..."
    query = st.text_input(
        search_label,
        value="",
        placeholder=search_placeholder,
        key="machine_param_search",
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

    def matches_query(label, value):
        if not query:
            return True
        haystack = f"{label} {value}".lower()
        return query in haystack

    groups = []
    used = set()

    for group_title, cols in group_defs:
        pairs = []
        for col in cols:
            value = visible_value(col)
            if value is None:
                continue
            label = param_label(col, language)
            used.add(col)
            if matches_query(label, value):
                pairs.append((label, value))
        if pairs:
            groups.append((group_title, pairs))

    extra_pairs = []
    for col in selected_row.index:
        if col in used or str(col).startswith("Unnamed") or col in {"Note", "Prodotto"}:
            continue
        value = visible_value(col)
        if value is None:
            continue
        label = param_label(col, language)
        if matches_query(label, value):
            extra_pairs.append((label, value))

    if extra_pairs:
        groups.append(("Altri parametri" if language == "IT" else "Other parameters", extra_pairs))

    if not groups:
        st.info("Nessun parametro trovato." if language == "IT" else "No parameter found.")
        return

    st.markdown(
        """
        <style>
        .machine-group-head-native{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:14px;
            margin:16px 0 10px 0;
            padding:14px 18px;
            border-radius:18px 18px 0 0;
            background:linear-gradient(90deg, rgba(197,126,90,0.18), transparent);
            border:1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            border-bottom:none;
        }
        .machine-group-title-native{
            font-size:15px;
            font-weight:950;
            letter-spacing:0.065em;
            text-transform:uppercase;
            color:var(--text-color);
        }
        .machine-group-count-native{
            min-width:30px;
            height:30px;
            padding:0 10px;
            border-radius:999px;
            background:#C57E5A;
            color:#fff;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:13px;
            font-weight:950;
        }
        .machine-card-native{
            min-height:104px;
            padding:14px 16px;
            border-radius:16px;
            border:1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--text-color) 4%, transparent),
                color-mix(in srgb, var(--text-color) 7%, transparent)
            );
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            gap:12px;
            box-sizing:border-box;
        }
        .machine-card-label-native{
            font-size:11px;
            line-height:1.22;
            font-weight:820;
            text-transform:uppercase;
            letter-spacing:0.04em;
            color:color-mix(in srgb, var(--text-color) 68%, transparent);
            word-break:break-word;
            min-height:30px;
            display:flex;
            align-items:flex-start;
        }
        .machine-card-value-native{
            font-size:26px;
            line-height:1.02;
            font-weight:950;
            color:var(--text-color);
            letter-spacing:-0.01em;
            word-break:break-word;
            min-height:32px;
            display:flex;
            align-items:flex-end;
        }
        @media (max-width: 900px){
            .machine-card-native{
                min-height:80px;
            }
            .machine-card-value-native{
                font-size:22px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for group_title, pairs in groups:
        st.markdown(
            f"""
            <div class="machine-group-head-native">
                <div class="machine-group-title-native">{html.escape(str(group_title))}</div>
                <div class="machine-group-count-native">{len(pairs)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            cols_per_row = 4

            for i in range(0, len(pairs), cols_per_row):
                row_pairs = pairs[i:i + cols_per_row]
                cols = st.columns(cols_per_row, gap="medium")
                for col_ui, pair in zip(cols, row_pairs):
                    label, value = pair
                    with col_ui:
                        st.markdown(
                            f"""
                            <div class="machine-card-native">
                                <div class="machine-card-label-native">{html.escape(str(label))}</div>
                                <div class="machine-card-value-native">{html.escape(str(value))}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                # leave remaining columns empty if row_pairs shorter


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
        <div class="semaphore-card {html.escape(tone)}">
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

    st.markdown(
        f"""
        <style>
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
        </style>
        <div class="checklist-hero">
            <div class="checklist-hero-head">
                <div class="checklist-title">{html.escape(title)}</div>
                <div class="checklist-subtitle">{html.escape(subtitle)}</div>
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

    for area_name, sections in groups:
        st.markdown(f'<div class="checklist-area-title">{html.escape(area_name)}</div>', unsafe_allow_html=True)
        area_cols = st.columns(len(sections), gap="large")
        for col_ui, (section_name, items) in zip(area_cols, sections):
            with col_ui:
                st.markdown(
                    f'<div class="checklist-section-title">{html.escape(section_name)}</div>',
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    for item in items:
                        key_base = f"{area_name}_{section_name}_{item}".lower()
                        key = "check_cambio_" + "".join(ch if ch.isalnum() else "_" for ch in key_base)
                        st.checkbox(item, key=key)









def render_elegant_panel_open(title=None, subtitle=None, tag=None):
    title_html = f'<div class="elegant-panel-title">{html.escape(str(title))}</div>' if title else ""
    subtitle_html = f'<div class="elegant-panel-subtitle">{html.escape(str(subtitle))}</div>' if subtitle else ""
    tag_html = f'<div class="elegant-panel-tag">{html.escape(str(tag))}</div>' if tag else ""

    st.markdown(
        f"""
        <style>
        .elegant-panel {{
            margin:8px 0 18px 0;
            border-radius:24px;
            overflow:hidden;
            border:1px solid color-mix(in srgb, var(--text-color) 11%, transparent);
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            box-shadow:0 10px 26px rgba(0,0,0,0.07);
        }}
        .elegant-panel-head {{
            display:flex;
            align-items:flex-start;
            justify-content:space-between;
            gap:16px;
            padding:15px 18px 13px 18px;
            border-bottom:1px solid color-mix(in srgb, var(--text-color) 8%, transparent);
            background:linear-gradient(90deg, rgba(197,126,90,0.08), transparent 58%);
        }}
        .elegant-panel-title {{
            font-size:15px;
            line-height:1.15;
            font-weight:950;
            letter-spacing:0.055em;
            text-transform:uppercase;
            color:var(--text-color);
        }}
        .elegant-panel-subtitle {{
            margin-top:4px;
            font-size:12px;
            line-height:1.25;
            font-weight:650;
            color:color-mix(in srgb, var(--text-color) 62%, transparent);
        }}
        .elegant-panel-tag {{
            flex:0 0 auto;
            border-radius:999px;
            padding:6px 10px;
            font-size:11px;
            line-height:1;
            font-weight:900;
            letter-spacing:0.065em;
            text-transform:uppercase;
            border:1px solid rgba(197,126,90,0.20);
            background:rgba(197,126,90,0.08);
            color:color-mix(in srgb, var(--text-color) 75%, transparent);
        }}
        .elegant-panel-body {{
            padding:14px;
        }}
        </style>
        <div class="elegant-panel">
            <div class="elegant-panel-head">
                <div>{title_html}{subtitle_html}</div>
                {tag_html}
            </div>
            <div class="elegant-panel-body">
        """,
        unsafe_allow_html=True,
    )


def render_elegant_panel_close():
    st.markdown("</div></div>", unsafe_allow_html=True)

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

tab_production, tab_tech_sheet, tab_checklist = st.tabs([
    production_label,
    tech_sheet_label,
    checklist_label,
])

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
    st.markdown(f"### {production_label}")
    render_workflow_bar(lang)

    render_section_header(
        "Selezione prodotto" if lang == "IT" else "Product selection",
        "Scegli un preset: i valori si caricano automaticamente nel render." if lang == "IT" else "Choose a preset: values are automatically loaded into the render.",
        "①",
    )

    top_left, top_right = st.columns([1.6, 1.0], gap="large")

    with top_left:
        selected_product = st.selectbox(
            t["select_product"],
            preset_names,
            key="selected_preset_product",
        )

    selected_row = presets_df[presets_df["Prodotto"] == selected_product].iloc[0]

    # Auto-load preset when product changes. This removes the old "select + load + switch tab" flow.
    last_auto_loaded = st.session_state.get("last_auto_loaded_preset")
    if last_auto_loaded != selected_product:
        apply_preset_to_calculator(selected_row)
        st.session_state["last_auto_loaded_preset"] = selected_product
        st.session_state["loaded_preset_name"] = selected_product
        st.session_state["show_preset_loaded_success"] = False

    # If the user changes any calculator value manually, the selected preset becomes only a base.
    sync_active_preset_state()
    preset_modified = st.session_state.get("loaded_preset_name") != selected_product

    with top_right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        render_active_preset_card(selected_product, lang, modified=preset_modified)

    render_preset_summary_strip(selected_product, selected_row, lang, modified=preset_modified)

    render_section_header(
        "Parametri principali" if lang == "IT" else "Main parameters",
        "Solo i valori essenziali per simulare e verificare l’avvolgimento." if lang == "IT" else "Only the essential values for simulating and checking the winding.",
        "②",
    )

    colA, colB, colC = st.columns([0.95, 1.25, 1.0], gap="large")

    with colA:
        st.markdown(f"**{t['bobina']}**")
        diametro_aspo = st.number_input(t["diam_aspo"], step=10.0, key="calc_diametro_aspo")
        spalla = st.number_input(t["spalla"], step=1.0, key="calc_spalla")

    with colB:
        st.markdown(f"**{t['tubo']}**")
        rame_options = list(COPPER_SIZES_MM.keys())

        tube_layout_label = st.radio(
            "Tipo tubo",
            ["Singolo", "Doppio"],
            horizontal=True,
            key="calc_tube_layout",
        )

        if tube_layout_label == "Singolo":
            if st.session_state.get("calc_rame") not in rame_options:
                st.session_state["calc_rame"] = "1/4"
            rame = st.selectbox(t["rame"], rame_options, key="calc_rame")
            spessore = st.number_input(t["isolamento"], step=1.0, key="calc_spessore")
            lunghezza = st.number_input(t["lunghezza"], step=5.0, key="calc_lunghezza")

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
                rame_inf = st.selectbox("Rame inferiore", rame_options, key="calc_rame_inf")
                spessore_inf = st.number_input("Guaina inferiore (mm)", step=1.0, key="calc_spessore_inf")

            with c_sup:
                if st.session_state.get("calc_rame_sup") not in rame_options:
                    st.session_state["calc_rame_sup"] = "1/4"
                rame_sup = st.selectbox("Rame superiore", rame_options, key="calc_rame_sup")
                spessore_sup = st.number_input("Guaina superiore (mm)", step=1.0, key="calc_spessore_sup")

            lunghezza = st.number_input(t["lunghezza"], step=5.0, key="calc_lunghezza")

            d_tubo_lower = COPPER_SIZES_MM[rame_inf] + 2.0 * spessore_inf
            d_tubo_upper = COPPER_SIZES_MM[rame_sup] + 2.0 * spessore_sup

            d_tubo_sim = max(d_tubo_lower, d_tubo_upper)
            d_tubo = d_tubo_sim
            d_tubo_footprint = d_tubo_sim

            tube_layout_code = "double"
            tube_diameter_label = f"Inferiore {d_tubo_lower:.2f} / Superiore {d_tubo_upper:.2f} mm"
            passo_consigliato = d_tubo_lower + d_tubo_upper
            incremento_consigliato = max(d_tubo_lower, d_tubo_upper)

    with colC:
        st.markdown(f"**{t['avvolg']}**")
        passo_visuale = st.number_input(t["passo_assiale"], step=0.5, key="calc_passo_visuale")
        incremento_visuale = st.number_input(t["incremento"], step=0.5, key="calc_incremento_visuale")
        rit_b = st.number_input(t["rit_min"], step=1.0, key="calc_rit_b")
        rit_t = st.number_input(t["rit_max"], step=1.0, key="calc_rit_t")

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

    visual_metrics = compute_metrics(local_points, d_tubo_footprint)

    coil_footprint_for_status = float(visual_metrics["max_xy_span"])
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
                "note": "Preset caricato e parametri principali presenti",
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
                "note": "Preset loaded and main parameters present",
                "tone": "ok" if machine_complete else "warn",
            },
        ]

    render_status_semaphore(status_items, lang)

    st.divider()

    render_section_header(
        "Render 3D" if lang == "IT" else "3D render",
        "Usa Avvolgimento per controllare la bobina e Packaging per verificare pallet/scatola/torretta." if lang == "IT" else "Use Winding to check the coil and Packaging to verify pallet/box/tower.",
        "③",
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

    render_elegant_panel_open(
        "Render 3D" if lang == "IT" else "3D render",
        "Vista integrata per controllare avvolgimento, packaging e ingombri." if lang == "IT" else "Integrated view to check winding, packaging and footprint.",
        selected_product,
    )

    components.html(
        viewer(
            diametro_aspo,
            spalla,
            d_tubo,
            660,
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
            initial_scene="packaging" if view_mode == t["scene_packaging"] else "winding",
            packaging_mode=packaging_mode_selected,
            container_mode=container_mode_selected,
            pack_roll_count=pack_roll_count,
            tube_layout=tube_layout_code,
            d_tubo_lower=d_tubo_lower,
            d_tubo_upper=d_tubo_upper,
            tube_diameter_label=tube_diameter_label,
        ),
        height=660,
    )

    render_elegant_panel_close()

    st.divider()

    render_section_header(
        "Risultati" if lang == "IT" else "Results",
        "Prima una lettura rapida, poi il dettaglio tecnico se serve." if lang == "IT" else "First a quick reading, then the technical detail if needed.",
        "④",
    )

    pallet_size_mm = 750.0
    coil_footprint_mm = float(visual_metrics["max_xy_span"])

    render_quick_reading(
        lang,
        tube_layout_code,
        tube_diameter_label,
        passo_visuale,
        incremento_visuale,
        visual_metrics,
        coil_footprint_mm,
        pallet_size_mm,
    )

    detail_label = "Dettaglio tecnico" if lang == "IT" else "Technical detail"
    with st.expander(detail_label, expanded=False):
        render_summary_cards(
                t["results"],
            [
                {"label": t["metric1"], "value": tube_diameter_label, "note": ("Configurazione verticale" if tube_layout_code == "double" else "")},
                {"label": t["metric2"], "value": f"{passo_visuale:.2f} mm"},
                {"label": t["metric3"], "value": f"{incremento_visuale:.2f} mm"},
                {"label": t["metric4"], "value": f"{visual_metrics['diam_radiale']:.1f} mm"},
                {"label": t["metric5"], "value": f"{visual_metrics['max_xy_span']:.1f} mm"},
                {"label": t["metric6"], "value": f"{visual_metrics['wound_length_m']:.3f} m"},
            ],
            cards_per_row=3,
        )

    if coil_footprint_mm > pallet_size_mm:
        st.warning(t["warning"])

with tab_tech_sheet:
    st.markdown(f"### {tech_sheet_label}")

    render_section_header(
        "Consultazione preset" if lang == "IT" else "Preset reference",
        "Qui trovi anteprima, lettura rapida e parametri completi del CSV in una vista più ordinata." if lang == "IT" else "Here you find preview, quick reading and full CSV parameters in a cleaner layout.",
        "ⓘ",
    )

    selected_product = st.session_state.get("selected_preset_product", preset_names[0])
    selected_row = presets_df[presets_df["Prodotto"] == selected_product].iloc[0]

    st.markdown(
        f"""
        <div style="
            margin-top:12px;
            margin-bottom:18px;
            padding:22px 24px;
            border-radius:18px;
            background:linear-gradient(180deg,
                color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color)),
                color-mix(in srgb, var(--secondary-background-color) 98%, var(--background-color))
            );
            border:1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            box-shadow:0 8px 22px rgba(0,0,0,0.08);
            border-left:6px solid #C57E5A;
        ">
            <div style="font-size:13px; color:color-mix(in srgb, var(--text-color) 62%, transparent); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; font-weight:700;">
                {t["preset_sheet"]}
            </div>
            <div style="font-size:30px; font-weight:800; color:var(--text-color); line-height:1.15;">
                {selected_product}
            </div>
            <div style="font-size:14px; color:color-mix(in srgb, var(--text-color) 68%, transparent); margin-top:8px;">
                {t["preset_subtitle"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_tech_snapshot_cards(selected_row, lang)

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
        components.html(make_preset_visual(selected_row, lang), height=470, scrolling=False)

    with machine_sheet_tab:
        render_section_header(
            "Scheda parametri macchina" if lang == "IT" else "Machine parameter sheet",
            "Vista unica ordinata per introdurre tutti i valori in macchina senza separarli tra render e consultivi." if lang == "IT" else "A single grouped view for entering all values into the machine.",
            "B",
        )

        render_machine_parameter_groups(selected_row, lang)

with tab_checklist:
    render_startup_checklist(lang)

