import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
# =========================

if "lang" not in st.session_state:
    st.session_state.lang = "IT"

lang_option = st.selectbox(
    "🌍 Language",
    ["🇮🇹 Italiano", "🇺🇸 English (US)"],
    index=0 if st.session_state.lang == "IT" else 1
)

st.session_state.lang = "IT" if "Italiano" in lang_option else "EN"
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
        "gradi_start": "Gradi iniziali (°)",
        "pinza": "Lunghezza pinza (m)",
        "altezza": "Altezza",
        "animazione": "Animazione",
        "velocita": "Velocità",
        "metric1": "Diametro tubo",
        "metric2": "Passo assiale",
        "metric3": "Incremento strato",
        "metric4": "Diametro esterno",
        "warning": "⚠️ Diametro esterno superiore a 750 mm."
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
        "gradi_start": "Initial degrees (°)",
        "pinza": "Clamp length (m)",
        "altezza": "Height",
        "animazione": "Animation",
        "velocita": "Speed",
        "metric1": "Tube diameter",
        "metric2": "Axial pitch",
        "metric3": "Layer increment",
        "metric4": "Outer diameter",
        "warning": "⚠️ Outer diameter exceeds 750 mm."
    }
}

t = TEXTS[lang]

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1, 7])
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

def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def trim_polyline(points: np.ndarray, target_length: float) -> np.ndarray:
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
    p_trim = p0 + alpha * (p1 - p0)

    return np.vstack([points[:idx + 1], p_trim])

def compute_total_turns(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    theta = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    return float(np.sum(np.abs(np.diff(theta))) / (2 * np.pi))

def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

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
    gradi_start_deg,
    lunghezza_pinza_m,
):
    lunghezza_totale_mm = float(lunghezza_m) * 1000.0
    lunghezza_pinza_mm = max(0.0, float(lunghezza_pinza_m) * 1000.0)
    lunghezza_visibile_mm = max(0.0, lunghezza_totale_mm - lunghezza_pinza_mm)

    d_tubo = float(d_rame_mm) + 2.0 * float(spessore_guaina_mm)
    r_tubo = d_tubo / 2.0

    passo_assiale = max(float(passo_assiale), EPS)
    passo_radiale = max(float(passo_radiale), EPS)
    spalla_mm = max(float(spalla_mm), d_tubo + EPS)

    ritardo_bottom_deg = max(0.0, float(ritardo_min_deg))
    ritardo_top_deg = max(0.0, float(ritardo_max_deg))
    gradi_start_deg = max(0.0, float(gradi_start_deg))

    # Centerline limits so wall touches base/spalla
    z_min = r_tubo
    z_max = spalla_mm - r_tubo

    # First centerline radius so wall is tangent to mandrel
    r0 = d_aspo_mm / 2.0 + r_tubo

    # Angular resolution
    theta_step_deg = 4.0
    theta_step = np.deg2rad(theta_step_deg)

    # Clockwise winding => theta decreases
    theta_sign = -1.0

    dz_per_rad = passo_assiale / (2.0 * np.pi)

    theta = 0.0
    z = z_min
    r = r0

    # +1 = towards spalla, -1 = towards base
    z_dir = +1

    points = []
    state_marks = []

    def add_point(theta_val, r_val, z_val, state_name):
        x = r_val * np.cos(theta_val)
        y = r_val * np.sin(theta_val)
        points.append([x, y, z_val])
        state_marks.append(state_name)

    add_point(theta, r, z, "start")

    # Initial start degrees: keep at base with clockwise rotation
    if gradi_start_deg > EPS and lunghezza_visibile_mm > EPS:
        n0 = max(4, int(np.ceil(gradi_start_deg / theta_step_deg)))
        step0 = np.deg2rad(gradi_start_deg) / n0

        for _ in range(n0):
            theta += theta_sign * step0
            add_point(theta, r, z_min, "start_wrap")
            if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
                break

    while True:
        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
            break

        # =========================
        # NORMAL HELICAL WINDING
        # =========================
        while True:
            theta += theta_sign * theta_step
            z += z_dir * dz_per_rad * theta_step

            if z_dir == +1 and z >= z_max:
                z = z_max
                add_point(theta, r, z, "reach_spalla")
                break

            if z_dir == -1 and z <= z_min:
                z = z_min
                add_point(theta, r, z, "reach_base")
                break

            add_point(theta, r, z, "helical")

            if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
                break

        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visible_mm:
            break

        # =========================
        # RADIAL SHIFT WITH Z FIXED
        # =========================
        at_spalla = (z_dir == +1)
        z_const = z_max if at_spalla else z_min
        ritardo_deg = ritardo_top_deg if at_spalla else ritardo_bottom_deg

        r_old = r
        r_new = r_old + passo_radiale

        if ritardo_deg > EPS:
            n_shift = max(8, int(np.ceil(ritardo_deg / theta_step_deg)))
            step_shift = np.deg2rad(ritardo_deg) / n_shift

            for i in range(1, n_shift + 1):
                theta += theta_sign * step_shift
                u = smoothstep01(i / n_shift)
                r_curr = r_old + (r_new - r_old) * u
                add_point(theta, r_curr, z_const, "radial_shift")

                if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
                    break
        else:
            # Instant radial change, still keep a point at the edge
            r = r_new
            add_point(theta, r, z_const, "radial_shift_instant")

        r = r_new

        if len(points) > 2 and polyline_length(np.array(points, dtype=float)) >= lunghezza_visibile_mm:
            break

        # Reverse axial direction after layer change
        z_dir *= -1

    path = np.array(points, dtype=float)

    if lunghezza_visibile_mm > EPS and len(path) > 1:
        path = trim_polyline(path, lunghezza_visibile_mm)

    # Center coil vertically in viewer
    if len(path) > 0:
        path[:, 2] -= spalla_mm / 2.0

    r_path = np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2)
    r_max = float(np.max(r_path)) if len(r_path) > 0 else r0
    diam_ext = 2.0 * (r_max + r_tubo)

    meta = {
        "DiametroTubo": d_tubo,
        "PassoAssiale": passo_assiale,
        "IncrementoStrato": passo_radiale,
        "DiametroEsterno": diam_ext,
        "MandrelHeight": spalla_mm,
        "Rmax": r_max,
        "R0": r0,
        "Turns": compute_total_turns(path),
    }

    return path, meta

# =========================
# VIEWER HTML BUILDER
# =========================

def build_viewer_html(points, d_tubo, viewer_height, animazione, velocita, d_aspo_mm, spalla_mm, r_max_mm):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    r_mandrel = d_aspo_mm / 2.0
    flange_radius = max(r_max_mm + d_tubo * 1.25, r_mandrel + 55.0)

    # Keep it lighter
    tubular_segments = min(3200, max(700, int(len(pts) * 0.42)))
    radial_segments = 18

    anim_js = "true" if animazione else "false"
    init_progress = "0.0" if animazione else "1.0"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #000;
    }}
    #viewer {{
      width: 100%;
      height: {viewer_height}px;
      display: block;
      overflow: hidden;
    }}
  </style>
</head>
<body>
  <div id="viewer"></div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

  <script>
    function initViewer() {{
      const container = document.getElementById("viewer");
      if (!container) return;

      const width = container.offsetWidth || 900;
      const height = {viewer_height};

      if (width < 10 || height < 10) {{
        setTimeout(initViewer, 80);
        return;
      }}

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000);

      const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100000);
      camera.up.set(0, 0, 1);
      camera.position.set(760, -1050, 300);
      camera.lookAt(0, 0, 0);

      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
      renderer.setSize(width, height);
      container.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.target.set(0, 0, 0);

      scene.add(new THREE.HemisphereLight(0xffffff, 0x202020, 0.92));

      const light1 = new THREE.DirectionalLight(0xffffff, 0.60);
      light1.position.set(900, -900, 1200);
      scene.add(light1);

      const light2 = new THREE.DirectionalLight(0xffffff, 0.24);
      light2.position.set(-1100, 650, 750);
      scene.add(light2);

      const rawPoints = {points_json};
      const vectors = rawPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));

      if (vectors.length < 2) return;

      class CurvePath extends THREE.Curve {{
        constructor(pts) {{
          super();
          this.pts = pts;
        }}
        getPoint(t) {{
          const n = this.pts.length;
          if (n === 1) return this.pts[0].clone();
          const f = t * (n - 1);
          const i = Math.floor(f);
          const i0 = Math.max(0, Math.min(i, n - 2));
          const i1 = i0 + 1;
          const tt = f - i0;
          return new THREE.Vector3().lerpVectors(this.pts[i0], this.pts[i1], tt);
        }}
      }}

      const coilGroup = new THREE.Group();
      scene.add(coilGroup);

      const curve = new CurvePath(vectors);

      let tubeGeom = new THREE.TubeGeometry(
        curve,
        {tubular_segments},
        {r_tubo},
        {radial_segments},
        false
      );
      tubeGeom = tubeGeom.toNonIndexed();

      const tubeMat = new THREE.MeshStandardMaterial({{
        color: 0xe4e4e4,
        roughness: 0.86,
        metalness: 0.06
      }});

      const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
      coilGroup.add(tubeMesh);

      const mandrelHeight = {spalla_mm};

      const mandrelGeom = new THREE.CylinderGeometry({r_mandrel}, {r_mandrel}, mandrelHeight, 64, 1, false);
      const mandrelMat = new THREE.MeshStandardMaterial({{
        color: 0x5e5e5e,
        roughness: 0.72,
        metalness: 0.45,
        transparent: true,
        opacity: 0.42
      }});
      const mandrelMesh = new THREE.Mesh(mandrelGeom, mandrelMat);
      mandrelMesh.rotation.x = Math.PI / 2;
      scene.add(mandrelMesh);

      const hubRadius = Math.max({r_mandrel} * 0.35, 28.0);
      const hubGeom = new THREE.CylinderGeometry(hubRadius, hubRadius, mandrelHeight, 42);
      const hubMat = new THREE.MeshStandardMaterial({{
        color: 0x2f5fa6,
        roughness: 0.82,
        metalness: 0.16
      }});
      const hubMesh = new THREE.Mesh(hubGeom, hubMat);
      hubMesh.rotation.x = Math.PI / 2;
      scene.add(hubMesh);

      const flangeRadius = {flange_radius};
      const flangeThickness = 6.0;
      const flangeMat = new THREE.MeshStandardMaterial({{
        color: 0x2f6ab8,
        roughness: 0.86,
        metalness: 0.14
      }});

      const flangeGeom = new THREE.CylinderGeometry(flangeRadius, flangeRadius, flangeThickness, 84);

      const baseMesh = new THREE.Mesh(flangeGeom, flangeMat);
      baseMesh.rotation.x = Math.PI / 2;
      baseMesh.position.z = -mandrelHeight / 2 - flangeThickness / 2;
      scene.add(baseMesh);

      const topMesh = new THREE.Mesh(flangeGeom, flangeMat);
      topMesh.rotation.x = Math.PI / 2;
      topMesh.position.z = mandrelHeight / 2 + flangeThickness / 2;
      scene.add(topMesh);

      function createCap(position, direction, color) {{
        const geom = new THREE.CircleGeometry({r_tubo}, 24);
        const mat = new THREE.MeshBasicMaterial({{
          color: color,
          side: THREE.DoubleSide
        }});
        const cap = new THREE.Mesh(geom, mat);

        const up = new THREE.Vector3(0, 0, 1);
        const dir = direction.clone().normalize();
        if (dir.length() > 1e-9) {{
          cap.quaternion.setFromUnitVectors(up, dir);
        }}
        cap.position.copy(position);
        return cap;
      }}

      const startCap = createCap(
        vectors[0],
        vectors[1].clone().sub(vectors[0]).multiplyScalar(-1),
        0x00ff00
      );
      coilGroup.add(startCap);

      const endCap = createCap(
        vectors[vectors.length - 1],
        vectors[vectors.length - 1].clone().sub(vectors[vectors.length - 2]),
        0xff0000
      );
      coilGroup.add(endCap);

      // Guide column at left
      const guideColumnX = -(flangeRadius + 120);
      const guideColumnHeight = mandrelHeight + 180;

      const guideColumnGeom = new THREE.BoxGeometry(18, 18, guideColumnHeight);
      const guideColumnMat = new THREE.MeshStandardMaterial({{
        color: 0x5b5b5b,
        roughness: 0.78,
        metalness: 0.35
      }});
      const guideColumn = new THREE.Mesh(guideColumnGeom, guideColumnMat);
      guideColumn.position.x = guideColumnX;
      scene.add(guideColumn);

      const guideGroup = new THREE.Group();
      scene.add(guideGroup);

      const carriageGeom = new THREE.BoxGeometry(30, 24, 24);
      const carriageMat = new THREE.MeshStandardMaterial({{
        color: 0xc8c8c8,
        roughness: 0.82,
        metalness: 0.18
      }});
      const carriageMesh = new THREE.Mesh(carriageGeom, carriageMat);
      guideGroup.add(carriageMesh);

      const guideArmLen = Math.abs(guideColumnX) - 44;
      const armGeom = new THREE.BoxGeometry(guideArmLen, 12, 12);
      const armMat = new THREE.MeshStandardMaterial({{
        color: 0x989898,
        roughness: 0.74,
        metalness: 0.28
      }});
      const armMesh = new THREE.Mesh(armGeom, armMat);
      armMesh.position.x = guideArmLen / 2;
      guideGroup.add(armMesh);

      const guideNozzleLen = 34;
      const nozzleRadius = Math.max({r_tubo} * 0.95, 4.5);
      const nozzleGeom = new THREE.CylinderGeometry(nozzleRadius, nozzleRadius * 0.9, guideNozzleLen, 22);
      const nozzleMat = new THREE.MeshStandardMaterial({{
        color: 0xb0b0b0,
        roughness: 0.62,
        metalness: 0.42
      }});
      const nozzleMesh = new THREE.Mesh(nozzleGeom, nozzleMat);
      nozzleMesh.rotation.z = Math.PI / 2;
      nozzleMesh.position.x = guideArmLen + guideNozzleLen / 2;
      guideGroup.add(nozzleMesh);

      // Straight feed tube between guide nozzle and tangent/deposition point
      const feedGeom = new THREE.CylinderGeometry({r_tubo}, {r_tubo}, 1, 18, 1, false);
      const feedMesh = new THREE.Mesh(feedGeom, tubeMat);
      scene.add(feedMesh);

      function setCylinderBetween(mesh, p0, p1) {{
        const dir = new THREE.Vector3().subVectors(p1, p0);
        const len = dir.length();
        if (len < 1e-6) {{
          mesh.visible = false;
          return;
        }}
        mesh.visible = true;
        mesh.position.copy(new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5));
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        mesh.scale.set(1, len, 1);
      }}

      const total = tubeGeom.attributes.position.count;
      let progress = {init_progress};
      const isAnimating = {anim_js};

      tubeGeom.setDrawRange(0, isAnimating ? 0 : total);

      // Keep actual contact visually on the left side
      const thetaContact = -Math.PI / 2;

      function getCurrentPoint(t) {{
        return curve.getPoint(Math.max(0.0005, Math.min(1.0, t)));
      }}

      function updateMachine(t) {{
        const pLocal = getCurrentPoint(t);

        // rotate the wound part so active point stays on left
        const thetaCurrent = Math.atan2(pLocal.y, pLocal.x);
        coilGroup.rotation.z = thetaContact - thetaCurrent;
        coilGroup.updateMatrixWorld(true);

        // active deposition point in world
        const pWorld = coilGroup.localToWorld(pLocal.clone());
        const rCurrent = Math.sqrt(pWorld.x * pWorld.x + pWorld.y * pWorld.y);

        // guidatubo follows radial + axial position
        guideGroup.position.set(guideColumnX, -rCurrent, pWorld.z);

        const outlet = new THREE.Vector3(guideArmLen + guideNozzleLen, 0, 0);
        guideGroup.localToWorld(outlet);

        // contact point stays on left side at current radius and z
        const tangentPoint = new THREE.Vector3(0, -rCurrent, pWorld.z);
        setCylinderBetween(feedMesh, outlet, tangentPoint);
      }}

      updateMachine(progress);

      // Cosmetic rotation of mandrel/flanges/hub
      let mandrelSpin = 0.0;

      function animate() {{
        requestAnimationFrame(animate);

        if (isAnimating && progress < 1.0) {{
          progress += {velocita} * 0.002;
          if (progress > 1.0) progress = 1.0;
          tubeGeom.setDrawRange(0, Math.max(2, Math.floor(progress * total)));
          mandrelSpin -= {velocita} * 0.06; // clockwise
        }}

        mandrelMesh.rotation.z = mandrelSpin;
        hubMesh.rotation.z = mandrelSpin;
        baseMesh.rotation.z = mandrelSpin;
        topMesh.rotation.z = mandrelSpin;

        updateMachine(progress);
        controls.update();
        renderer.render(scene, camera);
      }}

      animate();

      window.addEventListener("resize", () => {{
        const newWidth = container.offsetWidth || 900;
        camera.aspect = newWidth / height;
        camera.updateProjectionMatrix();
        renderer.setSize(newWidth, height);
      }});
    }}

    if (document.readyState === "complete" || document.readyState === "interactive") {{
      setTimeout(initViewer, 50);
    }} else {{
      document.addEventListener("DOMContentLoaded", () => setTimeout(initViewer, 50));
    }}
  </script>
</body>
</html>
"""
    return html

# =========================
# UI
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"#### {t['bobina']}")
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0, step=1.0)
    spalla = st.number_input(t["spalla"], value=95.0, step=1.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame_label = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore_guaina = st.number_input(t["isolamento"], value=7.0, step=0.1)
    lunghezza = st.number_input(t["lunghezza"], value=50.0, step=1.0)
    d_rame = COPPER_SIZES_MM[rame_label]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo_assiale = st.number_input(t["passo_assiale"], value=20.0, step=0.1)
    incremento_strato = st.number_input(t["incremento"], value=20.0, step=0.1)
    ritardo_min = st.number_input(t["rit_min"], min_value=0.0, max_value=720.0, value=180.0, step=1.0)
    ritardo_max = st.number_input(t["rit_max"], min_value=0.0, max_value=720.0, value=180.0, step=1.0)
    gradi_start = st.number_input(t["gradi_start"], min_value=0.0, max_value=720.0, value=30.0, step=1.0)
    lunghezza_pinza = st.number_input(t["pinza"], min_value=0.0, value=0.30, step=0.01)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    animazione = st.checkbox(t["animazione"], False)
    velocita = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD & RENDER
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
    gradi_start,
    lunghezza_pinza,
)

html = build_viewer_html(
    path,
    meta["DiametroTubo"],
    altezza,
    animazione,
    velocita,
    diametro_aspo,
    spalla,
    meta["Rmax"]
)

components.html(html, height=altezza + 20)

# =========================
# METRICS
# =========================

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric(t["metric1"], f"{meta['DiametroTubo']:.2f} mm")
m2.metric(t["metric2"], f"{meta['PassoAssiale']:.2f} mm")
m3.metric(t["metric3"], f"{meta['IncrementoStrato']:.2f} mm")
m4.metric(t["metric4"], f"{meta['DiametroEsterno']:.1f} mm")

if meta["DiametroEsterno"] > 750:
    st.warning(t["warning"])
