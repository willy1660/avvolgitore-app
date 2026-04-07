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

# =========================
# UTILS
# =========================

def polyline_length(points):
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def trim_polyline(points, target_length_mm):
    if len(points) < 2:
        return points

    pts = np.asarray(points, dtype=float)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])

    if cum[-1] <= target_length_mm:
        return pts

    idx = np.searchsorted(cum, target_length_mm, side="right") - 1
    idx = max(0, min(idx, len(segs) - 1))

    remain = target_length_mm - cum[idx]
    if segs[idx] < 1e-9:
        extra = pts[idx + 1]
    else:
        alpha = remain / segs[idx]
        extra = pts[idx] + alpha * (pts[idx + 1] - pts[idx])

    out = np.vstack([pts[:idx + 1], extra])
    return out


# =========================
# GEOMETRY
# Governada per aspo + guidatubo
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo, incremento, rit_b, rit_t, gradi_start, pinza):
    d_tubo = d_rame + 2 * spessore
    r_start = d_aspo / 2 + d_tubo / 2

    z_min = -spalla / 2
    z_max = spalla / 2

    deg_step = 4.0
    rad_step = np.deg2rad(deg_step)

    theta = np.deg2rad(gradi_start)
    r_layer = r_start
    z_guide = z_min
    direction = 1.0
    delay_deg = 0.0
    pending_layer_jump = False

    pts = []

    axial_per_rad = passo / (2 * np.pi)

    for _ in range(40000):
        # rotació imposada per l’aspo
        theta += rad_step

        # moviment axial imposat pel guidatubo
        if delay_deg > 0:
            delay_deg -= deg_step
            if delay_deg < 0:
                delay_deg = 0
        else:
            if pending_layer_jump:
                r_layer += incremento
                pending_layer_jump = False

            z_guide += direction * axial_per_rad * rad_step

            if z_guide >= z_max:
                z_guide = z_max
                delay_deg = rit_t
                pending_layer_jump = True
                direction = -1.0

            elif z_guide <= z_min:
                z_guide = z_min
                delay_deg = rit_b
                pending_layer_jump = True
                direction = 1.0

        # el tub queda definit per la rotació + posició del guidatubo
        x = r_layer * np.cos(theta)
        y = r_layer * np.sin(theta)
        z = z_guide

        pts.append([x, y, z])

        if len(pts) > 2:
            if polyline_length(np.array(pts)) >= lunghezza * 1000.0:
                break

    pts = np.array(pts, dtype=float)
    pts = trim_polyline(pts, lunghezza * 1000.0)

    # petit tram inicial recte, coherent amb el guidatubo a l'esquerra
    # es fa servir pinza com a longitud màxima d'entrada visible
    if len(pts) >= 2:
        p0 = pts[0].copy()
        z0 = p0[2]
        x_tan = -np.sqrt(max((np.linalg.norm(p0[:2]) ** 2) - (p0[1] ** 2), 0.0))
        p_tan = np.array([x_tan, p0[1], z0], dtype=float)

        nozzle_x = -d_aspo / 2 - max(120.0, d_tubo * 3.0)
        p_nozzle = np.array([nozzle_x, p_tan[1], z0], dtype=float)

        lead = np.vstack([p_nozzle, p_tan])

        lead_len = polyline_length(lead)
        max_lead = max(80.0, pinza * 1000.0)
        if lead_len > max_lead and lead_len > 1e-9:
            alpha = max_lead / lead_len
            p_nozzle_trim = p_tan + alpha * (p_nozzle - p_tan)
            lead = np.vstack([p_nozzle_trim, p_tan])

        pts = np.vstack([lead, pts])

    return pts


# =========================
# VIEWER
# =========================

def viewer(points, d_aspo, spalla, d_tubo, altezza, anim, vel):
    points_js = json.dumps(np.asarray(points, dtype=float).tolist())
    anim_js = "true" if anim else "false"

    return f"""
    <div id="viewer-wrap" style="width:100%;height:{altezza}px;background:#dfe3ea;overflow:hidden;position:relative;">
        <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const oldCanvas = document.querySelector("#viewer canvas");
        if (oldCanvas) oldCanvas.remove();

        const container = document.getElementById("viewer");
        const W = container.clientWidth || 1200;
        const H = container.clientHeight || {altezza};

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xdfe3ea);

        const camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 10000);
        camera.position.set(-420, -980, 250);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, H);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(40, 0, 0);

        // =====================
        // DATA
        // =====================

        const ptsArr = {points_js};
        const pts = ptsArr.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const dAspo = {float(d_aspo)};
        const spalla = {float(spalla)};
        const dTubo = {float(d_tubo)};
        const rMandrel = dAspo / 2.0;
        const rTube = dTubo / 2.0;

        const zMin = -spalla / 2.0;
        const zMax =  spalla / 2.0;

        const flangeR = Math.max(260, rMandrel + 120);
        const flangeTh = 6;
        const hubR = rMandrel;
        const hubLen = spalla;

        // =====================
        // LIGHTS
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.9));

        const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
        dir1.position.set(-600, -500, 700);
        scene.add(dir1);

        const dir2 = new THREE.DirectionalLight(0xffffff, 0.45);
        dir2.position.set(500, 300, 400);
        scene.add(dir2);

        // =====================
        // MACHINE GROUP
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        // =====================
        // ASPO VERMELL
        // =====================

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0xd92d2d,
            roughness: 0.55,
            metalness: 0.08
        }});

        const hub = new THREE.Mesh(
            new THREE.CylinderGeometry(hubR, hubR, hubLen, 64),
            redMat
        );
        hub.rotation.x = Math.PI / 2;
        machine.add(hub);

        const lowerFlange = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 64),
            redMat
        );
        lowerFlange.rotation.x = Math.PI / 2;
        lowerFlange.position.z = zMin - flangeTh / 2;
        machine.add(lowerFlange);

        const upperFlange = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 64),
            redMat
        );
        upperFlange.rotation.x = Math.PI / 2;
        upperFlange.position.z = zMax + flangeTh / 2;
        machine.add(upperFlange);

        // =====================
        // GUIDATUBO
        // Negre = eix vertical
        // Groc  = eix horitzontal
        // Blau  = guidatubo
        // =====================

        const guideAssembly = new THREE.Group();
        scene.add(guideAssembly);

        // posició base a l'esquerra, similar a la foto
        const guideBaseX = -flangeR - 165;
        const guideBaseY = -8;
        const guideBaseZ = zMin - 6;

        // eix horitzontal groc
        const yellowAxisLen = 145;
        const yellowAxis = new THREE.Mesh(
            new THREE.BoxGeometry(yellowAxisLen, 18, 18),
            new THREE.MeshStandardMaterial({{
                color: 0xc8c83a,
                roughness: 0.85,
                metalness: 0.05
            }})
        );
        yellowAxis.position.set(guideBaseX, guideBaseY, guideBaseZ);
        scene.add(yellowAxis);

        // eix vertical negre
        const blackCol = new THREE.Mesh(
            new THREE.BoxGeometry(22, 22, spalla + 145),
            new THREE.MeshStandardMaterial({{
                color: 0x1f1f1f,
                roughness: 0.8,
                metalness: 0.15
            }})
        );
        blackCol.position.set(
            guideBaseX + yellowAxisLen/2 - 10,
            guideBaseY,
            (zMin + zMax) / 2 + 25
        );
        scene.add(blackCol);

        // grup mòbil del guidatubo
        const guideSlide = new THREE.Group();
        scene.add(guideSlide);

        // braç metàl·lic fins a la zona tangent
        const nozzleX = pts.length > 0 ? pts[0].x : (-rMandrel - 120);
        const armStartX = blackCol.position.x;
        const armLen = Math.max(70, armStartX - nozzleX);

        const arm = new THREE.Mesh(
            new THREE.CylinderGeometry(5.5, 5.5, armLen, 18),
            new THREE.MeshStandardMaterial({{
                color: 0xcfcfcf,
                roughness: 0.35,
                metalness: 0.7
            }})
        );
        arm.rotation.z = Math.PI / 2;
        arm.position.x = (armStartX + nozzleX) / 2;
        guideSlide.add(arm);

        // bloc blau
        const blueBlock = new THREE.Mesh(
            new THREE.BoxGeometry(24, 22, 18),
            new THREE.MeshStandardMaterial({{
                color: 0x2146c7,
                roughness: 0.55,
                metalness: 0.15
            }})
        );
        blueBlock.position.set(nozzleX - 10, guideBaseY, 0);
        guideSlide.add(blueBlock);

        // broquet cilíndric gris-blanc
        const nozzle = new THREE.Mesh(
            new THREE.CylinderGeometry(6.5, 6.5, 24, 20),
            new THREE.MeshStandardMaterial({{
                color: 0xd9d9d9,
                roughness: 0.45,
                metalness: 0.6
            }})
        );
        nozzle.rotation.z = Math.PI / 2;
        nozzle.position.set(nozzleX - 24, guideBaseY, 0);
        guideSlide.add(nozzle);

        // =====================
        // TUB BLANC
        // =====================

        const tubeMat = new THREE.MeshStandardMaterial({{
            color: 0xf1f1f1,
            roughness: 0.65,
            metalness: 0.15
        }});

        function buildTubeMesh(points3d, radius, radialSegments=14) {{
            if (!points3d || points3d.length < 2) return null;
            const curve = new THREE.CatmullRomCurve3(points3d, false, "centripetal", 0.2);
            const tubularSegments = Math.max(40, Math.min(1800, points3d.length * 2));
            const geo = new THREE.TubeGeometry(curve, tubularSegments, radius, radialSegments, false);
            return new THREE.Mesh(geo, tubeMat);
        }}

        // tram de sortida des del guidatubo fins l'entrada al rotlle
        let leadEndIdx = 1;
        if (pts.length > 3) {{
            leadEndIdx = 2;
        }}

        const leadPts = pts.slice(0, Math.min(leadEndIdx + 1, pts.length));
        const leadMesh = buildTubeMesh(leadPts, rTube, 12);
        if (leadMesh) scene.add(leadMesh);

        // tub enrotllat
        const woundPts = pts.slice(Math.min(2, pts.length - 1));
        const woundMesh = buildTubeMesh(woundPts, rTube, 14);
        if (woundMesh) machine.add(woundMesh);

        // cap frontal del tub visible a la sortida del guidatubo
        if (pts.length > 0) {{
            const p0 = pts[0];
            const tubeEnd = new THREE.Mesh(
                new THREE.SphereGeometry(rTube, 16, 12),
                tubeMat
            );
            tubeEnd.position.copy(p0);
            scene.add(tubeEnd);
        }}

        // =====================
        // POSICIÓ INICIAL GUIDATUBO
        // =====================

        function nearestGuideInfo(indexFloat) {{
            const idx = Math.max(0, Math.min(pts.length - 1, Math.floor(indexFloat)));
            const p = pts[idx];
            return {{
                x: nozzleX,
                y: guideBaseY,
                z: p.z
            }};
        }}

        const initGuide = nearestGuideInfo(0);
        guideSlide.position.z = initGuide.z;

        // =====================
        // SHADOW FLOOR SUAU
        // =====================

        const floor = new THREE.Mesh(
            new THREE.CircleGeometry(flangeR + 260, 64),
            new THREE.MeshBasicMaterial({{
                color: 0xcfd4db
            }})
        );
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(0, 18, zMin - 40);
        floor.visible = false;

        // =====================
        // FRAME
        // =====================

        const bbox = new THREE.Box3().setFromObject(scene);
        const size = new THREE.Vector3();
        bbox.getSize(size);

        const maxDim = Math.max(size.x, size.y, size.z, flangeR * 2);
        camera.near = 0.1;
        camera.far = maxDim * 10;
        camera.updateProjectionMatrix();

        // =====================
        // ANIMATION
        // =====================

        const animEnabled = {anim_js};
        const speed = {float(vel)};
        let t = 0.0;

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled) {{
                machine.rotation.z += 0.01 * speed;

                if (pts.length > 2) {{
                    t += 0.9 * speed;
                    const gi = nearestGuideInfo(t);
                    guideSlide.position.z = gi.z;
                }}
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
            const w = container.clientWidth || 1200;
            const h = container.clientHeight || {altezza};
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }});
    }})();
    </script>
    """


# =========================
# UI (NO TOCADA)
# =========================

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"#### {t['bobina']}")
    diametro_aspo = st.number_input(t["diam_aspo"], value=450.0)
    spalla = st.number_input(t["spalla"], value=95.0)

with colB:
    st.markdown(f"#### {t['tubo']}")
    rame = st.selectbox(t["rame"], list(COPPER_SIZES_MM.keys()))
    spessore = st.number_input(t["isolamento"], value=7.0)
    lunghezza = st.number_input(t["lunghezza"], value=30.0)
    d_rame = COPPER_SIZES_MM[rame]

with colC:
    st.markdown(f"#### {t['avvolg']}")
    passo = st.number_input(t["passo_assiale"], value=20.0)
    incremento = st.number_input(t["incremento"], value=20.0)
    rit_b = st.number_input(t["rit_min"], value=180.0)
    rit_t = st.number_input(t["rit_max"], value=180.0)
    gradi_start = st.number_input(t["gradi_start"], value=30.0)
    pinza = st.number_input(t["pinza"], value=0.3)

with colD:
    st.markdown(f"#### {t['viewer']}")
    altezza = st.slider(t["altezza"], 400, 900, 700)
    anim = st.checkbox(t["animazione"], False)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

pts = build_coil(
    diametro_aspo,
    spalla,
    lunghezza,
    d_rame,
    spessore,
    passo,
    incremento,
    rit_b,
    rit_t,
    gradi_start,
    pinza
)

d_tubo = d_rame + 2 * spessore

components.html(
    viewer(pts, diametro_aspo, spalla, d_tubo, altezza, anim, vel),
    height=altezza
)

# =========================
# METRICS (INTACTES)
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_rame + 2 * spessore:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm")
m3.metric(t["metric3"], f"{incremento:.2f} mm")

rmax = np.max(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2))
m4.metric(t["metric4"], f"{2 * (rmax):.1f} mm")

if 2 * rmax > 750:
    st.warning(t["warning"])
