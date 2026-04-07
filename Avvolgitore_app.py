import os
import glob
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

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
        "isolamento": "Spessore guaina (mm)",
        "lunghezza": "Lunghezza rotolo (m)",
        "passo_assiale": "Passo assiale (mm/rev)",
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
        "isolamento": "Foam thickness (mm)",
        "lunghezza": "Coil length (m)",
        "passo_assiale": "Axial pitch (mm/rev)",
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

if logo_path:
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(logo_path, use_container_width=True)
    with c2:
        st.markdown(f"## {t['title']}")
else:
    st.markdown(f"## {t['title']}")

# =========================
# METRICS SIMULATION
# Approx local packing:
# - first layer against mandrel
# - following layers against previous deposited points
# =========================

def simulate_outer_diameter(
    d_aspo: float,
    spalla: float,
    d_tubo: float,
    passo: float,
    incremento: float,
    rit_b: float,
    rit_t: float,
    lunghezza_m: float,
):
    R = d_aspo / 2.0
    Rt = d_tubo / 2.0
    H = spalla
    max_len = lunghezza_m * 1000.0

    guideX = -(R + 80.0)
    guideRadius = R + Rt
    guideZ = Rt
    theta = 0.0

    deposited = []
    deposited_len = 0.0

    direction = 1
    mode = "axial"
    turn_progress = 0.0
    turn_delay = 0.0
    turn_start_radius = guideRadius
    turn_end_radius = guideRadius
    turn_z = guideZ

    deg_step = 3.0
    rad_step = np.deg2rad(deg_step)

    def smoothstep(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def guide_point():
        return np.array([guideX, guideRadius, guideZ], dtype=float)

    def mandrel_contact():
        tube_theta = -theta + np.pi
        x = guideRadius * np.cos(tube_theta)
        y = guideRadius * np.sin(tube_theta)
        return np.array([x, y, guideZ], dtype=float)

    def contact_with_previous(gp, pts):
        if len(pts) < 8:
            return mandrel_contact()

        best = None
        best_err = 1e18
        search_n = min(300, len(pts))

        # recta de sortida: +X
        for p in pts[-search_n:]:
            if p[0] < gp[0]:
                continue

            cand = np.array([p[0], gp[1], gp[2]], dtype=float)
            dist = np.linalg.norm(cand - p)
            err = abs(dist - 2.0 * Rt)

            if err < best_err:
                best_err = err
                best = cand

        if best is None:
            return mandrel_contact()

        min_rad = R + Rt
        cand_r = np.hypot(best[0], best[1])
        if cand_r < min_rad:
            scale = min_rad / max(cand_r, 1e-9)
            best[0] *= scale
            best[1] *= scale

        return best

    # punt inicial
    first_gp = guide_point()
    first_cp = mandrel_contact()
    deposited.append(first_cp)

    for _ in range(250000):
        prev = deposited[-1]

        theta += rad_step

        if mode == "axial":
            guideZ += direction * passo * (deg_step / 360.0)

            if guideZ >= H - Rt:
                guideZ = H - Rt
                mode = "turn"
                turn_progress = 0.0
                turn_delay = max(rit_t, 0.0)
                turn_start_radius = guideRadius
                turn_end_radius = guideRadius + incremento
                turn_z = guideZ

            elif guideZ <= Rt:
                guideZ = Rt
                mode = "turn"
                turn_progress = 0.0
                turn_delay = max(rit_b, 0.0)
                turn_start_radius = guideRadius
                turn_end_radius = guideRadius + incremento
                turn_z = guideZ

        else:
            if turn_delay <= 0.0:
                guideRadius = turn_end_radius
                mode = "axial"
                direction *= -1
            else:
                turn_progress += deg_step
                s = smoothstep(turn_progress / turn_delay)
                guideRadius = turn_start_radius + s * (turn_end_radius - turn_start_radius)
                guideZ = turn_z

                if turn_progress >= turn_delay:
                    guideRadius = turn_end_radius
                    mode = "axial"
                    direction *= -1

        gp = guide_point()
        cp = contact_with_previous(gp, deposited)

        seg = float(np.linalg.norm(cp - prev))
        if seg < max(0.6, Rt * 0.08):
            continue

        free_len = float(np.linalg.norm(gp - cp))
        if deposited_len + seg + free_len >= max_len:
            remain = max_len - (deposited_len + free_len)
            if remain > 0 and seg > 1e-9:
                alpha = remain / seg
                final_p = prev + alpha * (cp - prev)
                deposited.append(final_p)
            break

        deposited.append(cp)
        deposited_len += seg

    dep = np.array(deposited)
    radial = np.sqrt(dep[:, 0] ** 2 + dep[:, 1] ** 2)
    max_centerline_r = float(np.max(radial)) if len(radial) else (R + Rt)
    outer_diameter = 2.0 * (max_centerline_r + Rt)
    return outer_diameter

# =========================
# VIEWER
# =========================

def viewer(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza, altezza, anim, vel):
    anim_js = "true" if anim else "false"

    return f"""
    <div id="viewer_root" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{
        const host = document.getElementById("viewer_root");
        host.innerHTML = "";

        const W = Math.max(host.clientWidth, 600);
        const Hview = Math.max(host.clientHeight, 400);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(38, W / Hview, 0.1, 20000);
        camera.position.set(-520, -760, 420);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, Hview);
        host.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, {spalla}/2);

        // =====================
        // PARAMS
        // =====================

        const R = {float(d_aspo)} / 2.0;
        const Rt = {float(d_tubo)} / 2.0;
        const Hs = {float(spalla)};
        const passo = {float(passo)};
        const incremento = {float(incremento)};
        const ritB = {float(rit_b)};
        const ritT = {float(rit_t)};
        const maxLen = {float(lunghezza)} * 1000.0;
        const speed = {float(vel)};
        const animEnabled = {anim_js};

        // Guidatubo
        const guideX = -(R + 80.0);
        let guideRadius = R + Rt;
        let guideZ = Rt;

        // =====================
        // MATERIALS
        // =====================

        const redMat = new THREE.MeshStandardMaterial({{
            color: 0xff3333,
            roughness: 0.55,
            metalness: 0.08
        }});

        const blueMat = new THREE.MeshStandardMaterial({{
            color: 0x0044ff,
            roughness: 0.55,
            metalness: 0.10
        }});

        const tubeMat = new THREE.MeshStandardMaterial({{
            color: 0xffffff,
            roughness: 0.65,
            metalness: 0.15
        }});

        // =====================
        // ASP0
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, Hs, 96),
            redMat
        );
        mandrel.rotation.x = Math.PI / 2;
        mandrel.position.z = Hs / 2.0;
        machine.add(mandrel);

        const flangeR = R + 120.0;
        const flangeTh = 6.0;

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 96),
            redMat
        );
        base.rotation.x = Math.PI / 2;
        base.position.z = 0.0;
        machine.add(base);

        const top = new THREE.Mesh(
            new THREE.CylinderGeometry(flangeR, flangeR, flangeTh, 96),
            redMat
        );
        top.rotation.x = Math.PI / 2;
        top.position.z = Hs;
        machine.add(top);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30, 20, 20),
            blueMat
        );
        scene.add(guide);

        // =====================
        // DEPOSITION MODEL
        // =====================

        let depositedPoints = [];
        let depositedLength = 0.0;
        let finished = false;
        let tubeMesh = null;
        let lastRebuildCount = 0;

        function smoothstep(x) {{
            x = Math.max(0.0, Math.min(1.0, x));
            return x * x * (3.0 - 2.0 * x);
        }}

        function currentGuidePoint() {{
            return new THREE.Vector3(guideX, guideRadius, guideZ);
        }}

        function mandrelContact(thetaMachine) {{
            const thetaTube = -thetaMachine + Math.PI;
            const x = guideRadius * Math.cos(thetaTube);
            const y = guideRadius * Math.sin(thetaTube);
            return new THREE.Vector3(x, y, guideZ);
        }}

        // 1a capa contra mandrí, després contra capa anterior
        function contactWithPrevious(guidePoint, thetaMachine) {{
            if (depositedPoints.length < 8) {{
                return mandrelContact(thetaMachine);
            }}

            let best = null;
            let bestErr = 1e18;
            const searchN = Math.min(300, depositedPoints.length);

            for (let i = depositedPoints.length - searchN; i < depositedPoints.length; i++) {{
                const p = depositedPoints[i];

                // recta de sortida: normal a la cara del guidatubo = +X
                if (p.x < guidePoint.x) continue;

                const cand = new THREE.Vector3(p.x, guidePoint.y, guidePoint.z);
                const dist = cand.distanceTo(p);
                const err = Math.abs(dist - 2.0 * Rt);

                if (err < bestErr) {{
                    bestErr = err;
                    best = cand;
                }}
            }}

            if (!best) {{
                return mandrelContact(thetaMachine);
            }}

            // Evita penetració dins del mandrí
            const candR = Math.sqrt(best.x * best.x + best.y * best.y);
            const minR = R + Rt;
            if (candR < minR) {{
                const scale = minR / Math.max(candR, 1e-9);
                best.x *= scale;
                best.y *= scale;
            }}

            return best;
        }}

        function rebuildTubeMesh(guidePoint) {{
            if (depositedPoints.length < 2) return;

            const displayPoints = depositedPoints.slice();

            // tram final recte, normal a la cara del guidatubo
            const straightLen = Math.max(50.0, Rt * 6.0);
            const preGuide = new THREE.Vector3(
                guidePoint.x + straightLen,
                guidePoint.y,
                guidePoint.z
            );

            displayPoints.push(preGuide);
            displayPoints.push(guidePoint.clone());

            if (tubeMesh) {{
                scene.remove(tubeMesh);
                tubeMesh.geometry.dispose();
                tubeMesh.material.dispose();
                tubeMesh = null;
            }}

            const curve = new THREE.CatmullRomCurve3(displayPoints, false, "centripetal", 0.1);
            const tubularSegments = Math.max(32, Math.min(1800, displayPoints.length * 2));
            const geo = new THREE.TubeGeometry(curve, tubularSegments, Rt, 12, false);

            tubeMesh = new THREE.Mesh(geo, tubeMat);
            scene.add(tubeMesh);
        }}

        // =====================
        // INITIAL STATE
        // =====================

        let thetaMachine = 0.0;
        depositedPoints.push(mandrelContact(thetaMachine));

        // =====================
        // LIGHTS
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const dLight = new THREE.DirectionalLight(0xffffff, 0.7);
        dLight.position.set(500, -500, 800);
        scene.add(dLight);

        // =====================
        // MOTION STATE MACHINE
        // =====================

        let direction = 1;
        let mode = "axial";
        let turnProgress = 0.0;
        let turnDelay = 0.0;
        let turnStartRadius = guideRadius;
        let turnEndRadius = guideRadius;
        let turnZ = guideZ;

        function advanceMechanics() {{
            const degPerFrame = 2.0 * speed;
            const radPerFrame = THREE.MathUtils.degToRad(degPerFrame);

            // aspo horari
            thetaMachine -= radPerFrame;
            machine.rotation.z = thetaMachine;

            if (mode === "axial") {{
                // passo = mm per revolució
                guideZ += direction * passo * (degPerFrame / 360.0);

                if (guideZ >= Hs - Rt) {{
                    guideZ = Hs - Rt;
                    mode = "turn";
                    turnProgress = 0.0;
                    turnDelay = Math.max(ritT, 0.0);
                    turnStartRadius = guideRadius;
                    turnEndRadius = guideRadius + incremento;
                    turnZ = guideZ;
                }} else if (guideZ <= Rt) {{
                    guideZ = Rt;
                    mode = "turn";
                    turnProgress = 0.0;
                    turnDelay = Math.max(ritB, 0.0);
                    turnStartRadius = guideRadius;
                    turnEndRadius = guideRadius + incremento;
                    turnZ = guideZ;
                }}
            }} else {{
                if (turnDelay <= 0.0) {{
                    guideRadius = turnEndRadius;
                    mode = "axial";
                    direction *= -1;
                }} else {{
                    turnProgress += degPerFrame;
                    const s = smoothstep(turnProgress / turnDelay);
                    guideRadius = turnStartRadius + s * (turnEndRadius - turnStartRadius);
                    guideZ = turnZ;

                    if (turnProgress >= turnDelay) {{
                        guideRadius = turnEndRadius;
                        mode = "axial";
                        direction *= -1;
                    }}
                }}
            }}
        }}

        function addDepositedPoint(newPoint, guidePoint) {{
            const prev = depositedPoints[depositedPoints.length - 1];
            const seg = newPoint.distanceTo(prev);

            if (seg < Math.max(0.8, Rt * 0.10)) {{
                return;
            }}

            // longitud real = diposició + tram lliure recte
            const freeLen = guidePoint.distanceTo(newPoint);
            const totalIfAccepted = depositedLength + seg + freeLen;

            if (totalIfAccepted <= maxLen) {{
                depositedPoints.push(newPoint.clone());
                depositedLength += seg;
                return;
            }}

            // trim exacte
            let lo = 0.0;
            let hi = 1.0;

            for (let i = 0; i < 28; i++) {{
                const mid = 0.5 * (lo + hi);
                const candidate = prev.clone().lerp(newPoint, mid);
                const totalCandidate =
                    depositedLength +
                    prev.distanceTo(candidate) +
                    guidePoint.distanceTo(candidate);

                if (totalCandidate < maxLen) lo = mid;
                else hi = mid;
            }}

            const finalPoint = prev.clone().lerp(newPoint, lo);
            depositedPoints.push(finalPoint);
            depositedLength += prev.distanceTo(finalPoint);
            finished = true;
        }}

        function animate() {{
            requestAnimationFrame(animate);

            if (animEnabled && !finished) {{
                advanceMechanics();

                const guidePoint = currentGuidePoint();
                const contactPoint = contactWithPrevious(guidePoint, thetaMachine);

                guide.position.copy(guidePoint);
                addDepositedPoint(contactPoint, guidePoint);

                if (
                    tubeMesh === null ||
                    depositedPoints.length !== lastRebuildCount ||
                    finished
                ) {{
                    rebuildTubeMesh(guidePoint);
                    lastRebuildCount = depositedPoints.length;
                }}
            }} else {{
                const guidePoint = currentGuidePoint();
                guide.position.copy(guidePoint);

                if (tubeMesh === null && depositedPoints.length >= 2) {{
                    rebuildTubeMesh(guidePoint);
                    lastRebuildCount = depositedPoints.length;
                }}
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener("resize", () => {{
            const nw = Math.max(host.clientWidth, 600);
            const nh = Math.max(host.clientHeight, 400);
            camera.aspect = nw / nh;
            camera.updateProjectionMatrix();
            renderer.setSize(nw, nh);
        }});
    }})();
    </script>
    """

# =========================
# UI
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
    anim = st.checkbox(t["animazione"], True)
    vel = st.slider(t["velocita"], 0.1, 5.0, 1.0)

# =========================
# BUILD
# =========================

d_tubo = d_rame + 2 * spessore

diam_esterno = simulate_outer_diameter(
    diametro_aspo,
    spalla,
    d_tubo,
    passo,
    incremento,
    rit_b,
    rit_t,
    lunghezza
)

components.html(
    viewer(
        diametro_aspo,
        spalla,
        d_tubo,
        passo,
        incremento,
        rit_b,
        rit_t,
        lunghezza,
        altezza,
        anim,
        vel
    ),
    height=altezza
)

# =========================
# METRICS
# =========================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(t["metric1"], f"{d_tubo:.2f} mm")
m2.metric(t["metric2"], f"{passo:.2f} mm/rev")
m3.metric(t["metric3"], f"{incremento:.2f} mm")
m4.metric(t["metric4"], f"{diam_esterno:.1f} mm")

if diam_esterno > 750:
    st.warning(t["warning"])
