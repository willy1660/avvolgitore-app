import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="Avvolgimento", layout="wide")

# =========================
# LANGUAGE
# =========================

lang = "IT"

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
    return np.vstack([points[:idx + 1], p0 + alpha * (p1 - p0)])


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def append_point(pts, p):
    p = np.asarray(p, dtype=float)
    if len(pts) == 0:
        pts.append(p)
    else:
        if np.linalg.norm(p - pts[-1]) > 1e-8:
            pts.append(p)


def append_segment(pts, p0, p1, n=20):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n = max(2, int(n))
    for i in range(n):
        t = i / (n - 1)
        p = (1 - t) * p0 + t * p1
        append_point(pts, p)


def append_arc_constant_z(pts, r, theta0, theta1, z, n=48):
    n = max(2, int(n))
    for i in range(n):
        t = i / (n - 1)
        th = (1 - t) * theta0 + t * theta1
        p = np.array([r * np.cos(th), r * np.sin(th), z], dtype=float)
        append_point(pts, p)


def append_helix(pts, r, theta0, theta1, z0, z1, n=120):
    n = max(2, int(n))
    for i in range(n):
        t = i / (n - 1)
        th = (1 - t) * theta0 + t * theta1
        z = (1 - t) * z0 + t * z1
        p = np.array([r * np.cos(th), r * np.sin(th), z], dtype=float)
        append_point(pts, p)


def append_radial_step(pts, r0, r1, theta, z, n=16):
    n = max(2, int(n))
    for i in range(n):
        t = i / (n - 1)
        r = (1 - t) * r0 + t * r1
        p = np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=float)
        append_point(pts, p)


# =========================
# COIL MODEL
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore, passo_ax, passo_rad, rit_min, rit_max):
    """
    Model mecànic simplificat però coherent:
    - gir horari
    - pujada axial entre base i spalla
    - ritardo a top/base
    - només després incremento radial
    - canvi de sentit
    """

    L = safe_float(lunghezza, 50.0) * 1000.0
    d_aspo = safe_float(d_aspo, 450.0)
    spalla = safe_float(spalla, 95.0)
    d_rame = safe_float(d_rame, 9.52)
    spessore = safe_float(spessore, 7.0)
    passo_ax = max(0.1, safe_float(passo_ax, 20.0))
    passo_rad = max(0.1, safe_float(passo_rad, 20.0))
    rit_min = max(0.0, safe_float(rit_min, 180.0))
    rit_max = max(0.0, safe_float(rit_max, 180.0))

    d_tubo = d_rame + 2 * spessore
    r_tubo = d_tubo / 2.0

    # radi centreline primera capa
    r0 = d_aspo / 2.0 + r_tubo

    # z físic abans de centrar
    z_min = r_tubo
    z_max = max(z_min + d_tubo, spalla - r_tubo)

    # centre vertical per visualitzar amb l'aspo centrat
    z_shift = spalla / 2.0

    # pas angular discret
    dtheta = np.deg2rad(4.0)

    # relació axial per radià:
    # una volta completa (2pi) puja "passo_ax"
    dz_per_rad = passo_ax / (2.0 * np.pi)

    # punt inicial: part frontal inferior del mandrí
    # theta = -pi/2 dona punt (0, -r), amb tangent horària coherent
    theta = -np.pi / 2.0
    r = r0
    z = z_min

    pts = []

    # -------------------------
    # Entrada recta del guidatubo
    # -------------------------
    guide_x_far = r0 + max(80.0, d_aspo * 0.35)
    # la boca del guidatubo queda tangent al mandrí en el punt inicial
    guide_x_near = 0.0
    guide_y = -r0

    p_start = np.array([guide_x_far, guide_y, z_min - z_shift], dtype=float)
    p_entry = np.array([0.0, -r0, z_min - z_shift], dtype=float)

    append_segment(pts, p_start, p_entry, n=32)

    # -------------------------
    # Des d'aquí, alternança axial real
    # -------------------------
    direction = +1  # +1 puja, -1 baixa

    while polyline_length(np.array(pts)) < L:
        if direction == +1:
            # quants radians necessitem per pujar de z a z_max?
            dz_needed = z_max - z
            if dz_needed < EPS:
                dz_needed = 0.0

            dtheta_ax = dz_needed / dz_per_rad if dz_per_rad > EPS else 0.0
            theta_end = theta - dtheta_ax  # horari -> theta decreix

            # pujada axial
            n_ax = max(24, int(abs(dtheta_ax) / dtheta) + 2)
            append_helix(pts, r, theta, theta_end, z - z_shift, z_max - z_shift, n=n_ax)
            theta = theta_end
            z = z_max

            # ritardo top
            if rit_max > 0:
                theta_delay = theta - np.deg2rad(rit_max)
                n_delay = max(12, int(rit_max / 4) + 2)
                append_arc_constant_z(pts, r, theta, theta_delay, z - z_shift, n=n_delay)
                theta = theta_delay

            # incremento radial DESPRÉS del ritardo
            r_next = r + passo_rad
            append_radial_step(pts, r, r_next, theta, z - z_shift, n=14)
            r = r_next

            # canvi de sentit
            direction = -1

        else:
            dz_needed = z - z_min
            if dz_needed < EPS:
                dz_needed = 0.0

            dtheta_ax = dz_needed / dz_per_rad if dz_per_rad > EPS else 0.0
            theta_end = theta - dtheta_ax  # el mandrí continua girant horari

            # baixada axial
            n_ax = max(24, int(abs(dtheta_ax) / dtheta) + 2)
            append_helix(pts, r, theta, theta_end, z - z_shift, z_min - z_shift, n=n_ax)
            theta = theta_end
            z = z_min

            # ritardo base
            if rit_min > 0:
                theta_delay = theta - np.deg2rad(rit_min)
                n_delay = max(12, int(rit_min / 4) + 2)
                append_arc_constant_z(pts, r, theta, theta_delay, z - z_shift, n=n_delay)
                theta = theta_delay

            # incremento radial DESPRÉS del ritardo
            r_next = r + passo_rad
            append_radial_step(pts, r, r_next, theta, z - z_shift, n=14)
            r = r_next

            # canvi de sentit
            direction = +1

        if len(pts) > 150000:
            break

    path = np.array(pts, dtype=float)
    path = trim_polyline(path, L)

    return path, d_tubo


# =========================
# VIEWER
# =========================

def build_html(points, d_tubo, altura, anim, speed, d_aspo, spalla):
    pts = json.dumps(points.tolist())

    r_tubo = d_tubo / 2.0
    r_mandrel = d_aspo / 2.0

    # Base proporcional al mandrí, no al rotllo complet
    flange = r_mandrel + max(25.0, d_aspo * 0.08)

    # Spalla una mica més gran que el mandrí
    spalla_r = r_mandrel + max(18.0, d_aspo * 0.06)

    # Guidatubo tangent al mandrí al punt d'entrada
    guide_y = -(r_mandrel + r_tubo)
    guide_x_far = (r_mandrel + r_tubo) + max(80.0, d_aspo * 0.35)
    guide_x_near = 0.0
    guide_z = -spalla / 2.0 + r_tubo

    anim_js = "true" if anim else "false"
    speed_js = float(speed)

    return f"""
    <div id="viewer-wrap" style="width:100%;height:{altura}px;position:relative;background:#000;border-radius:12px;overflow:hidden;">
      <div id="viewer" style="width:100%;height:100%;"></div>
      <div id="errbox" style="position:absolute;left:12px;bottom:12px;color:#ffb3b3;font:12px monospace;white-space:pre-wrap;max-width:60%;display:none;background:rgba(0,0,0,0.55);padding:8px 10px;border-radius:8px;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (function() {{
      const errbox = document.getElementById("errbox");
      function showErr(msg) {{
        errbox.style.display = "block";
        errbox.textContent = String(msg);
      }}

      try {{
        const container = document.getElementById("viewer");
        const wrap = document.getElementById("viewer-wrap");

        const W = Math.max(300, container.clientWidth || wrap.clientWidth || 900);
        const H = Math.max(300, container.clientHeight || {altura});

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 50000);
        camera.position.set({d_aspo * 1.9}, -{d_aspo * 2.2}, {max(spalla * 1.6, d_aspo * 0.65)});

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(W, H, false);
        renderer.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, 0);

        const ptsRaw = {pts};

        if (!Array.isArray(ptsRaw) || ptsRaw.length < 2) {{
          throw new Error("Trajectòria buida o invàlida.");
        }}

        const vec = ptsRaw.map((p, i) => {{
          if (!Array.isArray(p) || p.length !== 3) {{
            throw new Error("Punt invàlid a índex " + i);
          }}
          const x = Number(p[0]), y = Number(p[1]), z = Number(p[2]);
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {{
            throw new Error("Punt amb NaN/Inf a índex " + i);
          }}
          return new THREE.Vector3(x, y, z);
        }});

        // =========================
        // AXES / GRID
        // =========================
        const axes = new THREE.AxesHelper({max(d_aspo * 0.7, 100)});
        scene.add(axes);

        const grid = new THREE.GridHelper({max(d_aspo * 4.0, 1200)}, 20);
        grid.rotation.x = Math.PI / 2;
        grid.position.z = -{spalla}/2 - 0.1;
        scene.add(grid);

        // =========================
        // LIGHTS
        // =========================
        const hemi = new THREE.HemisphereLight(0xffffff, 0x222222, 1.15);
        scene.add(hemi);

        const dir1 = new THREE.DirectionalLight(0xffffff, 0.85);
        dir1.position.set(500, -700, 900);
        scene.add(dir1);

        const dir2 = new THREE.DirectionalLight(0xffffff, 0.35);
        dir2.position.set(-400, 500, -300);
        scene.add(dir2);

        // =========================
        // MANDREL
        // =========================
        const mandrelGeom = new THREE.CylinderGeometry({r_mandrel}, {r_mandrel}, {spalla}, 72, 1, false);
        const mandrelMat = new THREE.MeshStandardMaterial({{
          color: 0x8a8a8a,
          roughness: 0.65,
          metalness: 0.25,
          transparent: true,
          opacity: 0.52
        }});
        const mandrel = new THREE.Mesh(mandrelGeom, mandrelMat);
        mandrel.rotation.x = Math.PI / 2;
        scene.add(mandrel);

        // =========================
        // BASE
        // =========================
        const baseGeom = new THREE.CylinderGeometry({flange}, {flange}, 8, 72);
        const baseMat = new THREE.MeshStandardMaterial({{
          color: 0x2e69b9,
          roughness: 0.55,
          metalness: 0.15
        }});
        const base = new THREE.Mesh(baseGeom, baseMat);
        base.rotation.x = Math.PI / 2;
        base.position.z = -{spalla}/2 - 4;
        scene.add(base);

        // =========================
        // SPALLA SUPERIORE
        // =========================
        const topGeom = new THREE.CylinderGeometry({spalla_r}, {spalla_r}, 8, 72);
        const topMat = new THREE.MeshStandardMaterial({{
          color: 0x355c8c,
          roughness: 0.55,
          metalness: 0.15
        }});
        const topFlange = new THREE.Mesh(topGeom, topMat);
        topFlange.rotation.x = Math.PI / 2;
        topFlange.position.z = {spalla}/2 + 4;
        scene.add(topFlange);

        // =========================
        // GUIDATUBO
        // =========================
        const guideMat = new THREE.MeshStandardMaterial({{
          color: 0xbdbdbd,
          roughness: 0.5,
          metalness: 0.3
        }});

        // eix radial del braç (governa aproximadament el moviment radial)
        const armLen = {guide_x_far - guide_x_near};
        const armGeom = new THREE.CylinderGeometry(6, 6, armLen, 18);
        const arm = new THREE.Mesh(armGeom, guideMat);
        arm.rotation.z = Math.PI / 2;
        arm.position.set(({guide_x_far} + {guide_x_near}) / 2, {guide_y}, {guide_z});
        scene.add(arm);

        // columna guia (governa aproximadament el moviment axial)
        const postGeom = new THREE.CylinderGeometry(8, 8, {max(spalla * 1.2, 120)}, 18);
        const post = new THREE.Mesh(postGeom, guideMat);
        post.rotation.x = Math.PI / 2;
        post.position.set({guide_x_far}, {guide_y}, 0);
        scene.add(post);

        // carro axial sobre la columna
        const carriageGeom = new THREE.BoxGeometry(24, 24, 18);
        const carriage = new THREE.Mesh(carriageGeom, guideMat);
        carriage.position.set({guide_x_far}, {guide_y}, {guide_z});
        scene.add(carriage);

        // tirant entre carro axial i braç radial
        const tieLen = {guide_x_far - guide_x_near};
        const tieGeom = new THREE.CylinderGeometry(4, 4, tieLen, 16);
        const tie = new THREE.Mesh(tieGeom, guideMat);
        tie.rotation.z = Math.PI / 2;
        tie.position.set(({guide_x_far} + {guide_x_near}) / 2, {guide_y}, {guide_z});
        scene.add(tie);

        const nozzleGeom = new THREE.CylinderGeometry(9, 9, 24, 18);
        const nozzle = new THREE.Mesh(nozzleGeom, guideMat);
        nozzle.rotation.z = Math.PI / 2;
        nozzle.position.set({guide_x_near}, {guide_y}, {guide_z});
        scene.add(nozzle);

        // =========================
        // LÍNIA DE LA TRAJECTÒRIA
        // =========================
        const lineGeom = new THREE.BufferGeometry().setFromPoints(vec);
        const lineMat = new THREE.LineBasicMaterial({{ color: 0xffffff }});
        const line = new THREE.Line(lineGeom, lineMat);
        scene.add(line);

        // =========================
        // TUB 3D
        // =========================
        class PolylineCurve extends THREE.Curve {{
          constructor(points) {{
            super();
            this.points = points;
            this.count = points.length;
          }}
          getPoint(t) {{
            const scaled = t * (this.count - 1);
            const i = Math.floor(scaled);
            const a = this.points[i];
            const b = this.points[Math.min(i + 1, this.count - 1)];
            const f = scaled - i;
            return new THREE.Vector3().lerpVectors(a, b, f);
          }}
        }}

        const curve = new PolylineCurve(vec);
        const tubularSegments = Math.max(64, Math.min(vec.length - 1, 6000));
        const radialSegments = 18;

        const tubeGeom = new THREE.TubeGeometry(curve, tubularSegments, {r_tubo}, radialSegments, false);
        const tubeMat = new THREE.MeshStandardMaterial({{
          color: 0xf2f2f2,
          roughness: 0.82,
          metalness: 0.02
        }});
        const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
        scene.add(tubeMesh);

        // =========================
        // FIT CAMERA
        // =========================
        const box = new THREE.Box3().setFromObject(tubeMesh);
        box.expandByObject(mandrel);
        box.expandByObject(base);
        box.expandByObject(topFlange);
        box.expandByObject(arm);
        box.expandByObject(post);
        box.expandByObject(carriage);
        box.expandByObject(tie);

        const center = new THREE.Vector3();
        const size = new THREE.Vector3();
        box.getCenter(center);
        box.getSize(size);

        controls.target.copy(center);

        const maxDim = Math.max(size.x, size.y, size.z, 200);
        const dist = maxDim * 1.55;
        camera.position.set(center.x + dist * 0.95, center.y - dist * 1.05, center.z + dist * 0.55);
        camera.near = Math.max(0.1, maxDim / 1000);
        camera.far = Math.max(5000, maxDim * 10);
        camera.updateProjectionMatrix();

        // =========================
        // RESIZE
        // =========================
        function onResize() {{
          const w = Math.max(300, container.clientWidth || wrap.clientWidth || W);
          const h = Math.max(300, container.clientHeight || {altura});
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h, false);
        }}

        window.addEventListener("resize", onResize);

        // =========================
        // ANIMACIÓ OPCIONAL
        // =========================
        let reveal = 1.0;
        const useAnim = {anim_js};
        const animSpeed = {speed_js};

        let revealedLine = null;
        let revealedTube = null;

        if (useAnim) {{
          reveal = 0.02;
          scene.remove(line);
          scene.remove(tubeMesh);

          revealedLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(vec.slice(0, Math.max(2, Math.floor(vec.length * reveal)))),
            lineMat
          );
          scene.add(revealedLine);
        }}

        function rebuildReveal(frac) {{
          const n = Math.max(2, Math.floor(vec.length * frac));

          if (revealedLine) {{
            scene.remove(revealedLine);
            revealedLine.geometry.dispose();
          }}

          revealedLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(vec.slice(0, n)),
            lineMat
          );
          scene.add(revealedLine);

          if (revealedTube) {{
            scene.remove(revealedTube);
            revealedTube.geometry.dispose();
          }}

          if (n > 8) {{
            const subCurve = new PolylineCurve(vec.slice(0, n));
            const subTubeGeom = new THREE.TubeGeometry(
              subCurve,
              Math.max(16, Math.min(n - 1, 2500)),
              {r_tubo},
              radialSegments,
              false
            );
            revealedTube = new THREE.Mesh(subTubeGeom, tubeMat);
            scene.add(revealedTube);
          }}
        }}

        function animate() {{
          requestAnimationFrame(animate);

          if (useAnim) {{
            reveal += 0.0018 * animSpeed;
            if (reveal > 1.0) reveal = 1.0;
            rebuildReveal(reveal);
          }}

          controls.update();
          renderer.render(scene, camera);
        }}

        animate();
        onResize();

      }} catch (err) {{
        console.error(err);
        showErr(err && err.stack ? err.stack : err);
      }}
    }})();
    </script>
    """


# =========================
# UI
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    d_aspo = st.number_input("Ø Aspo", value=450.0)
    spalla = st.number_input("Spalla", value=95.0)

with col2:
    rame = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))
    spess = st.number_input("Spessore", value=7.0)
    lung = st.number_input("Lunghezza", value=50.0)

with col3:
    passo_ax = st.number_input("Passo assiale", value=20.0)
    passo_rad = st.number_input("Incremento", value=20.0)
    rit_min = st.number_input("Rit base", value=180.0)
    rit_max = st.number_input("Rit top", value=180.0)

d_rame = COPPER_SIZES_MM[rame]

# =========================
# RUN
# =========================

path, d_tubo = build_coil(
    d_aspo, spalla, lung,
    d_rame, spess,
    passo_ax, passo_rad,
    rit_min, rit_max
)

# =========================
# METRICS
# =========================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Ø tubo ext", f"{d_tubo:.2f} mm")

with m2:
    st.metric("Punts trajectòria", f"{len(path)}")

with m3:
    st.metric("Longitud model", f"{polyline_length(path)/1000:.2f} m")

with m4:
    ext_diam = 2 * np.max(np.sqrt(path[:, 0]**2 + path[:, 1]**2)) if len(path) else 0.0
    st.metric("Ø ext estimat", f"{ext_diam:.1f} mm")

html = build_html(
    path,
    d_tubo,
    altura=700,
    anim=False,
    speed=1.0,
    d_aspo=d_aspo,
    spalla=spalla
)

components.html(html, height=700)
