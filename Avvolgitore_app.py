import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Avvolgimento", layout="wide")

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


# =========================
# MODEL (COHERENT MECÀNICAMENT)
# =========================

def build_coil(d_aspo, spalla, lunghezza, d_rame, spessore,
               passo_ax, passo_rad, rit_min, rit_max):

    L = lunghezza * 1000.0
    d_tubo = d_rame + 2 * spessore
    r_tubo = d_tubo / 2

    r0 = d_aspo / 2 + r_tubo
    z_min = r_tubo
    z_max = spalla - r_tubo

    theta = -np.pi/2  # entrada correcta
    z = z_min
    r = r0

    dz_per_rad = passo_ax / (2*np.pi)

    direction = +1
    pts = []

    def add():
        pts.append([
            r*np.cos(theta),
            r*np.sin(theta),
            z - spalla/2
        ])

    add()

    while polyline_length(np.array(pts)) < L:

        if direction == 1:
            dz = z_max - z
            dtheta = dz / dz_per_rad
            theta_end = theta - dtheta

            steps = max(20, int(abs(dtheta)/0.05))
            for i in range(steps):
                t = i/(steps-1)
                th = theta + (theta_end-theta)*t
                zz = z + dz*t
                pts.append([r*np.cos(th), r*np.sin(th), zz - spalla/2])

            theta = theta_end
            z = z_max

            # ritardo top
            if rit_max > 0:
                theta2 = theta - np.deg2rad(rit_max)
                steps = int(rit_max/4)+5
                for i in range(steps):
                    t = i/(steps-1)
                    th = theta + (theta2-theta)*t
                    pts.append([r*np.cos(th), r*np.sin(th), z - spalla/2])
                theta = theta2

            # incremento radial DESPRÉS
            r += passo_rad

            direction = -1

        else:
            dz = z - z_min
            dtheta = dz / dz_per_rad
            theta_end = theta - dtheta

            steps = max(20, int(abs(dtheta)/0.05))
            for i in range(steps):
                t = i/(steps-1)
                th = theta + (theta_end-theta)*t
                zz = z - dz*t
                pts.append([r*np.cos(th), r*np.sin(th), zz - spalla/2])

            theta = theta_end
            z = z_min

            # ritardo base
            if rit_min > 0:
                theta2 = theta - np.deg2rad(rit_min)
                steps = int(rit_min/4)+5
                for i in range(steps):
                    t = i/(steps-1)
                    th = theta + (theta2-theta)*t
                    pts.append([r*np.cos(th), r*np.sin(th), z - spalla/2])
                theta = theta2

            # incremento radial DESPRÉS
            r += passo_rad

            direction = +1

        if len(pts) > 100000:
            break

    path = np.array(pts)
    path = trim_polyline(path, L)

    return path, d_tubo


# =========================
# VIEWER ROBUST
# =========================

def build_html(points, d_tubo, altura, d_aspo, spalla):

    pts = json.dumps(points.tolist())

    r_mandrel = d_aspo/2
    flange = r_mandrel + 40

    return f"""
    <div id="viewer" style="width:100%;height:{altura}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

    <script>
    (function() {{

        const container = document.getElementById("viewer");
        const width = container.clientWidth;
        const height = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(45, width/height, 0.1, 20000);
        camera.position.set({d_aspo*2}, -{d_aspo*2}, {spalla*1.5});

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        const light = new THREE.HemisphereLight(0xffffff,0x222222,1.2);
        scene.add(light);

        const pts = {pts};

        const vec = pts.map(p => new THREE.Vector3(p[0],p[1],p[2]));

        const geom = new THREE.BufferGeometry().setFromPoints(vec);
        const mat = new THREE.LineBasicMaterial({{color:0xffffff}});
        const line = new THREE.Line(geom, mat);
        scene.add(line);

        const cyl = new THREE.Mesh(
            new THREE.CylinderGeometry({r_mandrel},{r_mandrel},{spalla},48),
            new THREE.MeshStandardMaterial({{color:0x666666,transparent:true,opacity:0.4}})
        );
        cyl.rotation.x = Math.PI/2;
        scene.add(cyl);

        const base = new THREE.Mesh(
            new THREE.CylinderGeometry({flange},{flange},6,48),
            new THREE.MeshStandardMaterial({{color:0x2e69b9}})
        );
        base.rotation.x = Math.PI/2;
        base.position.z = -{spalla}/2-3;
        scene.add(base);

        let angle = 0;

        function animate(){{
            requestAnimationFrame(animate);

            angle += 0.003;

            camera.position.x = Math.cos(angle) * {d_aspo*2};
            camera.position.y = Math.sin(angle) * {d_aspo*2};

            camera.lookAt(0,0,0);

            renderer.render(scene,camera);
        }}

        animate();

    }})();
    </script>
    """


# =========================
# UI (NO CANVIADA)
# =========================

col1,col2,col3 = st.columns(3)

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

html = build_html(path, d_tubo, 700, d_aspo, spalla)

components.html(html, height=700)
