import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# =========================
# PARAMS
# =========================

d_aspo = st.number_input("Ø Aspo", value=450.0)
spalla = st.number_input("Spalla", value=95.0)
lunghezza = st.number_input("Lunghezza (m)", value=30.0)

passo = st.number_input("Passo assiale", value=20.0)
incremento = st.number_input("Incremento strato", value=20.0)

rit_base = st.number_input("Ritardo base", value=180.0)
rit_top = st.number_input("Ritardo spalla", value=180.0)

# =========================
# COIL GENERATION
# =========================

def build():
    pts = []

    r = d_aspo/2 + 10
    z_min = 0
    z_max = spalla

    z = z_min
    theta = 0

    direction = 1
    delay = 0
    pending = False

    for i in range(20000):

        theta += np.deg2rad(4)

        if delay > 0:
            delay -= 4

        else:
            if pending:
                r += incremento
                pending = False

            z += direction * (passo/(2*np.pi)) * np.deg2rad(4)

            if z >= z_max:
                z = z_max
                delay = rit_top
                pending = True
                direction = -1

            elif z <= z_min:
                z = z_min
                delay = rit_base
                pending = True
                direction = 1

        x = r*np.cos(theta)
        y = r*np.sin(theta)

        pts.append([x,y,z])

        if len(pts) > 2:
            if np.sum(np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1)) > lunghezza*1000:
                break

    pts = np.array(pts)
    pts[:,2] -= spalla/2

    return pts

points = build()

# =========================
# VIEWER
# =========================

def viewer(points):

    pts = json.dumps(points.tolist())

    return f"""
    <div id="viewer" style="width:100%;height:700px;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    setTimeout(() => {{

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const container = document.getElementById("viewer");
        const w = container.clientWidth;
        const h = container.clientHeight;

        const camera = new THREE.PerspectiveCamera(45, w/h, 1, 10000);
        camera.position.set(700,-900,300);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(w,h);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // MACHINE GROUP (ROTATES)
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        // =====================
        // TUBE
        // =====================

        const raw = {pts};
        const vecs = raw.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

        class Curve extends THREE.Curve {{
            constructor(p){{super();this.p=p}}
            getPoint(t){{
                const f=t*(this.p.length-1);
                const i=Math.floor(f);
                const t2=f-i;
                return new THREE.Vector3().lerpVectors(this.p[i],this.p[i+1],t2);
            }}
        }}

        const curve = new Curve(vecs);

        const geo = new THREE.TubeGeometry(curve, 2000, 6, 16, false);
        const mat = new THREE.MeshStandardMaterial({{color:0xdddddd}});
        const tube = new THREE.Mesh(geo,mat);

        machine.add(tube);

        // =====================
        // MANDREL
        // =====================

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry({d_aspo/2},{d_aspo/2},{spalla},64),
            new THREE.MeshStandardMaterial({{color:0x555555,transparent:true,opacity:0.4}})
        );

        mandrel.rotation.x = Math.PI/2;
        machine.add(mandrel);

        // =====================
        // GUIDATUBO (FIX X)
        // =====================

        const guide = new THREE.Group();
        scene.add(guide);

        const baseX = {d_aspo/2 + 120};

        const block = new THREE.Mesh(
            new THREE.BoxGeometry(20,20,20),
            new THREE.MeshStandardMaterial({{color:0xffffff}})
        );
        guide.add(block);

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.HemisphereLight(0xffffff,0x444444));

        // =====================
        // ANIMATION
        // =====================

        let idx = 0;

        function animate(){{
            requestAnimationFrame(animate);

            // ROTACIÓ MANDRÍ
            machine.rotation.z += 0.02;

            if(idx < vecs.length){{
                const p = vecs[idx];

                // AXIAL
                const z = p.z;

                // RADIAL NOMÉS CAP A FORA
                const r = Math.sqrt(p.x*p.x + p.y*p.y);

                guide.position.set(baseX + (r - {d_aspo/2}), 0, z);

                idx++;
            }}

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }},100);
    </script>
    """

components.html(viewer(points), height=720)
