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

def smoothstep(u):
    return 0.5 - 0.5*np.cos(np.pi*u)

def polyline_length(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

# =========================
# GEOMETRY (CORRECTE)
# =========================

def build_coil(d_aspo, spalla, Lm, d_rame, sp, pitch, step_r):

    L = Lm * 1000
    d_tubo = d_rame + 2*sp
    R = d_tubo / 2

    z_min = R
    z_max = spalla - R
    r = d_aspo/2 + R

    theta = 0
    z = z_min
    dir = 1

    step = np.deg2rad(1)
    dz = pitch/(2*np.pi)

    pts = []

    def add():
        pts.append([r*np.cos(theta), r*np.sin(theta), z])

    add()

    while polyline_length(np.array(pts)) < L:

        while True:
            theta += step
            z += dir * dz * step
            add()

            if z >= z_max:
                z = z_max
                break
            if z <= z_min:
                z = z_min
                break

        # suavitzat radial
        r0 = r
        r1 = r + step_r

        for i in range(8):
            theta += step
            u = (i+1)/8
            r = r0 + (r1-r0)*smoothstep(u)
            add()

        r = r1
        dir *= -1

    pts = np.array(pts)

    # trim
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    if cum[-1] > L:
        i = np.searchsorted(cum, L)-1
        p0, p1 = pts[i], pts[i+1]
        a = (L-cum[i])/(np.linalg.norm(p1-p0)+EPS)
        pts = np.vstack([pts[:i+1], p0 + a*(p1-p0)])

    r_path = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    diam_ext = 2*np.max(r_path + R)

    meta = {
        "DiametroTubo": d_tubo,
        "DiametroEsterno": diam_ext
    }

    return pts, meta

# =========================
# VIEWER (FIX DEFINITIU)
# =========================

def viewer(points, d_tubo, h, anim, speed):

    pts = json.dumps(points.tolist())
    r = d_tubo/2

    return f"""
<div style="width:100%;height:{h}px;" id="v"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>
const el = document.getElementById("v");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(45, el.clientWidth/el.clientHeight,0.1,100000);

const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(el.clientWidth, el.clientHeight);
el.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1));

const light = new THREE.DirectionalLight(0xffffff,0.8);
light.position.set(5,5,5);
scene.add(light);

const pts = {pts};
const v = pts.map(p=>new THREE.Vector3(p[0],p[1],p[2]));

class C extends THREE.Curve {{
 getPoint(t){{
  const n=v.length;
  const f=t*(n-1);
  const i=Math.floor(f);
  const a=v[Math.max(0,Math.min(i,n-2))];
  const b=v[Math.max(1,Math.min(i+1,n-1))];
  return new THREE.Vector3().lerpVectors(a,b,f-i);
 }}
}}

const curve=new C();

let g=new THREE.TubeGeometry(curve,2000,{r},32,false);
g=g.toNonIndexed();

const m=new THREE.MeshStandardMaterial({{color:0xe6e6e6}});
const mesh=new THREE.Mesh(g,m);
scene.add(mesh);

const box=new THREE.Box3().setFromPoints(v);
const c=new THREE.Vector3();
box.getCenter(c);

const size=new THREE.Vector3();
box.getSize(size);

const d=Math.max(size.x,size.y,size.z)*1.8;

camera.position.set(c.x+d,c.y+d,c.z+d*0.6);
camera.lookAt(c);
controls.target.copy(c);

let p=0;
const total=g.attributes.position.count;

if ({str(anim).lower()}) g.setDrawRange(0,0);
else g.setDrawRange(0,total);

function loop(){{
 requestAnimationFrame(loop);

 if ({str(anim).lower()}){{
  p+= {speed}*0.002;
  if(p>1)p=1;
  g.setDrawRange(0,Math.floor(p*total));
 }}

 controls.update();
 renderer.render(scene,camera);
}}

loop();
</script>
"""

# =========================
# UI
# =========================

c1,c2,c3,c4 = st.columns(4)

with c1:
    d_aspo = st.number_input("Ø Aspo", value=450.0)
    spalla = st.number_input("Spalla", value=95.0)

with c2:
    size = st.selectbox("Rame", list(COPPER_SIZES_MM.keys()))
    sp = st.number_input("Guaina", value=7.0)
    L = st.number_input("Lunghezza", value=50.0)
    d_rame = COPPER_SIZES_MM[size]

with c3:
    pitch = st.number_input("Passo assiale", value=20.0)
    step_r = st.number_input("Passo radiale", value=20.0)

with c4:
    h = st.slider("Altezza", 400, 900, 700)
    anim = st.checkbox("Animazione", False)
    speed = st.slider("Velocità", 0.1, 5.0, 1.0)

# =========================
# RUN
# =========================

p, m = build_coil(d_aspo, spalla, L, d_rame, sp, pitch, step_r)

components.html(viewer(p, m["DiametroTubo"], h, anim, speed), height=h)

st.metric("Diametro tubo", f"{m['DiametroTubo']:.2f} mm")
st.metric("Diametro esterno", f"{m['DiametroEsterno']:.1f} mm")
