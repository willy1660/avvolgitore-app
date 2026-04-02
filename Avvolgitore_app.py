import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os

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

# =========================
# LENGTH
# =========================

def add_point(pts, x, y, z, total_len):
    if len(pts) > 0:
        total_len += np.linalg.norm(np.array([x,y,z]) - np.array(pts[-1]))
    pts.append([x,y,z])
    return total_len

def trim_polyline(points, target_length):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    if cum[-1] <= target_length:
        return points

    idx = np.searchsorted(cum, target_length) - 1
    p0, p1 = points[idx], points[idx+1]
    alpha = (target_length - cum[idx]) / (np.linalg.norm(p1 - p0) + EPS)

    return np.vstack([points[:idx+1], p0 + alpha*(p1 - p0)])

# =========================
# GEOMETRY
# =========================

def build_coil(d_aspo, spalla, Lm, d_rame, sp, passo_assiale, passo_radiale, rit_min, rit_max):

    L_target = Lm * 1000
    total_len = 0

    d_tubo = d_rame + 2*sp
    R = d_tubo/2

    z_min = R
    z_max = spalla - R
    r = d_aspo/2 + R

    theta = 0
    z = z_min
    direction = 1

    step = np.deg2rad(1)
    dz = passo_assiale/(2*np.pi)

    pts = []

    total_len = add_point(pts, r*np.cos(theta), r*np.sin(theta), z, total_len)

    # enganxament
    theta_attach = np.pi
    for _ in range(40):
        if total_len >= L_target: break
        theta += theta_attach/40
        total_len = add_point(pts, r*np.cos(theta), r*np.sin(theta), z, total_len)

    while total_len < L_target:

        while True:
            if total_len >= L_target: break

            theta += step
            z += direction*dz*step
            total_len = add_point(pts, r*np.cos(theta), r*np.sin(theta), z, total_len)

            if direction == 1 and z >= z_max:
                z = z_max
                break
            if direction == -1 and z <= z_min:
                z = z_min
                break

        if total_len >= L_target: break

        rit = rit_max if direction == 1 else rit_min
        theta_dwell = np.deg2rad(rit)

        if theta_dwell > 0:
            steps = max(6, int(rit/2))
            r0 = r
            r1 = r + passo_radiale

            for i in range(steps):
                if total_len >= L_target: break
                theta += theta_dwell/steps
                u = (i+1)/steps
                r = r0 + (r1-r0)*smoothstep(u)
                total_len = add_point(pts, r*np.cos(theta), r*np.sin(theta), z, total_len)

            r = r1
        else:
            r += passo_radiale

        direction *= -1

    pts = np.array(pts)
    pts = trim_polyline(pts, L_target)

    r_path = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    diam_ext = 2*np.max(r_path + R)

    return pts, {
        "DiametroTubo": d_tubo,
        "DiametroEsterno": diam_ext,
    }

# =========================
# VIEWER (FIX FINAL)
# =========================

def viewer(points, d_tubo, d_aspo, spalla, h, anim, speed):

    pts = json.dumps(points.tolist())
    r = d_tubo/2
    r_aspo = d_aspo/2

    return f"""
<div style="width:100%;height:{h}px;" id="viewer"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

<script>
const el = document.getElementById("viewer");

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

// =========================
// TUB
// =========================

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

const m=new THREE.MeshStandardMaterial({{
    color:0xcfcfcf,
    roughness:0.85
}});

const mesh=new THREE.Mesh(g,m);
scene.add(mesh);

// =========================
// CAPS
// =========================

function cap(pos,dir,color){{
 const geo=new THREE.CircleGeometry({r},32);
 const mat=new THREE.MeshBasicMaterial({{color:color,side:THREE.DoubleSide}});
 const c=new THREE.Mesh(geo,mat);

 const up=new THREE.Vector3(0,0,1);
 const q=new THREE.Quaternion().setFromUnitVectors(up,dir.clone().normalize());
 c.quaternion.copy(q);

 c.position.copy(pos);
 scene.add(c);
}}

if(v.length>1){{
 cap(v[0],v[1].clone().sub(v[0]).multiplyScalar(-1),0x00ff00);
 cap(v[v.length-1],v[v.length-1].clone().sub(v[v.length-2]),0xff0000);
}}

// =========================
// MANDRÍ (NOU)
// =========================

const cyl = new THREE.Mesh(
 new THREE.CylinderGeometry({r_aspo},{r_aspo},{spalla},64),
 new THREE.MeshStandardMaterial({{
  color:0x444444,
  roughness:0.9,
  metalness:0.2,
  transparent:true,
  opacity:0.4
 }})
);

cyl.rotation.x = Math.PI/2;
scene.add(cyl);

// =========================
// CAMERA
// =========================

const box=new THREE.Box3().setFromPoints(v);
const c=new THREE.Vector3();
box.getCenter(c);

const size=new THREE.Vector3();
box.getSize(size);

const d=Math.max(size.x,size.y,size.z)*1.8;

camera.position.set(c.x+d,c.y+d,c.z+d*0.6);
camera.lookAt(c);
controls.target.copy(c);

// =========================
// ANIM
// =========================

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
    passo = st.number_input("Passo assiale", value=20.0)
    step_r = st.number_input("Passo radiale", value=20.0)
    rit_min = st.number_input("Ritardo base", value=180.0)
    rit_max = st.number_input("Ritardo spalla", value=180.0)

with c4:
    h = st.slider("Altezza", 400, 900, 700)
    anim = st.checkbox("Animazione", False)
    speed = st.slider("Velocità", 0.1, 5.0, 1.0)

# =========================
# RUN
# =========================

p, m = build_coil(d_aspo, spalla, L, d_rame, sp, passo, step_r, rit_min, rit_max)

components.html(viewer(p, m["DiametroTubo"], d_aspo, spalla, h, anim, speed), height=h)

st.metric("Diametro esterno", f"{m['DiametroEsterno']:.1f} mm")
