def build_viewer_html(
    points,
    d_tubo,
    altezza,
    animazione,
    velocita,
    show_grid,
    show_axes,
    show_plane,
    light_mode,
    transparency,
    manual_progress,
    show_caps,
):
    pts = points.tolist()
    points_json = json.dumps(pts)

    r_tubo = d_tubo / 2.0
    tubular_segments = min(8000, max(1500, int(len(pts) * 0.9)))
    radial_segments = 56

    # 🎯 COLORS PRO
    if light_mode:
        bg_color = "0xf4f6f8"
        tube_color = "0x9aa4ad"   # gris tècnic (clau)
        grid_c1 = "0xb8c2cc"
        grid_c2 = "0xd8dee6"
        hemi_ground = "0xe6eaef"
        plane_opacity = 0.08
    else:
        bg_color = "0x0b0d10"
        tube_color = "0xcfd5db"   # gris clar (no blanc)
        grid_c1 = "0x3b4452"
        grid_c2 = "0x232933"
        hemi_ground = "0x20242c"
        plane_opacity = 0.18

    html = f"""
    <div style="width:100%;height:{altezza}px;border-radius:16px;overflow:hidden;background:{'#f4f6f8' if light_mode else '#0b0d10'};">
      <div id="viewer" style="width:100%;height:100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    const container = document.getElementById("viewer");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color({bg_color});

    const camera = new THREE.PerspectiveCamera(
      42,
      container.clientWidth / container.clientHeight,
      0.1,
      100000
    );

    const renderer = new THREE.WebGLRenderer({{
      antialias: true,
      powerPreference: "high-performance"
    }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // 💡 LIGHTS més contrast
    scene.add(new THREE.AmbientLight(0xffffff, {0.85 if light_mode else 0.55}));

    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(800,800,800);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0x9fb3c8, 0.35);
    fill.position.set(-800,-300,500);
    scene.add(fill);

    // GEOMETRY
    const pts = {points_json};
    const vec = pts.map(p => new THREE.Vector3(p[0], p[1], p[2]));

    class Curve extends THREE.Curve {{
      constructor(points) {{ super(); this.points = points; }}
      getPoint(t) {{
        const n = this.points.length;
        const f = t*(n-1);
        const i = Math.floor(f);
        return this.points[Math.min(i, n-1)];
      }}
    }}

    const curve = new Curve(vec);

    const tubeGeom = new THREE.TubeGeometry(curve, {tubular_segments}, {r_tubo}, {radial_segments}, false);

    // 🎯 MATERIAL MILLORAT (important)
    const tubeMat = new THREE.MeshStandardMaterial({{
        color: {tube_color},
        roughness: 0.55,
        metalness: 0.15,
        transparent: true,
        opacity: {1.0 - transparency}
    }});

    const mesh = new THREE.Mesh(tubeGeom, tubeMat);
    scene.add(mesh);

    // GRID
    if ({str(show_grid).lower()}) {{
        const grid = new THREE.GridHelper(2000, 40, {grid_c1}, {grid_c2});
        grid.material.opacity = 0.35;
        grid.material.transparent = true;
        scene.add(grid);
    }}

    // AXES
    if ({str(show_axes).lower()}) {{
        scene.add(new THREE.AxesHelper(200));
    }}

    // SHADOW PLANE
    if ({str(show_plane).lower()}) {{
        const plane = new THREE.Mesh(
            new THREE.PlaneGeometry(2000,2000),
            new THREE.MeshBasicMaterial({{color:0x000000, transparent:true, opacity:{plane_opacity}}})
        );
        plane.rotation.x = -Math.PI/2;
        scene.add(plane);
    }}

    camera.position.set(600,600,400);
    controls.target.set(0,0,200);

    // PROGRESS
    const total = tubeGeom.attributes.position.count;

    if ({str(animazione).lower()}) {{
        tubeGeom.setDrawRange(0, 0);
    }} else {{
        tubeGeom.setDrawRange(0, Math.floor(total * {manual_progress}));
    }}

    let progress = 0;

    function animate(){{
        requestAnimationFrame(animate);

        if ({str(animazione).lower()}) {{
            progress += {velocita} * 0.002;
            if (progress > 1) progress = 1;
            tubeGeom.setDrawRange(0, Math.floor(progress * total));
        }}

        controls.update();
        renderer.render(scene, camera);
    }}

    animate();
    </script>
    """
    return html
