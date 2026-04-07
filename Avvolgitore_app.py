def viewer(d_aspo, spalla, d_tubo, passo, incremento, rit_b, rit_t, lunghezza, altezza, anim, vel):

    return f"""
    <div id="viewer" style="width:100%;height:{altezza}px;background:#000;"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js"></script>

    <script>
    (() => {{

        const el = document.getElementById("viewer");
        el.innerHTML = "";

        const w = el.clientWidth;
        const h = el.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        const camera = new THREE.PerspectiveCamera(40, w/h, 0.1, 10000);
        camera.position.set(-500, -700, 400);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(w,h);
        el.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        // =====================
        // PARAMS
        // =====================

        const R = {d_aspo}/2;
        const H = {spalla};
        const Rt = {d_tubo}/2;

        const maxLen = {lunghezza} * 1000;

        const guideX = -(R + 80);

        let guideY = R + Rt;
        let guideZ = Rt;

        // =====================
        // ASPO
        // =====================

        const machine = new THREE.Group();
        scene.add(machine);

        const red = new THREE.MeshStandardMaterial({{color:0xff3333}});

        const mandrel = new THREE.Mesh(
            new THREE.CylinderGeometry(R, R, H, 80),
            red
        );
        mandrel.rotation.x = Math.PI/2;
        mandrel.position.z = H/2;
        machine.add(mandrel);

        // =====================
        // GUIDATUBO
        // =====================

        const guide = new THREE.Mesh(
            new THREE.BoxGeometry(30,20,20),
            new THREE.MeshStandardMaterial({{color:0x0044ff}})
        );
        scene.add(guide);

        // =====================
        // TUB
        // =====================

        let points = [];
        let totalLength = 0;

        const geometry = new THREE.BufferGeometry();
        const material = new THREE.LineBasicMaterial({{color:0xffffff}});
        const line = new THREE.Line(geometry, material);
        scene.add(line);

        function currentTubePoint() {{

            // 🔥 SOLUCIÓ REAL
            const theta = -machine.rotation.z + Math.PI;

            const x = guideY * Math.cos(theta);
            const y = guideY * Math.sin(theta);

            return new THREE.Vector3(x, y, guideZ);
        }}

        function addPoint(p) {{
            if (points.length > 0) {{
                const prev = points[points.length - 1];
                const d = p.distanceTo(prev);

                if (totalLength + d > maxLen) return;

                totalLength += d;
            }}

            points.push(p);
            geometry.setFromPoints(points);
        }}

        // =====================
        // LIGHT
        // =====================

        scene.add(new THREE.AmbientLight(0xffffff,0.8));

        // =====================
        // MOTION
        // =====================

        let dir = 1;
        let delay = 0;

        function animate(){{
            requestAnimationFrame(animate);

            // aspo horari
            machine.rotation.z -= 0.02 * {vel};

            if (delay > 0) delay--;
            else {{

                guideZ += dir * {passo} * 0.02 * {vel};

                if (guideZ >= H - Rt) {{
                    guideZ = H - Rt;
                    guideY += {incremento};
                    delay = {rit_t};
                    dir = -1;
                }}

                if (guideZ <= Rt) {{
                    guideZ = Rt;
                    guideY += {incremento};
                    delay = {rit_b};
                    dir = 1;
                }}
            }}

            guide.position.set(guideX, guideY, guideZ);

            addPoint(currentTubePoint());

            controls.update();
            renderer.render(scene,camera);
        }}

        animate();

    }})();
    </script>
    """
