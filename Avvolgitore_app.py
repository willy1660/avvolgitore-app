import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Three.js Test", layout="wide")
st.title("Three.js Minimal Test")

height = 600

html = f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ margin: 0; padding: 0; overflow: hidden; }}
    #viewer {{ width: 100%; height: {height}px; display: block; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  
  <script src="[cdnjs.cloudflare.com](https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js)"></script>
  <script src="[cdn.jsdelivr.net](https://cdn.jsdelivr.net/npm/three@0.128/examples/js/controls/OrbitControls.js)"></script>
  
  <script>
    // Wait for everything to be ready
    function init() {{
      const container = document.getElementById("viewer");
      
      // CRITICAL: Use explicit dimensions, not clientWidth/clientHeight
      const width = container.offsetWidth || {height * 1.5};
      const height = {height};
      
      // Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a2e);
      
      // Camera
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
      camera.position.set(300, 300, 300);
      camera.lookAt(0, 0, 0);
      
      // Renderer - CRITICAL: Set size explicitly
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);
      
      // Controls
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      
      // Lights
      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(100, 200, 100);
      scene.add(dirLight);
      
      // Test cube
      const geometry = new THREE.BoxGeometry(100, 100, 100);
      const material = new THREE.MeshStandardMaterial({{ color: 0x00ff88 }});
      const cube = new THREE.Mesh(geometry, material);
      scene.add(cube);
      
      // Grid helper
      const grid = new THREE.GridHelper(500, 20, 0x444444, 0x222222);
      grid.rotation.x = Math.PI / 2;
      scene.add(grid);
      
      // Axes
      scene.add(new THREE.AxesHelper(150));
      
      // Animation loop
      function animate() {{
        requestAnimationFrame(animate);
        cube.rotation.x += 0.005;
        cube.rotation.y += 0.01;
        controls.update();
        renderer.render(scene, camera);
      }}
      
      animate();
      
      console.log("Three.js initialized successfully");
    }}
    
    // CRITICAL: Wait for DOM
    if (document.readyState === 'complete') {{
      init();
    }} else {{
      window.addEventListener('load', init);
    }}
  </script>
</body>
</html>
"""

components.html(html, height=height + 10)

st.success("If you see a rotating green cube above, Three.js is working!")
