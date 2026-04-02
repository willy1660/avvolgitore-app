// ==============================
// CAPS REALS (discos)
// ==============================

function createCap(position, direction, color) {

    const geometry = new THREE.CircleGeometry(d/2, 32)

    const material = new THREE.MeshBasicMaterial({
        color: color,
        side: THREE.DoubleSide
    })

    const cap = new THREE.Mesh(geometry, material)

    const up = new THREE.Vector3(0,0,1)
    const dir = direction.clone().normalize()

    if (dir.length() > 0) {
        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir)
        cap.quaternion.copy(quat)
    }

    cap.position.copy(position)
    scene.add(cap)
}

// CAP INICI
if (pts.length >= 2) {
    const start = pts[0]
    const dirStart = pts[1].clone().sub(pts[0]).multiplyScalar(-1)
    createCap(start, dirStart, 0x00ff00)

    // CAP FINAL
    const end = pts[pts.length-1]
    const dirEnd = pts[pts.length-1].clone().sub(pts[pts.length-2])
    createCap(end, dirEnd, 0xff0000)
}
