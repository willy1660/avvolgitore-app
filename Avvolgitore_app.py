def build_coil_real(
    d_aspo_mm,
    spalla_mm,
    lunghezza_m,
    d_rame_mm,
    spessore_guaina_mm,
    passo_assiale,
    passo_radiale,
    ritardo_top_deg,
    ritardo_bottom_deg,
):

    lunghezza_mm = lunghezza_m * 1000
    d_tubo = d_rame_mm + 2*spessore_guaina_mm

    dz_dtheta = passo_assiale / (2*np.pi)

    r = d_aspo_mm/2 + d_tubo/2
    r0 = r

    theta = 0
    z = 0

    direction = 1

    points = []

    def add():
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        points.append([x,y,z])

    while True:

        if len(points) > 2 and polyline_length(np.array(points)) >= lunghezza_mm:
            break

        # =========================
        # RUN
        # =========================
        while True:

            theta += 0.05
            z += direction * dz_dtheta * 0.05

            add()

            if direction == 1 and z >= spalla_mm:
                z = spalla_mm
                break

            if direction == -1 and z <= 0:
                z = 0
                break

        # =========================
        # STOP (RITARDO REAL)
        # =========================

        ritardo = ritardo_top_deg if direction == 1 else ritardo_bottom_deg
        theta_stop = np.deg2rad(ritardo)

        steps = max(5, int(ritardo / 5))

        for _ in range(steps):
            theta += theta_stop / steps
            add()

        # =========================
        # CANVI DIRECCIÓ + CAPA
        # =========================

        direction *= -1
        r += passo_radiale

    path = trim_polyline(np.array(points), lunghezza_mm)

    r_max = np.max(np.sqrt(path[:,0]**2 + path[:,1]**2))
    diam_ext = 2*(r_max + d_tubo/2)

    meta = {
        "DiametroTubo": d_tubo,
        "DiametroEsterno": diam_ext,
        "Capes": int((r_max-r0)/passo_radiale)+1,
        "VolteTotali": compute_total_turns(path)
    }

    return path, meta
