# SUBSTITUEIX NOMÉS EL BLOC RITARDO PER AQUEST

if ritardo_max_deg > 0:

    ritardo_deg = np.random.uniform(ritardo_min_deg, ritardo_max_deg)
    dtheta_delay = math.radians(ritardo_deg)

    r_next = r + passo_radiale

    n = 60
    t = np.linspace(0,dtheta_delay,n)

    s = 0.5 - 0.5*np.cos(np.linspace(0,math.pi,n))

    r_vals = r + (r_next - r)*s
    theta_vals = theta + t

    # 🔥 FIX KINK → suavitzat també en Z
    dz_small = (z1 - z0) * 0.02
    z_vals = z1 + dz_small * (s - 0.5)

    x = r_vals*np.cos(theta_vals)
    y = r_vals*np.sin(theta_vals)

    delay = np.column_stack([x,y,z_vals])[1:]
    points.extend(delay.tolist())

    theta += dtheta_delay
    r = r_next
