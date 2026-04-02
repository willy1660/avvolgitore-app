# ==============================
# TRANSICIÓ CORRECTA (NO SPLINE)
# ==============================

ritardo_deg = 0.0
if ritardo_max_deg > 0:
    ritardo_deg = np.random.uniform(ritardo_min_deg, ritardo_max_deg)

total_turn = 0.2 + ritardo_deg / 360.0
dtheta = 2 * math.pi * total_turn

r_next = r + passo_radiale

n = 60
t = np.linspace(0, dtheta, n)

# suavitzat real (cosinus)
s = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n))

theta_vals = theta + t
r_vals = r + (r_next - r) * s
z_vals = np.full_like(theta_vals, z1)

x = r_vals * np.cos(theta_vals)
y = r_vals * np.sin(theta_vals)

transition = np.column_stack([x, y, z_vals])[1:]
points.extend(transition.tolist())

theta += dtheta
r = r_next
