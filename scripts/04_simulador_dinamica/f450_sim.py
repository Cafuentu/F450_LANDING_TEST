"""
f450_sim.py - Simulador 6-DOF del cuadricoptero F450 (proyecto SLSD)

Independiente de Blender/Isaac: solo numpy. Ejes cuerpo FLU (X adelante, Y izquierda, Z arriba),
mundo ENU-like con Z arriba. Rotores en orden PX4/Pegasus quad-X:
  0 = delantero derecho (CCW), 1 = trasero izquierdo (CCW), 2 = delantero izquierdo (CW), 3 = trasero derecho (CW)

Uso:  python f450_sim.py            -> corre la mision de prueba y escribe vuelo_prueba.csv
"""
import numpy as np

G = 9.80665
RHO = 1.225


class Params:
    # --- cuerpo (del modelo f450_real.blend) ---
    m = 1.1654                                   # kg
    I = np.diag([0.01291, 0.01353, 0.02342])     # kg m^2 en el CG
    r_cg = np.array([-0.0034, 0.0002, 0.0124])   # CG respecto del centro del frame (m)
    h_cg_ground = 0.1147 + 0.0124                # altura del CG con el tren apoyado (m)
    # --- rotores: motor 2216 880KV + helice 1045 ---
    arm = 0.225
    D = 0.254
    CT, CP = 0.110, 0.045                        # coeficientes estaticos tipicos de una 10x4.5
    kT = CT * RHO * D**4 / (4 * np.pi**2)        # N/(rad/s)^2
    kQ = CP / (2 * np.pi) * RHO * D**5 / (4 * np.pi**2)   # N m/(rad/s)^2
    w_max = 785.0                                # rad/s (~7500 rpm con 3S)
    w_idle = 120.0
    tau_motor = 0.05                             # s, constante de tiempo motor+ESC
    Jr = 6.0e-5                                  # kg m^2, inercia rotor+helice
    a = arm / np.sqrt(2)
    pos = np.array([[a, -a, 0], [-a, a, 0], [a, a, 0], [-a, -a, 0]], float)
    spin = np.array([+1, +1, -1, -1], float)     # +1 = CCW visto desde arriba
    # --- aerodinamica ---
    kd_lin = 0.10                                # N/(m/s)
    kd_quad = 0.018                              # N/(m/s)^2  (~0.5 rho Cd A)
    kd_rot = 0.002                               # N m/(rad/s)
    # --- suelo ---
    k_ground, c_ground, mu = 6000.0, 90.0, 0.9
    # --- energia ---
    eta = 0.75                                   # rendimiento motor+ESC
    P_avionics = 9.0                             # W: Raspberry Pi + camara (~6.5) y Pixhawk/GPS/radios (~2.5)
    E_batt_Wh = 57.7                             # 3S 11.1 V 5200 mAh
    usable = 0.80


def q_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def q_from_euler(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll/2), np.sin(roll/2), np.cos(pitch/2), np.sin(pitch/2), np.cos(yaw/2), np.sin(yaw/2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def euler_from_q(q):
    w, x, y, z = q
    return np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)), np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


class Quad:
    """Dinamica de cuerpo rigido + rotores de primer orden."""
    def __init__(self, P=Params, p0=(0, 0, None), yaw0=0.0):
        self.P = P
        z0 = P.h_cg_ground if p0[2] is None else p0[2]
        self.p = np.array([p0[0], p0[1], z0], float); self.v = np.zeros(3)
        self.q = q_from_euler(0, 0, yaw0); self.w = np.zeros(3)
        self.rot = np.zeros(4)                   # velocidad de cada rotor (rad/s)
        self.rot_angle = np.zeros(4)
        self.E_Wh = 0.0; self.contact = True; self.P_elec = 0.0; self.T = np.zeros(4)

    def step(self, w_cmd, dt):
        P = self.P
        self.rot += (np.clip(w_cmd, 0, P.w_max) - self.rot) * dt / P.tau_motor
        self.rot_angle += P.spin * self.rot * dt
        R = q_to_R(self.q)
        # efecto suelo (Cheeseman-Bennett), acotado
        z_rot = max(self.p[2] - P.r_cg[2] + 0.055, 0.05)
        ge = 1.0 / (1.0 - min((P.D/2 / (4*z_rot))**2, 0.25))
        T = P.kT * self.rot**2 * ge
        self.T = T
        F_b = np.array([0, 0, T.sum()])
        r = P.pos - P.r_cg
        tau = np.array([(r[:, 1]*T).sum(), (-r[:, 0]*T).sum(), (-P.spin * P.kQ * self.rot**2).sum()])
        tau -= np.cross(self.w, np.array([0, 0, P.Jr * (P.spin*self.rot).sum()]))      # giroscopico de rotores
        tau -= P.kd_rot * self.w
        vn = np.linalg.norm(self.v)
        F = R @ F_b + np.array([0, 0, -P.m*G]) - (P.kd_lin + P.kd_quad*vn) * self.v
        # contacto con el suelo (tren como resorte-amortiguador vertical + roce)
        pen = P.h_cg_ground - self.p[2]
        self.contact = pen > 0
        if self.contact:
            N = max(P.k_ground*pen - P.c_ground*self.v[2], 0.0)
            F[2] += N
            vh = self.v[:2]; nvh = np.linalg.norm(vh)
            if nvh > 1e-4: F[:2] -= min(P.mu*N, P.m*nvh/dt) * vh/nvh
            tau -= 0.6 * self.w                                   # el tren apoyado frena la rotacion
            e = euler_from_q(self.q); tau[:2] -= 8.0 * e[:2]      # y nivela el cuerpo
        self.v += F/P.m * dt
        self.p += self.v * dt
        self.w += np.linalg.solve(P.I, tau - np.cross(self.w, P.I @ self.w)) * dt
        self.q = q_mul(self.q, np.array([1, *(0.5*self.w*dt)])); self.q /= np.linalg.norm(self.q)
        self.P_elec = (P.kQ * self.rot**3).sum() / P.eta + P.P_avionics
        self.E_Wh += self.P_elec * dt / 3600


class Controller:
    """Cascada tipo PX4: posicion -> velocidad -> actitud -> tasas -> mezcla."""
    def __init__(self, P=Params):
        self.P = P
        self.kp_pos = np.array([1.0, 1.0, 1.2])
        self.kp_vel = np.array([1.8, 1.8, 4.5]); self.ki_vel = np.array([0.5, 0.5, 1.2]); self.i_vel = np.zeros(3)
        self.v_max_h, self.v_up, self.v_dn = 2.0, 2.0, 1.0
        self.kp_att = np.array([7.0, 7.0, 3.0])
        self.kp_rate = np.array([14.0, 14.0, 8.0])        # 1/s: aceleracion angular por error de tasa
        self.ki_rate = np.array([0.25, 0.25, 0.12]); self.kd_rate = np.array([0.0004, 0.0004, 0.0])
        self.i_rate = np.zeros(3); self.w_prev = np.zeros(3)
        self.tilt_max = np.radians(20)
        r = P.pos - P.r_cg
        A = np.vstack([np.ones(4), r[:, 1], -r[:, 0], -P.spin * P.kQ / P.kT])
        self.Ainv = np.linalg.inv(A)

    def update(self, quad, sp, dt):
        """sp: dict(mode='pos'|'att'|'idle'|'off', p, yaw, roll, pitch, vz_land)"""
        P = self.P
        if sp["mode"] == "off": return np.zeros(4)
        if sp["mode"] == "idle":
            self.i_vel[:] = 0; self.i_rate[:] = 0
            return np.full(4, P.w_idle)
        R = q_to_R(quad.q); yaw = euler_from_q(quad.q)[2]
        # --- posicion / velocidad ---
        v_sp = self.kp_pos * (sp["p"] - quad.p)
        nh = np.linalg.norm(v_sp[:2])
        if nh > self.v_max_h: v_sp[:2] *= self.v_max_h / nh
        v_sp[2] = np.clip(v_sp[2], -self.v_dn, self.v_up)
        if sp.get("vz") is not None: v_sp[2] = sp["vz"]
        ev = v_sp - quad.v
        self.i_vel = np.clip(self.i_vel + self.ki_vel * ev * dt, -3, 3)
        a_sp = self.kp_vel * ev + self.i_vel + np.array([0, 0, G])
        if sp["mode"] == "att":                                  # actitud mandada, altura sostenida
            q_sp = q_from_euler(sp["roll"], sp["pitch"], sp["yaw"])
            thrust = P.m * a_sp[2] / max(R[2, 2], 0.5)
            self.i_vel[:2] = 0
        else:
            a_sp[2] = max(a_sp[2], 0.3*G)
            zb = a_sp / np.linalg.norm(a_sp)
            tilt = np.arccos(np.clip(zb[2], -1, 1))
            if tilt > self.tilt_max:
                h = zb[:2] / np.linalg.norm(zb[:2]); zb = np.array([*(h*np.sin(self.tilt_max)), np.cos(self.tilt_max)])
            xc = np.array([np.cos(sp["yaw"]), np.sin(sp["yaw"]), 0.0])
            yb = np.cross(zb, xc); yb /= np.linalg.norm(yb); xb = np.cross(yb, zb)
            Rsp = np.column_stack([xb, yb, zb])
            q_sp = _R_to_q(Rsp)
            thrust = P.m * a_sp[2] / max(zb[2], 0.5)
        # --- actitud -> tasas ---
        qe = q_mul(q_conj(quad.q), q_sp)
        if qe[0] < 0: qe = -qe
        w_sp = self.kp_att * 2.0 * qe[1:]
        w_sp = np.clip(w_sp, -np.radians([220, 220, 90]), np.radians([220, 220, 90]))
        # --- tasas -> torques ---
        ew = w_sp - quad.w
        self.i_rate = np.clip(self.i_rate + self.ki_rate * ew * dt, -0.3, 0.3)
        dw = (quad.w - self.w_prev) / dt; self.w_prev = quad.w.copy()
        tau = P.I @ (self.kp_rate * ew) + self.i_rate - self.kd_rate * dw
        # --- mezcla ---
        T = self.Ainv @ np.array([thrust, *tau])
        T = np.clip(T, P.kT * P.w_idle**2, P.kT * P.w_max**2)
        return np.sqrt(T / P.kT)


def _R_to_q(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; q = np.array([0.25*s, (R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s])
    else:
        i = np.argmax(np.diag(R)); j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = np.zeros(4); q[1+i] = 0.25*s; q[0] = (R[k, j]-R[j, k])/s; q[1+j] = (R[j, i]+R[i, j])/s; q[1+k] = (R[k, i]+R[i, k])/s
    return q / np.linalg.norm(q)


def test_mission(home):
    """Lista de (t_inicio, nombre, setpoint). Derecha = -Y (ejes FLU)."""
    hx, hy = home; d = np.radians
    P = lambda x, y, z, yaw=0.0: dict(mode="pos", p=np.array([hx+x, hy+y, z]), yaw=yaw)
    A = lambda roll, pitch: dict(mode="att", p=np.array([hx, hy, 5.0]), roll=d(roll), pitch=d(pitch), yaw=0.0)
    return [(0.0, "armado (ralenti)", dict(mode="idle")),
            (2.0, "despegue a 5 m", P(0, 0, 5)),
            (10.0, "derecha 3 m", P(0, -3, 5)),
            (17.0, "izquierda 3 m", P(0, 3, 5)),
            (26.0, "volver al centro", P(0, 0, 5)),
            (32.0, "roll +15 deg", A(15, 0)), (33.0, "roll -15 deg", A(-15, 0)), (34.0, "recuperar posicion", P(0, 0, 5)),
            (40.0, "pitch +15 deg", A(0, 15)), (41.0, "pitch -15 deg", A(0, -15)), (42.0, "recuperar posicion", P(0, 0, 5)),
            (48.0, "yaw +90 deg", P(0, 0, 5, d(90))), (53.0, "yaw 0 deg", P(0, 0, 5, 0.0)),
            (58.0, "aterrizaje", dict(mode="land"))]


def run(home=(0.0, 0.0), dt=0.002, ctrl_div=2, log_hz=100, t_max=90.0):
    P = Params; quad = Quad(P, p0=(home[0], home[1], None)); ctl = Controller(P)
    mission = test_mission(home); k = 0; w_cmd = np.zeros(4); rows = []; landed_t = None; phase = mission[0][1]; sp = mission[0][2]
    n = int(t_max/dt); log_every = int(round(1/(log_hz*dt)))
    for i in range(n):
        t = i*dt
        while k+1 < len(mission) and t >= mission[k+1][0]: k += 1; phase, sp = mission[k][1], mission[k][2]
        cur = sp
        if sp["mode"] == "land":
            if landed_t is None:
                h = quad.p[2] - P.h_cg_ground
                cur = dict(mode="pos", p=np.array([home[0], home[1], 0.0]), yaw=0.0, vz=-0.7 if h > 1.0 else -0.3)
                if quad.contact and abs(quad.v[2]) < 0.05: landed_t = t
            if landed_t is not None:
                cur = dict(mode="idle") if t - landed_t < 2.0 else dict(mode="off"); phase = "en tierra"
                if t - landed_t > 4.0: break
        if i % ctrl_div == 0: w_cmd = ctl.update(quad, cur, dt*ctrl_div)
        quad.step(w_cmd, dt)
        if i % log_every == 0:
            e = np.degrees(euler_from_q(quad.q)); psp = cur.get("p", quad.p)
            rows.append([t, *quad.p, *quad.v, *quad.q, *e, *quad.rot, *quad.rot_angle, quad.T.sum(), quad.P_elec, quad.E_Wh, *psp, int(quad.contact), phase])
    cols = ["t", "x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg", "w0", "w1", "w2", "w3",
            "ang0", "ang1", "ang2", "ang3", "thrust_N", "P_elec_W", "E_Wh", "x_sp", "y_sp", "z_sp", "contact", "fase"]
    return cols, rows, mission


def hover_power(P=Params):
    w = np.sqrt(P.m*G/4/P.kT); return w, 4*P.kQ*w**3/P.eta + P.P_avionics


if __name__ == "__main__":
    import csv, sys
    home = (-0.3595, 0.0)          # centro de la franja libre inferior de la lamina A0
    cols, rows, _ = run(home)
    out = sys.argv[1] if len(sys.argv) > 1 else "vuelo_prueba.csv"
    with open(out, "w", newline="") as f: w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    w_h, p_h = hover_power()
    print(f"{len(rows)} muestras, {rows[-1][0]:.1f} s -> {out}")
    print(f"hover: {w_h:.0f} rad/s ({w_h*60/2/np.pi:.0f} rpm), {p_h:.0f} W -> autonomia {Params.E_batt_Wh*Params.usable/p_h*60:.1f} min")
