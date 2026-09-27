"""
mision_cruz.py - Mision de prueba sobre la lamina de la cruz (usa f450_sim.py)

Marcos:
  Mundo  : origen en el centro de la cruz (centro de la lamina A0), Z arriba.
  Cuerpo : origen en el centro de masa del drone, ejes FLU.
  Efector: la camara nadir. Los puntos de la mision son posiciones del EFECTOR.

Secuencia: parte apoyado con la camara sobre el origen -> sube a 4 m -> para cada rumbo
(0, 45, 135, 225 grados): derecha 1 m, centro, izquierda 1 m, centro, adelante 1 m, centro,
atras 1 m, centro (direcciones relativas al drone) -> rumbo 0 -> aterriza con la camara sobre el origen.
"""
import csv, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '04_simulador_dinamica'))
from f450_sim import Params, Quad, Controller, q_to_R, euler_from_q, hover_power

R_CAM = np.array([0.0680, 0.0, -0.0504]) - Params.r_cg      # camara respecto del CG, ejes cuerpo (m)
ALT = 4.0
YAWS_DEG = [0.0, 45.0, 135.0, 225.0]


def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def waypoints():
    wp = [("despegue a 4 m", np.array([0, 0, ALT]), 0.0)]
    for yd in YAWS_DEG:
        y = np.radians(yd)
        if yd != 0: wp.append((f"giro a {yd:.0f} deg", np.array([0, 0, ALT]), y))
        for name, d in (("derecha", [0, -1, 0]), ("izquierda", [0, 1, 0]), ("adelante", [1, 0, 0]), ("atras", [-1, 0, 0])):
            wp.append((f"{name} 1 m (rumbo {yd:.0f})", Rz(y) @ np.array(d, float) + [0, 0, ALT], y))
            wp.append((f"centro (rumbo {yd:.0f})", np.array([0, 0, ALT]), y))
    wp.append(("giro a 0 deg", np.array([0, 0, ALT]), 0.0))
    return wp


def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi


def run(dt=0.002, ctrl_div=2, log_hz=100, tol=0.04, hold=1.0, t_wp_max=20.0):
    P = Params
    p0 = -(R_CAM)[:2]                                   # CG tal que la camara queda sobre el origen
    quad = Quad(P, p0=(p0[0], p0[1], None)); ctl = Controller(P)
    wps = waypoints(); k = -1; t_arr = None; t_wp0 = 0.0; landed_t = None; rows = []; w_cmd = np.zeros(4)
    log_every = int(round(1/(log_hz*dt))); i = 0; phase = "armado (ralenti)"; cam_sp = np.array([0, 0, P.h_cg_ground + R_CAM[2]]); yaw_sp = 0.0
    while True:
        t = i*dt
        R = q_to_R(quad.q); cam = quad.p + R @ R_CAM; yaw = euler_from_q(quad.q)[2]
        if t < 2.0: cur = dict(mode="idle")
        elif k < len(wps):
            if k == -1: k = 0; t_wp0 = t; t_arr = None
            if k < len(wps):
                phase, cam_sp, yaw_sp = wps[k]
                ok = np.linalg.norm(cam - cam_sp) < tol and np.linalg.norm(quad.v) < 0.08 and abs(wrap(yaw - yaw_sp)) < np.radians(1.5)
                t_arr = (t if t_arr is None else t_arr) if ok else None
                if (t_arr is not None and t - t_arr >= hold) or t - t_wp0 > t_wp_max: k += 1; t_wp0 = t; t_arr = None
            cur = dict(mode="pos", p=cam_sp - Rz(yaw_sp) @ R_CAM, yaw=yaw_sp)
        if t >= 2.0 and k >= len(wps):
            phase = "aterrizaje"; yaw_sp = 0.0; cam_sp = np.array([0, 0, P.h_cg_ground + R_CAM[2]])
            if landed_t is None:
                h = quad.p[2] - P.h_cg_ground
                cur = dict(mode="pos", p=np.array([*(-(Rz(0) @ R_CAM)[:2]), 0.0]), yaw=0.0, vz=-0.7 if h > 1.0 else -0.3)
                if quad.contact and abs(quad.v[2]) < 0.05: landed_t = t
            else:
                phase = "en tierra"; cur = dict(mode="idle") if t - landed_t < 2.0 else dict(mode="off")
                if t - landed_t > 4.0: break
        if i % ctrl_div == 0: w_cmd = ctl.update(quad, cur, dt*ctrl_div)
        quad.step(w_cmd, dt)
        if i % log_every == 0:
            e = np.degrees(euler_from_q(quad.q)); zc = R[:, 2]
            g = cam - zc * (cam[2] / zc[2]) if zc[2] > 0.2 else cam            # punto del suelo que ve el centro de la imagen
            rows.append([t, *quad.p, *quad.v, *quad.q, *e, *cam, *cam_sp, np.degrees(yaw_sp), g[0], g[1], *quad.rot, *quad.rot_angle,
                         quad.T.sum(), quad.P_elec, quad.E_Wh, int(quad.contact), phase])
        i += 1
        if t > 600: break
    cols = ["t", "x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg", "cam_x", "cam_y", "cam_z",
            "cam_x_sp", "cam_y_sp", "cam_z_sp", "yaw_sp_deg", "img_centro_x", "img_centro_y", "w0", "w1", "w2", "w3", "ang0", "ang1", "ang2", "ang3",
            "thrust_N", "P_elec_W", "E_Wh", "contact", "fase"]
    return cols, rows


if __name__ == "__main__":
    cols, rows = run()
    out = sys.argv[1] if len(sys.argv) > 1 else "mision_cruz.csv"
    with open(out, "w", newline="") as f: w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    print(f"{len(rows)} muestras, {rows[-1][0]:.1f} s, energia {rows[-1][cols.index('E_Wh')]:.2f} Wh -> {out}")
