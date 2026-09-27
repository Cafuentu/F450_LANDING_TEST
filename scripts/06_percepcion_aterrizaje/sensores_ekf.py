"""
sensores_ekf.py - Sensores simulados del F450 y estimador de estado (etapa 06)

Sensores (modelos simples pero realistas para un Pixhawk + GPS M8N):
  IMU   : acelerometro (fuerza especifica en ejes cuerpo) y giroscopo, con ruido blanco y sesgo fijo. 250 Hz.
  GPS   : posicion con ruido blanco + error lento (Gauss-Markov, tipico del GPS) y velocidad. 5 Hz.
  Baro  : altura con ruido blanco. 20 Hz.

Estimador:
  Actitud  : filtro complementario (integra giroscopo, corrige inclinacion con el acelerometro).
  Posicion : EKF de 9 estados  x = [p(3), v(3), sesgo_acel(3)]
             prediccion con la IMU, correccion con GPS (posicion y velocidad) y barometro.
"""
import numpy as np
from f450_sim import q_mul, q_to_R, q_from_euler, euler_from_q, G

g_w = np.array([0.0, 0.0, -G])


class Sensores:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.acc_sigma, self.gyr_sigma = 0.08, 0.004                # m/s^2, rad/s (por muestra)
        self.acc_bias = np.array([0.06, -0.04, 0.09]); self.gyr_bias = np.array([0.0015, -0.0010, 0.0008])
        self.gps_sigma_h, self.gps_sigma_v, self.gps_vel_sigma = 0.20, 0.40, 0.08
        self.gm_sigma_h, self.gm_sigma_v, self.gm_tau = 0.45, 0.8, 90.0   # error lento del GPS
        self.gm = np.array([0.35, -0.30, 0.4])                          # valor inicial del error lento
        self.baro_sigma = 0.10
        self.v_prev = None

    def imu(self, quad, dt):
        """Fuerza especifica f_b = R^T (a - g) y velocidad angular, medidas en ejes cuerpo."""
        if self.v_prev is None: self.v_prev = quad.v.copy()
        a_w = (quad.v - self.v_prev) / dt; self.v_prev = quad.v.copy()
        R = q_to_R(quad.q)
        f_b = R.T @ (a_w - g_w)
        return (f_b + self.acc_bias + self.rng.normal(0, self.acc_sigma, 3),
                quad.w + self.gyr_bias + self.rng.normal(0, self.gyr_sigma, 3))

    def gps(self, quad, dt_gps):
        a = np.exp(-dt_gps / self.gm_tau); s = np.array([self.gm_sigma_h, self.gm_sigma_h, self.gm_sigma_v])
        self.gm = a*self.gm + np.sqrt(1 - a*a) * s * self.rng.normal(0, 1, 3)
        n = self.rng.normal(0, 1, 3) * np.array([self.gps_sigma_h, self.gps_sigma_h, self.gps_sigma_v])
        return quad.p + self.gm + n, quad.v + self.rng.normal(0, self.gps_vel_sigma, 3)

    def baro(self, quad):
        return quad.p[2] + self.rng.normal(0, self.baro_sigma)


class Estimador:
    def __init__(self, p0, yaw0=0.0):
        self.q = q_from_euler(0, 0, yaw0)
        self.x = np.zeros(9); self.x[:3] = p0
        self.P = np.diag([1.0, 1.0, 0.5, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05])**2
        self.kp_att = 0.6
        self.q_acc, self.q_bias = 0.35, 0.002
        self.w = np.zeros(3)

    # --- actitud ---
    def _actitud(self, f_b, w_m, dt):
        e = np.zeros(3); n = np.linalg.norm(f_b)
        if abs(n - G) < 1.5:                                    # solo corrige cuando casi no hay aceleracion
            up_est = q_to_R(self.q).T @ np.array([0, 0, 1.0])
            e = np.cross(f_b / n, up_est)
        w = w_m + self.kp_att * e
        self.q = q_mul(self.q, np.array([1, *(0.5*w*dt)])); self.q /= np.linalg.norm(self.q)
        self.w = w_m

    # --- EKF ---
    def predecir(self, f_b, w_m, dt):
        self._actitud(f_b, w_m, dt)
        R = q_to_R(self.q); p, v, ba = self.x[:3], self.x[3:6], self.x[6:]
        a = R @ (f_b - ba) + g_w
        self.x[:3] = p + v*dt + 0.5*a*dt*dt; self.x[3:6] = v + a*dt
        F = np.eye(9); F[:3, 3:6] = np.eye(3)*dt; F[3:6, 6:] = -R*dt
        Q = np.zeros((9, 9)); Q[3:6, 3:6] = np.eye(3)*(self.q_acc**2)*dt; Q[:3, :3] = np.eye(3)*(0.5*self.q_acc*dt)**2*dt; Q[6:, 6:] = np.eye(3)*(self.q_bias**2)*dt
        self.P = F @ self.P @ F.T + Q

    def _corregir(self, z, H, Rm):
        y = z - H @ self.x; S = H @ self.P @ H.T + Rm; K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(9) - K @ H) @ self.P

    def corregir_gps(self, p_m, v_m):
        H = np.zeros((6, 9)); H[:3, :3] = np.eye(3); H[3:, 3:6] = np.eye(3)
        Rm = np.diag([1.5, 1.5, 2.0, 0.12, 0.12, 0.2])**2      # posicion GPS poco confiable a corto plazo; la velocidad si
        self._corregir(np.concatenate([p_m, v_m]), H, Rm)

    def corregir_baro(self, z_m):
        H = np.zeros((1, 9)); H[0, 2] = 1.0
        self._corregir(np.array([z_m]), H, np.array([[0.12**2]]))

    @property
    def p(self): return self.x[:3]
    @property
    def v(self): return self.x[3:6]


class EstadoNav:
    """Lo que ve el controlador: estado estimado (no la verdad)."""
    def __init__(self, p, v, q, w): self.p, self.v, self.q, self.w = p, v, q, w
