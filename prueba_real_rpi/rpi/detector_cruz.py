"""
detector_cruz.py - Detector de la cruz por filtro adaptado (kernel + FFT, correlacion normalizada) - version standalone para la Raspberry.
Es el mismo algoritmo validado en la simulacion (experimento 05). Solo depende de numpy y OpenCV.

Convenciones:
  imagen 640x480 de la Pi Camera v2 (campo completo). Intrinsecos en intrinsecos_picam_v2.json (reemplazar por los calibrados).
  Camara fija al cuerpo mirando a -Z; arriba de la imagen = adelante del drone, derecha de la imagen = derecha del drone.
  Cruz de la lamina A0: cuadro de 250 mm, brazos de 76.4 mm.
"""
import json, os
import numpy as np
import cv2

_INTR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "intrinsecos_picam_v2.json")))
W, H = _INTR["ancho"], _INTR["alto"]
FX, FY, CX, CY = _INTR["fx"], _INTR["fy"], _INTR["cx"], _INTR["cy"]
DIST = np.array(_INTR.get("dist", [0, 0, 0, 0, 0]), float)
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
R_CB = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])      # cuerpo (FLU) -> camara (x der, y abajo, z optico)
CRUZ_LADO = 0.250
CRUZ_BRAZO = 0.250 * 110 / 360
_MAP = cv2.initUndistortRectifyMap(K, DIST, None, K, (W, H), cv2.CV_16SC2) if np.any(DIST) else None


def preparar(frame_bgr_o_gris):
    """Escala a 640x480, pasa a gris y corrige distorsion si hay coeficientes."""
    g = frame_bgr_o_gris if frame_bgr_o_gris.ndim == 2 else cv2.cvtColor(frame_bgr_o_gris, cv2.COLOR_BGR2GRAY)
    if g.shape[1] != W or g.shape[0] != H: g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    if _MAP is not None: g = cv2.remap(g, _MAP[0], _MAP[1], cv2.INTER_LINEAR)
    return g


def rot_rpy(roll, pitch, yaw):
    """Matriz cuerpo->mundo (FLU / Z arriba) a partir de roll, pitch, yaw en radianes (convencion MAVLink ATTITUDE)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])


class DetectorCruz:
    def __init__(self, umbral=0.55):
        self.umbral = umbral; self.cache = {}

    def kernel(self, L, yaw_deg=0.0, modo="completa"):
        key = (modo, round(L*4)/4, int(round(yaw_deg/2.0))*2)
        if key not in self.cache:
            L = key[1]; n = int(np.ceil(L*(1.3 if modo == "completa" else 2.4))) | 1
            ss = 8 if n < 80 else (4 if n < 200 else 1); N = n*ss; c = N/2
            a = L*ss/2 if modo == "completa" else N; b = (L*110/360 if modo == "completa" else L)*ss/2
            k = np.zeros((N, N), np.float32); r = lambda x: int(round(np.clip(x, 0, N)))
            k[r(c-b):r(c+b), r(c-a):r(c+a)] = 1; k[r(c-a):r(c+a), r(c-b):r(c+b)] = 1
            if key[2] % 90 != 0: k = cv2.warpAffine(k, cv2.getRotationMatrix2D((c, c), key[2], 1.0), (N, N), flags=cv2.INTER_LINEAR)
            if ss > 1: k = cv2.resize(k, (n, n), interpolation=cv2.INTER_AREA)
            self.cache[key] = k
        return self.cache[key]

    def detectar(self, img, L_pred, yaw_deg=0.0, ancho=0.25, n_esc=11, modo="completa", devolver_ncc=True):
        I = 255.0 - img.astype(np.float32)
        F_I = np.fft.rfft2(I)
        mejor = None; escalas = L_pred * np.geomspace(1-ancho, 1+ancho, n_esc); punt = []
        for L in escalas:
            if L < 8 or L*(1.3 if modo == "completa" else 2.4) > min(W, H) - 2: punt.append(0.0); continue
            k = self.kernel(L, yaw_deg, modo); n = k.shape[0]; k0 = k - k.mean()
            pad = np.zeros((H, W), np.float32); pad[:n, :n] = k0; pad = np.roll(pad, (-(n//2), -(n//2)), axis=(0, 1))
            corr = np.fft.irfft2(F_I * np.conj(np.fft.rfft2(pad)), s=(H, W))
            S1 = cv2.boxFilter(I, -1, (n, n), normalize=False, borderType=cv2.BORDER_CONSTANT)
            S2 = cv2.boxFilter(I*I, -1, (n, n), normalize=False, borderType=cv2.BORDER_CONSTANT)
            ncc = corr / (np.sqrt(np.maximum(S2 - S1*S1/(n*n), 1e-3)) * np.sqrt((k0*k0).sum()))
            m = n//2 + 1; ncc[:m, :] = 0; ncc[-m:, :] = 0; ncc[:, :m] = 0; ncc[:, -m:] = 0
            v, u = np.unravel_index(np.argmax(ncc), ncc.shape); sc = float(ncc[v, u]); punt.append(sc)
            if mejor is None or sc > mejor["score"]: mejor = dict(score=sc, u=u, v=v, L=float(L), ncc=ncc, kernel=k)
        if mejor is None: return dict(valido=False, score=0.0, modo=modo)
        ncc = mejor["ncc"]; u, v = mejor["u"], mejor["v"]
        def par(a, b, c): d = a - 2*b + c; return 0.0 if abs(d) < 1e-9 else 0.5*(a - c)/d
        du = par(ncc[v, u-1], ncc[v, u], ncc[v, u+1]); dv = par(ncc[v-1, u], ncc[v, u], ncc[v+1, u])
        i = int(np.argmax(punt)); L = mejor["L"]
        if 0 < i < len(punt)-1 and punt[i-1] > 0 and punt[i+1] > 0:
            L = float(np.exp(np.log(escalas[i]) + par(punt[i-1], punt[i], punt[i+1]) * np.log(escalas[1]/escalas[0])))
        out = dict(modo=modo, valido=mejor["score"] >= self.umbral, score=mejor["score"], u=float(u+du), v=float(v+dv), L=L, escalas=escalas.tolist(), puntajes=punt)
        if devolver_ncc: out["ncc"] = ncc; out["kernel"] = mejor["kernel"]
        return out

    def detectar_auto(self, img, h_pred, yaw_deg=0.0, L_prev=None):
        """Busca la cruz completa alrededor de la altura prevista; bajo ~0.75 m, si no cabe, usa el kernel del centro."""
        L_pred = FX*CRUZ_LADO/max(h_pred, 0.15) if L_prev is None else L_prev
        d = self.detectar(img, L_pred, yaw_deg, ancho=0.30 if L_prev is None else 0.15, n_esc=13 if L_prev is None else 9)
        if not d["valido"] and h_pred < 0.75:
            d = self.detectar(img, FX*CRUZ_BRAZO/max(h_pred, 0.10), yaw_deg, ancho=0.25, n_esc=9, modo="centro")
        return d

    @staticmethod
    def a_mundo(det, R_wb):
        """Centro (u,v) y tamano L -> posicion de la camara respecto del centro de la cruz, en el mundo (m). z = altura de la camara."""
        Zc = FX * (CRUZ_LADO if det["modo"] == "completa" else CRUZ_BRAZO) / det["L"]
        d_c = np.array([(det["u"]-CX)/FX, (det["v"]-CY)/FY, 1.0]) * Zc
        return -(R_wb @ (R_CB.T @ d_c))
