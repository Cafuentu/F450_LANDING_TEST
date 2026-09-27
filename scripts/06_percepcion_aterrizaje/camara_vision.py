"""
camara_vision.py - Camara nadir sintetica, detector de la cruz (kernel + FFT) y odometria ORB (etapa 06)

Camara   : como el suelo es plano, la imagen exacta de la camara es una homografia de la textura del suelo.
           Intrinsecos de la Pi Camera v2 leidos de intrinsecos_picam_v2.json (640x480, campo completo). Fija al cuerpo, mirando a -Z.
           Imagen: arriba = adelante del drone (+X cuerpo), derecha = derecha del drone (-Y cuerpo).
Detector : filtro adaptado (kernel con forma de cruz, media cero) correlacionado en frecuencia (FFT),
           normalizado (NCC), con banco de escalas -> centro de la cruz (px), tamano (px) -> altura y posicion.
ORB      : descriptores ORB entre cuadros -> cambio de escala -> altura relativa y velocidad de descenso.
"""
from collections import deque
import numpy as np
import cv2
from f450_sim import q_to_R

import json, os
_INTR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "intrinsecos_picam_v2.json")))   # intrinsecos de la Pi Camera v2
W, H = _INTR["ancho"], _INTR["alto"]
FX, FY, CX, CY = _INTR["fx"], _INTR["fy"], _INTR["cx"], _INTR["cy"]
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
R_CB = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])     # cuerpo -> camara (x der, y abajo, z adelante optico)
CRUZ_LADO = 0.250                                            # m, lado del cuadro de la cruz
CRUZ_BRAZO = 0.250 * 110/360                                 # m, ancho de los brazos
LAMINA = (1.189, 0.841)                                      # m, largo en X, ancho en Y


# --------------------------------------------------------------------------- mundo plano
class Suelo:
    def __init__(self, lado=8.0, res=0.001, seed=3):
        self.res, self.lado = res, lado; n = int(lado/res); self.min = -lado/2
        rng = np.random.default_rng(seed)
        tex = np.full((n, n), 118.0, np.float32)
        for cell, amp in ((500, 9), (120, 8), (30, 8), (8, 9), (3, 8)):           # cemento: ruido multi-escala
            m = n // cell + 2
            tex += cv2.resize(rng.normal(0, amp, (m, m)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        k = int(n*n*2.5e-5)                                                        # aridos / manchas
        for val, rad in ((60, 4), (175, 3), (80, 7)):
            pts = rng.integers(0, n, (k, 2))
            for (u, v) in pts: cv2.circle(tex, (int(u), int(v)), int(rad * rng.uniform(0.6, 1.4)), float(val), -1)
        tex = cv2.GaussianBlur(tex, (0, 0), 0.8)
        px = lambda m_: int(round((m_ - self.min)/res))                            # textura[v(y), u(x)]
        tex[px(-LAMINA[1]/2):px(LAMINA[1]/2), px(-LAMINA[0]/2):px(LAMINA[0]/2)] = 238.0
        a, b = CRUZ_LADO/2, CRUZ_BRAZO/2
        tex[px(-b):px(b), px(-a):px(a)] = 22.0; tex[px(-a):px(a), px(-b):px(b)] = 22.0
        self.niveles = [np.clip(tex, 0, 255).astype(np.uint8)]
        for _ in range(3): self.niveles.append(cv2.resize(self.niveles[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


class Camara:
    def __init__(self, suelo, seed=11):
        self.s = suelo; self.rng = np.random.default_rng(seed)

    def imagen(self, p_cam, R_wb):
        """p_cam: posicion de la camara en el mundo; R_wb: cuerpo->mundo (verdad)."""
        R = R_CB @ R_wb.T; t = -R @ p_cam
        Hwi = K @ np.column_stack([R[:, 0], R[:, 1], t])                           # plano z=0 -> imagen
        mm_px = max(p_cam[2], 0.02) / FX / self.s.res
        lvl = int(np.clip(np.floor(np.log2(max(mm_px, 1.0))), 0, 3)); f = 2**lvl
        T = np.array([[self.s.res*f, 0, self.s.min + (f-1)/2*self.s.res], [0, self.s.res*f, self.s.min + (f-1)/2*self.s.res], [0, 0, 1.0]])
        img = cv2.warpPerspective(self.s.niveles[lvl], Hwi @ T, (W, H), flags=cv2.INTER_LINEAR, borderValue=118)
        img = cv2.GaussianBlur(img, (3, 3), 0.7).astype(np.float32) + self.rng.normal(0, 2.0, (H, W)).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- detector kernel + FFT
class DetectorCruz:
    def __init__(self, umbral=0.55):
        self.umbral = umbral; self.cache = {}

    def kernel(self, L, yaw_deg=0.0, modo="completa"):
        """modo 'completa': cruz de lado L px en ventana de 1.3 L.
           modo 'centro'  : solo el cruce de los brazos (ancho de brazo L px, brazos hasta el borde) en ventana de 2.4 L;
                            sirve cuando la cruz ya no cabe en la imagen (camara muy cerca).
           Se dibuja sobre-muestreado y se reduce con promedio de area: el tamano no queda cuantizado a pixeles enteros."""
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

    def detectar(self, img, L_pred, yaw_deg=0.0, ancho=0.25, n_esc=11, etapas=False, modo="completa"):
        I = 255.0 - img.astype(np.float32)                                         # lo oscuro pasa a ser senal
        F_I = np.fft.rfft2(I)
        mejor = None; escalas = L_pred * np.geomspace(1-ancho, 1+ancho, n_esc); punt = []
        for L in escalas:
            if L < 8 or L*(1.3 if modo == "completa" else 2.4) > min(W, H) - 2: punt.append(0.0); continue
            k = self.kernel(L, yaw_deg, modo); n = k.shape[0]; k0 = k - k.mean()
            pad = np.zeros((H, W), np.float32); pad[:n, :n] = k0
            pad = np.roll(pad, (-(n//2), -(n//2)), axis=(0, 1))                    # kernel centrado en el origen
            corr = np.fft.irfft2(F_I * np.conj(np.fft.rfft2(pad)), s=(H, W))       # correlacion cruzada por FFT
            S1 = cv2.boxFilter(I, -1, (n, n), normalize=False, borderType=cv2.BORDER_CONSTANT)
            S2 = cv2.boxFilter(I*I, -1, (n, n), normalize=False, borderType=cv2.BORDER_CONSTANT)
            den = np.sqrt(np.maximum(S2 - S1*S1/(n*n), 1e-3)) * np.sqrt((k0*k0).sum())
            ncc = corr / den
            m = n//2 + 1; ncc[:m, :] = 0; ncc[-m:, :] = 0; ncc[:, :m] = 0; ncc[:, -m:] = 0   # ventana completa dentro de la imagen
            v, u = np.unravel_index(np.argmax(ncc), ncc.shape); sc = float(ncc[v, u]); punt.append(sc)
            if mejor is None or sc > mejor["score"]: mejor = dict(score=sc, u=u, v=v, L=float(L), ncc=ncc, kernel=k)
        if mejor is None: return dict(valido=False, score=0.0)
        # refinamiento sub-pixel (parabola) en posicion y en escala
        ncc = mejor["ncc"]; u, v = mejor["u"], mejor["v"]
        def par(a, b, c): d = a - 2*b + c; return 0.0 if abs(d) < 1e-9 else 0.5*(a - c)/d
        du = par(ncc[v, u-1], ncc[v, u], ncc[v, u+1]); dv = par(ncc[v-1, u], ncc[v, u], ncc[v+1, u])
        i = int(np.argmax(punt)); L = mejor["L"]
        if 0 < i < len(punt)-1 and punt[i-1] > 0 and punt[i+1] > 0:
            L = float(np.exp(np.log(escalas[i]) + par(punt[i-1], punt[i], punt[i+1]) * np.log(escalas[1]/escalas[0])))
        out = dict(modo=modo, valido=mejor["score"] >= self.umbral, score=mejor["score"], u=u+du, v=v+dv, L=L, escalas=escalas, puntajes=np.array(punt))
        if etapas: out.update(I=I, espectro=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(I)))), ncc=ncc, kernel=mejor["kernel"])
        return out

    @staticmethod
    def a_mundo(det, R_wb_est):
        """Pixel del centro + tamano -> posicion de la camara respecto de la cruz (origen del mundo)."""
        Zc = FX * (CRUZ_LADO if det.get("modo", "completa") == "completa" else CRUZ_BRAZO) / det["L"]                                             # profundidad sobre el eje optico
        d_c = np.array([(det["u"]-CX)/FX, (det["v"]-CY)/FY, 1.0]) * Zc
        delta = R_wb_est @ (R_CB.T @ d_c)                                          # camara -> cruz, en el mundo
        return -delta                                                              # posicion de la camara (z = altura)


# --------------------------------------------------------------------------- odometria ORB
class OdometriaORB:
    def __init__(self, base_s=0.3):
        self.orb = cv2.ORB_create(nfeatures=900, fastThreshold=10, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.hist = deque(maxlen=12); self.base_s = base_s; self.h = None

    def iniciar(self, h0): self.h = h0; self.hist.clear()

    def paso(self, img, t, guardar=False):
        kp, des = self.orb.detectAndCompute(img, None)
        out = dict(n_kp=len(kp), n_match=0, n_inl=0, escala=np.nan, h=self.h, vz=np.nan, ok=False)
        ref = None
        for e in self.hist:
            if t - e["t"] >= self.base_s - 1e-6: ref = e
        if ref is not None and des is not None and ref["des"] is not None and len(kp) >= 8:
            # la cruz es simetrica (esquinas identicas): entre los 4 mejores candidatos se elige el primero que este cerca en la imagen
            m = []
            for cand in self.bf.knnMatch(ref["des"], des, k=min(4, len(kp))):
                for x in cand:
                    if x.distance < 64 and np.hypot(*(np.array(ref["kp"][x.queryIdx].pt) - np.array(kp[x.trainIdx].pt))) < 70: m.append(x); break
            out["n_match"] = len(m)
            if len(m) >= 6:
                a = np.float32([ref["kp"][x.queryIdx].pt for x in m]); b = np.float32([kp[x.trainIdx].pt for x in m])
                M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None and inl is not None and inl.sum() >= 5:
                    s = float(np.hypot(M[0, 0], M[1, 0])); out.update(n_inl=int(inl.sum()), escala=s)
                    if 0.9 < s < 1.4:                                    # compuerta: en descenso la escala solo puede crecer moderadamente
                        h = ref["h"] / s; out.update(h=h, vz=(h - ref["h"]) / (t - ref["t"]), ok=True); self.h = h
                        if guardar: out.update(a=a[inl.ravel() == 1], b=b[inl.ravel() == 1], img_ref=ref["img"])
        self.hist.append(dict(t=t, kp=kp, des=des, h=self.h, img=img if guardar or True else None))
        return out
