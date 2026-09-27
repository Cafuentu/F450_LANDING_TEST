"""
calibracion_camara.py - Prueba de calibracion de la camara (etapa 06)

Sin argumentos : PRUEBA SINTETICA. Genera vistas de un tablero de ajedrez con la misma camara sintetica de la simulacion,
                 calibra con OpenCV y compara los intrinsecos recuperados con los del archivo intrinsecos_picam_v2.json.
                 Si coinciden, la camara de la simulacion respeta esos intrinsecos.
--fotos CARPETA: CALIBRACION REAL. Usa fotos del tablero tomadas con la Pi Camera v2 (a 640x480, campo completo) y escribe
                 intrinsecos_calibrados.json para reemplazar los valores nominales.
Tablero: 9x6 esquinas interiores, cuadros de 30 mm.
"""
import os, sys, glob, json
import numpy as np, cv2
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(AQUI, "..", "04_simulador_dinamica"))
from f450_sim import q_from_euler, q_to_R
from camara_vision import Camara, K, W, H, _INTR
PATRON, CUADRO = (9, 6), 0.030
RES = os.path.join(AQUI, "resultados"); os.makedirs(RES, exist_ok=True)


class PlanoTablero:
    def __init__(self, lado=1.6, res=0.00025):
        self.res, self.min = res, -lado/2; n = int(lado/res); tex = np.full((n, n), 120, np.uint8); px = lambda m: int(round((m - self.min)/res))
        nx, ny = PATRON[0]+1, PATRON[1]+1; x0, y0 = -nx*CUADRO/2, -ny*CUADRO/2
        tex[px(y0-0.04):px(-y0+0.04), px(x0-0.04):px(-x0+0.04)] = 240
        for i in range(nx):
            for j in range(ny):
                if (i+j) % 2 == 0: tex[px(y0+j*CUADRO):px(y0+(j+1)*CUADRO), px(x0+i*CUADRO):px(x0+(i+1)*CUADRO)] = 20
        self.niveles = [tex]
        for _ in range(3): self.niveles.append(cv2.resize(self.niveles[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def vistas_sinteticas(n=24, seed=5):
    rng = np.random.default_rng(seed); cam = Camara(PlanoTablero()); out = []
    while len(out) < n:
        h = rng.uniform(0.32, 0.65); e = np.radians([rng.uniform(-28, 28), rng.uniform(-28, 28), rng.uniform(-180, 180)])
        R = q_to_R(q_from_euler(*e)); eje = R @ np.array([0, 0, -1.0])                 # eje optico en el mundo
        p = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), h]); p[:2] -= eje[:2] * (h / -eje[2])   # apunta cerca del centro del tablero
        out.append(cam.imagen(p, R))
    return out


def calibrar(imagenes):
    objp = np.zeros((PATRON[0]*PATRON[1], 3), np.float32); objp[:, :2] = np.mgrid[0:PATRON[0], 0:PATRON[1]].T.reshape(-1, 2) * CUADRO
    obj, img_pts, usadas = [], [], []
    for im in imagenes:
        ok, c = cv2.findChessboardCorners(im, PATRON, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            c = cv2.cornerSubPix(im, c, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)); obj.append(objp); img_pts.append(c); usadas.append(im)
    rms, Kc, dist, rv, tv = cv2.calibrateCamera(obj, img_pts, (imagenes[0].shape[1], imagenes[0].shape[0]), None, None)
    por_vista = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, Kc, dist)[0] - p)**2, axis=2)))) for o, p, r, t in zip(obj, img_pts, rv, tv)]
    return rms, Kc, dist.ravel(), por_vista, usadas, img_pts


if __name__ == "__main__":
    real = "--fotos" in sys.argv
    if real:
        carpeta = sys.argv[sys.argv.index("--fotos")+1]; ims = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in sorted(glob.glob(os.path.join(carpeta, "*.*")))]
        ims = [cv2.resize(i, (W, H), interpolation=cv2.INTER_AREA) for i in ims if i is not None]
    else: ims = vistas_sinteticas()
    rms, Kc, dist, por_vista, usadas, pts = calibrar(ims)
    res = dict(vistas_usadas=len(usadas), vistas_totales=len(ims), rms_px=round(rms, 4), fx=Kc[0, 0], fy=Kc[1, 1], cx=Kc[0, 2], cy=Kc[1, 2], dist=dist.tolist())
    nominal = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
    print(f"vistas usadas {len(usadas)}/{len(ims)}   RMS de reproyeccion = {rms:.3f} px")
    for k in nominal: print(f"  {k}: nominal {nominal[k]:8.2f}   calibrado {res[k]:8.2f}   diferencia {res[k]-nominal[k]:+6.2f} px ({100*(res[k]-nominal[k])/nominal[k]:+.2f} %)")
    print("  distorsion (k1 k2 p1 p2 k3):", np.round(dist, 4))
    if real:
        json.dump(dict(_INTR, fx=float(Kc[0, 0]), fy=float(Kc[1, 1]), cx=float(Kc[0, 2]), cy=float(Kc[1, 2]), dist=dist.tolist(), calibrada=True, fuente=f"calibracion real, RMS {rms:.3f} px, {len(usadas)} fotos"),
                  open(os.path.join(AQUI, "intrinsecos_calibrados.json"), "w"), indent=2)
    json.dump(dict(res, nominal=nominal), open(os.path.join(RES, "calibracion_resultado.json"), "w"), indent=2)
    # grafico
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = ["#2a78d6", "#eb6834"]; GRID, MUTED, SURF = "#e4e3df", "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID, "legend.frameon": False})
    fig = plt.figure(figsize=(14, 8)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1], hspace=0.3, wspace=0.3)
    for j in range(min(6, len(usadas))):
        ax = fig.add_subplot(gs[0, j]); vis = cv2.cvtColor(usadas[j], cv2.COLOR_GRAY2BGR); cv2.drawChessboardCorners(vis, PATRON, pts[j], True); ax.imshow(vis[:, :, ::-1]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"vista {j+1}: {por_vista[j]:.2f} px", fontsize=9)
    ax = fig.add_subplot(gs[1, :3]); ks = list(nominal); x = np.arange(4)
    ax.bar(x-0.19, [nominal[k] for k in ks], 0.36, color=C[0], label="nominal (archivo de intrínsecos)"); ax.bar(x+0.19, [res[k] for k in ks], 0.36, color=C[1], label="recuperado por calibración")
    for i, k in enumerate(ks): ax.text(i, max(nominal[k], res[k])+8, f"{res[k]-nominal[k]:+.2f} px", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("px"); ax.legend(loc="upper right"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_title("Intrínsecos: nominal vs calibrado", loc="left", fontweight="bold")
    ax = fig.add_subplot(gs[1, 3:]); ax.bar(np.arange(1, len(por_vista)+1), por_vista, color=C[0]); ax.axhline(rms, color=MUTED, lw=1, ls=(0, (2, 2))); ax.text(len(por_vista), rms, f" RMS {rms:.3f} px", va="bottom", ha="right", fontsize=9)
    ax.set_xlabel("vista"); ax.set_ylabel("error de reproyección (px)"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.set_title("Error de reproyección por vista", loc="left", fontweight="bold")
    fig.suptitle(("Calibración real" if real else "Prueba de calibración con la cámara sintética") + " — Pi Camera v2 a 640×480", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(RES, "00_calibracion_camara.png"), dpi=120)
