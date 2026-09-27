"""graficos_percepcion.py - Grafica todas las etapas de la mision de percepcion (lee ./resultados)."""
import os, pickle
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
AQUI = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(AQUI, "resultados")
d = pd.read_csv(os.path.join(RES, "vuelo.csv")); v = pd.read_csv(os.path.join(RES, "vision.csv")); et = pickle.load(open(os.path.join(RES, "etapas.pkl"), "rb"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False})
ini = d.loc[d.fase != d.fase.shift(), ["t", "fase"]].reset_index(drop=True); figs = []
gps = d.iloc[::10]                                                    # el GPS entrega a 5 Hz
def deco(ax, titulo, yl, fases=False):
    ax.set_title(titulo, loc="left", fontsize=10.5, fontweight="bold", pad=5); ax.set_ylabel(yl); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for t in ini.t: ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xlim(0, d.t.max())
def rotulos(ax):
    y1 = ax.get_ylim()[1]
    for t, f in zip(ini.t, ini.fase): ax.text(t+0.2, y1, f, rotation=50, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=10.5, fontweight="bold", pad=78)
def guardar(fig, nombre): fig.savefig(os.path.join(RES, nombre + ".png"), dpi=120); figs.append(fig)

# 1 --- trayectoria: verdad vs EKF vs GPS vs vision
fig, axs = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True, gridspec_kw=dict(hspace=0.25))
for ax, c, n in zip(axs, "xyz", ["x mundo (adelante)", "y mundo (izquierda)", "altura de la cámara"]):
    deco(ax, f"Efector (cámara): {n}", "m")
    ax.plot(gps.t, gps["gps_"+c], ".", color=C[1], ms=2.5, alpha=0.55, label="GPS (medición)")
    ax.plot(d.t, d["ekf_"+c], color=C[0], lw=1.6, label="EKF (IMU + GPS + barómetro)")
    ax.plot(v.t, v["vis_"+c], ".", color=C[2], ms=3.5, label="visión: kernel + FFT")
    if c == "z": ax.plot(v.t, v.h_orb, color=C[3], lw=2, label="visión: ORB (escala)")
    ax.plot(d.t, d["cam_"+c], color=INK, lw=1.3, label="verdad (ground truth)")
    ax.plot(d.t, d["obj_"+c], color=MUTED, lw=0.9, ls=(0, (2, 2)), label="referencia")
axs[2].legend(ncol=3, loc="upper right"); rotulos(axs[0]); axs[2].set_xlabel("tiempo (s)")
fig.suptitle("1 · Trayectoria del efector: verdad vs EKF vs GPS vs visión", x=0.06, y=0.995, ha="left", fontsize=13, fontweight="bold"); fig.subplots_adjust(top=0.83, left=0.06, right=0.985, bottom=0.05); guardar(fig, "01_trayectoria_estimacion")

# 2 --- errores
fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(hspace=0.3))
deco(axs[0], "Error horizontal respecto de la verdad", "m")
axs[0].plot(d.t, np.hypot(d.ekf_x-d.cam_x, d.ekf_y-d.cam_y), color=C[0], lw=1.6, label="EKF (absoluto: arrastra el error lento del GPS)")
axs[0].plot(d.t, np.hypot(d.nav_x-d.cam_x, d.nav_y-d.cam_y), color=C[1], lw=1.6, label="navegación usada por el control")
axs[0].plot(v.t, np.hypot(v.vis_x-v.cam_x, v.vis_y-v.cam_y), ".", color=C[2], ms=3.5, label="visión (kernel + FFT)"); axs[0].legend(ncol=3, loc="upper right")
deco(axs[1], "Error de altura respecto de la verdad", "m")
axs[1].plot(d.t, d.ekf_z-d.cam_z, color=C[0], lw=1.6, label="EKF"); axs[1].plot(v.t, v.vis_z-v.cam_z, ".", color=C[2], ms=3.5, label="kernel (tamaño de la cruz)")
axs[1].plot(v.t, v.h_orb-v.cam_z, color=C[3], lw=2, label="ORB (escala acumulada)"); axs[1].legend(ncol=3, loc="upper right"); axs[1].set_ylim(-0.5, 0.5); axs[1].set_xlabel("tiempo (s)")
rotulos(axs[0]); fig.suptitle("2 · Errores de estimación", x=0.06, y=0.995, ha="left", fontsize=13, fontweight="bold"); fig.subplots_adjust(top=0.80, left=0.06, right=0.985, bottom=0.07); guardar(fig, "02_errores")

# 3 --- IMU y actitud
fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(hspace=0.3))
deco(axs[0], "IMU: acelerómetro (fuerza específica, ejes cuerpo)", "m/s²")
for k, c in enumerate("xyz"): axs[0].plot(d.t, d["acc_"+c], color=C[k], lw=0.8, label=c)
axs[0].legend(ncol=3, loc="center right")
deco(axs[1], "IMU: giróscopo", "rad/s")
for k, c in enumerate("xyz"): axs[1].plot(d.t, d["gyr_"+c], color=C[k], lw=0.8, label=c)
axs[1].legend(ncol=3, loc="upper right")
for ax, c in zip(axs[2:], ["roll", "pitch"]):
    deco(ax, f"Actitud: {c} — verdad vs estimado", "grados"); ax.plot(d.t, d["ekf_"+c], color=C[0], lw=1.6, label="estimado (filtro complementario)"); ax.plot(d.t, d[c], color=INK, lw=1, label="verdad"); ax.legend(ncol=2, loc="upper right")
rotulos(axs[0]); axs[3].set_xlabel("tiempo (s)"); fig.suptitle("3 · IMU y estimación de actitud", x=0.06, y=0.995, ha="left", fontsize=13, fontweight="bold"); fig.subplots_adjust(top=0.85, left=0.06, right=0.985, bottom=0.05); guardar(fig, "03_imu_actitud")

# 4 --- etapas del detector kernel + FFT
filas = [k for k in et if not k.startswith("orb")]
fig, axs = plt.subplots(len(filas), 6, figsize=(17, 3.1*len(filas)+0.8))
tit = ["imagen de la cámara + detección", "señal (imagen invertida)", "kernel ganador", "espectro |FFT| (log)", "mapa de correlación NCC", "puntaje vs tamaño del kernel"]
for r, k in enumerate(filas):
    e = et[k]; a = axs[r]
    a[0].imshow(e["img"], cmap="gray", vmin=0, vmax=255); a[0].plot(e["u"], e["v"], "+", color="#e34948", ms=14, mew=2); hw = e["L"]*0.65
    a[0].add_patch(Rectangle((e["u"]-hw, e["v"]-hw), 2*hw, 2*hw, fill=False, ec="#1baf7a", lw=1.2))
    a[1].imshow(e["I"], cmap="gray"); a[2].imshow(e["kernel"], cmap="gray"); a[3].imshow(e["espectro"], cmap="magma"); im = a[4].imshow(e["ncc"], cmap="viridis", vmin=-0.2, vmax=1); a[4].plot(e["u"], e["v"], "o", mfc="none", mec="white", ms=12, mew=1.5)
    a[5].plot(e["escalas"], e["puntajes"], "-o", color=C[0], lw=2, ms=4); a[5].axvline(e["L"], color=MUTED, lw=1, ls=(0, (2, 2))); a[5].set_xlabel("lado del kernel (px)"); a[5].set_ylabel("NCC máx"); a[5].grid(color=GRID, lw=0.8)
    a[5].text(0.03, 0.06, f"L = {e['L']:.1f} px\nNCC = {e['score']:.2f}\nh = {529.6*0.25/e['L']:.2f} m (verdad {e['h']:.2f})", transform=a[5].transAxes, fontsize=8, color=INK, va="bottom")
    a[0].set_ylabel(k.replace("kernel ON: ", ""), fontsize=10, fontweight="bold", color=INK)
    for j in range(5): a[j].set_xticks([]); a[j].set_yticks([])
    if r == 0:
        for j in range(6): a[j].set_title(tit[j], fontsize=9.5, fontweight="bold")
fig.suptitle("4 · Etapas del detector de la cruz (filtro adaptado en frecuencia)", x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.97)); guardar(fig, "04_etapas_kernel_fft")

# 5 --- salidas del kernel en el tiempo
fig, axs = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True, gridspec_kw=dict(hspace=0.3)); vv = v[v.score.notna()]
deco(axs[0], "Puntaje de detección (NCC) — umbral 0,55", "NCC"); axs[0].plot(vv.t, vv.score, color=C[0], lw=1.5); axs[0].axhline(0.55, color=MUTED, lw=1, ls=(0, (2, 2))); axs[0].set_ylim(0, 1.05)
deco(axs[1], "Tamaño del kernel ganador", "px"); m = vv.modo == "completa"; axs[1].plot(vv.t[m], vv.L[m], ".", color=C[0], ms=4, label="cruz completa (lado)"); axs[1].plot(vv.t[~m], vv.L[~m], ".", color=C[1], ms=4, label="kernel de centro (ancho de brazo)"); axs[1].legend(ncol=2, loc="upper left")
deco(axs[2], "Altura de la cámara desde el tamaño de la cruz", "m"); axs[2].plot(vv.t, vv.vis_z, ".", color=C[2], ms=4, label="kernel"); axs[2].plot(d.t, d.cam_z, color=INK, lw=1, label="verdad"); axs[2].legend(ncol=2, loc="upper right")
deco(axs[3], "Centro de la cruz en la imagen (el centro de la imagen es 320, 240)", "px"); axs[3].plot(vv.t, vv.u, color=C[0], lw=1.3, label="u (columna)"); axs[3].plot(vv.t, vv.v, color=C[1], lw=1.3, label="v (fila)"); axs[3].legend(ncol=2, loc="upper right"); axs[3].set_xlabel("tiempo (s)")
rotulos(axs[0]); fig.suptitle("5 · Salidas del kernel a lo largo del vuelo", x=0.06, y=0.995, ha="left", fontsize=13, fontweight="bold"); fig.subplots_adjust(top=0.85, left=0.06, right=0.985, bottom=0.05); guardar(fig, "05_kernel_en_el_tiempo")

# 6 --- ORB
o = v[v.fase == "descenso final (ORB)"]; ko = [k for k in et if k.startswith("orb")]
fig = plt.figure(figsize=(14, 11)); gs = fig.add_gridspec(3, max(len(ko), 1)*2, height_ratios=[1.25, 1, 1], hspace=0.38, wspace=0.35)
for j, k in enumerate(ko):
    e = et[k]; ax = fig.add_subplot(gs[0, 2*j:2*j+2]); par = np.hstack([e["img_ref"], e["img"]]); ax.imshow(par, cmap="gray", vmin=0, vmax=255)
    for (x0, y0), (x1, y1) in zip(e["a"][::max(1, len(e["a"])//60)], e["b"][::max(1, len(e["b"])//60)]): ax.plot([x0, x1+640], [y0, y1], color="#eda100", lw=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"h = {e['h']:.2f} m · {len(e['a'])} pares válidos · escala {e['escala']:.3f}", fontsize=9.5, fontweight="bold")
n2 = max(len(ko), 1)
ax = fig.add_subplot(gs[1, :n2]); ax.plot(o.t, o.n_kp, color=C[0], lw=1.8, label="puntos ORB"); ax.plot(o.t, o.n_match, color=C[1], lw=1.8, label="pares"); ax.plot(o.t, o.n_inl, color=C[2], lw=1.8, label="válidos (RANSAC)"); ax.set_yscale("log"); ax.legend(); ax.grid(color=GRID, lw=0.8); ax.set_title("Cantidad de descriptores", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[1, n2:]); ax.plot(o.t, o.escala, "o-", color=C[0], ms=3.5, lw=1.2); ax.axhline(1, color=MUTED, lw=1, ls=(0, (2, 2))); ax.grid(color=GRID, lw=0.8); ax.set_title("Cambio de escala entre cuadros (0,3 s)", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[2, :n2]); ax.plot(o.t, o.h_orb, color=C[3], lw=2.2, label="ORB"); ax.plot(o.t, o.vis_z, ".", color=C[2], ms=5, label="kernel"); ax.plot(o.t, o.cam_z, color=INK, lw=1.2, label="verdad"); ax.axhline(0.0643, color=MUTED, lw=1, ls=(0, (2, 2))); ax.text(o.t.min(), 0.075, "tren apoyado: cámara a 64 mm", fontsize=8, color=MUTED)
ax.legend(); ax.grid(color=GRID, lw=0.8); ax.set_title("Altura de la cámara", loc="left", fontweight="bold"); ax.set_ylabel("m"); ax.set_xlabel("tiempo (s)")
ax = fig.add_subplot(gs[2, n2:]); dd = d[d.fase == "descenso final (ORB)"]; ax.plot(o.t, o.vz_orb, "o", color=C[3], ms=4, label="ORB"); ax.plot(dd.t, dd.ekf_vz, color=C[0], lw=1.5, label="EKF"); ax.plot(dd.t, dd.vz, color=INK, lw=1.2, label="verdad")
ax.legend(); ax.grid(color=GRID, lw=0.8); ax.set_title("Velocidad de descenso", loc="left", fontweight="bold"); ax.set_ylabel("m/s"); ax.set_xlabel("tiempo (s)"); ax.set_ylim(-0.6, 0.3)
fig.suptitle("6 · Descenso final con descriptores ORB", x=0.02, y=0.98, ha="left", fontsize=13, fontweight="bold"); guardar(fig, "06_orb_descenso")

# 7 --- planta y aterrizaje
fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 6.6))
for ax in (a, b):
    ax.add_patch(Rectangle((-0.4205, -0.5945), 0.841, 1.189, fc="white", ec=MUTED, lw=1)); ax.add_patch(Rectangle((-0.038, -0.125), 0.076, 0.25, fc=INK)); ax.add_patch(Rectangle((-0.125, -0.038), 0.25, 0.076, fc=INK))
    ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("← izquierda (+y)        derecha (−y) →   [m]"); ax.set_ylabel("adelante (x) [m]")
a.plot(-gps.gps_y, gps.gps_x, ".", color=C[1], ms=2.5, alpha=0.5, label="GPS"); a.plot(-d.ekf_y, d.ekf_x, color=C[0], lw=1.4, label="EKF"); a.plot(-v.vis_y, v.vis_x, ".", color=C[2], ms=3, label="visión (kernel)"); a.plot(-d.cam_y, d.cam_x, color=INK, lw=1.4, label="verdad")
a.set_xlim(-1.2, 2.2); a.set_ylim(-1.5, 1.7); a.legend(loc="upper left", ncol=2); a.set_title("Planta completa", loc="left", fontweight="bold")
fin = d.iloc[-1]; cg = np.array([fin.cam_x - 0.0714, fin.cam_y]); h = 0.343/2
b.add_patch(Rectangle((-cg[1]-h, cg[0]-h), 2*h, 2*h, fill=False, ec=C[0], lw=1.6)); 
for sx in (-1, 1):
    for sy in (-1, 1): b.plot(-cg[1]+sx*h, cg[0]+sy*h, "o", color=C[0], ms=7)
b.plot(-fin.cam_y, fin.cam_x, "x", color="#e34948", ms=11, mew=2.5); dd = d[d.cam_z < 1.05]; b.plot(-dd.cam_y, dd.cam_x, color=INK, lw=1)
err = np.hypot(fin.cam_x, fin.cam_y)*100; b.set_xlim(-0.55, 0.55); b.set_ylim(-0.7, 0.7); b.set_title(f"Aterrizaje: cámara a {err:.1f} cm del centro de la cruz", loc="left", fontweight="bold")
b.text(0.02, 0.02, "× cámara al tocar suelo · cuadrado azul: apoyos del tren (343 mm)", transform=b.transAxes, fontsize=8, color=MUTED)
fig.suptitle("7 · Vista en planta", x=0.04, y=0.99, ha="left", fontsize=13, fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.96)); guardar(fig, "07_planta_aterrizaje")

# 8 --- actuadores y energia
fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw=dict(hspace=0.3))
deco(axs[0], "Velocidad de los rotores", "rpm")
for k in range(4): axs[0].plot(d.t, d[f"rpm{k+1}"], color=C[k], lw=0.9, label=f"M{k+1}")
axs[0].legend(ncol=4, loc="lower center"); deco(axs[1], f"Potencia eléctrica total — energía usada: {d.E_Wh.iloc[-1]:.2f} Wh", "W"); axs[1].plot(d.t, d.P_elec_W, color=C[0], lw=1.2); axs[1].set_xlabel("tiempo (s)")
rotulos(axs[0]); fig.suptitle("8 · Actuadores y energía", x=0.06, y=0.995, ha="left", fontsize=13, fontweight="bold"); fig.subplots_adjust(top=0.78, left=0.06, right=0.985, bottom=0.08); guardar(fig, "08_actuadores_energia")

with PdfPages(os.path.join(RES, "informe_graficos.pdf")) as pdf:
    for f in figs: pdf.savefig(f)
vv = v[v.valido == 1]
print("error vision horizontal medio (cm):", round(100*np.hypot(vv.vis_x-vv.cam_x, vv.vis_y-vv.cam_y).mean(), 2), "| error altura kernel (cm):", round(100*(vv.vis_z-vv.cam_z).abs().mean(), 2),
      "| error ORB altura (cm):", round(100*(o.h_orb-o.cam_z).abs().mean(), 2), "| error EKF horizontal medio (m):", round(np.hypot(d.ekf_x-d.cam_x, d.ekf_y-d.cam_y).mean(), 2), "| aterrizaje (cm):", round(err, 1))
