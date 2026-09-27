import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
d = pd.read_csv("mision_cruz.csv")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})
yaw = np.degrees(np.unwrap(np.radians(d.yaw_deg))); ysp = d.yaw_sp_deg.copy()
blocks = [(d.t[d.fase.str.contains(f"rumbo {r}\\)")].min(), d.t[d.fase.str.contains(f"rumbo {r}\\)")].max(), r) for r in (0, 45, 135, 225)]
fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True, gridspec_kw=dict(hspace=0.32, height_ratios=[1.2, 0.8, 0.9, 1, 1, 0.8]))
def deco(ax, title, yl):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=5); ax.set_ylabel(yl); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for t0, t1, r in blocks: ax.axvline(t0, color=GRID, lw=1)
ax = axs[0]; deco(ax, "Efector (cámara): posición horizontal en el mundo — llena: real, punteada: referencia", "m")
for k, (c, n) in enumerate([("cam_x", "x mundo"), ("cam_y", "y mundo")]):
    ax.plot(d.t, d[c], color=C[k], lw=1.8, label=n); ax.plot(d.t, d[c+"_sp"], color=C[k], lw=1, ls=(0, (2, 2)))
ax.legend(ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.2))
for t0, t1, r in blocks: ax.text((t0+t1)/2, 1.22, f"rumbo {r}°", ha="center", fontsize=9, color=MUTED)
ax.set_ylim(-1.2, 1.45)
ax = axs[1]; deco(ax, "Efector: altura", "m"); ax.plot(d.t, d.cam_z, color=C[2], lw=1.8); ax.plot(d.t, d.cam_z_sp, color=C[2], lw=1, ls=(0, (2, 2)))
ax = axs[2]; deco(ax, "Rumbo (yaw)", "grados"); ax.plot(d.t, yaw, color=C[3], lw=1.8); ax.set_yticks([0, 45, 135, 225, 360]); ax.set_yticklabels(["0", "45", "135", "225", "360 (=0)"])
ax = axs[3]; deco(ax, "Actitud: roll y pitch", "grados"); ax.plot(d.t, d.roll_deg, color=C[0], lw=1.5, label="roll"); ax.plot(d.t, d.pitch_deg, color=C[1], lw=1.5, label="pitch"); ax.legend(ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.22))
ax = axs[4]; deco(ax, "Velocidad de los rotores", "rpm")
for k in range(4): ax.plot(d.t, d[f"w{k}"]*60/2/np.pi, color=C[k], lw=1.1, label=f"M{k+1}")
ax.legend(ncol=4, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.22))
ax = axs[5]; deco(ax, "Potencia eléctrica total", "W"); ax.plot(d.t, d.P_elec_W, color=C[0], lw=1.3); ax.set_xlabel("tiempo (s)"); ax.set_xlim(0, d.t.max())
fig.suptitle("F450 — misión sobre la cruz: 4 m, patrón ±1 m en rumbos 0°, 45°, 135° y 225°", x=0.06, y=0.995, ha="left", fontsize=13, fontweight="bold")
fig.subplots_adjust(top=0.945, left=0.06, right=0.985, bottom=0.04)
fig.savefig("mision_cruz_graficos.png", dpi=125); fig.savefig("mision_cruz_graficos.pdf")

fig2, axs2 = plt.subplots(1, 2, figsize=(12.5, 6.2))
for ax, (cx, cy, ttl) in zip(axs2, [("cam_x", "cam_y", "Trayectoria del efector (cámara) en planta"), ("img_centro_x", "img_centro_y", "Punto del suelo que ve el centro de la imagen")]):
    ax.add_patch(Rectangle((-0.4205, -0.5945), 0.841, 1.189, fc="white", ec=MUTED, lw=1, zorder=1))
    ax.add_patch(Rectangle((-0.038, -0.125), 0.076, 0.25, fc=INK, zorder=2)); ax.add_patch(Rectangle((-0.125, -0.038), 0.25, 0.076, fc=INK, zorder=2))
    for k, (t0, t1, r) in enumerate(blocks):
        s = d[(d.t >= t0) & (d.t <= t1)]; ax.plot(-s[cy], s[cx], color=C[k], lw=1.6, label=f"rumbo {r}°", zorder=3)
    ax.set_aspect("equal"); ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("← +y mundo (izquierda)        −y mundo (derecha) →   [m]"); ax.set_ylabel("x mundo (adelante) [m]"); ax.set_title(ttl, loc="left", fontweight="bold")
axs2[0].legend(frameon=False, loc="upper right", ncol=2)
fig2.tight_layout(); fig2.savefig("mision_cruz_planta.png", dpi=125)
m = d[(d.cam_z > 3.9)]; print("desvio max centro imagen respecto camara (m):", round(float(np.hypot(m.img_centro_x-m.cam_x, m.img_centro_y-m.cam_y).max()), 2))
