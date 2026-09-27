import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv("vuelo_prueba.csv")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "font.family": "DejaVu Sans"})
ph = d.groupby("fase", sort=False).t.min()
starts = d.loc[d.fase != d.fase.shift(), ["t", "fase"]]
fig, axs = plt.subplots(5, 1, figsize=(11, 12.5), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1, 1, 1, 0.9], hspace=0.28))
def deco(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=6); ax.set_ylabel(ylabel); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for t in starts.t: ax.axvline(t, color=GRID, lw=0.8, zorder=0)
ax = axs[0]; deco(ax, "Posición (línea llena) y referencia (punteada)", "m")
for k, (c, n) in enumerate(zip("xyz", ["x (adelante)", "y (izquierda)", "z (altura)"])):
    ax.plot(d.t, d[c], color=C[k], lw=2, label=n); ax.plot(d.t, d[c+"_sp"], color=C[k], lw=1, ls=(0, (2, 2)))
ax.legend(ncol=3, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.22))
ax = axs[1]; deco(ax, "Actitud: roll y pitch", "grados")
ax.plot(d.t, d.roll_deg, color=C[0], lw=2, label="roll"); ax.plot(d.t, d.pitch_deg, color=C[1], lw=2, label="pitch"); ax.legend(ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.25))
ax = axs[2]; deco(ax, "Rumbo (yaw)", "grados"); ax.plot(d.t, d.yaw_deg, color=C[2], lw=2)
ax = axs[3]; deco(ax, "Velocidad de los rotores", "rpm")
for k in range(4): ax.plot(d.t, d[f"w{k}"]*60/2/np.pi, color=C[k], lw=1.4, label=f"M{k+1}")
ax.legend(ncol=4, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.25))
ax = axs[4]; deco(ax, "Potencia eléctrica total (motores + Raspberry Pi, cámara y aviónica)", "W"); ax.plot(d.t, d.P_elec_W, color=C[0], lw=1.6); ax.set_xlabel("tiempo (s)")
ax.set_xlim(0, d.t.max())
top = axs[0]; ymax = top.get_ylim()[1]
for i, (t, f) in enumerate(zip(starts.t, starts.fase)):
    top.text(t+0.2, ymax*1.30 + (0 if i % 2 == 0 else ymax*0.0), f, rotation=55, fontsize=7.2, color=MUTED, ha="left", va="bottom")
fig.suptitle("F450 — vuelo de prueba simulado (6-DOF)", x=0.07, y=0.995, ha="left", fontsize=13, fontweight="bold")
fig.subplots_adjust(top=0.86, left=0.07, right=0.98, bottom=0.05)
fig.savefig("vuelo_prueba_graficos.png", dpi=130); fig.savefig("vuelo_prueba_graficos.pdf")
# trayectoria en planta y perfil
fig2, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
a.plot(-d.y, d.x, color=C[0], lw=2); a.set_aspect("equal"); a.set_xlabel("← izquierda      derecha → (m)"); a.set_ylabel("adelante (m)"); a.grid(color=GRID, lw=0.8)
a.set_title("Trayectoria en planta", loc="left", fontweight="bold"); a.plot(-d.y.iloc[0], d.x.iloc[0], "o", color=INK, ms=6); a.annotate("despegue / aterrizaje", (-d.y.iloc[0], d.x.iloc[0]), (8, -14), textcoords="offset points", fontsize=8, color=MUTED)
a.add_patch(plt.Rectangle((-0.4205, -0.5945), 0.841, 1.189, fill=False, ec=MUTED, lw=1)); a.annotate("lámina A0", (0.43, 0.45), fontsize=8, color=MUTED)
b.plot(d.t, d.z - d.z.iloc[0], color=C[2], lw=2); b.set_xlabel("tiempo (s)"); b.set_ylabel("altura sobre el suelo (m)"); b.grid(axis="y", color=GRID, lw=0.8); b.set_title("Perfil de altura", loc="left", fontweight="bold")
fig2.tight_layout(); fig2.savefig("vuelo_prueba_trayectoria.png", dpi=130)
s = d.groupby("fase", sort=False).agg(roll=("roll_deg", lambda v: v.abs().max()), pitch=("pitch_deg", lambda v: v.abs().max()))
print(s.round(1).to_string()); print("z_max", round(d.z.max()-d.z.iloc[0],3), "E_Wh", round(d.E_Wh.iloc[-1],2), "t_end", d.t.max())
land = d[(d.fase=="aterrizaje")&(d.contact==1)]; print("vz toque", round(land.vz.iloc[0],3), "error xy final (mm)", round(1000*np.hypot(d.x.iloc[-1]-d.x.iloc[0], d.y.iloc[-1]-d.y.iloc[0]),1))
