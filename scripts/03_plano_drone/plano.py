import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg, numpy as np
from matplotlib.patches import Rectangle, Circle
src = "/mnt/user-data/uploads/Proyecto Drones/F450_modelo/planos/"
PW, PH = 420, 297; SC = 4.0            # A3 apaisado, escala 1:4
fig = plt.figure(figsize=(PW/25.4, PH/25.4)); ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,PW); ax.set_ylim(0,PH); ax.axis("off"); ax.set_aspect("equal")
ax.add_patch(Rectangle((6,6), PW-12, PH-12, fill=False, lw=1.2))
K = dict(color="#0b3d91", lw=0.6); TXT = dict(color="#0b3d91", fontsize=6.5, ha="center", va="center", backgroundcolor="white")
def dim(p, q, text, off=0.0, txt_off=(0,0)):
    p, q = np.array(p, float), np.array(q, float); d = (q-p)/np.linalg.norm(q-p); n = np.array([-d[1], d[0]])
    a, b = p + n*off, q + n*off
    if off: ax.plot(*zip(p, a), **K, ls=":"); ax.plot(*zip(q, b), **K, ls=":")
    ax.annotate("", a, b, arrowprops=dict(arrowstyle="<|-|>", color=K["color"], lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))
    m = (a+b)/2 + np.array(txt_off); rot = np.degrees(np.arctan2(d[1], d[0])); rot = rot-180 if rot > 90 or rot <= -90 else rot
    ax.text(*m, text, rotation=rot, rotation_mode="anchor", **TXT)
def leader(pt, txt_pt, text, ha="left"):
    ax.annotate(text, pt, txt_pt, fontsize=6.2, ha=ha, va="center", arrowprops=dict(arrowstyle="-|>", lw=0.5, color="k", mutation_scale=5))

# ---------- PLANTA ----------
cx, cy, half = 112, 178, 800/SC/2
ax.imshow(mpimg.imread(src+"vista_planta.png"), extent=(cx-half, cx+half, cy-half, cy+half), zorder=1)
T = lambda X, Y: (cx - Y/SC, cy + X/SC)          # modelo (mm, FLU) -> pagina
m = 225/np.sqrt(2)
dim(T(m, m), T(-m, -m), "450 (entre ejes)", 0, (-19, 19))
dim(T(m, m), T(m, -m), "318,2", off=(305-m)/SC)
dim(T(m, m), T(-m, m), "318,2", off=-(305-m)/SC)
for a in np.linspace(0, 2*np.pi, 80): pass
ax.add_patch(Circle(T(-m, -m), 130.4/SC, fill=False, ls="--", **K))
dim(T(-m-130.4*0.7071, -m+130.4*0.7071), T(-m+130.4*0.7071, -m-130.4*0.7071), "Ø260 hélice 10×4,5", 0, (0, -5.5))
h = 343/2
ax.add_patch(Rectangle((cx-h/SC, cy-h/SC), 343/SC, 343/SC, fill=False, ls="-.", color="#b00020", lw=0.7, zorder=3))
ax.text(cx, cy-h/SC-2.5, "apoyos del tren: cuadrado 343 × 343", color="#b00020", fontsize=6.2, ha="center", va="top")
ax.annotate("FRENTE (+X)", (cx, cy+half-3), (cx, cy+half-16), fontsize=7, ha="center", arrowprops=dict(arrowstyle="-|>", lw=0.8))
for i, (sx, sy, sp) in enumerate([(1,-1,"CCW"),(-1,1,"CCW"),(1,1,"CW"),(-1,-1,"CW")]):
    x, y = T(sx*m, sy*m); ax.text(x + (-sy)*14, y + sx*15, f"M{i+1} ({sp})", fontsize=6.2, ha="center", color="k", backgroundcolor="white")
ax.text(cx-half+2, cy+half-4, "PLANTA", fontsize=9, weight="bold")

# ---------- FRONTAL ----------
fx, fy, fw, fh = 318, 232, 800/SC, 350/SC
ax.imshow(mpimg.imread(src+"vista_frontal.png"), extent=(fx-fw/2, fx+fw/2, fy-fh/2, fy+fh/2), zorder=1)
F = lambda Y, Z: (fx + Y/SC, fy + (Z-20)/SC)
ax.plot([fx-fw/2+5, fx+fw/2-5], [F(0,-114.7)[1]]*2, color="k", lw=0.8); 
dim(F(-263.7, -114.7), F(263.7, -114.7), "527 (ancho total con hélices)", off=-7)
dim(F(285, -114.7), F(285, 158.9), "273,6", off=0)
dim(F(-285, -114.7), F(-285, 65.8), "180,5", off=0)
ax.text(fx-fw/2+2, fy+fh/2-1, "VISTA FRONTAL", fontsize=9, weight="bold", va="top")

# ---------- LATERAL ----------
lx, ly = 318, 128
ax.imshow(mpimg.imread(src+"vista_lateral.png"), extent=(lx-fw/2, lx+fw/2, ly-fh/2, ly+fh/2), zorder=1)
L = lambda X, Z: (lx + X/SC, ly + (Z-20)/SC)
ax.plot([lx-fw/2+5, lx+fw/2-5], [L(0,-114.7)[1]]*2, color="k", lw=0.8)
dim(L(-245, -114.7), L(245, -114.7), "490 (largo total con hélices)", off=-7)
dim(L(120, -114.7), L(120, -50.4), "64,3", off=0, txt_off=(7,0))
leader(L(-40, 152), L(-150, 150), "GPS M8N + mástil", ha="right")
leader(L(20, 34), L(150, 120), "Pixhawk")
leader(L(-30, 28), L(-190, 95), "Telemetría / receptor RC", ha="right")
leader(L(60, -7), L(230, 135), "Batería 3S 5200 mAh")
leader(L(-20, -42), L(-100, -100), "Raspberry Pi 4", ha="left")
leader(L(68, -50), L(25, -100), "Cámara nadir", ha="left")
leader(L(-67, -25), L(-215, -60), "UBEC 5 A", ha="right")
leader(L(-85, -8), L(-230, 30), "Power module", ha="right")
leader(L(205, -60), L(235, -20), "Tren PLA (×4)")
ax.text(lx-fw/2+2, ly+fh/2-1, "VISTA LATERAL (frente a la derecha)", fontsize=9, weight="bold", va="top")

# ---------- TABLAS ----------
comp = [("Frame F450 (2 placas + 4 brazos)", "282"), ("Motor brushless 2216 880 KV ×4", "232"), ("Hélice 1045 ×4", "40"), ("ESC 30 A ×4", "100"),
        ("Tren de aterrizaje PLA (f450_leg) ×4", "40"), ("Batería LiPo 3S 11,1 V 5200 mAh *", "220"), ("Pixhawk", "38"), ("GPS M8N + mástil", "45"),
        ("Telemetría + receptor RC", "35"), ("Power module + UBEC 5 A (2-8S→5 V)", "37"), ("Raspberry Pi 4", "46"), ("Cámara nadir + soporte", "10"), ("Cableado y tornillería", "40")]
x0, y0 = 214, 78
ax.text(x0, y0, "COMPONENTES Y MASAS [g]", fontsize=8, weight="bold")
for i, (n, g) in enumerate(comp):
    y = y0 - 5 - i*4.1; ax.text(x0, y, n, fontsize=6.6); ax.text(x0+86, y, g, fontsize=6.6, ha="right")
ax.plot([x0, x0+86], [y0-5-len(comp)*4.1+2.6]*2, color="k", lw=0.5)
ax.text(x0, y0-5-len(comp)*4.1-1, "TOTAL", fontsize=7, weight="bold"); ax.text(x0+86, y0-5-len(comp)*4.1-1, "1165", fontsize=7, weight="bold", ha="right")
x1 = 312
ax.text(x1, y0, "CARACTERÍSTICAS FÍSICAS (modelo)", fontsize=8, weight="bold")
phys = ["Masa total: 1,165 kg   (cuerpo sin hélices: 1,125 kg)", "Centro de gravedad (desde centro del frame): x = −3,4  y = 0,2  z = 12,4 mm",
        "Inercia en el CG:  Ixx = 0,0129   Iyy = 0,0135   Izz = 0,0234 kg·m²", "Distancia entre ejes: 450 mm (brazo 225 mm), configuración X",
        "Hélices 10×4,5\": M1 (del. der.) y M2 (tras. izq.) CCW; M3 y M4 CW", "Tren: 4 apoyos en cuadrado de 343 mm; despeje a cámara 64 mm",
        "Dimensiones totales: 490 × 527 × 274 mm (con hélices y mástil GPS)", "Cámara nadir: 68 mm delante del centro, eje óptico −Z;",
        "   óptica supuesta Pi Camera v2 (f = 3,04 mm, FOV 62,4° × 48,8°)", "Ejes cuerpo FLU: X adelante, Y izquierda, Z arriba"]
for i, t in enumerate(phys): ax.text(x1, y0-5-i*4.1, t, fontsize=6.4)
ax.text(x1, y0-5-len(phys)*4.1-1.5, "* Masa de batería informada por el usuario; demás masas de catálogo o estimadas.\n   Inercias calculadas a partir del modelo 3D y las masas de la tabla.", fontsize=5.8, style="italic", va="top")
# cajetin
ax.add_patch(Rectangle((6,6), 202, 16, fill=False, lw=0.8))
ax.text(9, 16.5, "Cuadricóptero F450 — Plano general, componentes y características físicas", fontsize=9, weight="bold")
ax.text(9, 10, "Proyecto SLSD · Escala 1:4 (A3) · Cotas en mm · Modelo: f450_real.blend · 2026-09-19", fontsize=7)
fig.savefig("plano_F450.pdf", dpi=400); fig.savefig("plano_F450_prev.png", dpi=110)
