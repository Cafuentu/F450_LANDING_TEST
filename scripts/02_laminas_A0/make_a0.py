import math, json, numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import black, white, Color
W, H = 841.0, 1189.0            # A0 vertical, mm
S = 250.0                       # lado del cuadro que ocupa la figura
EDGE = 60.0                     # distancia del cuadro al borde del papel
LAYOUTS = [
  dict(id="SLSD-A0-01", fig="cruz",     c=(W/2, H/2)),
  dict(id="SLSD-A0-02", fig="estrella", c=(W/2, H/2)),
  dict(id="SLSD-A0-03", fig="cara",     c=(W/2, H/2)),
]
def draw_fig(cv, kind, cx, cy, s, u):
    r = s/2
    cv.setFillColor(black); cv.setStrokeColor(black)
    if kind == "cruz":
        a = 110.0/360*s
        cv.rect((cx-a/2)*u, (cy-r)*u, a*u, s*u, stroke=0, fill=1)
        cv.rect((cx-r)*u, (cy-a/2)*u, s*u, a*u, stroke=0, fill=1)
    elif kind == "estrella":
        R = r/ math.cos(math.radians(18)) * 1.0   # ancho total = 2 R cos18 = s
        R = r/math.cos(math.radians(18)); ri = R*0.382
        cyy = cy - (R - (R + R*math.cos(math.radians(36)))/2)   # centrar bbox vertical
        p = cv.beginPath()
        for k in range(10):
            rr = R if k % 2 == 0 else ri; ang = math.radians(90 + 36*k)
            x, y = cx + rr*math.cos(ang), cyy + rr*math.sin(ang)
            (p.moveTo if k == 0 else p.lineTo)(x*u, y*u)
        p.close(); cv.drawPath(p, stroke=0, fill=1)
    elif kind == "cara":
        t = 28.0/360*s
        cv.setLineWidth(t*u); cv.circle(cx*u, cy*u, (r - t/2)*u, stroke=1, fill=0)
        for sx in (-1, 1): cv.circle((cx+sx*0.33*r)*u, (cy+0.28*r)*u, 0.11*r*u, stroke=0, fill=1)
        cv.setLineCap(1); rs = 0.55*r
        cv.arc((cx-rs)*u, (cy-0.05*r-rs)*u, (cx+rs)*u, (cy-0.05*r+rs)*u, startAng=205, extent=130)
def free_circle(c):
    xs = np.arange(0, W+1, 2.0); ys = np.arange(0, H+1, 2.0); X, Y = np.meshgrid(xs, ys)
    d_edge = np.minimum.reduce([X, W-X, Y, H-Y])
    dx = np.maximum(np.abs(X-c[0]) - S/2, 0); dy = np.maximum(np.abs(Y-c[1]) - S/2, 0)
    d = np.minimum(d_edge, np.hypot(dx, dy)); d[(dx == 0) & (dy == 0)] = 0
    i = np.unravel_index(d.argmax(), d.shape); return float(X[i]), float(Y[i]), float(d[i])
spec = []
for L in LAYOUTS:
    cv = canvas.Canvas(f"{L['id']}_{L['fig']}.pdf", pagesize=(W*mm, H*mm))
    cv.setTitle(f"{L['id']} {L['fig']} (A0, escala 1:1)"); cv.setAuthor("SLSD - F450")
    cv.setFillColor(white); cv.rect(0, 0, W*mm, H*mm, stroke=0, fill=1)
    draw_fig(cv, L["fig"], *L["c"], S, mm)
    cv.setFillColor(Color(0.6,0.6,0.6)); cv.setFont("Helvetica", 9)
    cv.drawCentredString(L["c"][0]*mm, (L["c"][1]-S/2-8)*mm, f"{L['id']}  {L['fig']}  cuadro {S:.0f} mm  A0 1:1")
    cv.showPage(); cv.save()
    fx, fy, fr = W/2, (L['c'][1]-S/2)/2, (L['c'][1]-S/2)/2
    spec.append(dict(id=L["id"], figura=L["fig"], papel_mm=[W, H], cuadro_figura_mm=S, centro_figura_mm=list(L["c"]),
                     bbox_figura_mm=[L["c"][0]-S/2, L["c"][1]-S/2, L["c"][0]+S/2, L["c"][1]+S/2],
                     mayor_circulo_libre=dict(centro_mm=[fx, fy], diametro_mm=round(2*fr, 1)),
                     zonas_libres_rect_mm=dict(inferior=[0,0,W,L['c'][1]-S/2], superior=[0,L['c'][1]+S/2,W,H], izquierda=[0,0,L['c'][0]-S/2,H], derecha=[L['c'][0]+S/2,0,W,H])))
json.dump(spec, open("laminas_A0_spec.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(spec, indent=1))

# hoja resumen A4 apaisada
A4W, A4H = 297.0, 210.0; k = 0.105
cv = canvas.Canvas("laminas_A0_resumen.pdf", pagesize=(A4W*mm, A4H*mm)); cv.setTitle("Láminas A0 - geometría")
cv.setFont("Helvetica-Bold", 12); cv.drawString(12*mm, 198*mm, "Láminas de prueba A0 (841 × 1189 mm) — geometría y zona libre de aterrizaje")
cv.setFont("Helvetica", 8); cv.drawString(12*mm, 193*mm, "Origen en la esquina inferior izquierda de la lámina. Figura negra inscrita en un cuadro de 250 × 250 mm, centrado en la lámina.")
cv.drawString(12*mm, 189*mm, "Tren de aterrizaje F450: 4 apoyos en un cuadrado de 343 mm de lado (círculo Ø486 mm); envolvente con hélices Ø710 mm.")
for n, sp in enumerate(spec):
    ox, oy = 14 + n*95, 40
    cv.saveState(); cv.translate(ox*mm, oy*mm); u = k*mm
    cv.setStrokeColor(black); cv.setLineWidth(0.6); cv.setFillColor(white); cv.rect(0, 0, W*u, H*u, stroke=1, fill=1)
    draw_fig(cv, sp["figura"], *sp["centro_figura_mm"], S, u)
    b = sp["bbox_figura_mm"]; cv.setLineWidth(0.4); cv.setDash(2, 2); cv.setStrokeColor(Color(0.85,0.1,0.1)); cv.rect(b[0]*u, b[1]*u, S*u, S*u, stroke=1, fill=0)
    fc = sp["mayor_circulo_libre"]; fx, fy = fc["centro_mm"]; fr = fc["diametro_mm"]/2
    cv.setStrokeColor(Color(0.1,0.5,0.1)); cv.circle(fx*u, fy*u, fr*u, stroke=1, fill=0)
    cv.setDash(); cv.setStrokeColor(Color(0.1,0.2,0.8)); cv.setLineWidth(0.5)
    h = 343/2; cv.rect((fx-h)*u, (fy-h)*u, 343*u, 343*u, stroke=1, fill=0)
    for sx in (-1, 1):
        for sy in (-1, 1): cv.circle((fx+sx*h)*u, (fy+sy*h)*u, 8*u, stroke=0, fill=1) if False else cv.circle((fx+sx*h)*u, (fy+sy*h)*u, 6*u, stroke=1, fill=0)
    cv.setDash()
    cv.restoreState()
    cv.setFillColor(black); cv.setFont("Helvetica-Bold", 9); cv.drawString(ox*mm, (oy-5)*mm, f"{sp['id']} — {sp['figura']}")
    cv.setFont("Helvetica", 7.5)
    cv.drawString(ox*mm, (oy-9)*mm, f"Cuadro figura: x {b[0]:.0f}–{b[2]:.0f}, y {b[1]:.0f}–{b[3]:.0f} mm")
    cv.drawString(ox*mm, (oy-12.5)*mm, f"Mayor círculo libre: Ø{fc['diametro_mm']:.0f} mm, centro ({fx:.0f}, {fy:.0f}) mm")
cv.setFont("Helvetica", 7.5); cv.setFillColor(black)
cv.drawString(12*mm, 12*mm, "Rojo: cuadro ocupado por la figura.   Verde: mayor círculo libre de figura dentro de la lámina.")
cv.drawString(12*mm, 8*mm, "Azul: apoyos del tren (cuadrado 343 mm) centrados en la franja libre inferior (841 × 469,5 mm): margen de ±63 mm en el eje largo; cabe con giro de hasta ±30° respecto de la lámina.")
cv.showPage(); cv.save()
