#!/usr/bin/env python3
"""
monitor_pc.py - Monitor en tiempo real EN EL PC de lo que hace la Raspberry (agente_rpi.py).

Conexion: abre el tunel SSH en otra terminal   ssh -L 5555:localhost:5555 pi@<ip_de_la_raspberry>
          y luego                                python monitor_pc.py            (se conecta a localhost:5555)
          (o directo por la red:                 python monitor_pc.py --host <ip_de_la_raspberry>)

Ventana con 4 paneles, igual que en la simulacion:
  arriba-izq   camara nadir con la deteccion          arriba-der   mapa de correlacion NCC + kernel + puntaje vs tamano
  abajo-izq    proceso de decision                     abajo-der    graficas en vivo: NCC, altura, posicion, comandos
Teclas:  i = iniciar secuencia   p = parar (vuelve a espera)   l = LAND   g = grabar/pausar video   q = salir
Todo lo recibido se guarda en logs_pc/<fecha>.csv (y el video de la ventana si se graba).
"""
import argparse, base64, collections, csv, json, os, socket, time
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
AZ, NA, VE, OC, GR, BL = (208, 111, 31), (31, 87, 226), (106, 153, 18), (0, 133, 201), (150, 150, 150), (255, 255, 255)
FASES = ["espera", "buscar cruz", "centrado visual", "descenso a 3 m", "descenso a 1 m", "descenso final", "en tierra"]
def txt(im, s, xy, sc=0.5, col=BL, th=1): cv2.putText(im, s, xy, FONT, sc, col, th, cv2.LINE_AA)
def dejpg(s): return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)

def panel_correlacion(f):
    P = np.full((480, 640, 3), 30, np.uint8); txt(P, "CORRELACION KERNEL + FFT", (200, 16), 0.5, OC)
    if "ncc_img" in f:
        P[24:264, 240:560] = cv2.resize(dejpg(f["ncc_img"]), (320, 240))
        k = cv2.resize(dejpg(f["kernel_img"]), (100, 100), interpolation=cv2.INTER_NEAREST); P[60:160, 60:160] = k; cv2.rectangle(P, (60, 60), (160, 160), GR, 1)
        txt(P, "kernel", (60, 52), 0.42, GR); txt(P, "cruz completa" if f["modo_kernel"] == "completa" else "centro (brazos)", (40, 180), 0.42); txt(P, f"lado {f['L']:.1f} px", (40, 198), 0.42)
        e, p = f.get("escalas") or [], f.get("puntajes") or []
        if len(e) > 1:
            x0, y0, w, h = 40, 300, 560, 110; cv2.rectangle(P, (x0, y0), (x0+w, y0+h), GR, 1)
            pts = [(int(x0 + i*w/(len(e)-1)), int(y0 + h - max(min(pp, 1), 0)*h)) for i, pp in enumerate(p)]
            cv2.polylines(P, [np.array(pts)], False, AZ, 2); [cv2.circle(P, q, 3, BL, -1) for q in pts]
            yu = int(y0 + h - 0.55*h); cv2.line(P, (x0, yu), (x0+w, yu), NA, 1); txt(P, "umbral 0.55", (x0+w-90, yu-4), 0.4, NA)
            txt(P, f"NCC max vs tamano del kernel ({e[0]:.0f}-{e[-1]:.0f} px)", (x0, y0-6), 0.42, GR)
        txt(P, f"centro ({f['u']:.0f}, {f['v']:.0f}) px", (40, 440), 0.5); txt(P, f"NCC = {f['ncc']:.3f}", (420, 440), 0.65, VE if f["valido"] else NA, 2)
        if f.get("cam"): txt(P, f"h = {f['cam'][2]:.2f} m", (40, 464), 0.5)
    else: txt(P, "kernel sin resultado", (200, 240), 0.7, GR, 2)
    return P

def panel_decision(f, conectado):
    P = np.full((480, 640, 3), 30, np.uint8); txt(P, "PROCESO DE DECISION", (20, 22), 0.6, OC, 2); txt(P, f"t = {f.get('t', 0):6.1f} s", (480, 22), 0.55)
    ki = FASES.index(f["fase"]) if f.get("fase") in FASES else 0
    for i, n in enumerate(FASES):
        y = 50 + i*22; col = VE if i < ki else (BL if i == ki else GR)
        if i == ki: cv2.rectangle(P, (14, y-16), (300, y+6), (60, 60, 60), -1)
        txt(P, f"{i+1}. {n}", (20, y), 0.48, col, 2 if i == ki else 1)
    txt(P, "ACCION", (20, 220), 0.45, GR); txt(P, str(f.get("nota", ""))[:52], (20, 240), 0.42)
    x = 330; txt(P, "TELEMETRIA PIXHAWK", (x, 50), 0.45, GR)
    for i, s in enumerate([f"modo {f.get('modo_fc','?')}   {'ARMADO' if f.get('armado') else 'desarmado'}", f"bateria {f.get('bat_v', 0):.2f} V", f"roll {f.get('roll', 0):+.1f}  pitch {f.get('pitch', 0):+.1f}  yaw {f.get('yaw', 0):.0f} deg",
                           f"altura FC {f.get('alt_fc', 0):.2f} m ({f.get('alt_fuente','')})"]): txt(P, s, (x, 72 + i*20), 0.45)
    txt(P, "VISION (camara respecto de la cruz)", (x, 170), 0.45, GR)
    c = f.get("cam"); txt(P, f"x {c[0]:+.2f}  y {c[1]:+.2f}  h {c[2]:.2f} m" if c else "sin deteccion", (x, 190), 0.45, BL if c else NA)
    txt(P, "INSTRUCCION DE VUELO (vx, vy, vz) m/s", (x, 230), 0.45, GR); v = f.get("cmd", [0, 0, 0])
    txt(P, f"{v[0]:+.2f}  {v[1]:+.2f}  {v[2]:+.2f}", (x, 252), 0.55, BL, 2)
    txt(P, "ENVIADA AL PIXHAWK" if f.get("enviado") else ("DRY-RUN: no se envia" if f.get("dry_run") else "no enviada (modo/armado)"), (x, 276), 0.45, VE if f.get("enviado") else NA)
    cv2.rectangle(P, (20, 300), (300, 320), GR, 1); s = max(min(f.get("ncc", 0), 1), 0); cv2.rectangle(P, (21, 301), (21+int(278*s), 319), VE if s >= 0.55 else NA, -1)
    cv2.line(P, (21+int(278*0.55), 296), (21+int(278*0.55), 324), BL, 2); txt(P, "NCC (umbral 0.55)", (20, 292), 0.42, GR)
    txt(P, f"lazo en la RPi: {f.get('dt_ms', 0):.0f} ms", (20, 350), 0.42, GR); txt(P, "conectado a la RPi" if conectado else "SIN CONEXION", (20, 372), 0.45, VE if conectado else NA)
    txt(P, "teclas: i iniciar | p parar | l LAND | g grabar | q salir", (20, 460), 0.42, GR)
    return P

def panel_graficas(hist):
    P = np.full((480, 640, 3), 30, np.uint8); txt(P, "SERIES EN VIVO (ultimos 60 s)", (200, 16), 0.5, OC)
    if len(hist) < 2: return P
    t = np.array([h["t"] for h in hist]); t0 = max(t[-1] - 60, t[0])
    def serie(y, yl, yh, y0, h, col, etiq, dots=False):
        x0, w = 50, 570; cv2.rectangle(P, (x0, y0), (x0+w, y0+h), GR, 1); txt(P, etiq, (x0, y0-4), 0.4, GR); txt(P, f"{yh:g}", (4, y0+10), 0.35, GR); txt(P, f"{yl:g}", (4, y0+h), 0.35, GR)
        pts = [(int(x0 + w*(tt - t0)/max(t[-1]-t0, 1e-3)), int(y0 + h - h*(min(max(yy, yl), yh) - yl)/(yh - yl))) for tt, yy in zip(t, y) if tt >= t0 and yy is not None and np.isfinite(yy)]
        if len(pts) > 1: (cv2.polylines(P, [np.array(pts)], False, col, 1) if not dots else [cv2.circle(P, q, 2, col, -1) for q in pts])
    serie([h["ncc"] for h in hist], 0, 1, 34, 80, AZ, "NCC"); cv2.line(P, (50, int(34+80-80*0.55)), (620, int(34+80-80*0.55)), NA, 1)
    serie([h["cam"][2] if h.get("cam") else np.nan for h in hist], 0, 5, 140, 80, VE, "altura camara [m]: kernel (verde) y Pixhawk (gris)", dots=True); serie([h["alt_fc"] for h in hist], 0, 5, 140, 80, GR, "")
    serie([h["cam"][0] if h.get("cam") else np.nan for h in hist], -1.5, 1.5, 246, 80, AZ, "posicion respecto de la cruz [m]: x (azul), y (naranja)"); serie([h["cam"][1] if h.get("cam") else np.nan for h in hist], -1.5, 1.5, 246, 80, NA, "")
    serie([h["cmd"][0] for h in hist], -0.4, 0.4, 352, 80, AZ, "instruccion [m/s]: vx (azul), vy (naranja), vz (verde)"); serie([h["cmd"][1] for h in hist], -0.4, 0.4, 352, 80, NA, ""); serie([h["cmd"][2] for h in hist], -0.4, 0.4, 352, 80, VE, "")
    return P

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="localhost"); ap.add_argument("--puerto", type=int, default=5555); ap.add_argument("--log", default="logs_pc"); a = ap.parse_args()
    os.makedirs(a.log, exist_ok=True); nombre = time.strftime("monitor_%Y%m%d_%H%M%S")
    fcsv = open(os.path.join(a.log, nombre + ".csv"), "w", newline=""); wcsv = csv.writer(fcsv); wcsv.writerow(["t_pc", "t_rpi", "fase", "modo_fc", "armado", "bat_v", "roll", "pitch", "yaw", "alt_fc", "valido", "ncc", "modo_kernel", "L", "u", "v", "cam_x", "cam_y", "cam_z", "vx", "vy", "vz", "enviado", "dt_ms"])
    hist = collections.deque(maxlen=1200); f = {}; sock = None; buf = b""; grab = None; cv2.namedWindow("F450 - prueba real", cv2.WINDOW_NORMAL)
    def conectar():
        s = socket.create_connection((a.host, a.puerto), timeout=3); s.settimeout(0.05); print("[monitor] conectado a", a.host, a.puerto); return s
    while True:
        if sock is None:
            try: sock = conectar()
            except OSError: time.sleep(1)
        if sock is not None:
            try:
                chunk = sock.recv(1 << 16)
                if not chunk: raise OSError("cerrado")
                buf += chunk
                while b"\n" in buf:
                    linea, buf = buf.split(b"\n", 1)
                    try: f = json.loads(linea)
                    except json.JSONDecodeError: continue
                    hist.append(f); c = f.get("cam") or ["", "", ""]
                    wcsv.writerow([round(time.time(), 3), f["t"], f["fase"], f["modo_fc"], f["armado"], f["bat_v"], f["roll"], f["pitch"], f["yaw"], f["alt_fc"], int(f["valido"]), f["ncc"], f["modo_kernel"], f["L"], f["u"], f["v"], *c, *f["cmd"], int(f["enviado"]), f["dt_ms"]])
            except socket.timeout: pass
            except OSError: sock = None; print("[monitor] conexion perdida, reintentando")
        cam = cv2.resize(dejpg(f["img"]), (640, 480)) if f.get("img") else np.full((480, 640, 3), 30, np.uint8)
        txt(cam, "CAMARA NADIR (RPi)", (8, 470), 0.5, (0, 255, 255))
        marco = np.vstack([np.hstack([cam, panel_correlacion(f) if f else np.full((480, 640, 3), 30, np.uint8)]), np.hstack([panel_decision(f, sock is not None), panel_graficas(hist)])])
        if grab is not None: grab.write(marco); cv2.circle(marco, (1260, 20), 8, (0, 0, 255), -1)
        cv2.imshow("F450 - prueba real", marco); k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k in (ord("i"), ord("p"), ord("l")) and sock is not None:
            orden = {ord("i"): b"iniciar", ord("p"): b"parar", ord("l"): b"land"}[k]; sock.sendall(orden + b"\n"); print("[orden]", orden.decode())
        if k == ord("g"):
            if grab is None: grab = cv2.VideoWriter(os.path.join(a.log, nombre + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 960)); print("[grabando]")
            else: grab.release(); grab = None; print("[grabacion detenida]")
    if grab is not None: grab.release()
    fcsv.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
