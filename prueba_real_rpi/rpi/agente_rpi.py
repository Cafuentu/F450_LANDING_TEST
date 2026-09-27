#!/usr/bin/env python3
"""
agente_rpi.py - Agente que corre EN LA RASPBERRY PI (F450 + Pi Camera v2 + Pixhawk/ArduPilot).

Hace, a ~10 Hz:
  1. captura un cuadro de la camara nadir
  2. lee la telemetria del Pixhawk (actitud, altura, modo, armado, bateria) por MAVLink
  3. corre el detector de la cruz (kernel + FFT) y calcula la posicion de la camara respecto de la cruz
  4. decide (maquina de estados igual a la simulacion) y genera la instruccion de vuelo
  5. envia la instruccion al Pixhawk SOLO si el modo de seguridad lo permite (por defecto NO envia: dry-run)
  6. publica todo (imagen con deteccion, mapa de correlacion, numeros, decision) por TCP para verlo en el PC
  7. registra todo en CSV

Uso tipico (drone anclado, sin enviar comandos):
    python3 agente_rpi.py --camara picam --mav /dev/ttyACM0 --baud 115200
Para ver el stream en el PC:  ssh -L 5555:localhost:5555 pi@<ip>   y luego  python3 monitor_pc.py
Prueba sin hardware:          python3 agente_rpi.py --camara sim --mav sim

SEGURIDAD: --enviar activa el envio de setpoints de velocidad (solo en modo GUIDED, armado y con el drone anclado).
Sin --enviar, el agente imprime y publica las instrucciones pero no manda nada al Pixhawk.
"""
import argparse, base64, csv, json, math, os, socket, threading, time
import numpy as np
import cv2
from detector_cruz import DetectorCruz, preparar, rot_rpy, W, H, FX, CRUZ_LADO, CRUZ_BRAZO

# ============================================================== CONFIGURACION (Armando: ajustar aqui) ==============================================================
MAV_PUERTO   = "/dev/ttyACM0"     # Pixhawk por USB en la Raspberry. Por TELEM2 suele ser /dev/serial0 o /dev/ttyAMA0
MAV_BAUD     = 115200             # USB: 115200. TELEM2: 57600 o 921600 (segun SERIAL2_BAUD del Pixhawk)
PUERTO_TCP   = 5555               # puerto del stream hacia el PC (se tuneliza por SSH)
FREC_HZ      = 10                 # frecuencia del lazo de percepcion
CAM_OFFSET_B = np.array([0.0714, -0.0002, -0.0628])   # camara respecto del centro de masa, ejes cuerpo FLU (m) - del modelo 3D
H_CAM_TIERRA = 0.0643             # altura de la camara con el tren apoyado (m)
V_MAX        = 0.30               # m/s, limite de velocidad de las instrucciones (drone anclado)
BATT_ABORTA  = 10.5               # V (3S): bajo esto -> LAND
# =======================================================================================================================================================================


# ------------------------------------------------------------------ fuentes de imagen
class CamaraPi:
    """Pi Camera v2 por picamera2, campo de vision completo (modo binned 1640x1232 -> 640x480)."""
    def __init__(self):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}, raw={"size": (1640, 1232)}, controls={"FrameRate": 30})
        self.cam.configure(cfg); self.cam.start(); time.sleep(0.5)
    def leer(self):
        rgb = self.cam.capture_array(); return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

class CamaraCV:
    """Camara USB o archivo de video via OpenCV (indice numerico o ruta)."""
    def __init__(self, fuente):
        self.cap = cv2.VideoCapture(int(fuente) if str(fuente).isdigit() else fuente)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    def leer(self):
        ok, f = self.cap.read()
        if not ok: raise RuntimeError("sin cuadro de la camara")
        return f

class CamaraSim:
    """Imagen sintetica de la lamina (misma camara de la simulacion) siguiendo la telemetria simulada. Solo para probar sin hardware."""
    def __init__(self, mav):
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim_local"))
        from camara_vision import Suelo, Camara
        self.cam = Camara(Suelo()); self.mav = mav
    def leer(self):
        t = self.mav.estado(); R = rot_rpy(t["roll"], t["pitch"], t["yaw"])
        p = np.array([t["sim_x"], t["sim_y"], t["alt"]]) + R @ CAM_OFFSET_B
        return self.cam.imagen(p, R)


# ------------------------------------------------------------------ MAVLink
class Mav:
    """Lector de telemetria y emisor de instrucciones (ArduPilot). Hilo propio."""
    def __init__(self, puerto, baud, enviar):
        from pymavlink import mavutil
        self.mavutil = mavutil; self.enviar = enviar
        self.m = mavutil.mavlink_connection(puerto, baud=baud) if not (puerto.startswith("udp") or puerto.startswith("tcp")) else mavutil.mavlink_connection(puerto)
        print("[mav] esperando heartbeat en", puerto); self.m.wait_heartbeat(timeout=20); print("[mav] heartbeat ok, sistema", self.m.target_system)
        for msg_id, hz in ((30, 20), (33, 5), (1, 2), (173, 10)):        # ATTITUDE, GLOBAL_POSITION_INT, SYS_STATUS, RANGEFINDER
            self.m.mav.command_long_send(self.m.target_system, self.m.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, int(1e6/hz), 0, 0, 0, 0, 0)
        self.st = dict(roll=0.0, pitch=0.0, yaw=0.0, alt=0.0, alt_fuente="", modo="?", armado=False, bat_v=0.0, t_att=0.0)
        self.lock = threading.Lock(); threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while True:
            msg = self.m.recv_match(blocking=True, timeout=1)
            if msg is None: continue
            t = msg.get_type()
            with self.lock:
                if t == "ATTITUDE": self.st.update(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw, t_att=time.time())
                elif t == "GLOBAL_POSITION_INT" and self.st["alt_fuente"] != "rangefinder": self.st.update(alt=msg.relative_alt/1000.0, alt_fuente="baro")
                elif t == "RANGEFINDER": self.st.update(alt=msg.distance, alt_fuente="rangefinder")
                elif t == "HEARTBEAT": self.st.update(modo=self.mavutil.mode_string_v10(msg), armado=bool(msg.base_mode & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
                elif t == "SYS_STATUS": self.st["bat_v"] = msg.voltage_battery/1000.0
    def estado(self):
        with self.lock: return dict(self.st)
    def velocidad_cuerpo(self, vx, vy, vz, yaw_rate=0.0):
        """Setpoint de velocidad en ejes cuerpo (FLU -> FRD de MAVLink). Solo se envia con --enviar, GUIDED y armado."""
        st = self.estado(); ok = self.enviar and st["modo"] == "GUIDED" and st["armado"]
        if ok:
            mv = self.mavutil.mavlink
            self.m.mav.set_position_target_local_ned_send(0, self.m.target_system, self.m.target_component, mv.MAV_FRAME_BODY_OFFSET_NED,
                0b0000011111000111, 0, 0, 0, float(vx), float(-vy), float(-vz), 0, 0, 0, 0, float(yaw_rate))
        return ok
    def land(self):
        if self.enviar: self.m.set_mode("LAND")

class MavSim:
    """Telemetria simulada: el drone 'obedece' las instrucciones con un retardo. Para probar sin hardware."""
    def __init__(self):
        self.p = np.array([0.0, -1.0, 4.0]); self.v = np.zeros(3); self.rpy = np.zeros(3); self.t0 = time.time(); self.enviar = True
    def estado(self):
        return dict(roll=self.rpy[0], pitch=self.rpy[1], yaw=0.0, alt=self.p[2], alt_fuente="sim", modo="GUIDED", armado=True, bat_v=11.6, t_att=time.time(), sim_x=self.p[0], sim_y=self.p[1])
    def velocidad_cuerpo(self, vx, vy, vz, yaw_rate=0.0):
        dt = 0.1; self.v += (np.array([vx, vy, vz]) - self.v)*0.3; self.p += self.v*dt; self.p[2] = max(self.p[2], H_CAM_TIERRA + 0.0628)
        self.rpy[:2] = [-0.15*self.v[1], 0.15*self.v[0]]; return True
    def land(self): pass


# ------------------------------------------------------------------ stream TCP hacia el PC
class Publicador:
    def __init__(self, puerto):
        self.srv = socket.socket(); self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); self.srv.bind(("0.0.0.0", puerto)); self.srv.listen(2)
        self.clientes = []; threading.Thread(target=self._aceptar, daemon=True).start(); print(f"[tcp] publicando en puerto {puerto}")
    def _aceptar(self):
        while True:
            c, a = self.srv.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); self.clientes.append(c); print("[tcp] cliente", a)
    def enviar(self, obj):
        data = (json.dumps(obj) + "\n").encode()
        for c in list(self.clientes):
            try: c.sendall(data)
            except OSError: self.clientes.remove(c)

def jpg64(img, q=70): return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


# ------------------------------------------------------------------ maquina de estados (misma logica que la simulacion)
FASES = ["espera", "buscar cruz", "centrado visual", "descenso a 3 m", "descenso a 1 m", "descenso final", "en tierra"]

class Decision:
    def __init__(self):
        self.fase = 0; self.t_fase = time.time(); self.t_ok = None; self.hist = []
    def paso(self, det_ok, pos, alt, contacto):
        """pos: posicion de la camara respecto de la cruz (x adelante, y izquierda, z altura). Devuelve (vx, vy, vz) en ejes cuerpo y texto."""
        f = FASES[self.fase]; vx = vy = vz = 0.0; nota = ""
        e_h = math.hypot(pos[0], pos[1]) if det_ok else float("nan")
        def ir(obj_z, kp=0.6, vz_max=0.25):
            nonlocal vx, vy, vz
            if det_ok: vx = float(np.clip(-kp*pos[0], -V_MAX, V_MAX)); vy = float(np.clip(-kp*pos[1], -V_MAX, V_MAX))
            vz = float(np.clip(0.8*(obj_z - alt), -vz_max, vz_max))
        def llegado(tol, espera):
            ok = det_ok and e_h < tol and abs(pos[2] - obj) < 0.15
            self.t_ok = (self.t_ok or time.time()) if ok else None
            return self.t_ok is not None and time.time() - self.t_ok >= espera
        if f == "espera": nota = "esperando orden de inicio"
        elif f == "buscar cruz":
            nota = "hover; kernel activo"; vz = 0.0
            if det_ok and time.time() - self.t_fase > 3: self.sig()
        elif f == "centrado visual":
            obj = alt if not hasattr(self, "z0") else self.z0
            if not hasattr(self, "z0"): self.z0 = alt
            ir(obj); nota = f"centrar sobre la cruz (err {e_h*100:.0f} cm)" if det_ok else "sin deteccion: hover"
            if llegado(0.10, 2.0): self.sig()
        elif f == "descenso a 3 m": obj = 3.0; ir(obj); nota = "bajar a 3 m manteniendo centro"; (self.sig() if llegado(0.10, 1.0) else None)
        elif f == "descenso a 1 m": obj = 1.0; ir(obj); nota = "bajar a 1 m manteniendo centro"; (self.sig() if llegado(0.08, 1.0) else None)
        elif f == "descenso final":
            despeje = alt - H_CAM_TIERRA; vz = -0.25 if despeje > 0.30 else -0.12
            if det_ok: vx = float(np.clip(-0.6*pos[0], -0.15, 0.15)); vy = float(np.clip(-0.6*pos[1], -0.15, 0.15))
            nota = f"descenso final, despeje {despeje*100:.0f} cm"
            if contacto: self.sig()
        else: nota = "en tierra: motores a ralenti / desarmar"
        return vx, vy, vz, nota
    def sig(self):
        if self.fase < len(FASES)-1: self.fase += 1; self.t_fase = time.time(); self.t_ok = None
    def iniciar(self):
        if self.fase == 0: self.sig()


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", default="picam", help="picam | sim | indice USB (0) | ruta de video")
    ap.add_argument("--mav", default=MAV_PUERTO, help="puerto MAVLink (/dev/ttyACM0, udp:..., o 'sim')")
    ap.add_argument("--baud", type=int, default=MAV_BAUD)
    ap.add_argument("--enviar", action="store_true", help="ENVIA setpoints de velocidad al Pixhawk (solo GUIDED y armado). Sin esto: dry-run")
    ap.add_argument("--auto", action="store_true", help="inicia la secuencia sin esperar la orden del monitor")
    ap.add_argument("--puerto", type=int, default=PUERTO_TCP)
    ap.add_argument("--log", default="logs")
    a = ap.parse_args()

    mav = MavSim() if a.mav == "sim" else Mav(a.mav, a.baud, a.enviar)
    cam = CamaraSim(mav) if a.camara == "sim" else (CamaraPi() if a.camara == "picam" else CamaraCV(a.camara))
    det = DetectorCruz(); dec = Decision(); pub = Publicador(a.puerto)
    if a.auto: dec.iniciar()
    os.makedirs(a.log, exist_ok=True); nombre = time.strftime("vuelo_%Y%m%d_%H%M%S")
    fcsv = open(os.path.join(a.log, nombre + ".csv"), "w", newline=""); wcsv = csv.writer(fcsv)
    wcsv.writerow(["t", "fase", "modo_fc", "armado", "bat_v", "roll_deg", "pitch_deg", "yaw_deg", "alt_fc", "alt_fuente", "valido", "ncc", "modo_kernel", "L_px", "u", "v", "cam_x", "cam_y", "cam_z", "vx_cmd", "vy_cmd", "vz_cmd", "enviado", "dt_ms"])
    vid = cv2.VideoWriter(os.path.join(a.log, nombre + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FREC_HZ, (W, H))
    # ordenes que llegan desde el monitor (por el mismo socket): 'iniciar', 'parar', 'land'
    def escuchar():
        while True:
            for c in list(pub.clientes):
                try:
                    c.settimeout(0.01); linea = c.recv(64)
                    if b"iniciar" in linea: dec.iniciar(); print("[orden] iniciar")
                    if b"parar" in linea: dec.fase = 0; print("[orden] parar")
                    if b"land" in linea: mav.land(); dec.fase = len(FASES)-1; print("[orden] LAND")
                except (socket.timeout, OSError): pass
            time.sleep(0.1)
    threading.Thread(target=escuchar, daemon=True).start()

    L_prev = None; t0 = time.time(); k = 0; print("[agente] modo", "ENVIO ACTIVO" if a.enviar else "DRY-RUN (no envia comandos)")
    while True:
        t_ini = time.time(); t = t_ini - t0
        img = preparar(cam.leer()); st = mav.estado()
        R = rot_rpy(st["roll"], st["pitch"], st["yaw"])
        h_pred = max(st["alt"] + (R @ CAM_OFFSET_B)[2], 0.15)
        d = det.detectar_auto(img, h_pred, yaw_deg=0.0, L_prev=L_prev)
        pos = None
        if d["valido"]:
            pos = DetectorCruz.a_mundo(d, R); L_prev = d["L"] if d["modo"] == "completa" else d["L"]*CRUZ_LADO/CRUZ_BRAZO
        elif d["score"] < 0.3: L_prev = None
        alt_cam = pos[2] if pos is not None else h_pred
        contacto = alt_cam < H_CAM_TIERRA + 0.03 and abs(st["alt"]) < 0.15
        vx, vy, vz, nota = dec.paso(pos is not None, pos if pos is not None else np.zeros(3), alt_cam, contacto)
        # seguridad
        if st["bat_v"] and st["bat_v"] < BATT_ABORTA and a.mav != "sim": mav.land(); nota = "BATERIA BAJA -> LAND"; vx = vy = vz = 0.0
        if time.time() - st["t_att"] > 1.0 and a.mav != "sim": vx = vy = vz = 0.0; nota = "sin telemetria -> hover"
        enviado = mav.velocidad_cuerpo(vx, vy, vz) if dec.fase not in (0, len(FASES)-1) else False
        dt_ms = (time.time() - t_ini)*1000
        # visual
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if d["valido"]:
            cv2.drawMarker(vis, (int(d["u"]), int(d["v"])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            r = int(d["L"]*(0.65 if d["modo"] == "completa" else 1.2)); cv2.rectangle(vis, (int(d["u"])-r, int(d["v"])-r), (int(d["u"])+r, int(d["v"])+r), (0, 200, 0), 1)
        cv2.putText(vis, f"t={t:6.1f}s NCC={d['score']:.2f} h={alt_cam:.2f}m {FASES[dec.fase]}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        vid.write(vis)
        ncc_img = None
        if "ncc" in d:
            ncc_img = cv2.applyColorMap((np.clip((d["ncc"] + 0.2)/1.2, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            cv2.circle(ncc_img, (int(d["u"]), int(d["v"])), 14, (255, 255, 255), 2); ncc_img = cv2.resize(ncc_img, (320, 240))
            kern = cv2.resize((d["kernel"]*255).astype(np.uint8), (80, 80), interpolation=cv2.INTER_NEAREST)
        fila = dict(t=round(t, 3), fase=FASES[dec.fase], nota=nota, modo_fc=st["modo"], armado=st["armado"], bat_v=round(st["bat_v"], 2),
                    roll=round(math.degrees(st["roll"]), 2), pitch=round(math.degrees(st["pitch"]), 2), yaw=round(math.degrees(st["yaw"]), 1), alt_fc=round(st["alt"], 3), alt_fuente=st["alt_fuente"],
                    valido=bool(d["valido"]), ncc=round(d["score"], 3), modo_kernel=d["modo"], L=round(d.get("L", 0), 1), u=round(d.get("u", 0), 1), v=round(d.get("v", 0), 1),
                    cam=[round(float(x), 3) for x in pos] if pos is not None else None, cmd=[round(vx, 3), round(vy, 3), round(vz, 3)], enviado=enviado, dry_run=not a.enviar, dt_ms=round(dt_ms, 1),
                    escalas=d.get("escalas"), puntajes=d.get("puntajes"))
        wcsv.writerow([fila["t"], fila["fase"], st["modo"], st["armado"], fila["bat_v"], fila["roll"], fila["pitch"], fila["yaw"], fila["alt_fc"], st["alt_fuente"], int(d["valido"]), fila["ncc"], d["modo"], fila["L"], fila["u"], fila["v"],
                       *(fila["cam"] or ["", "", ""]), vx, vy, vz, int(enviado), fila["dt_ms"]])
        if k % 1 == 0:
            fila["img"] = jpg64(cv2.resize(vis, (320, 240)))
            if ncc_img is not None: fila["ncc_img"] = jpg64(ncc_img); fila["kernel_img"] = jpg64(kern, 80)
            pub.enviar(fila)
        if k % 10 == 0: print(f"t={t:6.1f} {FASES[dec.fase]:16s} NCC={d['score']:.2f} h={alt_cam:.2f} cmd=({vx:+.2f},{vy:+.2f},{vz:+.2f}) {'ENVIADO' if enviado else 'dry'} {dt_ms:.0f} ms | {nota}")
        k += 1; fcsv.flush()
        time.sleep(max(0.0, 1.0/FREC_HZ - (time.time() - t_ini)))

if __name__ == "__main__":
    main()
