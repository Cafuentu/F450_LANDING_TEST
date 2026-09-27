# 06 · Prueba real con la Raspberry Pi — F450 anclado, Pi Camera v2, Pixhawk

**Estado.** Scripts listos y probados de punta a punta sin hardware (fuente simulada + MAVLink simulado): detección, fases, instrucciones y stream funcionan; lazo de ~90 ms por cuadro en el PC. Falta la prueba con el hardware real (Armando ajusta el puerto MAVLink).

Repite en el drone real el proceso validado en la simulación (experimento 05): la Raspberry captura la cámara nadir, corre el kernel + FFT, decide con la misma máquina de estados y genera las instrucciones de vuelo; el PC ve todo en tiempo real por el túnel SSH.

```
 Raspberry Pi (agente_rpi.py)                          PC (monitor_pc.py)
 ┌─────────────────────────────────────┐   túnel SSH   ┌───────────────────────────┐
 │ Pi Camera v2 ─► kernel+FFT ─► decisión │ ──(TCP 5555)──►│ 4 paneles en vivo + CSV │
 │ Pixhawk ◄── MAVLink (telemetría/setpoints)│ ◄─ órdenes ──│ teclas: i / p / l / g   │
 └─────────────────────────────────────┘                └───────────────────────────┘
```

## Archivos

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `rpi/agente_rpi.py` | Raspberry | Lazo de percepción + decisión + MAVLink + stream + log |
| `rpi/detector_cruz.py` | Raspberry | Detector kernel + FFT (mismo del experimento 05, sin dependencias de la simulación) |
| `rpi/intrinsecos_picam_v2.json` | Raspberry | Intrínsecos de la cámara (reemplazar por los calibrados con `calibracion_camara.py` del exp. 04) |
| `pc/monitor_pc.py` | PC | Ventana con cámara + detección, correlación, decisión y gráficas; CSV y video |
| `sim_local/` | ambos | Solo para la prueba sin hardware (`--camara sim --mav sim`) |

## 1. Instalación

**Raspberry Pi (Raspberry Pi OS Bookworm, Python 3):**
```bash
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
pip3 install pymavlink --break-system-packages
# copiar la carpeta rpi/ (y sim_local/ si se quiere probar sin hardware)
```
Verificar la cámara: `libcamera-hello -t 2000` (o `rpicam-hello`). Verificar el Pixhawk: `ls /dev/ttyACM* /dev/serial*`.

**PC (Windows/Linux):** `pip install opencv-python numpy`

## 2. Configuración que ajusta Armando (arriba de `agente_rpi.py`)

| Parámetro | Valor por defecto | Nota |
|---|---|---|
| `MAV_PUERTO` | `/dev/ttyACM0` | Pixhawk por USB. Por TELEM2: `/dev/serial0` o `/dev/ttyAMA0` (habilitar UART en `raspi-config`, quitar la consola serie) |
| `MAV_BAUD` | `115200` | USB: 115200. TELEM2: el `SERIAL2_BAUD` del Pixhawk (57600 o 921600) |
| `PUERTO_TCP` | `5555` | Puerto del stream (se tuneliza por SSH, no hace falta abrirlo) |
| `V_MAX` | `0.30 m/s` | Tope de las instrucciones de velocidad |
| `BATT_ABORTA` | `10.5 V` | Bajo esto el agente pide LAND |
| `CAM_OFFSET_B` | del modelo 3D | Posición de la cámara respecto del centro de masa (ejes cuerpo). Ajustar si la cámara va en otro lugar |

También se pueden pasar por línea de comandos: `--mav`, `--baud`, `--puerto`.

En el Pixhawk (ArduPilot) conviene tener `SERIALx_PROTOCOL = 2` (MAVLink 2) en el puerto usado y, si se va a enviar comandos, el modo GUIDED disponible. Con la telemetría SiK conectada a Mission Planner al mismo tiempo no hay conflicto: la Raspberry usa otro puerto físico.

## 3. Ejecución

**Paso 0 — prueba sin hardware (en el PC o en la RPi):**
```bash
cd rpi && python3 agente_rpi.py --camara sim --mav sim --auto --puerto 5555
# en otra terminal
cd pc && python3 monitor_pc.py
```
Debe verse el drone simulado centrarse sobre la cruz y bajar, igual que en Blender.

**Paso 1 — en la Raspberry (drone anclado, hélices puestas o no según la prueba):**
```bash
cd rpi
python3 agente_rpi.py --camara picam --mav /dev/ttyACM0 --baud 115200
```
Sin `--enviar` el agente está en **dry-run**: calcula y publica las instrucciones pero NO manda nada al Pixhawk. Es el modo para las primeras pruebas.

**Paso 2 — en el PC, túnel SSH y monitor (dos terminales):**
```bash
ssh -L 5555:localhost:5555 pi@<ip_de_la_raspberry>      # dejar abierta
python monitor_pc.py                                     # se conecta a localhost:5555
```
Si prefieres sin túnel (misma red wifi): `python monitor_pc.py --host <ip_de_la_raspberry>`.

**Paso 3 — secuencia:** en la ventana del monitor, tecla `i` inicia la secuencia (buscar cruz → centrado → 3 m → 1 m → descenso final → en tierra); `p` la detiene y vuelve a espera; `l` manda LAND; `g` graba el video de la ventana; `q` sale. Con el drone anclado, la secuencia se puede observar completa en dry-run: el monitor muestra qué instrucción se enviaría en cada instante.

**Paso 4 — enviar instrucciones de verdad (solo con el drone anclado y en GUIDED):**
```bash
python3 agente_rpi.py --camara picam --mav /dev/ttyACM0 --enviar
```
El agente envía setpoints de velocidad en ejes cuerpo (`SET_POSITION_TARGET_LOCAL_NED`, frame `BODY_OFFSET_NED`) **únicamente** si el Pixhawk está en GUIDED y armado; en cualquier otro modo no envía. Límite `V_MAX` = 0,3 m/s. Si se pierde la telemetría por más de 1 s la instrucción pasa a cero; si la batería baja de `BATT_ABORTA`, pide LAND.

## 4. Qué se guarda

- Raspberry: `rpi/logs/vuelo_<fecha>.csv` (todo el lazo a 10 Hz: telemetría, detección, posición, instrucciones, tiempo de cómputo) y `vuelo_<fecha>.mp4` (cámara con la detección).
- PC: `pc/logs_pc/monitor_<fecha>.csv` (lo recibido) y, con `g`, `monitor_<fecha>.mp4` (la ventana de 4 paneles).

## 5. Qué mirar en la primera prueba

- **NCC** con la lámina a 1–4 m bajo la cámara: en simulación da 0,95–0,98; en real esperar 0,7–0,9 por iluminación y foco. Si queda bajo 0,55 sostenido, revisar foco de la cámara, brillo (papel mate) y que la cruz mida 250 mm.
- **h (kernel) vs altura del Pixhawk**: deben coincidir dentro de unos 10 cm. Si difieren mucho, los intrínsecos no son los correctos (calibrar).
- **Tiempo de lazo** (`dt_ms`): en una Pi 4 a 640×480 se esperan 80–150 ms por cuadro (10 Hz justo). Si sube, bajar `n_esc` en `detectar_auto` o reducir a 5 Hz (`FREC_HZ`).
- **Instrucciones**: con el drone quieto sobre la cruz deben ser ≈ 0; al desplazar la lámina 20 cm a la derecha, `vy` debe apuntar a la derecha (negativo) y volver a 0 al centrarla.

## Pendiente para la segunda etapa (integridad del drone)
Geofence por software, tope de altura, hover automático ante NCC bajo sostenido, confirmación humana antes del descenso final, y fusión con la altura del Pixhawk.
