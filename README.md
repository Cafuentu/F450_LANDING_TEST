# F450 — Aterrizaje autónomo sobre objetivo visual (prueba inicial)

Espacio de trabajo del cuadricóptero **F450** del proyecto SLSD (*Safe Landing Site Detection*): modelo 3D completo en Blender, láminas de prueba, simulación y scripts para la prueba real con Raspberry Pi + Pi Camera v2 + Pixhawk.

## Descargar

```bash
git clone https://github.com/Cafuentu/F450_LANDING_TEST.git
cd F450_LANDING_TEST
```
Para traer cambios nuevos más adelante: `git pull`.

## Contenido

| Carpeta | Qué hay |
|---|---|
| `modelo/` | **`f450_real.blend`**: F450 armado con el CAD real (placas, brazos, motores 2216, hélices 1045), tren de aterrizaje `f450_leg`, Pixhawk, GPS con mástil, ESC ×4, batería 3S 5200 mAh, telemetría, receptor, power module, UBEC, Raspberry Pi 4 y cámara nadir. Cada pieza tiene su masa como propiedad `mass_g`. `f450_real.usda`: el mismo modelo en USD (Isaac Sim / Pegasus). `f450.blend`: versión simplificada (útil para colisiones). |
| `stl/` | Piezas sueltas en STL (convertidas desde el STEP del CAD) y la pata del tren. |
| `planos/` | `plano_F450.pdf`: plano A3 con cotas, componentes y masas. |
| `datos/` | Masas, posiciones y dimensiones de cada componente; masa total, centro de gravedad e inercias; geometría de las láminas. |
| `laminas_A0/` | Láminas A0 de prueba (cruz, estrella, cara) a escala 1:1 para imprimir, resumen y texturas para Blender. |
| `scripts/` | Simulador 6-DOF, misiones, sensores + EKF, detector de la cruz (kernel + FFT), ORB, calibración de cámara. Ver `scripts/INDICE.md`. |
| `prueba_real_rpi/` | Agente para la Raspberry y monitor en tiempo real para el PC. Ver `prueba_real_rpi/README.md`. |

## Abrir el modelo

Blender 4.x o superior (hecho en 5.2): `File → Open → modelo/f450_real.blend`. El archivo es autocontenido.

## Marcos de referencia

- **Cuerpo:** ejes FLU (X adelante, Y izquierda, Z arriba), origen en el centro de masa para la dinámica.
- **Rotores (orden PX4 / Pegasus quad-X):** `rotor0` delantero derecho (CCW), `rotor1` trasero izquierdo (CCW), `rotor2` delantero izquierdo (CW), `rotor3` trasero derecho (CW).
- **Mundo (pruebas):** origen en el centro de la cruz de la lámina, Z arriba.
- **Efector:** cámara nadir, 71,4 mm delante y 62,8 mm bajo el centro de masa.

## Características principales (modelo)

| | |
|---|---|
| Distancia entre ejes | 450 mm |
| Masa total | 1,165 kg (batería 220 g, informada; resto de catálogo) |
| Centro de gravedad | (−3,4; 0,2; 12,4) mm desde el centro del frame |
| Inercias en el CG | Ixx 0,0129 · Iyy 0,0135 · Izz 0,0234 kg·m² |
| Tren de aterrizaje | 4 apoyos en cuadrado de 343 mm; cámara a 64 mm del suelo apoyado |
| Cámara | Pi Camera v2, 640×480, f ≈ 529,6 px (nominal, falta calibrar) |
