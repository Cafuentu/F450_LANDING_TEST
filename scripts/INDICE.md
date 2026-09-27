# Scripts del proyecto F450 / SLSD, por etapa

Cada carpeta es un punto del trabajo. Los de la etapa 05 y 06 importan el simulador de la 04 (no mover las carpetas entre sí).

| Carpeta | Qué hace | Cómo correr |
|---|---|---|
| `01_conversion_step_a_stl` | `conv.py`: convierte las piezas STEP del CAD a STL (OpenCascade). | `python conv.py` (requiere `cadquery-ocp`) |
| `02_laminas_A0` | `make_a0.py`: genera las 3 láminas A0 (cruz, estrella, cara), el resumen y el JSON de geometría. | `python make_a0.py` |
| `03_plano_drone` | `plano.py`: arma el plano A3 del drone a partir de las vistas renderizadas en Blender. | `python plano.py` |
| `04_simulador_dinamica` | `f450_sim.py`: dinámica 6-DOF, rotores, control en cascada, energía. `graficos.py`: gráficos del vuelo de prueba. | `python f450_sim.py` |
| `05_mision_cruz` | `mision_cruz.py`: patrón ±1 m en 4 rumbos con el efector (cámara) como referencia. | `python mision_cruz.py` y `python graficos_mision.py` |
| `06_percepcion_aterrizaje` | `sensores_ekf.py` (IMU, GPS, barómetro, EKF) · `camara_vision.py` (cámara sintética, kernel+FFT, ORB) · `mision_percepcion.py` (misión) · `graficos_percepcion.py` · `calibracion_camara.py` · `intrinsecos_picam_v2.json` | `python mision_percepcion.py`, luego `python graficos_percepcion.py`; `python calibracion_camara.py` |
| `blender` | `bake_vuelo.py`: pasa un CSV de vuelo a la animación de `escenarios_A0.blend`. | desde la pestaña Scripting de Blender |

Requisitos: Python 3 con numpy, opencv-python, pandas, matplotlib (y reportlab para la 02).
