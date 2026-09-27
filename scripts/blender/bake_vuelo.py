"""
bake_vuelo.py - Pasa un vuelo simulado (CSV) a la animacion de Blender. Ejecutar dentro de escenarios_A0.blend (pestana Scripting).
El CSV debe traer: t, x, y, z (centro de masa), qw, qx, qy, qz, ang0..ang3 (angulo acumulado de cada rotor, rad).
El objeto 'f450' tiene su origen en el centro de masa. Las helices se muestran 12x mas lentas para que el giro se vea a 30 fps.
"""
import bpy, csv
import numpy as np
from mathutils import Quaternion

CSV = r"C:\Users\Jose\Desktop\Proyecto Drones\F450_modelo\scripts\06_percepcion_aterrizaje\resultados\vuelo.csv"
FPS, LENTO = 30, 12.0

rows = list(csv.DictReader(open(CSV))); T = np.array([float(r["t"]) for r in rows])
nf = int(T[-1]*FPS) + 1; idx = np.clip(np.searchsorted(T, np.arange(nf)/FPS), 0, len(T)-1)
root = bpy.data.objects["f450"]; root.rotation_mode = 'QUATERNION'; root.animation_data_clear(); prev = None
for f, i in enumerate(idx):
    r = rows[i]; q = Quaternion((float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])))
    if prev is not None and q.dot(prev) < 0: q.negate()
    prev = q.copy(); root.location = (float(r["x"]), float(r["y"]), float(r["z"])); root.rotation_quaternion = q
    root.keyframe_insert("location", frame=f+1); root.keyframe_insert("rotation_quaternion", frame=f+1)
for k in range(4):
    ob = bpy.data.objects[f"rotor{k}"]; ob.rotation_mode = 'XYZ'; ob.animation_data_clear()
    for f, i in enumerate(idx):
        ob.rotation_euler.z = float(rows[i][f"ang{k}"]) / LENTO; ob.keyframe_insert("rotation_euler", index=2, frame=f+1)
for s in bpy.data.scenes:
    if s.name.startswith("Esc"): s.render.fps = FPS; s.frame_start = 1; s.frame_end = nf; s.frame_set(1)
print("animacion lista:", nf, "cuadros")
