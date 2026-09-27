"""
mision_percepcion.py - Aterrizaje sobre la cruz guiado por vision (etapa 06)

Marcos: mundo = centro de la cruz; cuerpo = centro de masa; efector = camara nadir.
El control vuela SOLO con estados estimados (EKF con IMU+GPS+barometro); la verdad se usa para simular y comparar.

Fases:
  armado -> despegue a 4 m -> derecha 1 m -> KERNEL ON: observar a 4 m -> centrado visual a 4 m
  -> descenso a 3 m -> observar a 3 m -> descenso a 1 m -> observar a 1 m -> descenso final con ORB -> en tierra

Vision (10 Hz): el detector (kernel + FFT) entrega la posicion de la camara respecto de la cruz. Sin fusion bayesiana:
la diferencia (vision - EKF) se filtra con un pasa-bajos y se usa como correccion horizontal de la navegacion.
Bajo 1 m, ORB mide el cambio de escala entre cuadros -> altura relativa y velocidad de descenso.
"""
import os, sys, csv, pickle
import numpy as np
import cv2
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "04_simulador_dinamica"))
from f450_sim import Params, Quad, Controller, q_to_R, euler_from_q
from sensores_ekf import Sensores, Estimador, EstadoNav
from camara_vision import Suelo, Camara, DetectorCruz, OdometriaORB, FX, CRUZ_LADO, CRUZ_BRAZO, W, H

R_CAM = np.array([0.0680, 0.0, -0.0504]) - Params.r_cg          # camara respecto del CG (ejes cuerpo)
H_CAM_TIERRA = Params.h_cg_ground + R_CAM[2]                    # altura de la camara con el tren apoyado (0.0643 m)
SALIDA = os.path.join(AQUI, "resultados"); os.makedirs(SALIDA, exist_ok=True)


def correr(dt=0.002, seed=7, video=True):
    P = Params; quad = Quad(P, p0=(-R_CAM[0], -R_CAM[1], None)); ctl = Controller(P); ctl.v_dn = 0.6
    sen = Sensores(seed); est = Estimador(p0=quad.p + sen.gm)     # el EKF parte donde el GPS cree que esta
    cam = Camara(Suelo()); det = DetectorCruz(); orb = OdometriaORB()
    fases = ["armado", "despegue a 4 m", "derecha 1 m", "kernel ON: observar a 4 m", "centrado visual a 4 m", "descenso a 3 m", "observar a 3 m",
             "descenso a 1 m", "observar a 1 m", "descenso final (ORB)", "en tierra"]
    k = 0; t_f0 = 0.0; t_ok = None; vision = False; off = np.zeros(2); off_ok = False; servo = False; L_prev = None
    h_kernel_1m = []; landed_t = None; w_cmd = np.zeros(4); log, vlog, etapas = [], [], {}
    obj = np.array([0.0, 0.0, 4.0]); origen_ekf = est.p[:2].copy() + (q_to_R(est.q) @ R_CAM)[:2]   # "origen" segun el EKF al partir
    vw = cv2.VideoWriter(os.path.join(SALIDA, "camara_nadir.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H)) if video else None
    i = 0; f_b = np.array([0, 0, 9.81]); w_m = np.zeros(3); last_det = None; last_orb = None; h_orb = np.nan; vz_orb = np.nan
    while True:
        t = i*dt; R_true = q_to_R(quad.q); cam_true = quad.p + R_true @ R_CAM
        # ------------------------------------------------ sensores y estimador
        if i % 2 == 0:
            f_b, w_m = sen.imu(quad, 2*dt); est.predecir(f_b, w_m, 2*dt)
        if i % 100 == 0: gps_p, gps_v = sen.gps(quad, 0.2); est.corregir_gps(gps_p, gps_v)
        if i % 25 == 0: est.corregir_baro(sen.baro(quad))
        R_est = q_to_R(est.q); cam_ekf = est.p + R_est @ R_CAM
        cam_nav = cam_ekf.copy(); cam_nav[:2] -= origen_ekf                         # EKF referido al punto de partida
        if off_ok and servo: cam_nav[:2] = cam_ekf[:2] + off                        # navegacion corregida por vision
        # ------------------------------------------------ vision a 10 Hz
        if i % 50 == 0 and k >= 1:
            img = cam.imagen(cam_true, R_true); fila = dict(t=t, fase=fases[k], modo="", score=np.nan, L=np.nan, u=np.nan, v=np.nan, valido=0,
                                                            vis_x=np.nan, vis_y=np.nan, vis_z=np.nan, n_kp=0, n_match=0, n_inl=0, escala=np.nan, h_orb=np.nan, vz_orb=np.nan)
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if vision:
                h_pred = max(cam_ekf[2], 0.15) if L_prev is None else FX*CRUZ_LADO/L_prev
                guardar = fases[k].startswith("observar") or fases[k].startswith("kernel ON")
                d = det.detectar(img, FX*CRUZ_LADO/h_pred, yaw_deg=0.0, ancho=0.30 if L_prev is None else 0.15, n_esc=13 if L_prev is None else 9,
                                 etapas=guardar and fases[k] not in etapas)
                if not d["valido"] and h_pred < 0.75:                                # la cruz ya no cabe: kernel del centro (cruce de brazos)
                    d = det.detectar(img, FX*CRUZ_BRAZO/h_pred, yaw_deg=0.0, ancho=0.25, n_esc=9, modo="centro")
                fila.update(score=d["score"], valido=int(d["valido"]), modo=d.get("modo", ""))
                if d["valido"]:
                    pv = DetectorCruz.a_mundo(d, R_est); L_prev = d["L"] if d["modo"] == "completa" else d["L"]*CRUZ_LADO/CRUZ_BRAZO; last_det = (t, pv)
                    fila.update(L=d["L"], u=d["u"], v=d["v"], vis_x=pv[0], vis_y=pv[1], vis_z=pv[2])
                    nuevo = pv[:2] - cam_ekf[:2]
                    if not off_ok: off = nuevo; off_ok = True
                    elif np.linalg.norm(nuevo - off) < 0.6: off += 0.2*(nuevo - off)
                    if fases[k] == "observar a 1 m": h_kernel_1m.append(pv[2])
                    cv2.drawMarker(vis_img, (int(d["u"]), int(d["v"])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                    r = int(d["L"]*(0.65 if d["modo"] == "completa" else 1.2)); cv2.rectangle(vis_img, (int(d["u"])-r, int(d["v"])-r), (int(d["u"])+r, int(d["v"])+r), (0, 200, 0), 1)
                    if "ncc" in d and t - t_f0 > 1.5: etapas[fases[k]] = dict(img=img, I=d["I"], espectro=d["espectro"], ncc=d["ncc"], kernel=d["kernel"], u=d["u"], v=d["v"], L=d["L"],
                                                                         score=d["score"], escalas=d["escalas"], puntajes=d["puntajes"], h=cam_true[2])
                else: L_prev = None if d["score"] < 0.3 else L_prev
            if fases[k] == "descenso final (ORB)":
                o = orb.paso(img, t, guardar=True); h_orb = o["h"] if o["h"] is not None else np.nan
                if o["ok"]: vz_orb = o["vz"]; last_orb = t
                fila.update(n_kp=o["n_kp"], n_match=o["n_match"], n_inl=o["n_inl"], escala=o["escala"], h_orb=h_orb, vz_orb=o["vz"])
                if o["ok"] and "a" in o:
                    for (x0, y0), (x1, y1) in zip(o["a"][::6], o["b"][::6]): cv2.line(vis_img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 160, 0), 1)
                    for nombre, hh in (("orb_0.8m", 0.8), ("orb_0.4m", 0.4), ("orb_0.2m", 0.2)):
                        if nombre not in etapas and cam_true[2] <= hh: etapas[nombre] = dict(img=img, img_ref=o["img_ref"], a=o["a"], b=o["b"], h=cam_true[2], escala=o["escala"])
            fila.update(cam_x=cam_true[0], cam_y=cam_true[1], cam_z=cam_true[2]); vlog.append(fila)
            if vw is not None:
                cv2.putText(vis_img, f"t={t:5.1f}s  h={cam_true[2]:.2f} m  {fases[k]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                vw.write(vis_img)
        # ------------------------------------------------ maquina de estados (con estados estimados)
        err = np.linalg.norm(cam_nav - obj); quieto = np.linalg.norm(est.v) < 0.20
        def llegar(tol, espera, t_max):
            nonlocal t_ok
            t_ok = (t if t_ok is None else t_ok) if (err < tol and quieto) else None
            return (t_ok is not None and t - t_ok >= espera) or (t - t_f0 > t_max)
        sig = False; vz_cmd = None
        f = fases[k]
        if f == "armado": sig = t >= 2.0
        elif f == "despegue a 4 m": obj = np.array([0, 0, 4.0]); sig = llegar(0.10, 1.0, 14)
        elif f == "derecha 1 m": obj = np.array([0, -1.0, 4.0]); sig = llegar(0.10, 1.0, 12)
        elif f == "kernel ON: observar a 4 m": vision = True; sig = t - t_f0 >= 4.0
        elif f == "centrado visual a 4 m": servo = True; obj = np.array([0, 0, 4.0]); sig = llegar(0.10, 2.0, 18)
        elif f == "descenso a 3 m": obj = np.array([0, 0, 3.0]); sig = llegar(0.08, 0.5, 12)
        elif f == "observar a 3 m": sig = t - t_f0 >= 3.0
        elif f == "descenso a 1 m": obj = np.array([0, 0, 1.0]); sig = llegar(0.08, 0.5, 14)
        elif f == "observar a 1 m":
            sig = t - t_f0 >= 3.0
            if sig: orb.iniciar(float(np.median(h_kernel_1m)) if h_kernel_1m else cam_ekf[2])      # altura inicial de ORB = la del kernel
        elif f == "descenso final (ORB)":
            obj = np.array([0, 0, 0.0]); despeje = (h_orb if np.isfinite(h_orb) else cam_ekf[2]) - H_CAM_TIERRA
            vz_cmd = -0.25 if despeje > 0.30 else -0.12
            if quad.contact and abs(est.v[2]) < 0.08 and landed_t is None: landed_t = t; sig = True
        if sig and k < len(fases)-1: k += 1; t_f0 = t; t_ok = None
        # ------------------------------------------------ control
        if f == "armado": sp = dict(mode="idle")
        elif f == "en tierra":
            sp = dict(mode="idle") if t - landed_t < 2.0 else dict(mode="off")
            if t - landed_t > 3.5: break
        else:
            cg_obj = obj - R_est @ R_CAM; cg_obj[2] = obj[2] - (R_est @ R_CAM)[2]
            sp = dict(mode="pos", p=cg_obj, yaw=0.0, vz=vz_cmd)
        nav_cg = cam_nav - R_est @ R_CAM
        if i % 2 == 0: w_cmd = ctl.update(EstadoNav(nav_cg, est.v, est.q, w_m), sp, 2*dt)
        quad.step(w_cmd, dt)
        if i % 10 == 0:
            e_t = np.degrees(euler_from_q(quad.q)); e_e = np.degrees(euler_from_q(est.q))
            log.append([t, fases[k], *cam_true, *cam_ekf, *cam_nav, *(gps_p + R_est @ R_CAM), *quad.v, *est.v, *e_t, *e_e, *f_b, *w_m, *obj,
                        *(quad.rot*60/2/np.pi), quad.P_elec, quad.E_Wh, *off, int(servo and off_ok), int(quad.contact), *quad.p, *quad.q, *quad.rot_angle])
        i += 1
        if t > 200: break
    if vw is not None: vw.release()
    cols = ["t", "fase", "cam_x", "cam_y", "cam_z", "ekf_x", "ekf_y", "ekf_z", "nav_x", "nav_y", "nav_z", "gps_x", "gps_y", "gps_z", "vx", "vy", "vz", "ekf_vx", "ekf_vy", "ekf_vz",
            "roll", "pitch", "yaw", "ekf_roll", "ekf_pitch", "ekf_yaw", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "obj_x", "obj_y", "obj_z",
            "rpm1", "rpm2", "rpm3", "rpm4", "P_elec_W", "E_Wh", "off_x", "off_y", "servo_visual", "contacto",
            "x", "y", "z", "qw", "qx", "qy", "qz", "ang0", "ang1", "ang2", "ang3"]
    return cols, log, vlog, etapas, quad


if __name__ == "__main__":
    cols, log, vlog, etapas, quad = correr()
    with open(os.path.join(SALIDA, "vuelo.csv"), "w", newline="") as f_: w = csv.writer(f_); w.writerow(cols); w.writerows(log)
    with open(os.path.join(SALIDA, "vision.csv"), "w", newline="") as f_: w = csv.DictWriter(f_, fieldnames=list(vlog[0].keys())); w.writeheader(); w.writerows(vlog)
    pickle.dump(etapas, open(os.path.join(SALIDA, "etapas.pkl"), "wb"))
    R = q_to_R(quad.q); c = quad.p + R @ R_CAM
    print(f"fin t={log[-1][0]:.1f} s; camara final (mm): x={c[0]*1000:.1f} y={c[1]*1000:.1f} z={c[2]*1000:.1f}; etapas guardadas: {list(etapas)}")
