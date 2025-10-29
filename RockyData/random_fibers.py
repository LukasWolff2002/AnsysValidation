#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fibras 3D sin solapamientos en la UNIÓN de dos zonas conectadas.

CSV final (Rocky): rocky_id,x,y,z,nx,ny,nz,angle
- (nx,ny,nz) = eje unitario de rotación
- angle (rad) = ángulo de rotación tal que R(n,angle) * ŷ = u
- ŷ = (0,1,0) es la orientación inicial de referencia
"""

import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Sequence
import numpy as np
from pathlib import Path

# ===================== ZONAS ================================================
# Zona A (tu dominio original)
XA_min, XA_max = 0.0, 0.085
YA_min, YA_max = -0.031, 0.031
ZA_min, ZA_max = 0.00, 0.2

# Zona B (conectada a A; ajusta a tu caso)
XB_min, XB_max = 0.085, 0.135
YB_min, YB_max = -0.031, 0.031
ZB_min, ZB_max = 0.00, 0.02

# ===================== FIBRAS / PACKING =====================================
margin = 0.002
N = 1600
fiber_diameter = 0.00016
fiber_length   = 0.013

out_csv     = "RockyData/fibers_random_packing.csv"
preview_png = "RockyData/fibers_preview.png"

rng_seed = 1234
max_attempts_per_fiber = 2000
touch_tolerance = 1e-8

SHOW_PLOT = True
SAVE_PLOT = True
# Si True, el margen se aplica a la SUPERFICIE (margin + radio); si False, al eje:
MARGIN_APLICA_A_SUPERFICIE = False

# ============================ ESTRUCTURAS ====================================

@dataclass
class Box:
    xmin: float; xmax: float
    ymin: float; ymax: float
    zmin: float; zmax: float
    def size(self): return np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin])

@dataclass
class Fiber:
    center: np.ndarray      # (3,)
    u: np.ndarray           # (3,) unitario (dirección fibra)
    length: float
    diameter: float
    @property
    def radius(self): return 0.5*self.diameter
    def endpoints(self): 
        h = 0.5*self.length
        return self.center - h*self.u, self.center + h*self.u

# ============================ UTILIDADES =====================================

def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Muestra uniforme en S^2."""
    mu = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0*np.pi)
    sin_t = math.sqrt(max(0.0, 1.0 - mu*mu))
    return np.array([sin_t*math.cos(phi), sin_t*math.sin(phi), mu], dtype=float)

def segment_segment_distance_sq(p1,q1,p2,q2) -> float:
    u = q1-p1; v=q2-p2; w0=p1-p2
    a,b,c = np.dot(u,u), np.dot(u,v), np.dot(v,v)
    d,e = np.dot(u,w0), np.dot(v,w0)
    denom = a*c - b*b
    if denom < 1e-18:
        sc, tc = 0.0, (e/c if c>1e-18 else 0.0)
    else:
        sN = (b*e - c*d); tN = (a*e - b*d)
        sD = tD = denom
        if sN < 0.0: sN=0.0; tN=e; tD=c
        elif sN > sD: sN=sD; tN=e+b; tD=c
        if tN < 0.0:
            tN=0.0
            if -d < 0.0: sN=0.0
            elif -d > a: sN=sD
            else: sN=-d; sD=a
        elif tN > tD:
            tN=tD
            if (-d+b) < 0.0: sN=0.0
            elif (-d+b) > a: sN=sD
            else: sN=(-d+b); sD=a
        sc = 0.0 if abs(sN)<1e-18 else sN/sD
        tc = 0.0 if abs(tN)<1e-18 else tN/tD
    dP = w0 + sc*u - tc*v
    return float(np.dot(dP,dP))

def point_in_box(p, box: Box, margin_eff: float) -> bool:
    return (box.xmin+margin_eff - touch_tolerance <= p[0] <= box.xmax-margin_eff + touch_tolerance and
            box.ymin+margin_eff - touch_tolerance <= p[1] <= box.ymax-margin_eff + touch_tolerance and
            box.zmin+margin_eff - touch_tolerance <= p[2] <= box.zmax-margin_eff + touch_tolerance)

def point_in_union(p, boxes: Sequence[Box], margin_eff: float) -> bool:
    return any(point_in_box(p,b,margin_eff) for b in boxes)

def fits_in_union(f: Fiber, boxes: Sequence[Box], margin: float, aplica_superficie: bool) -> bool:
    m_eff = margin + (f.radius if aplica_superficie else 0.0)
    p1,p2 = f.endpoints()
    return point_in_union(p1, boxes, m_eff) and point_in_union(p2, boxes, m_eff)

def overlap(f1: Fiber, f2: Fiber) -> bool:
    p1,p2 = f1.endpoints(); q1,q2 = f2.endpoints()
    return math.sqrt(segment_segment_distance_sq(p1,p2,q1,q2)) + touch_tolerance < (f1.radius+f2.radius)

def union_bounds(boxes: Sequence[Box]) -> Tuple[float,float,float,float,float,float]:
    return (min(b.xmin for b in boxes), max(b.xmax for b in boxes),
            min(b.ymin for b in boxes), max(b.ymax for b in boxes),
            min(b.zmin for b in boxes), max(b.zmax for b in boxes))

# ---- Eje y ángulo a partir de u (referencia ŷ=(0,1,0)) ---------------------

def axis_angle_from_y_to_u(u: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    """
    Devuelve (n, angle) tal que R(n,angle) * ŷ = u.
    n = normalize(ŷ × u), angle = arccos( clamp(ŷ·u, -1, 1) ).
    Casos degenerados:
      * u ≈ +ŷ: angle = 0, n = (1,0,0)
      * u ≈ -ŷ: angle = π, n = (1,0,0)
    """
    yhat = np.array([0.0, 1.0, 0.0], dtype=float)
    u = u / max(np.linalg.norm(u), eps)
    dot = float(np.clip(np.dot(yhat, u), -1.0, 1.0))
    angle = float(math.acos(dot))
    cross = np.cross(yhat, u)
    n_norm = float(np.linalg.norm(cross))
    if n_norm < eps:
        n = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n = cross / n_norm
    return n, angle

# ---- Rodrigues para verificación opcional -----------------------------------

def rodrigues_rotate(v: np.ndarray, n: np.ndarray, angle: float) -> np.ndarray:
    """Rota v alrededor de n (unitario) por 'angle' rad (fórmula de Rodrigues)."""
    n = n / max(np.linalg.norm(n), 1e-12)
    v_par = np.dot(v, n) * n
    v_perp = v - v_par
    v_perp_rot = v_perp * math.cos(angle) + np.cross(n, v) * math.sin(angle)
    return v_par + v_perp_rot

# ============================ GENERACIÓN =====================================

def sample_center_in_union(rng: np.random.Generator, boxes: Sequence[Box], margin_eff: float) -> np.ndarray:
    vols = []
    for b in boxes:
        sx = max(0.0, (b.xmax-b.xmin) - 2*margin_eff)
        sy = max(0.0, (b.ymax-b.ymin) - 2*margin_eff)
        sz = max(0.0, (b.zmax-b.zmin) - 2*margin_eff)
        vols.append(sx*sy*sz)
    vols = np.array(vols, dtype=float)
    if vols.sum() <= 0:
        raise ValueError("El margen efectivo anuló todo el volumen útil.")
    probs = vols / vols.sum()
    idx = int(np.random.default_rng().choice(len(boxes), p=probs))
    b = boxes[idx]
    cx = rng.uniform(b.xmin+margin_eff, b.xmax-margin_eff)
    cy = rng.uniform(b.ymin+margin_eff, b.ymax-margin_eff)
    cz = rng.uniform(b.zmin+margin_eff, b.zmax-margin_eff)
    return np.array([cx, cy, cz], dtype=float)

def generate_fibers_union(N: int, boxes: Sequence[Box], margin: float,
                          L: float, D: float, aplica_superficie: bool,
                          rng_seed=None) -> List[Fiber]:
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()
    fibers: List[Fiber] = []

    extra = 0.5*D if aplica_superficie else 0.0
    feasible = False
    for b in boxes:
        dom = b.size() - 2*(margin + extra)
        if np.all(dom > 0.0) and L <= np.linalg.norm(dom):
            feasible = True; break
    if not feasible:
        raise ValueError("Con margen efectivo y longitudes actuales, no hay caja con espacio suficiente.")

    for i in range(N):
        placed = False
        for _ in range(max_attempts_per_fiber):
            u = random_unit_vector(rng)
            center = sample_center_in_union(rng, boxes, margin + extra)
            f = Fiber(center=center, u=u/np.linalg.norm(u), length=L, diameter=D)
            if not fits_in_union(f, boxes, margin, aplica_superficie): continue
            if any(overlap(f, g) for g in fibers): continue
            fibers.append(f); placed=True; break
        if not placed:
            raise RuntimeError(f"No se pudo ubicar la fibra {i+1} (ajusta N/length/diameter o margen).")
    print(f"[OK] Colocadas {N} fibras.")
    return fibers

# ============================ EXPORTACIÓN CSV ================================

def save_to_rocky_csv(fibers: List[Fiber], path: str, verify=False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    yhat = np.array([0.0, 1.0, 0.0], dtype=float)
    with open(path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["rocky_id","x","y","z","nx","ny","nz","angle"])
        for i,fib in enumerate(fibers,1):
            n, angle = axis_angle_from_y_to_u(fib.u)  # n unitario, angle rad
            w.writerow([i, fib.center[0], fib.center[1], fib.center[2],
                        n[0], n[1], n[2], angle])
            # Verificación opcional de que R(n,angle)*ŷ ≈ u
            if verify:
                u_rec = rodrigues_rotate(yhat, n, angle)
                if np.linalg.norm(u_rec - fib.u) > 1e-6:
                    print(f"[WARN] Fib {i}: reconstrucción difiere (‖Δ‖={np.linalg.norm(u_rec - fib.u):.2e})")
    print(f"[SAVE] {path}")

# ============================ GRAFICADO ======================================

def plot_boxes(ax, boxes: Sequence[Box], alpha=0.6):
    for b in boxes:
        xs=[b.xmin,b.xmax]; ys=[b.ymin,b.ymax]; zs=[b.zmin,b.zmax]
        edges = [
            ((xs[0],ys[0],zs[0]),(xs[1],ys[0],zs[0])), ((xs[0],ys[1],zs[0]),(xs[1],ys[1],zs[0])),
            ((xs[0],ys[0],zs[1]),(xs[1],ys[0],zs[1])), ((xs[0],ys[1],zs[1]),(xs[1],ys[1],zs[1])),
            ((xs[0],ys[0],zs[0]),(xs[0],ys[1],zs[0])), ((xs[1],ys[0],zs[0]),(xs[1],ys[1],zs[0])),
            ((xs[0],ys[0],zs[1]),(xs[0],ys[1],zs[1])), ((xs[1],ys[0],zs[1]),(xs[1],ys[1],zs[1])),
            ((xs[0],ys[0],zs[0]),(xs[0],ys[0],zs[1])), ((xs[1],ys[0],zs[0]),(xs[1],ys[0],zs[1])),
            ((xs[0],ys[1],zs[0]),(xs[0],ys[1],zs[1])), ((xs[1],ys[1],zs[0]),(xs[1],ys[1],zs[1])),
        ]
        for (x0,y0,z0),(x1,y1,z1) in edges:
            ax.plot([x0,x1],[y0,y1],[z0,z1], linewidth=1, alpha=alpha)

def plot_fibers(fibers: List[Fiber], boxes: Sequence[Box], save=None, show=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    xmin,xmax,ymin,ymax,zmin,zmax = union_bounds(boxes)
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    plot_boxes(ax, boxes, alpha=0.5)
    for fib in fibers:
        p1,p2=fib.endpoints()
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], lw=1.2)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(f"Fibras N={len(fibers)}  |  L={fiber_length*1e3:.2f} mm, D={fiber_diameter*1e3:.2f} mm")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=200, bbox_inches="tight")
        print(f"[PLOT] {save}")
    if show: plt.show()
    else: plt.close(fig)

# ============================ MAIN ===========================================
if __name__ == "__main__":
    boxes = [
        Box(XA_min,XA_max,YA_min,YA_max,ZA_min,ZA_max),
        Box(XB_min,XB_max,YB_min,YB_max,ZB_min,ZB_max),
    ]
    fibers = generate_fibers_union(N, boxes, margin, fiber_length, fiber_diameter,
                                   MARGIN_APLICA_A_SUPERFICIE, rng_seed)
    save_to_rocky_csv(fibers, out_csv, verify=True)  # pon verify=False si no quieres el check
    if SHOW_PLOT or SAVE_PLOT:
        plot_fibers(fibers, boxes, save=(preview_png if SAVE_PLOT else None), show=SHOW_PLOT)
