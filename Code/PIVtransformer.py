#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convierte exportes de PIVlab a un TXT de perfiles por tiempo, leyendo la carpeta en orden.

- Selecciona puntos con x ≈ X_TARGET (tolerancia global estimada con el primer archivo si es None).
- Normaliza y globalmente: y_norm = y - y_min_global (=> 0 para el “fondo”), igualando origen con CFD.
- Acepta Vector type = 1 (válido) y, opcionalmente, 2 (interpolado).
- Colapsa x a la mediana por perfil para evitar jitter en el archivo final.
- Salida con header y línea en blanco entre perfiles.

Uso rápido: ajusta INPUT_DIR, X_TARGET y OUTPUT_PATH, y ejecuta.
"""

from pathlib import Path
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# =========================
# Configuración
# =========================
start = 0.10 #m
finish = 0.16 #m
diff = int((finish-start)*100+1) #*2 para hacerlo cada 5 mm
X_TARGET  = np.linspace(start, finish, num=diff)
TOLERANCIA_X     = None                            # None => se estima con el primer archivo
ACCEPT_INTERP    = True                            # True: acepta Vector type 2 (interpolados)
FLIP_V_SIGN      = False                           # True: invierte el signo de v al exportar


# ======================
# Utilidades
# ======================
def _read_pivlab_txt(path: Path) -> pd.DataFrame:
    """Lee un .txt PIVlab y retorna DataFrame con columnas estandarizadas y bandera is_valid."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    # localizar header
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("x [m],y [m],u [m/s],v [m/s],vector type"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Header PIVlab no encontrado en {path.name}")

    df = pd.read_csv(
        StringIO("\n".join(lines[header_idx:])),
        sep=",",
        na_values=["NaN", "nan", ""],
    )
    df.columns = [c.strip() for c in df.columns]

    # columna "Vector type"
    vt_cols = [c for c in df.columns if c.lower().startswith("vector type")]
    if not vt_cols:
        raise ValueError(f"Columna 'Vector type' no encontrada en {path.name}")
    vt_col = vt_cols[0]

    valid_types = [1, 2] if ACCEPT_INTERP else [1]
    df["is_valid"] = df[vt_col].isin(valid_types) & df["u [m/s]"].notna() & df["v [m/s]"].notna()
    return df


def _estimate_tol_x(df: pd.DataFrame) -> float:
    xs = np.sort(df["x [m]"].dropna().unique())
    if xs.size < 2:
        return 1e-6
    dx = np.median(np.diff(xs))
    return (dx / 2.0) if dx > 0 else 1e-6


# ======================
# Pipeline directo
# ======================
def piv_to_perfiles_txt(
    input_dir: Path,
    pattern: str,
    x_target: float,
    tol_x: Optional[float],
    out_path: Path,
) -> None:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay archivos '{pattern}' en {input_dir}")

    # 1) Estimar tolerancia global (si no está definida) usando el primer archivo
    df0 = _read_pivlab_txt(files[0])
    tol_global = tol_x if tol_x is not None else _estimate_tol_x(df0)
    if tol_global <= 0:
        tol_global = 1e-6

    # 2) Primer barrido: obtener y_min_global con la selección aplicada
    ymins = []
    for p in files:
        try:
            df = _read_pivlab_txt(p)
            sel = df.loc[
                (np.abs(df["x [m]"] - x_target) <= tol_global) & df["is_valid"],
                ["y [m]"]
            ].dropna()
            if not sel.empty:
                ymins.append(sel["y [m]"].min())
        except Exception:
            # si un archivo falla, lo saltamos (seguimos con los demás)
            continue

    if not ymins:
        raise RuntimeError("No se encontraron puntos válidos para estimar y_min_global. Revisa X_TARGET/tolerancia.")
    y_min_global = float(np.min(ymins))

    # 3) Segundo barrido: escribir el TXT final (orden por nombre de archivo)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("x[m], y[m], u[m/s], v[m/s]\n")

        for p in files:
            try:
                df = _read_pivlab_txt(p)
                sel = df.loc[
                    (np.abs(df["x [m]"] - x_target) <= tol_global) & df["is_valid"],
                    ["x [m]", "y [m]", "u [m/s]", "v [m/s]"]
                ].dropna().sort_values("y [m]").reset_index(drop=True)

                if not sel.empty:
                    # x constante por perfil (mediana para evitar jitter numérico)
                    x_ref = float(np.median(sel["x [m]"].to_numpy()))
                    y_vals = sel["y [m]"].to_numpy() - y_min_global  # normalización global
                    u_vals = sel["u [m/s]"].to_numpy()
                    v_vals = sel["v [m/s]"].to_numpy()
                    if FLIP_V_SIGN:
                        v_vals = -v_vals

                    for yv, uv, vv in zip(y_vals, u_vals, v_vals):
                        f.write(f"{x_ref:.9g}, {yv:.9g}, {uv:.9g}, {vv:.9g}\n")

                # separador entre perfiles (línea en blanco)
                f.write("\n")
                print(f"[OK] {p.name}: {len(sel)} filas (tol_x={tol_global:.2e})")

            except Exception as e:
                print(f"[WARN] {p.name}: {e}")
                f.write("\n")  # mantener bloque vacío

    print(f"[DONE] Archivo generado: {out_path}")


def conversion_factor(path, x_global_target, x_global_anchor):
    path = path / f"1_PIVlab_0001.txt"
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()          # line 1
        frame = f.readline()      # line 2 contains conversion factors

    m = re.search(r"conversion factor xy\s*\(px\s*->\s*m\)\s*:\s*([0-9.eE+-]+)", frame, flags=re.I)
    if not m:
        raise ValueError("Could not find 'conversion factor xy (px -> m): ...' in line 2.")
    px_to_m = float(m.group(1))   # e.g., 0.00012843 m/px

    # --- 2) Load data (skip preamble), compute speed, mask invalid vectors ---
    df = pd.read_csv(path, skiprows=2)
    df["speed"] = np.sqrt(df["u [m/s]"]**2 + df["v [m/s]"]**2)
    df.loc[(df["Vector type [-]"] == 0) | df["u [m/s]"].isna() | df["v [m/s]"].isna(), "speed"] = np.nan

    # --- 3) Pivot to a regular grid ---
    ys = np.sort(df["y [m]"].unique())
    xs = np.sort(df["x [m]"].unique())
    grid = df.pivot(index="y [m]", columns="x [m]", values="speed").reindex(index=ys, columns=xs).values

    # --- 4) Find the stair (first valid→NaN transition per row, using NaN-cell CENTERS) ---
    boundary_idx = []
    for i in range(grid.shape[0]):
        row = grid[i]
        mnan = np.isnan(row)
        trans = np.where((~mnan[:-1]) & (mnan[1:]))[0] + 1  # j where prev valid, current NaN
        boundary_idx.append(trans[0] if trans.size else np.nan)

    boundary_idx = np.array(boundary_idx, dtype=float)
    valid_rows = ~np.isnan(boundary_idx)
    x_boundary = xs[boundary_idx[valid_rows].astype(int)]   # local x (m) from your TXT
    y_boundary = ys[valid_rows]

    # Detect main stair (largest jump) and take a robust local median
    dx = np.diff(x_boundary)
    if dx.size:
        j = int(np.argmax(np.abs(dx)))
        lo, hi = max(0, j-2), min(len(x_boundary), j+3)
        x_local_anchor = float(np.median(x_boundary[lo:hi]))   # local x (m) of the anchor (stair)
    else:
        # Fallback: single boundary value
        x_local_anchor = float(np.median(x_boundary)) if x_boundary.size else np.nan

    if np.isnan(x_local_anchor):
        raise RuntimeError("Could not determine the local stair position (no valid→NaN transitions found).")

    # --- 5) Convert target global x to local x using the conversion factor ---
    # Pixel delta from anchor in global coordinates:
    delta_px = (x_global_target - x_global_anchor) / px_to_m
    # Local x in meters (same scale), starting from the local anchor:
    x_local_target = x_local_anchor + delta_px * px_to_m  # equals x_local_anchor + (x_global_target - x_global_anchor)


    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    plt.figure(figsize=(6, 5), dpi=140)
    pc = plt.pcolormesh(xs, ys, grid, shading="nearest", cmap=cmap)
    cbar = plt.colorbar(pc)
    cbar.set_label("|v| [m/s]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Velocity magnitude heatmap (white = missing)")
    plt.gca().invert_yaxis()

    # Draw the anchor for reference (comment out if not needed)
    plt.axvline(x_local_anchor, color="gray", lw=1.5, ls="--", label=fr"anchor: $x_g={x_global_anchor:.3f}$ m")

    # Draw the requested global-x line at its local position
    plt.axvline(x_local_target, color="red", lw=2,
                label=fr"$x_g={x_global_target:.3f}$ m  →  $x_\mathrm{{local}}={x_local_target:.4f}$ m")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/f"x_anchored={x_global_target:.2f}.png")



    return x_local_target

PATTERN = "*.txt"
x_anchor = 0.10 


for n in range(1, 4):
        INPUT_DIR        = Path(f"PIV_Solution/PIV_{n}")
        OUTPUT_DIR       = Path(f"PIV_Profiles/PIV_{n}")
        for x_trgt in X_TARGET:
            x_local = conversion_factor(INPUT_DIR, x_trgt, x_anchor)
            OUTPUT_PATH      = OUTPUT_DIR / f"perfiles_x={x_trgt:.2f}.txt"
            if __name__ == "__main__":
                piv_to_perfiles_txt(
                INPUT_DIR,
                PATTERN,
                x_local,
                TOLERANCIA_X,
                OUTPUT_PATH,
                )

