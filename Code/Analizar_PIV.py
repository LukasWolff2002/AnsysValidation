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


# =========================
# Configuración
# =========================
INPUT_DIR        = Path("Results/PIV_1")       # carpeta con .txt de PIVlab
PATTERN          = "*.txt"                         # patrón de archivos
X_TARGET         = 0.11                            # x objetivo [m]
TOLERANCIA_X     = None                            # None => se estima con el primer archivo
ACCEPT_INTERP    = True                            # True: acepta Vector type 2 (interpolados)
FLIP_V_SIGN      = False                           # True: invierte el signo de v al exportar
OUTPUT_DIR       = Path("PERFILES_PIV")            # carpeta de salida
OUTPUT_PATH      = OUTPUT_DIR / f"perfiles_x={X_TARGET:.6g}.txt"  # archivo final


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


# ======================
# Main
# ======================
if __name__ == "__main__":
    piv_to_perfiles_txt(
        INPUT_DIR,
        PATTERN,
        X_TARGET,
        TOLERANCIA_X,
        OUTPUT_PATH,
    )
