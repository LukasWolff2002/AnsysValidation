# -*- coding: utf-8 -*-
"""
Promedio temporal de velocidades para cada malla (Size3/4/5)
y extracción de cortes en x usando una banda de tolerancia.
Además genera:
  - esquema tipo "mapa de la L" con los cortes marcados, y
  - archivos 'SizeN_perfiles_x=<...>.csv' con TODOS los perfiles
    (todos los timesteps) para cada corte y cada malla.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN EDITABLE
# =========================

# Raíz donde tienes las soluciones exportadas de Fluent
BASE = Path("CFD_Solution")
MESH_NAME = "HexSweep"          # CFD_Solution/<MESH_NAME>/SizeN/...
SUBDIR = "CarbopolSolution"     # subcarpeta con los ASCII

# Mallas a procesar: Size3, Size4, Size5
SIZES = (3, 4, 5)

# Prefijo de archivos: ej. "HexSweep3-0001"
PREFIX_FMT = "{mesh}{size}-"

# Cortes en x donde quieres perfiles (en metros)
# Usa aquí los mismos valores que estás usando en el análisis
X_CUTS = [0.13, 0.17, 0.25, 0.35, 0.43]

# Carpeta de salida
OUT = Path("out_timeavg_cuts")
OUT.mkdir(parents=True, exist_ok=True)

# Cuántos puntos máximo usar en el quiver (para no morir con millones de flechas)
MAX_QUIVER_POINTS = 4000

# =========================
# MAPEOS DE COLUMNAS (Fluent)
# =========================

CANON = {
    "nodenumber": "id",
    "x-coordinate": "x",
    "y-coordinate": "y",
    "z-coordinate": "z",
    "phase-carbopol-x-velocity": "ux",
    "phase-carbopol-y-velocity": "uy",
    "phase-carbopol-z-velocity": "uz",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas de Fluent a algo manejable.
    Asumimos que trabajas en el plano x–z y que necesitas ux y uz.
    """
    cols = [c.strip().lower() for c in df.columns]
    new = []
    for c in cols:
        if c in CANON:
            new.append(CANON[c])
        else:
            new.append(c.replace(" ", "_"))
    df.columns = new

    need = {"x", "z", "ux", "uz"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Faltan columnas requeridas en archivo Fluent: {miss}")
    return df

# =========================
# UTILIDADES DE LECTURA
# =========================

def list_step_files(dir_path: Path, prefix: str):
    """
    Lista todos los archivos de tiempo para una malla dada,
    ordenados por el número final (HexSweep3-0001, 0002, etc.).
    """
    cands = [p for p in dir_path.iterdir()
             if p.is_file() and p.name.startswith(prefix)]
    if not cands:
        raise FileNotFoundError(f"No hay archivos con prefijo '{prefix}' en {dir_path}")

    def _num(p: Path):
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", p.name)
        return int(m.group(1)) if m else 10**12

    cands.sort(key=_num)
    return cands

def time_averaged_field(mesh_size: int) -> pd.DataFrame:
    """
    Lee TODOS los timesteps para Size<mesh_size> y devuelve
    un DataFrame con el promedio temporal en cada (x,z):
        columnas: x, z, ux, uz
    """
    folder = BASE / MESH_NAME / f"Size{mesh_size}" / SUBDIR
    prefix = PREFIX_FMT.format(mesh=MESH_NAME, size=mesh_size)
    files = list_step_files(folder, prefix)

    chunks = []
    for fp in files:
        df = pd.read_csv(fp, header=0, skipinitialspace=True)
        df = standardize_columns(df)
        # nos quedamos solo con lo que nos interesa
        chunks.append(df[["x", "z", "ux", "uz"]].dropna())

    big = pd.concat(chunks, axis=0, ignore_index=True)

    # promedio temporal en cada punto espacial (x,z)
    grouped = big.groupby(["x", "z"], as_index=False).mean()
    return grouped

# =========================
# TOLERANCIA EN X PARA CORTES
# =========================

def auto_tol_x(x_values: np.ndarray, frac: float = 0.005, floor: float = 1e-12) -> float:
    """
    Estima una tolerancia 'razonable' en x.
    Combina:
      - una fracción del rango total
      - media del delta mínimo entre puntos consecutivos
    para evitar cortes con un solo punto.
    """
    x = np.sort(np.unique(x_values))
    if x.size < 2:
        return floor

    dx = np.diff(x)
    dx_pos = dx[dx > 0]
    if dx_pos.size == 0:
        dx_min = floor
    else:
        dx_min = float(dx_pos.min())

    span = float(x.max() - x.min())
    tol_span = frac * max(span, floor)

    tol = max(0.5 * dx_min, tol_span, floor)
    return tol

# =========================
# CORTES Y FIGURAS POR MALLA (PROMEDIO TEMPORAL)
# =========================

def extract_cuts_and_plot_for_mesh(mesh_size: int, x_cuts):
    """
    Para una malla (SizeN):
      1) calcula el promedio temporal (x,z, ux_mean, uz_mean),
      2) calcula tolerancia en x,
      3) para cada x_target:
           - elige x_ref cercano,
           - extrae banda |x - x_ref| <= tol_x,
           - guarda CSV con perfil (x,z,u_mean,v_mean),
      4) genera figura con quiver de (ux,uz) promediado + líneas verticales en cada x_ref.
    """
    print(f"\n=== Procesando malla Size{mesh_size} (promedio temporal + cortes) ===")
    df_avg = time_averaged_field(mesh_size)

    if df_avg.empty:
        print("  [ADVERTENCIA] Campo promedio vacío para esta malla.")
        return

    x_vals = df_avg["x"].values
    z_vals = df_avg["z"].values

    x_unique = np.sort(np.unique(x_vals))
    tol_x = auto_tol_x(x_vals, frac=0.005, floor=1e-12)
    print(f"  Tolerancia en x (tol_x) ~ {tol_x:.3e}")

    cortes_info = []  # para acumular (x_target, x_ref)

    for x_target in x_cuts:
        # x_ref = valor real de la malla más cercano a x_target
        j = int(np.argmin(np.abs(x_unique - x_target)))
        x_ref = float(x_unique[j])

        sel = np.abs(df_avg["x"].values - x_ref) <= tol_x
        df_cut = df_avg.loc[sel].copy()

        if df_cut.empty:
            print(f"  [ADVERTENCIA] Corte en x_target={x_target:.6f} -> x_ref={x_ref:.6f} no tiene puntos dentro de la banda.")
            continue

        df_cut.sort_values("z", inplace=True)

        df_out = pd.DataFrame({
            "x": df_cut["x"].values,
            "z": df_cut["z"].values,
            "u_mean": df_cut["ux"].values,
            "v_mean": df_cut["uz"].values,
        })

        # archivo de perfil promedio para ESTA malla y ESTE corte
        out_name = OUT / f"Size{mesh_size}_cut_x_{x_target}.csv"
        df_out.to_csv(out_name, index=False)

        cortes_info.append((x_target, x_ref))
        print(f"  Corte en x_target={x_target:.6f} -> x_ref={x_ref:.6f} | puntos={len(df_out)} | archivo={out_name.name}")

    # ===== Figura tipo "mapa de la L" con cortes marcados =====
    # Submuestreo para quiver si hay demasiados puntos
    n_points = len(df_avg)
    if n_points > MAX_QUIVER_POINTS:
        df_plot = df_avg.sample(n=MAX_QUIVER_POINTS, random_state=0)
        print(f"  Quiver: usando submuestreo de {MAX_QUIVER_POINTS} de {n_points} puntos.")
    else:
        df_plot = df_avg

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.quiver(
        df_plot["x"], df_plot["z"],
        df_plot["ux"], df_plot["uz"],
        angles="xy", scale_units="xy", scale=5, width=0.002, alpha=0.8
    )

    # Rango de z para las líneas de corte
    zmin = float(z_vals.min())
    zmax = float(z_vals.max())

    for i, (x_target, x_ref) in enumerate(cortes_info, start=1):
        ax.axvline(x_ref, color="red", linestyle="--", linewidth=1.5)
        ax.text(
            x_ref, zmax + 0.02 * (zmax - zmin),
            f"Corte {i}\nx={x_ref:.3f}",
            rotation=90,
            va="bottom", ha="center", fontsize=8, color="red"
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Campo promediado en el tiempo y cortes en x (Size{mesh_size})")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig_name = OUT / f"Size{mesh_size}_timeavg_quiver_cortes.png"
    plt.savefig(fig_name, dpi=200)
    plt.close(fig)
    print(f"  Figura de cortes guardada como: {fig_name.name}")

# =========================
# PERFILES TIPO CFDtransformer (TODOS LOS TIMESTEPS)
# =========================

def extract_all_profiles_for_mesh(mesh_size: int, x_cuts):
    """
    Para una malla (SizeN) y cada x_target:
      - usa una banda en x alrededor de x_ref (más cercano a x_target),
      - recorre TODOS los archivos de tiempo,
      - para cada archivo extrae el perfil (x,z,ux,uz) en la banda,
      - concatena todo en un único CSV:
            SizeN_perfiles_x=<x_target>.csv
        con columnas: step, x, z, ux, uz
    Es análogo a los 'perfiles_x=...' del CFDtransformer,
    pero en formato CSV y con columna 'step'.
    """
    print(f"\n=== Procesando malla Size{mesh_size} (perfiles tipo CFDtransformer) ===")
    folder = BASE / MESH_NAME / f"Size{mesh_size}" / SUBDIR
    prefix = PREFIX_FMT.format(mesh=MESH_NAME, size=mesh_size)
    files = list_step_files(folder, prefix)

    # Usamos el primer archivo para definir tolerancia y x_unique
    df0 = pd.read_csv(files[0], header=0, skipinitialspace=True)
    df0 = standardize_columns(df0)
    x_vals0 = df0["x"].values
    x_unique0 = np.sort(np.unique(x_vals0))
    tol_x = auto_tol_x(x_vals0, frac=0.005, floor=1e-12)
    print(f"  Tolerancia en x (tol_x) para perfiles ~ {tol_x:.3e}")

    for x_target in x_cuts:
        # x_ref más cercano en la malla
        j = int(np.argmin(np.abs(x_unique0 - x_target)))
        x_ref = float(x_unique0[j])

        all_rows = []

        for step_idx, fp in enumerate(files, start=1):
            df = pd.read_csv(fp, header=0, skipinitialspace=True)
            df = standardize_columns(df)
            sel = np.abs(df["x"].values - x_ref) <= tol_x
            df_cut = df.loc[sel, ["x", "z", "ux", "uz"]].dropna().copy()
            if df_cut.empty:
                continue
            df_cut.sort_values("z", inplace=True)
            df_cut["step"] = step_idx
            all_rows.append(df_cut)

        if not all_rows:
            print(f"  [perfiles] No se obtuvieron perfiles para x_target={x_target:.6f} (x_ref={x_ref:.6f}).")
            continue

        df_all = pd.concat(all_rows, axis=0, ignore_index=True)
        # reordenamos columnas para que 'step' quede primero
        df_all = df_all[["step", "x", "z", "ux", "uz"]]

        out_name = OUT / f"Size{mesh_size}_perfiles_x={x_target}.csv"
        df_all.to_csv(out_name, index=False)
        print(f"  [perfiles] x_target={x_target:.6f} -> x_ref={x_ref:.6f} | filas={len(df_all)} -> {out_name.name}")

# =========================
# MAIN
# =========================

def main():
    for sz in SIZES:
        # 1) Perfiles promediados en el tiempo + figura de la L con cortes
        extract_cuts_and_plot_for_mesh(sz, X_CUTS)
        # 2) Perfiles "brutos" de todos los timesteps tipo CFDtransformer
        extract_all_profiles_for_mesh(sz, X_CUTS)

if __name__ == "__main__":
    main()
