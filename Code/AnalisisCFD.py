#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recorre todos los archivos ASCII separados por comas (sin extensión) con nombre 'carbopol_fluent-####'
dentro de CarbopolSolution/. Para cada archivo calcula el perfil u_x(z) en un x fijo (tomado del primer
archivo como el x más cercano a X_TARGET), y guarda TODOS los perfiles en un único TXT con bloques
separados por línea en blanco, con columnas:

    x[m], z[m], u[m/s], v[m/s]

Además, SOLO para el primer archivo genera:
  - Figura del perfil u_x(z)
  - Quiver (u_x, u_z) en X–Z con una línea roja en el x del perfil

Requisitos: pandas, numpy, matplotlib
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===================== Parámetros =====================
DATA_DIR  = Path("AnsysData/CarbopolSolution")          # carpeta con archivos sin extensión
OUTDIR    = Path("PERFILES_CFD")         # carpeta de salida
X_TARGET  = 0.15  
ruta = "perfiles_x="+str(X_TARGET)+".txt"                             # x deseado (m)
INVERT_Z_AXIS = False                          # True si quieres 0 "abajo" en eje Z
OUTPUT_TXT = OUTDIR / ruta  # archivo acumulado con TODOS los perfiles

# Mapeo de nombres esperados a nombres cortos
CANON_MAP = {
    "nodenumber": "id",
    "x-coordinate": "x",
    "y-coordinate": "y",
    "z-coordinate": "z",
    "phase-carbopol-x-velocity": "ux",  # u (en X)
    "phase-carbopol-y-velocity": "uy",
    "phase-carbopol-z-velocity": "uz",  # v (en Z) para el formato pedido
}


# ===================== Utilidades =====================
def _list_step_files(folder: Path, prefix: str = "carbopol_fluent-") -> list[Path]:
    """Devuelve la lista de archivos (sin extensión) ordenados por el sufijo numérico."""
    cands = [p for p in folder.iterdir() if p.is_file() and p.name.startswith(prefix)]
    if not cands:
        raise FileNotFoundError(f"No se encontraron archivos con prefijo '{prefix}' en {folder}")

    def _num(path: Path) -> int:
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", path.name)
        return int(m.group(1)) if m else 10**12

    cands.sort(key=_num)
    return cands


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a {id,x,y,z,ux,uy,uz} y valida requeridas."""
    cols_norm = [c.strip().lower() for c in df.columns]
    new_cols = []
    for c in cols_norm:
        if c in CANON_MAP:
            new_cols.append(CANON_MAP[c])
        else:
            new_cols.append(c.replace(" ", "_"))
    df.columns = new_cols
    needed = {"x", "z", "ux", "uz"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    return df


def _auto_tol(vals: np.ndarray, frac: float = 0.005, floor: float = 1e-12) -> float:
    """Tolerancia adaptativa: % del rango o piso absoluto (para seleccionar x)."""
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    return max((vmax - vmin) * frac, floor)


# ===================== Flujo principal =====================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Lista todos los archivos (ordenados por step)
    files = _list_step_files(DATA_DIR, prefix="carbopol_fluent-")
    print(f"[INFO] Encontrados {len(files)} archivos en {DATA_DIR}")

    # 2) Leer PRIMER archivo para fijar x_ref (x más cercano a X_TARGET)
    first_path = files[0]
    df0 = pd.read_csv(first_path, header=0, skipinitialspace=True)
    df0 = _standardize_columns(df0)

    x_unique0 = np.unique(df0["x"].to_numpy())
    if x_unique0.size == 0:
        raise RuntimeError("No hay valores de 'x' en el primer archivo.")
    x_ref = float(x_unique0[np.argmin(np.abs(x_unique0 - X_TARGET))])
    tol_x = _auto_tol(df0["x"].to_numpy(), frac=0.005, floor=1e-12)
    print(f"[INFO] X_TARGET={X_TARGET:.6f} -> x_ref (primer archivo)={x_ref:.6f} (tol_x={tol_x:.2e})")

    # 3) Abrir archivo TXT de salida y escribir encabezado
    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        fout.write("x[m], y[m], u[m/s], v[m/s]\n")

        # 4) Recorre todos los archivos y escribe cada perfil como un bloque
        for idx, path in enumerate(files, start=1):
            df = pd.read_csv(path, header=0, skipinitialspace=True)
            df = _standardize_columns(df)

            # Selección por banda en x alrededor de x_ref (SIN filtro en y; incluye todos los puntos)
            sel = (np.abs(df["x"] - x_ref) <= tol_x)
            prof = df.loc[sel, ["x", "z", "ux", "uz"]].dropna().sort_values("z").reset_index(drop=True)

            print(f"[INFO] {path.name}: puntos perfil = {len(prof)} / total = {len(df)}")

            # Escribe bloque del perfil (mismo formato solicitado), usando TODOS los puntos
            for _, r in prof.iterrows():
                # u := ux, v := uz (X–Z)
                fout.write(f"{r['x']:.10g}, {r['z']:.10g}, {r['ux']:.10g}, {r['uz']:.10g}\n")

            # Línea en blanco separando perfiles (excepto si es el último, igual no hace daño)
            fout.write("\n")

            # 5) Solo para el PRIMER archivo: gráficos de referencia
            if idx == 1:
                # 5.a) Perfil u_x(z)
                plt.figure(figsize=(6, 5))
                plt.plot(prof["ux"], prof["z"], "-o", lw=1.4, ms=3.5, label=r"$u_x(z)$")
                plt.axvline(0.0, lw=0.8, alpha=0.5)
                plt.xlabel(r"$u_x$ [m/s]")
                plt.ylabel("z [m]")
                plt.title(f"Perfil u_x(z) en x ≈ {x_ref:.4f} m (primer archivo)")
                plt.grid(True, alpha=0.3)
                if INVERT_Z_AXIS:
                    plt.gca().invert_yaxis()
                plt.tight_layout()
                fig1 = OUTDIR / f"perfil_ux_vs_z_x={x_ref:.6f}.png"
                plt.savefig(fig1, dpi=180)
                plt.close()
                print(f"[OK] Figura perfil -> {fig1}")

                # 5.b) Quiver (u_x, u_z) en X–Z con línea roja en x_ref
                df_quiv = df0[["x", "z", "ux", "uz"]].dropna().copy()
                plt.figure(figsize=(7, 5))
                plt.quiver(
                    df_quiv["x"], df_quiv["z"],
                    df_quiv["ux"], df_quiv["uz"],
                    angles="xy", scale_units="xy", scale=5, width=0.002, alpha=0.85
                )
                zmin, zmax = np.nanmin(df0["z"]), np.nanmax(df0["z"])
                plt.plot([x_ref, x_ref], [zmin, zmax], "r-", lw=2, label=f"Perfil x={x_ref:.4f} m")

                plt.xlabel("x [m]")
                plt.ylabel("z [m]")
                plt.title("Campo de velocidad (u_x, u_z) en X–Z (primer archivo)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.axis("equal")
                if INVERT_Z_AXIS:
                    plt.gca().invert_yaxis()
                plt.tight_layout()
                fig2 = OUTDIR / f"quiver_xz_con_linea_x={x_ref:.6f}.png"
                plt.savefig(fig2, dpi=180)
                plt.close()
                print(f"[OK] Figura quiver -> {fig2}")

    print(f"[OK] TXT con todos los perfiles -> {OUTPUT_TXT}")
    print("[DONE]")

if __name__ == "__main__":
    main()
