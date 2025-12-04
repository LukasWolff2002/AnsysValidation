# -*- coding: utf-8 -*-
"""
Plotea perfiles de velocidad para cada corte y cada malla.

Para cada combinación (SizeN, x_cut):
  - lee SizeN_perfiles_x={x_cut}.csv  -> todos los timesteps
  - lee SizeN_cut_x_{x_cut}.csv      -> perfil promedio temporal
  - genera una figura con:
      * arriba: u(z) (ux)
      * abajo: v(z) (uz)
    donde:
      - todas las curvas de cada timestep se plotean en gris
      - el perfil promedio se plotea en color marcado
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================

OUT = Path("out_timeavg_cuts")   # misma carpeta de salida del mesh_cuts.py

SIZES = (3, 4, 5)                # Size3, Size4, Size5
X_CUTS = [0.13, 0.17, 0.25, 0.35, 0.43]   # mismos cortes que en mesh_cuts.py

# Si quieres cambiar fuente / estilo, puedes hacerlo aquí
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
})


def plot_profiles_for_mesh_and_cut(mesh_size: int, x_cut: float):
    """
    Hace el plot para una malla (SizeN) y un corte en x_cut:
      - lee SizeN_perfiles_x={x_cut}.csv
      - lee SizeN_cut_x_{x_cut}.csv
      - genera figura PNG en OUT
    """
    # archivos de entrada
    f_perfiles = OUT / f"Size{mesh_size}_perfiles_x={x_cut}.csv"
    f_mean     = OUT / f"Size{mesh_size}_cut_x_{x_cut}.csv"

    if not f_perfiles.exists():
        print(f"[AVISO] No existe {f_perfiles.name}, se omite.")
        return
    if not f_mean.exists():
        print(f"[AVISO] No existe {f_mean.name}, se omite.")
        return

    df_all = pd.read_csv(f_perfiles)
    df_mean = pd.read_csv(f_mean)

    # Asegurarnos de que estén ordenados en z
    df_mean = df_mean.sort_values("z")
    # pasos de tiempo únicos
    steps = np.sort(df_all["step"].unique())

    # ========= FIGURA =========
    plt.figure(figsize=(6, 8))

    # ----- Componente u (ux) -----
    for s in steps:
        g = df_all[df_all["step"] == s].sort_values("z")
        plt.plot(g["ux"], g["z"], color="0.8", linewidth=0.5)  # gris claro

    plt.plot(df_mean["u_mean"], df_mean["z"],
              color="tab:blue", linewidth=2, label="Promedio temporal")
    plt.xlabel("u [m/s]")
    plt.ylabel("z [m]")
    plt.title(f"Size{mesh_size} – Corte en x = {x_cut}")
    plt.legend(loc="best")

    plt.tight_layout()

    fig_name = OUT / f"Size{mesh_size}_perfil_promedio_x_{x_cut}.png"
    plt.savefig(fig_name, dpi=200)
    plt.close()

    print(f"[OK] Figura guardada: {fig_name.name}")


def main():
    for sz in SIZES:
        for x_cut in X_CUTS:
            plot_profiles_for_mesh_and_cut(sz, x_cut)


if __name__ == "__main__":
    main()
