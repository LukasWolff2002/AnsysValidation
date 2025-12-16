# -*- coding: utf-8 -*-
"""
Análisis de perfiles por corte para independencia de malla.

Requisitos de entrada (salida de mesh_cuts.py):
    out_timeavg_cuts/Size3_perfiles_x={x}.csv
    out_timeavg_cuts/Size4_perfiles_x={x}.csv
    out_timeavg_cuts/Size5_perfiles_x={x}.csv

    out_timeavg_cuts/Size3_cut_x_{x}.csv
    out_timeavg_cuts/Size4_cut_x_{x}.csv
    out_timeavg_cuts/Size5_cut_x_{x}.csv

Donde:
    - archivos "perfiles_x=" tienen columnas: step, x, z, ux, uz
      (TODAS las muestras de todos los timesteps para ese corte)
    - archivos "cut_x_" tienen columnas: x, z, u_mean, v_mean
      (perfil promedio temporal del corte)

Para cada corte x_cut, el script produce:

1) Histograma de u (ux) para las tres mallas (subplots):
       hist_u_todosTiempos_x_{x_cut}.png

2) Cálculo de RMSE entre malla fina (Size3) y cada malla más gruesa,
   interpolando la malla gruesa a los z de la malla fina.

3) Gráficos u_fina vs u_otra con la diagonal y el RMSE en el título:
       scatter_u_Size3_vs_Size4_x_{x_cut}.png
       scatter_u_Size3_vs_Size5_x_{x_cut}.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================

OUT = Path("out_timeavg_cuts")   # carpeta donde están los CSV que generó mesh_cuts.py

# mallas (primera se toma como "fina")
MESHES = [3, 4, 5]               # Size3, Size4, Size5
MESH_LABELS = {3: "fina", 4: "media", 5: "gruesa"}

# cortes (deben coincidir con los usados en mesh_cuts.py)
X_CUTS = [0.13, 0.17, 0.25, 0.35, 0.43]

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
})


# =========================
# FUNCIONES AUXILIARES
# =========================

def load_profile_mean(mesh_size: int, x_cut: float) -> pd.DataFrame:
    """
    Lee el archivo de perfil promedio para una malla y un corte:
        out_timeavg_cuts/SizeN_cut_x_<x_cut>.csv

    Devuelve un DataFrame con columnas: x, z, u_mean, v_mean
    """
    fname = OUT / f"Size{mesh_size}_cut_x_{x_cut}.csv"
    if not fname.exists():
        raise FileNotFoundError(f"No se encontró {fname}")
    df = pd.read_csv(fname)
    return df.sort_values("z").reset_index(drop=True)


def load_perfiles_all(mesh_size: int, x_cut: float) -> pd.DataFrame:
    """
    Lee el archivo con TODOS los perfiles (todos los timesteps) para
    una malla y un corte:
        out_timeavg_cuts/SizeN_perfiles_x={x_cut}.csv

    Devuelve un DataFrame con columnas: step, x, z, ux, uz
    """
    fname = OUT / f"Size{mesh_size}_perfiles_x={x_cut}.csv"
    if not fname.exists():
        raise FileNotFoundError(f"No se encontró {fname}")
    df = pd.read_csv(fname)
    return df.dropna(subset=["z", "ux"]).reset_index(drop=True)


def rmse_interp(df_fine: pd.DataFrame, df_coarse: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calcula el RMSE entre dos perfiles u_mean(z):

    - interpola la malla gruesa (df_coarse) a los z de la malla fina,
    - devuelve (rmse, u_fine, u_interp_coarse)
    """
    z_f = df_fine["z"].values
    u_f = df_fine["u_mean"].values

    z_c = df_coarse["z"].values
    u_c = df_coarse["u_mean"].values

    # interpolamos u_c en los z de la malla fina
    u_c_int = np.interp(z_f, z_c, u_c)

    err = u_f - u_c_int
    rmse = float(np.sqrt(np.mean(err**2)))
    return rmse, u_f, u_c_int


# =========================
# 1) HISTOGRAMAS (TODAS LAS VELOCIDADES)
# =========================

def make_histograms_for_cut(x_cut: float, mallas_perfiles: dict[int, pd.DataFrame]):
    """
    Histograma REAL de u para cada malla:
    usa los archivos SizeN_perfiles_x=...csv,
    es decir: TODAS las velocidades de todos los timesteps
    para ese corte y esa malla.
    """
    # concatenamos TODAS las muestras para fijar rango común
    all_u = np.concatenate(
        [mallas_perfiles[m]["ux"].values for m in MESHES if m in mallas_perfiles]
    )
    u_min, u_max = float(all_u.min()), float(all_u.max())
    bins = np.linspace(u_min, u_max, 30)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)

    for ax, m in zip(axes, MESHES):
        if m not in mallas_perfiles:
            ax.set_title(f"Size{m} (sin datos)")
            ax.set_xlabel("u [m/s]")
            ax.set_ylabel("Frecuencia")
            continue

        dfp = mallas_perfiles[m]
        ax.hist(dfp["ux"].values, bins=bins, color="0.7", edgecolor="k")
        ax.set_title(f"Size{m} ({MESH_LABELS.get(m,'')})")
        ax.set_xlabel("u [m/s]")
        ax.set_ylabel("Frecuencia")

    fig.suptitle(f"Histograma de u (TODOS LOS TIMESTEPS)\nCorte en x = {x_cut}", fontsize=12)
    plt.tight_layout(rect=[0, 0.0, 1, 0.90])

    out_fig = OUT / f"hist_u_todosTiempos_x_{x_cut}.png"
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"[OK] Histograma (todos los timesteps) guardado: {out_fig.name}")


# =========================
# 2 & 3) RMSE + SCATTER u1 vs u2
# =========================

def make_scatter_for_cut(x_cut: float,
                         df_fine: pd.DataFrame,
                         df_other: pd.DataFrame,
                         fine_mesh: int,
                         other_mesh: int):
    """
    Hace el gráfico u_fina vs u_otra (interpolada), con la diagonal y
    el RMSE en el título.
    """
    rmse, u_f, u_o_int = rmse_interp(df_fine, df_other)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(u_f, u_o_int, s=15, alpha=0.7, edgecolors="none")

    # diagonal
    u_min = min(u_f.min(), u_o_int.min())
    u_max = max(u_f.max(), u_o_int.max())
    ax.plot([u_min, u_max], [u_min, u_max], "r--", label="y = x")

    ax.set_xlabel(f"u (Size{fine_mesh}, {MESH_LABELS.get(fine_mesh,'fina')}) [m/s]")
    ax.set_ylabel(f"u interp. (Size{other_mesh}, {MESH_LABELS.get(other_mesh,'')}) [m/s]")

    ax.set_title(f"Corte x = {x_cut} – RMSE = {rmse:.3e}")
    ax.legend(loc="best")

    plt.tight_layout()
    out_fig = OUT / f"scatter_u_Size{fine_mesh}_vs_Size{other_mesh}_x_{x_cut}.png"
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"[OK] Scatter guardado: {out_fig.name}")


# =========================
# MAIN
# =========================

def main():
    fine_mesh = MESHES[0]  # Size3 como malla fina

    for x_cut in X_CUTS:
        print(f"\n=== Corte en x = {x_cut} ===")

        # -------- 1) Histogramas: usamos archivos "perfiles_x" --------
        mallas_perfiles = {}
        for m in MESHES:
            try:
                mallas_perfiles[m] = load_perfiles_all(m, x_cut)
            except FileNotFoundError as e:
                print(f"  [AVISO] {e}")
        if len(mallas_perfiles) >= 1:
            make_histograms_for_cut(x_cut, mallas_perfiles)
        else:
            print("  [AVISO] No hay datos de perfiles para este corte; sin histograma.")

        # -------- 2 & 3) RMSE + u1 vs u2: usamos perfiles promedio --------
        profiles_mean = {}
        for m in MESHES:
            try:
                profiles_mean[m] = load_profile_mean(m, x_cut)
            except FileNotFoundError as e:
                print(f"  [AVISO] {e}")

        if fine_mesh not in profiles_mean:
            print("  [ERROR] No se pudo cargar la malla fina para este corte; se omiten RMSE y scatter.")
            continue

        df_fine = profiles_mean[fine_mesh]

        for m in MESHES[1:]:
            if m not in profiles_mean:
                continue
            df_other = profiles_mean[m]
            make_scatter_for_cut(x_cut, df_fine, df_other, fine_mesh, m)


if __name__ == "__main__":
    main()
