# -*- coding: utf-8 -*-
"""
Super-plot de convergencia temporal + independencia de malla.

Lee:
  out_groups_fullfield/groups_residual_like.csv
  out_groups_fullfield/groups_metrics.csv

y genera, para cada componente (u, v, ...), una figura 2x2 con:
  - residual sintético (3 mallas)
  - RMSE_norm (W&Z)
  - CV(RMSE) (Lee 2020)
  - R² (Lee 2020)

Incluye líneas verticales:
  - g_estacionario: primer grupo donde los 3 residuales son suficientemente bajos
  - g_indep_Lee_21: primer grupo donde medium vs fine cumple CV<=thr y R²>=thr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = Path("out_groups_fullfield")
CSV_RES  = BASE_DIR / "groups_residual_like.csv"
CSV_MET  = BASE_DIR / "groups_metrics.csv"

OUT_DIR  = Path("plots_groups")
OUT_DIR.mkdir(exist_ok=True)

# umbral relativo para "estacionario" (residual sintético)
# se define como fracción del primer residual no-NaN de la malla fina
RESID_REL_THRESHOLD = 0.05     # 5% del valor inicial, ajustable

# umbrales de Lee (deberían coincidir con los del análisis)
LEE_CV_THRESHOLD  = 0.10       # 10%
LEE_R2_THRESHOLD  = 0.95

# para pintar
MESH_ORDER = ["fine", "medium", "coarse"]
MESH_LABEL = {"fine": "fine", "medium": "medium", "coarse": "coarse"}
MESH_STYLE = {"fine": ("tab:blue",  "o-"),
              "medium": ("tab:orange", "s-"),
              "coarse": ("tab:green", "^-")}

# =========================
# FUNCIONES AUXILIARES
# =========================

def find_stationary_group(residual_df, component):
    """
    Devuelve el primer grupo donde TODAS las mallas tienen residual
    < RESID_REL_THRESHOLD * residual_inicial_fine.

    Si no se encuentra, devuelve None.
    """
    sub = residual_df[residual_df["component"] == component].copy()

    # Pivot: filas=group, columnas=mesh
    pivot = sub.pivot_table(index="group", columns="mesh", values="residual_like")
    pivot = pivot.sort_index()

    # referencia: primer residual no-NaN de la malla fina
    if "fine" not in pivot.columns:
        return None

    fine_series = pivot["fine"].dropna()
    if fine_series.empty:
        return None

    ref0 = fine_series.iloc[0]
    thr  = RESID_REL_THRESHOLD * ref0

    mask_all = (pivot <= thr).all(axis=1)  # True donde todas las mallas están bajo umbral
    candidates = pivot.index[mask_all]

    if len(candidates) == 0:
        return None
    return int(candidates[0])


def find_lee_independence_group(metrics_df, component):
    """
    Devuelve el primer grupo donde medium vs fine cumple el criterio de Lee:
      CV(RMSE)_21 <= LEE_CV_THRESHOLD y R2_21 >= LEE_R2_THRESHOLD.
    Si no se encuentra, devuelve None.
    """
    sub = metrics_df[metrics_df["component"] == component].copy()
    sub = sub.sort_values("group")

    mask = (sub["cv_rmse_21"] <= LEE_CV_THRESHOLD) & (sub["r2_21"] >= LEE_R2_THRESHOLD)
    candidates = sub.loc[mask, "group"]

    if candidates.empty:
        return None
    return int(candidates.iloc[0])


def make_superplot_for_component(residual_df, metrics_df, component):
    # Filtrar datos de este componente
    res_sub = residual_df[residual_df["component"] == component].copy()
    met_sub = metrics_df[metrics_df["component"] == component].copy()

    if res_sub.empty or met_sub.empty:
        print(f"[WARN] No hay datos para componente '{component}'")
        return

    # Detectar grupos especiales
    g_stat = find_stationary_group(residual_df, component)
    g_lee  = find_lee_independence_group(metrics_df, component)

    print(f"\nComponente {component}:")
    print(f"  Grupo estacionario (residual sintético) ≈ {g_stat}")
    print(f"  Grupo independencia Lee (medium vs fine) ≈ {g_lee}")

    # Pivot residuals: group vs mesh
    pivot_res = res_sub.pivot_table(index="group", columns="mesh",
                                    values="residual_like").sort_index()
    groups_res = pivot_res.index.values

    # Ordenar metrics por group
    met_sub = met_sub.sort_values("group")
    groups_met = met_sub["group"].values

    # Crear figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # --- subplot 1: residual sintético ---
    for mesh in MESH_ORDER:
        if mesh not in pivot_res.columns:
            continue
        color, style = MESH_STYLE[mesh]
        ax1.semilogy(groups_res, pivot_res[mesh].values,
                     style, color=color, label=MESH_LABEL[mesh])
    ax1.set_xlabel("Grupo / tanda temporal")
    ax1.set_ylabel("RMSE entre grupos consecutivos")
    ax1.set_title("Residual sintético (todo el dominio)")
    ax1.grid(True, which="both", alpha=0.4)
    ax1.legend(title="Malla")

    # --- subplot 2: RMSE_norm (W&Z) ---
    ax2.plot(groups_met, met_sub["rmse_norm_21"], "o-", label="RMSE_norm 21 (med vs fine)")
    ax2.plot(groups_met, met_sub["rmse_norm_31"], "s-", label="RMSE_norm 31 (coarse vs fine)")
    ax2.axhline(0.10, color="gray", ls="--", lw=1, label="Umbral 10%")
    ax2.set_xlabel("Grupo / tanda temporal")
    ax2.set_ylabel("RMSE_norm [-]")
    ax2.set_title("Wang & Zhai – RMSE normalizado")
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    # --- subplot 3: CV(RMSE) (Lee) ---
    ax3.plot(groups_met, met_sub["cv_rmse_21"], "o-", label="CV(RMSE) 21")
    ax3.plot(groups_met, met_sub["cv_rmse_31"], "s-", label="CV(RMSE) 31")
    ax3.axhline(LEE_CV_THRESHOLD, color="gray", ls="--", lw=1, label=f"Umbral {LEE_CV_THRESHOLD:.0%}")
    ax3.set_xlabel("Grupo / tanda temporal")
    ax3.set_ylabel("CV(RMSE) [-]")
    ax3.set_title("Lee 2020 – Coef. variación de RMSE")
    ax3.grid(True, alpha=0.4)
    ax3.legend()

    # --- subplot 4: R² (Lee) ---
    ax4.plot(groups_met, met_sub["r2_21"], "o-", label="R² 21 (med vs fine)")
    ax4.plot(groups_met, met_sub["r2_31"], "s-", label="R² 31 (coarse vs fine)")
    ax4.axhline(LEE_R2_THRESHOLD, color="gray", ls="--", lw=1, label=f"Umbral {LEE_R2_THRESHOLD:.2f}")
    ax4.set_xlabel("Grupo / tanda temporal")
    ax4.set_ylabel("R² [-]")
    ax4.set_title("Lee 2020 – Coeficiente de determinación")
    ax4.grid(True, alpha=0.4)
    ax4.legend()

    # --- líneas verticales de referencia ---
    for ax in (ax1, ax2, ax3, ax4):
        if g_stat is not None:
            ax.axvline(g_stat, color="purple", ls="--", lw=1,
                       label="Inicio régimen estacionario" if ax is ax1 else None)
        if g_lee is not None:
            ax.axvline(g_lee, color="red", ls=":", lw=1.2,
                       label="Inicio independencia Lee (21)" if ax is ax1 else None)

    # Añadir leyenda extra al primer subplot si pusimos líneas
    if g_stat is not None or g_lee is not None:
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="best", title="Malla / referencias")

    fig.suptitle(f"Convergencia temporal + independencia de malla\nComponente: {component}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = OUT_DIR / f"superplot_{component}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"  → Figura guardada en {out_file}")


# =========================
# MAIN
# =========================

def main():
    df_res = pd.read_csv(CSV_RES)
    df_met = pd.read_csv(CSV_MET)

    components = df_res["component"].unique()
    # por seguridad, intersectar con los que sí tienen métricas
    components = [c for c in components if c in df_met["component"].unique()]

    if not components:
        print("No se encontraron componentes comunes en residuals y metrics.")
        return

    for comp in components:
        make_superplot_for_component(df_res, df_met, comp)


if __name__ == "__main__":
    main()
