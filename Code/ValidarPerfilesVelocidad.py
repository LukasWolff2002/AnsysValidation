from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Parámetros por defecto -----------------------------
PIV_TXT_DEFAULT = Path("PERFILES_PIV") / "perfiles_x=0.09.txt"
CFD_TXT_DEFAULT = Path("PERFILES_CFD") / "perfiles_x=0.12.txt"  # admite z en el header
OUTDIR_DEFAULT  = Path("Results/SALIDAS_PIV_vs_CFD")
DT_PIV_DEFAULT: float = 0.1
DT_CFD_DEFAULT: float = 0.01
MAX_TIME_S_DEFAULT: Optional[float] = 10  # None para usar todos


# ----------------------------- Utilidades de lectura -----------------------------
def leer_perfiles_txt(path: Path) -> List[pd.DataFrame]:
    """
    Lee un archivo de perfiles en bloques separados por líneas vacías.
    Devuelve una lista de DataFrames con columnas: ["x[m]", "y[m]", "u[m/s]", "v[m/s]"],
    ordenados crecientemente por y. Si el archivo trae 'z[m]' en vez de 'y[m]', lo remapea.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    # Detecta header (acepta y o z)
    hdr = None
    header_alternatives = (
        "x[m], y[m], u[m/s], v[m/s]",
        "x[m],y[m],u[m/s],v[m/s]",
        "x[m], z[m], u[m/s], v[m/s]",
        "x[m],z[m],u[m/s],v[m/s]",
    )
    low_lines = [ln.lower() for ln in lines]
    for i, ln in enumerate(low_lines):
        if any(ln.startswith(h.lower()) for h in header_alternatives):
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"No se encontró el header en {path}")

    # Separa bloques por líneas vacías
    data_lines = lines[hdr + 1 :]
    perfiles_raw: List[List[str]] = []
    bloque: List[str] = []
    for ln in data_lines:
        if ln == "":
            if bloque:
                perfiles_raw.append(bloque)
                bloque = []
        else:
            bloque.append(ln)
    if bloque:
        perfiles_raw.append(bloque)

    # Parseo de cada bloque a DataFrame
    perfiles: List[pd.DataFrame] = []
    for blk in perfiles_raw:
        rows = []
        for row in blk:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 4:
                continue
            try:
                rows.append(
                    [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
                )
            except Exception:
                continue

        if not rows:
            continue

        cols = ["x[m]", "y_or_z[m]", "u[m/s]", "v[m/s]"]
        df = pd.DataFrame(rows, columns=cols).dropna()
        # si vino z, mapea a y para homogenizar el pipeline
        df = df.rename(columns={"y_or_z[m]": "y[m]"}).sort_values("y[m]").reset_index(drop=True)
        perfiles.append(df)

    return perfiles


# ----------------------------- Interpolación a malla común -----------------------------
def interp_series_to_grid(
    dfs: List[pd.DataFrame], y_grid: np.ndarray, col: str
) -> np.ndarray:
    """
    Interpola una lista de perfiles (DataFrames) a una malla común y_grid para la columna `col`.
    Usa extrapolación por valores extremos (left/right).
    Retorna una matriz (n_perfiles, len(y_grid)).
    """
    out = np.full((len(dfs), len(y_grid)), np.nan, dtype=float)
    for i, df in enumerate(dfs):
        y = df["y[m]"].to_numpy()
        val = df[col].to_numpy()
        if len(y) >= 2:
            out[i, :] = np.interp(y_grid, y, val, left=val[0], right=val[-1])
        elif len(y) == 1:
            out[i, :] = val[0]
    return out


def _build_second_bins(n: int, dt: float) -> List[np.ndarray]:
    """
    Devuelve una lista de arrays de índices, uno por intervalo de 1 s:
    bin 0: t in [0,1), bin 1: [1,2), ...
    """
    if n == 0:
        return []
    times = np.arange(n) * dt
    t_end = times[-1]
    n_bins = int(np.floor(t_end)) + 1  # incluye el último parcial
    bins: List[np.ndarray] = []
    for b in range(n_bins):
        idx = np.where((times >= b) & (times < b + 1.0))[0]
        if idx.size > 0:
            bins.append(idx)
    return bins


# ----------------------------- Lógica principal -----------------------------
def main(
    piv_path: Path,
    cfd_path: Path,
    outdir: Path,
    dt_piv: float,
    dt_cfd: float,
    tmax: Optional[float],
    n_y: int = 200,
    invert_y_on_plot: bool = True,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    piv = leer_perfiles_txt(piv_path)
    cfd = leer_perfiles_txt(cfd_path)

    # Acotar por tmax de forma independiente
    n_piv = len(piv)
    n_cfd = len(cfd)
    if n_piv == 0 or n_cfd == 0:
        raise RuntimeError("No hay perfiles válidos en PIV/CFD.")

    if tmax is not None:
        k_piv = int(np.floor(tmax / dt_piv)) + 1
        k_cfd = int(np.floor(tmax / dt_cfd)) + 1
        n_piv = min(n_piv, k_piv)
        n_cfd = min(n_cfd, k_cfd)

    piv = piv[:n_piv]
    cfd = cfd[:n_cfd]

    if len(piv) == 0 or len(cfd) == 0:
        raise RuntimeError("Tras aplicar tmax, no quedan perfiles en PIV o CFD.")

    print(f"[INFO] N_perfiles: PIV={len(piv)} (dt={dt_piv}s), CFD={len(cfd)} (dt={dt_cfd}s)")

    # Malla común en y (intersección del primer perfil)
    y_min = max(piv[0]["y[m]"].min(), cfd[0]["y[m]"].min())
    y_max = min(piv[0]["y[m]"].max(), cfd[0]["y[m]"].max())
    if y_max <= y_min:
        y_grid = piv[0]["y[m]"].to_numpy()
    else:
        y_grid = np.linspace(y_min, y_max, n_y)

    # Interpolación y estadísticos globales
    U_piv = interp_series_to_grid(piv, y_grid, "u[m/s]")
    U_cfd = interp_series_to_grid(cfd, y_grid, "u[m/s]")

    piv_mean = np.nanmean(U_piv, axis=0)
    piv_std  = np.nanstd(U_piv,  axis=0)
    cfd_mean = np.nanmean(U_cfd, axis=0)
    cfd_std  = np.nanstd(U_cfd,  axis=0)

    # ---------------- Gráfico global (promedio ±σ) ----------------
    plt.figure(figsize=(7, 5))
    plt.plot(piv_mean, y_grid, label="PIV: ū(y)", linewidth=2)
    plt.fill_betweenx(y_grid, piv_mean - piv_std, piv_mean + piv_std, alpha=0.25, label="PIV: ±σ")
    plt.plot(cfd_mean, y_grid, label="CFD: ū(y)", linewidth=2)
    plt.fill_betweenx(y_grid, cfd_mean - cfd_std, cfd_mean + cfd_std, alpha=0.25, label="CFD: ±σ")
    plt.xlabel("u [m/s]")
    plt.ylabel("y [m]")
    plt.title(f"Promedio temporal ±σ | PIV Δt={dt_piv}s, CFD Δt={dt_cfd}s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if invert_y_on_plot:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_path = outdir / "perfil_promedio_u_con_sombra.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()

    # ---------------- Curvas y error por intervalos de 1 s (PIV y CFD con sus propios Δt) ----------------
    bins_piv = _build_second_bins(len(piv), dt_piv)
    bins_cfd = _build_second_bins(len(cfd), dt_cfd)

    # Alineamos por segundo: usamos hasta el mínimo número de bins disponible en ambos
    n_bins = min(len(bins_piv), len(bins_cfd))
    err_rows = []

    for b in range(n_bins):
        idx_piv = bins_piv[b]
        idx_cfd = bins_cfd[b]

        # Promedios y σ por segundo (cada uno con sus muestras)
        piv_b_mean = np.nanmean(U_piv[idx_piv, :], axis=0)
        piv_b_std  = np.nanstd( U_piv[idx_piv, :], axis=0)
        cfd_b_mean = np.nanmean(U_cfd[idx_cfd, :], axis=0)
        cfd_b_std  = np.nanstd( U_cfd[idx_cfd, :], axis=0)

        # Figura comparativa PIV vs CFD (con ±σ) para este segundo
        plt.figure(figsize=(7, 5))
        plt.plot(piv_b_mean, y_grid, label=f"PIV ū(y) [{b},{b+1})s (N={len(idx_piv)})", linewidth=2)
        plt.fill_betweenx(y_grid, piv_b_mean - piv_b_std, piv_b_mean + piv_b_std, alpha=0.25, label="PIV: ±σ")
        plt.plot(cfd_b_mean, y_grid, label=f"CFD ū(y) [{b},{b+1})s (N={len(idx_cfd)})", linewidth=2)
        plt.fill_betweenx(y_grid, cfd_b_mean - cfd_b_std, cfd_b_mean + cfd_b_std, alpha=0.25, label="CFD: ±σ")
        plt.xlabel("u [m/s]")
        plt.ylabel("y [m]")
        plt.title(f"Comparación PIV vs CFD por segundo [{b},{b+1})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if invert_y_on_plot:
            plt.gca().invert_yaxis()
        plt.tight_layout()
        fsec = outdir / f"comparacion_PIV_CFD_seg_{b:02d}.png"
        plt.savefig(fsec, dpi=180)
        plt.close()

        # Errores (entre las curvas promedio de ese segundo)
        diff_b = piv_b_mean - cfd_b_mean
        mse_b  = float(np.nanmean(diff_b**2))
        rmse_b = float(np.sqrt(mse_b))
        mae_b  = float(np.nanmean(np.abs(diff_b)))

        err_rows.append({
            "segundo": b,
            "N_PIV": int(len(idx_piv)),
            "N_CFD": int(len(idx_cfd)),
            "MSE": mse_b,
            "RMSE": rmse_b,
            "MAE": mae_b
        })

    # Serie de error por segundo (RMSE)
    err_df = pd.DataFrame(err_rows)
    err_csv = outdir / "errores_por_segundo.csv"
    err_df.to_csv(err_csv, index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(err_df["segundo"], err_df["RMSE"], marker="o", linewidth=1.8)
    plt.xlabel("Segundo")
    plt.ylabel("RMSE [m/s]")
    plt.title("Error PIV vs CFD por segundo (RMSE entre curvas promedio)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    err_fig = outdir / "error_por_segundo_RMSE.png"
    plt.savefig(err_fig, dpi=180)
    plt.close()

    # ---------------- CSV global ----------------
    df_out = pd.DataFrame(
        {
            "y[m]": y_grid,
            "u_mean_PIV[m/s]": piv_mean,
            "u_std_PIV[m/s]":  piv_std,
            "u_mean_CFD[m/s]": cfd_mean,
            "u_std_CFD[m/s]":  cfd_std,
        }
    )
    csv_path = outdir / "perfil_promedio_u.csv"
    df_out.to_csv(csv_path, index=False)

    print(f"OK  -> figura global: {fig_path}")
    print(f"OK  -> {n_bins} figuras de comparación por segundo (comparacion_PIV_CFD_seg_XX.png)")
    print(f"OK  -> serie error por segundo: {err_fig}")
    print(f"CSV -> perfiles globales: {csv_path}")
    print(f"CSV -> errores por segundo: {err_csv}")


if __name__ == "__main__":
    args = sys.argv[1:]

    piv_arg  = Path(args[0]) if len(args) >= 1 else PIV_TXT_DEFAULT
    cfd_arg  = Path(args[1]) if len(args) >= 2 else CFD_TXT_DEFAULT
    outd_arg = Path(args[2]) if len(args) >= 3 else OUTDIR_DEFAULT

    # Admite 1 o 2 Δt por CLI: <dt> o <dt_piv> <dt_cfd>
    if len(args) >= 5:
        dt_piv_arg = float(args[3])
        dt_cfd_arg = float(args[4])
        next_idx = 5
    elif len(args) == 4:
        dt_piv_arg = dt_cfd_arg = float(args[3])
        next_idx = 4
    else:
        dt_piv_arg = DT_PIV_DEFAULT
        dt_cfd_arg = DT_CFD_DEFAULT
        next_idx = 3

    if len(args) > next_idx and args[next_idx].strip().lower() != "none":
        tmax_arg = float(args[next_idx])
    else:
        tmax_arg = MAX_TIME_S_DEFAULT  # o pon None si quieres usar todos los perfiles

    print(f"[INFO] PIV: {piv_arg}")
    print(f"[INFO] CFD: {cfd_arg}")
    print(f"[INFO] OUT: {outd_arg}")
    print(f"[INFO] dt_piv: {dt_piv_arg}, dt_cfd: {dt_cfd_arg}, tmax: {tmax_arg}")

    main(piv_arg, cfd_arg, outd_arg, dt_piv_arg, dt_cfd_arg, tmax_arg)
