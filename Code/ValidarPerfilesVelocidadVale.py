from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Parámetros por defecto -----------------------------
PIV_TXT_DEFAULT = Path("PIVData") / "unido.txt"
CFD_TXT_DEFAULT = Path("PERFILES_CFD") / "perfiles_x=0.11.txt"  # admite z en el header
OUTDIR_DEFAULT  = Path("Results/SALIDAS_PIV_vs_CFD")
DT_PIV_DEFAULT: float = 1/250        # 0.004 s
DT_CFD_DEFAULT: float = 0.01
MAX_TIME_S_DEFAULT: Optional[float] = 10  # None para usar todos

# Binning (elige uno)
BIN_MODE = "seconds"   # "seconds" o "frames"
WINDOW_S = 1.0         # tamaño de ventana en segundos si BIN_MODE == "seconds"
FRAMES_PIV = 250       # frames/bin para PIV si BIN_MODE == "frames"
FRAMES_CFD = 100       # frames/bin para CFD si BIN_MODE == "frames"


# ----------------------------- Utilidades de lectura -----------------------------
def leer_perfiles_txt(path: Path) -> List[pd.DataFrame]:
    """
    Lee un archivo de perfiles en bloques separados por líneas vacías.
    Soporta dos formatos:
      1) 4 columnas: x,y(u),u(m/s),v(m/s) con separador coma/;/\t
      2) 2 columnas: y-coordinate [m] <sep> Velocity magnitude [m/s] (tab/espacios)

    Devuelve una lista de DataFrames con columnas:
      ["x[m]", "y[m]", "u[m/s]", "v[m/s]"]
    ordenados crecientemente por y.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n\r") for ln in f]

    if not lines:
        return []

    header_4col_candidates = (
        "x[m], y[m], u[m/s], v[m/s]",
        "x[m],y[m],u[m/s],v[m/s]",
        "x[m], z[m], u[m/s], v[m/s]",
        "x[m],z[m],u[m/s],v[m/s]",
        "x[m]; y[m]; u[m/s]; v[m/s]",
        "x[m];y[m];u[m/s];v[m/s]",
        "x[m]\ty[m]\tu[m/s]\tv[m/s]",
        "x[m]\tz[m]\tu[m/s]\tv[m/s]",
    )
    header_2col_candidates = (
        "y-coordinate [m]\tvelocity magnitude [m/s]",
        "y-coordinate [m] velocity magnitude [m/s]",
        "y [m]\tvelocity magnitude [m/s]",
        "y [m] velocity magnitude [m/s]",
    )

    def _smart_split(row: str) -> List[str]:
        for sep in [",", ";", "\t"]:
            if sep in row:
                return [p.strip() for p in row.split(sep)]
        return [p.strip() for p in row.split()]

    # Localiza header
    hdr_idx = None
    for i, ln in enumerate(lines):
        l = ln.strip().lower()
        if any(l.startswith(h.lower()) for h in header_4col_candidates + header_2col_candidates):
            hdr_idx = i
            break
    if hdr_idx is None:
        parts0 = _smart_split(lines[0])
        if len(parts0) == 2:
            hdr_idx = 0
        else:
            raise ValueError(f"No se encontró un header reconocido en {path}")

    header_line = lines[hdr_idx].strip().lower()
    is_2col = any(header_line.startswith(h.lower()) for h in header_2col_candidates)

    # Líneas de datos tras el header
    data_lines = lines[hdr_idx + 1 :]

    # Divide en bloques por líneas vacías
    perfiles_raw: List[List[str]] = []
    bloque: List[str] = []
    for ln in data_lines:
        if ln.strip() == "":
            if bloque:
                perfiles_raw.append(bloque)
                bloque = []
        else:
            bloque.append(ln)
    if bloque:
        perfiles_raw.append(bloque)

    perfiles: List[pd.DataFrame] = []

    for blk in perfiles_raw:
        rows_4 = []
        rows_2 = []
        for row in blk:
            parts = _smart_split(row)
            # Normaliza posibles decimales europeos si no hay punto
            parts = [p.replace(",", ".") if ("," in p and "." not in p) else p for p in parts]

            if is_2col:
                if len(parts) < 2:
                    continue
                try:
                    yv = float(parts[0])
                    umag = float(parts[1]) if parts[1].lower() != "nan" else np.nan
                    rows_2.append([np.nan, yv, umag, np.nan])  # x=NaN, u=|v|, v=NaN
                except Exception:
                    continue
            else:
                if len(parts) < 4:
                    continue
                try:
                    xv = float(parts[0])
                    yzv = float(parts[1])
                    u  = float(parts[2]) if parts[2].lower() != "nan" else np.nan
                    v  = float(parts[3]) if parts[3].lower() != "nan" else np.nan
                    rows_4.append([xv, yzv, u, v])
                except Exception:
                    continue

        if is_2col:
            if not rows_2:
                continue
            df = pd.DataFrame(rows_2, columns=["x[m]", "y[m]", "u[m/s]", "v[m/s]"])
        else:
            if not rows_4:
                continue
            df = pd.DataFrame(rows_4, columns=["x[m]", "y[m]", "u[m/s]", "v[m/s]"])

        df = df[np.isfinite(df["y[m]"])].copy()  # exige y finito; permite NaN en u/v
        df = df.sort_values("y[m]").reset_index(drop=True)
        perfiles.append(df)

    return perfiles


# ----------------------------- Normalización eje y (shift a 0) -----------------------------
def normalizar_y_shift(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Resta el valor mínimo de y para que cada perfil comience en 0."""
    out = []
    for df in dfs:
        df2 = df.copy()
        df2["y[m]"] = df2["y[m]"] - df2["y[m]"].min()
        out.append(df2)
    return out


# ----------------------------- Interpolación a malla común -----------------------------
def _make_strictly_monotonic(y: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordena por y ascendente y colapsa duplicados promediando val.
    """
    order = np.argsort(y)
    y_sorted = y[order]
    val_sorted = val[order]
    uniq_y, idx_start = np.unique(y_sorted, return_index=True)
    val_avg = []
    for k in range(len(idx_start)):
        i0 = idx_start[k]
        i1 = idx_start[k+1] if k+1 < len(idx_start) else len(y_sorted)
        val_avg.append(np.nanmean(val_sorted[i0:i1]))
    return uniq_y, np.asarray(val_avg)


def interp_series_to_grid(
    dfs: List[pd.DataFrame], y_grid: np.ndarray, col: str
) -> np.ndarray:
    """
    Interpola perfiles a y_grid sin extrapolar:
    - Dentro del rango [y.min, y.max]: interpola lineal.
    - Fuera del rango: NaN (evita líneas planas falsas por extrapolación).
    """
    out = np.full((len(dfs), len(y_grid)), np.nan, dtype=float)
    for i, df in enumerate(dfs):
        if df.empty:
            continue
        y = df["y[m]"].to_numpy()
        val = df[col].to_numpy()
        if len(y) == 0:
            continue
        if len(y) == 1:
            idx_closest = np.argmin(np.abs(y_grid - y[0]))
            out[i, idx_closest] = val[0]
            continue

        y, val = _make_strictly_monotonic(y, val)
        ymin, ymax = y[0], y[-1]

        interp_vals = np.interp(y_grid, y, val, left=val[0], right=val[-1])
        mask_in = (y_grid >= ymin) & (y_grid <= ymax)
        row = np.full_like(interp_vals, np.nan, dtype=float)
        row[mask_in] = interp_vals[mask_in]
        out[i, :] = row
    return out


# ----------------------------- Binning configurable -----------------------------
def build_time_bins(n: int, dt: float, window_s: float = 1.0) -> List[np.ndarray]:
    """
    Bins por ventanas de duración 'window_s' segundos: [0,window_s), [window_s, 2*window_s), ...
    """
    if n == 0:
        return []
    times = np.arange(n) * dt
    t_end = times[-1]
    n_bins = int(np.floor(t_end / window_s)) + 1
    bins: List[np.ndarray] = []
    for b in range(n_bins):
        t0 = b * window_s
        t1 = (b + 1) * window_s
        idx = np.where((times >= t0) & (times < t1))[0]
        if idx.size > 0:
            bins.append(idx)
    return bins


def build_frame_bins(n: int, frames_per_bin: int) -> List[np.ndarray]:
    """
    Bins por un número fijo de frames (útil si quieres que 'N' sea exactamente ese por bin).
    """
    if n == 0 or frames_per_bin <= 0:
        return []
    bins: List[np.ndarray] = []
    for start in range(0, n, frames_per_bin):
        stop = min(start + frames_per_bin, n)
        idx = np.arange(start, stop)
        if idx.size > 0:
            bins.append(idx)
    return bins


# ----------------------------- Utilidades varias -----------------------------
def _global_y_span(dfs: List[pd.DataFrame]) -> Tuple[float, float]:
    """
    Devuelve (ymin_global, ymax_global) tomando la intersección entre
    los spans de TODOS los perfiles de la serie.
    """
    ymins, ymaxs = [], []
    for df in dfs:
        if not df.empty and np.isfinite(df["y[m]"]).any():
            ymins.append(df["y[m]"].min())
            ymaxs.append(df["y[m]"].max())
    if not ymins:
        raise RuntimeError("No hay y válidos para calcular span global.")
    return max(ymins), min(ymaxs)


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

    # Normalizar y -> alinear en 0
    piv = normalizar_y_shift(piv)
    cfd = normalizar_y_shift(cfd)

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

    # ---- Malla en y: usa intersección si existe; si no, usa UNIÓN (y evita extrapolar en la interp) ----
    ymin_piv, ymax_piv = _global_y_span(piv)
    ymin_cfd, ymax_cfd = _global_y_span(cfd)

    y_int_min = max(ymin_piv, ymin_cfd)
    y_int_max = min(ymax_piv, ymax_cfd)

    if y_int_max > y_int_min:
        # hay solapamiento real
        y_grid = np.linspace(y_int_min, y_int_max, n_y)
        print("[DEBUG] y-grid (INTERSECCIÓN):", y_int_min, "->", y_int_max, "(len =", len(y_grid), ")")
    else:
        # NO hay solapamiento: usar UNIÓN
        y_union_min = min(ymin_piv, ymin_cfd)
        y_union_max = max(ymax_piv, ymax_cfd)
        y_grid = np.linspace(y_union_min, y_union_max, n_y)
        print("[WARN] Sin intersección PIV/CFD en y. Usando y-grid (UNIÓN):",
              y_union_min, "->", y_union_max, "(len =", len(y_grid), ")")

    print("[DEBUG] y-span PIV:", ymin_piv, "->", ymax_piv, "| y-span CFD:", ymin_cfd, "->", ymax_cfd)

    # Interpolación y estadísticos globales (sin extrapolar)
    U_piv = interp_series_to_grid(piv, y_grid, "u[m/s]")
    U_cfd = interp_series_to_grid(cfd, y_grid, "u[m/s]")

    piv_mean = np.nanmean(U_piv, axis=0)
    piv_std  = np.nanstd(U_piv,  axis=0)
    cfd_mean = np.nanmean(U_cfd, axis=0)
    cfd_std  = np.nanstd(U_cfd,  axis=0)

    # Debug valores CFD
    try:
        print("[DEBUG] U_cfd shape:", U_cfd.shape,
              "| cfd_mean[min,max]:", float(np.nanmin(cfd_mean)), float(np.nanmax(cfd_mean)))
    except Exception:
        pass

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

    # ---------------- Curvas y error por bin (configurable) ----------------
    if BIN_MODE == "seconds":
        bins_piv = build_time_bins(len(piv), dt_piv, window_s=WINDOW_S)
        bins_cfd = build_time_bins(len(cfd), dt_cfd, window_s=WINDOW_S)
        bin_label = lambda b: f"[{b*WINDOW_S:.2f},{(b+1)*WINDOW_S:.2f})s"
    elif BIN_MODE == "frames":
        bins_piv = build_frame_bins(len(piv), FRAMES_PIV)
        bins_cfd = build_frame_bins(len(cfd), FRAMES_CFD)
        bin_label = lambda b: f"bin {b}"
    else:
        raise ValueError("BIN_MODE debe ser 'seconds' o 'frames'")

    n_bins = min(len(bins_piv), len(bins_cfd))
    err_rows = []

    # DEBUG bins
    print("[DEBUG] dt_piv=", dt_piv, "dt_cfd=", dt_cfd, "BIN_MODE=", BIN_MODE)
    print("[DEBUG] PIV bins sizes (primeros 5):", [len(b) for b in bins_piv[:5]])
    print("[DEBUG] CFD bins sizes (primeros 5):", [len(b) for b in bins_cfd[:5]])

    for b in range(n_bins):
        idx_piv = bins_piv[b]
        idx_cfd = bins_cfd[b]

        piv_b_mean = np.nanmean(U_piv[idx_piv, :], axis=0)
        piv_b_std  = np.nanstd( U_piv[idx_piv, :], axis=0)
        cfd_b_mean = np.nanmean(U_cfd[idx_cfd, :], axis=0)
        cfd_b_std  = np.nanstd( U_cfd[idx_cfd, :], axis=0)

        plt.figure(figsize=(7, 5))
        plt.plot(piv_b_mean, y_grid, label=f"PIV ū(y) {bin_label(b)} (N={len(idx_piv)})", linewidth=2)
        plt.fill_betweenx(y_grid, piv_b_mean - piv_b_std, piv_b_mean + piv_b_std, alpha=0.25, label="PIV: ±σ")
        plt.plot(cfd_b_mean, y_grid, label=f"CFD ū(y) {bin_label(b)} (N={len(idx_cfd)})", linewidth=2)
        plt.fill_betweenx(y_grid, cfd_b_mean - cfd_b_std, cfd_b_mean + cfd_b_std, alpha=0.25, label="CFD: ±σ")
        plt.xlabel("u [m/s]")
        plt.ylabel("y [m]")
        plt.title(f"Comparación PIV vs CFD por bin {bin_label(b)}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if invert_y_on_plot:
            plt.gca().invert_yaxis()
        plt.tight_layout()
        fsec = outdir / f"comparacion_PIV_CFD_bin_{b:02d}.png"
        plt.savefig(fsec, dpi=180)
        plt.close()

        diff_b = piv_b_mean - cfd_b_mean
        mse_b  = float(np.nanmean(diff_b**2))
        rmse_b = float(np.sqrt(mse_b))
        mae_b  = float(np.nanmean(np.abs(diff_b)))

        err_rows.append({
            "bin": b,
            "label": bin_label(b),
            "N_PIV": int(len(idx_piv)),
            "N_CFD": int(len(idx_cfd)),
            "MSE": mse_b,
            "RMSE": rmse_b,
            "MAE": mae_b
        })

    err_df = pd.DataFrame(err_rows)
    err_csv = outdir / "errores_por_bin.csv"
    err_df.to_csv(err_csv, index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(err_df["bin"], err_df["RMSE"], marker="o", linewidth=1.8)
    plt.xlabel("Bin")
    plt.ylabel("RMSE [m/s]")
    plt.title(f"Error PIV vs CFD por bin (modo='{BIN_MODE}')")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    err_fig = outdir / "error_por_bin_RMSE.png"
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
    print(f"OK  -> {n_bins} figuras de comparación por bin (comparacion_PIV_CFD_bin_XX.png)")
    print(f"OK  -> serie error por bin: {err_fig}")
    print(f"CSV -> perfiles globales: {csv_path}")
    print(f"CSV -> errores por bin: {err_csv}")


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
    print(f"[INFO] BIN_MODE={BIN_MODE}, WINDOW_S={WINDOW_S}, FRAMES_PIV={FRAMES_PIV}, FRAMES_CFD={FRAMES_CFD}")

    main(piv_arg, cfd_arg, outd_arg, dt_piv_arg, dt_cfd_arg, tmax_arg)
