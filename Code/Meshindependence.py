# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ============================================================
# LECTURA Y PRE-PROCESO DE TUS .TXT (CFDtransformer)
# ============================================================

def load_blocks_from_txt(txt_path: Path) -> List[pd.DataFrame]:
    """
    Lee un TXT con encabezado 'x[m], y[m], u[m/s], v[m/s]' (y = z en tus datos),
    con múltiples bloques (separados por línea en blanco). Devuelve lista de DataFrames
    con columnas estandarizadas: x,z,u,v ordenadas por z ascendente.
    """
    blocks: List[pd.DataFrame] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # quitar encabezado
    if lines and lines[0].lower().startswith("x[m]"):
        lines = lines[1:]

    curr: List[str] = []
    for ln in lines + [""]:
        if ln == "":
            if curr:
                data = []
                for row in curr:
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) < 4:
                        continue
                    try:
                        x = float(parts[0])  # x[m]
                        z = float(parts[1])  # y[m] en header => realmente z
                        u = float(parts[2])  # u[m/s] = ux
                        v = float(parts[3])  # v[m/s] = uz
                    except ValueError:
                        continue
                    data.append((x, z, u, v))
                if data:
                    df = pd.DataFrame(data, columns=["x", "z", "u", "v"]).sort_values("z").reset_index(drop=True)
                    blocks.append(df)
            curr = []
        else:
            curr.append(ln)
    return blocks

def pick_profile(blocks: List[pd.DataFrame], mode: str = "mean_last_n", last_n: int = 5) -> pd.DataFrame:
    """
    Selecciona un perfil representativo:
      - 'last': usa el último bloque.
      - 'mean_last_n': interpola y promedia los últimos N bloques (estabiliza ruido).
    Devuelve DataFrame con columnas x,z,u,v.
    """
    if not blocks:
        raise ValueError("El TXT no contiene bloques de perfil.")
    if mode == "last" or len(blocks) == 1:
        return blocks[-1]

    use = blocks[-last_n:] if len(blocks) >= last_n else blocks
    zmin = max(b["z"].min() for b in use)
    zmax = min(b["z"].max() for b in use)
    zc = np.linspace(zmin, zmax, 500)

    U, V = [], []
    for b in use:
        U.append(np.interp(zc, b["z"].values, b["u"].values))
        V.append(np.interp(zc, b["z"].values, b["v"].values))

    u_mean = np.nanmean(np.vstack(U), axis=0)
    v_mean = np.nanmean(np.vstack(V), axis=0)
    return pd.DataFrame({"x": np.nan, "z": zc, "u": u_mean, "v": v_mean})

def align_to_common_z(dfs: List[pd.DataFrame], npts: int = 600) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Interpola todos los perfiles a un eje z común (intersección de rangos).
    Devuelve zc y listas [u_list], [v_list] ya interpoladas.
    """
    zmin = max(df["z"].min() for df in dfs)
    zmax = min(df["z"].max() for df in dfs)
    zc = np.linspace(zmin, zmax, npts)
    U = [np.interp(zc, df["z"].values, df["u"].values) for df in dfs]
    V = [np.interp(zc, df["z"].values, df["v"].values) for df in dfs]
    return zc, U, V

# ============================================================
# MÉTRICAS: p (Richardson), GCI, NRMSE
# ============================================================

def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    rng = np.max(y_true) - np.min(y_true)
    if rng <= 0:
        return np.nan
    return np.sqrt(np.mean((y_pred - y_true)**2)) / (rng + 1e-12)

def richardson_p_phi(phi_f, phi_m, phi_c, r):
    """Orden aparente p (punto a punto) y extrapolación de Richardson."""
    phi_f = np.asarray(phi_f); phi_m = np.asarray(phi_m); phi_c = np.asarray(phi_c)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_local = np.log(np.abs((phi_c - phi_m) / (phi_m - phi_f) + 1e-30)) / np.log(r + 1e-30)
    denom = np.power(r, p_local) - 1.0
    denom[np.abs(denom) < 1e-12] = np.nan
    phi_ext = phi_f + (phi_f - phi_m) / denom
    return p_local, phi_ext

def gci_pair(phi_coarse, phi_fine, r, p_eff, Fs=1.25):
    """GCI (Roache) entre par de mallas (coarse vs fine) con p efectivo escalar."""
    phi_c = np.asarray(phi_coarse); phi_f = np.asarray(phi_fine)
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.abs((phi_c - phi_f) / (phi_f + 1e-30))
        gci_val = Fs * eps / (np.power(r, p_eff) - 1.0)
    return gci_val

# ============================================================
# PLOTEO
# ============================================================

def plot_profile(z, curves: Dict[str, np.ndarray], title: str, xlabel: str, outfile: Path):
    plt.figure()
    for lab, arr in curves.items():
        plt.plot(arr, z, label=lab)
    plt.gca().invert_yaxis()  # si z=0 es pared/fondo
    plt.xlabel(xlabel); plt.ylabel("z [m]")
    plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=170); plt.close()

# ============================================================
# PIPELINE PRINCIPAL (SIN CLASES)
# ============================================================

def analyze_three_meshes_txt(
    base_mesh_name: str,
    size_triplet: Tuple[int, int, int],
    h_triplet: Tuple[float, float, float],
    x_targets: List[float],
    metric: str = "u",
    smooth_last_n: int = 5,
    out_root: Path = Path("out_from_txt"),
    exp_paths: Optional[Dict[float, Path]] = None
):
    """
    base_mesh_name: p.ej. 'HexSweep' o 'Tet' (carpeta bajo CFD_Profiles/<mesh>/SizeN/)
    size_triplet: (fine, medium, coarse) p.ej. (1,2,3)
    h_triplet: (h_fine, h_med, h_coarse)
    x_targets: lista de valores x EXACTOS como en nombre de archivo 'perfiles_x={x:.2f}.txt'
    metric: 'u' (ux) o 'v' (uz)
    exp_paths: opcional { x_value: ruta_csv/txt experimento } (formato: CSV con columnas z,phi o TXT igual a CFD)
    """
    out_root.mkdir(parents=True, exist_ok=True)
    size_f, size_m, size_c = size_triplet
    h_f, h_m, h_c = h_triplet
    r12, r23 = h_m/h_f, h_c/h_m
    print(f"[INFO] r12={r12:.3f}, r23={r23:.3f}")

    for xval in x_targets:
        # Rutas a tus TXT (exactamente como los genera CFDtransformer)
        txt_f = Path(f"CFD_Profiles/{base_mesh_name}/Size{size_f}/perfiles_x={xval:.2f}.txt")
        txt_m = Path(f"CFD_Profiles/{base_mesh_name}/Size{size_m}/perfiles_x={xval:.2f}.txt")
        txt_c = Path(f"CFD_Profiles/{base_mesh_name}/Size{size_c}/perfiles_x={xval:.2f}.txt")
        for p in (txt_f, txt_m, txt_c):
            if not p.exists():
                raise FileNotFoundError(f"No existe: {p}")

        # 1) Cargar bloques y elegir perfil representativo
        prof_f = pick_profile(load_blocks_from_txt(txt_f), mode="mean_last_n", last_n=smooth_last_n)
        prof_m = pick_profile(load_blocks_from_txt(txt_m), mode="mean_last_n", last_n=smooth_last_n)
        prof_c = pick_profile(load_blocks_from_txt(txt_c), mode="mean_last_n", last_n=smooth_last_n)

        # 2) Interpolar a z común
        zc, U_list, V_list = align_to_common_z([prof_f, prof_m, prof_c], npts=700)
        PHI = U_list if metric == "u" else V_list
        phi_f, phi_m, phi_c = PHI[0], PHI[1], PHI[2]

        # 3) Richardson + GCI
        p_local, phi_ext = richardson_p_phi(phi_f, phi_m, phi_c, r12)
        p_eff = np.nanmedian(p_local)
        gci_21 = gci_pair(phi_m, phi_f, r12, p_eff)
        gci_32 = gci_pair(phi_c, phi_m, r23, p_eff)

        # 4) Métricas globales
        L2_mf = np.nanmean(np.abs(phi_m - phi_f)) / (np.nanmean(np.abs(phi_f)) + 1e-12)
        L2_cm = np.nanmean(np.abs(phi_c - phi_m)) / (np.nanmean(np.abs(phi_m)) + 1e-12)

        # 5) Experimento (opcional) -> NRMSE
        phi_exp_interp = None
        nrmse_f = nrmse_m = nrmse_c = np.nan
        if exp_paths and xval in exp_paths and Path(exp_paths[xval]).exists():
            exp_path = Path(exp_paths[xval])
            if exp_path.suffix.lower() == ".csv":
                exp_df = pd.read_csv(exp_path)
                # se esperan columnas: z, phi  (phi= u o v según 'metric')
                exp_df = exp_df.dropna().sort_values("z")
                phi_exp_interp = np.interp(zc, exp_df["z"].values, exp_df["phi"].values)
            else:
                # TXT con mismo formato que CFDtransformer (bloques). Tomamos promedio últimos N.
                prof_exp = pick_profile(load_blocks_from_txt(exp_path), mode="mean_last_n", last_n=smooth_last_n)
                phi_raw = prof_exp["u"].values if metric == "u" else prof_exp["v"].values
                phi_exp_interp = np.interp(zc, prof_exp["z"].values, phi_raw)
            nrmse_f = nrmse(phi_exp_interp, phi_f)
            nrmse_m = nrmse(phi_exp_interp, phi_m)
            nrmse_c = nrmse(phi_exp_interp, phi_c)

        # 6) Salidas
        outdir = out_root / f"{base_mesh_name}_x{str(xval).replace('.','p')}_{metric}"
        outdir.mkdir(parents=True, exist_ok=True)

        # Gráficos
        plot_profile(
            zc,
            {f"{metric} fine(S{size_f})": phi_f,
             f"{metric} med(S{size_m})":  phi_m,
             f"{metric} coarse(S{size_c})": phi_c,
             "extrap.": phi_ext},
            title=f"{base_mesh_name} – x={xval:.2f} – {metric}(z)  p≈{p_eff:.2f}",
            xlabel=f"{metric} [m/s]",
            outfile=outdir / f"profiles_{metric}.png"
        )
        plot_profile(
            zc,
            {"GCI med/fine": gci_21, "GCI coarse/med": gci_32},
            title=f"{base_mesh_name} – x={xval:.2f} – GCI (p≈{p_eff:.2f})",
            xlabel="GCI [-]",
            outfile=outdir / "gci.png"
        )

        # CSV resumen por punto x
        df_sum = pd.DataFrame({
            "z": zc,
            f"{metric}_fine":  phi_f,
            f"{metric}_med":   phi_m,
            f"{metric}_coarse":phi_c,
            f"{metric}_extrap":phi_ext,
            "p_local": p_local,
            "GCI_21":  gci_21,
            "GCI_32":  gci_32,
        })
        if phi_exp_interp is not None:
            df_sum["phi_exp_interp"] = phi_exp_interp
        df_sum.to_csv(outdir / f"summary_{metric}.csv", index=False)

        # Log
        print(f"[x={xval:.2f}] p≈{p_eff:.2f}  L2(med,fine)={L2_mf:.2%}  L2(coarse,med)={L2_cm:.2%} "
              f"{'(NRMSE fine/med/coarse = ' + ', '.join(f'{v:.2%}' for v in [nrmse_f,nrmse_m,nrmse_c]) + ')' if phi_exp_interp is not None else ''}")

# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    # 1) Tipo de malla/carpetas bajo CFD_Profiles/<mesh>/SizeN/
    BASE_MESH = "HexSweep"        # cámbialo por el que estés usando

    # 2) Las tres mallas (Size1, Size2, Size3 por ejemplo) y sus h característicos
    SIZE_TRIPLET = (3, 4, 5)      # fine, medium, coarse
    H_TRIPLET    = (0.003, 0.004, 0.005)  # ajusta a tus Δ o h

    # 3) Lista de x (deben coincidir con el nombre 'perfiles_x={x:.2f}.txt' que generó tu script)
    X_TARGETS = [0.10, 0.11, 0.12, 0.13]     # edita según lo que exportaste

    # 4) Métrica a analizar: 'u' (ux) o 'v' (uz)
    METRIC = "u"

    # 5) (Opcional) rutas de experimento por x; CSV (z,phi) o TXT con el MISMO formato CFD
    EXP_PATHS = {
        # 0.12: Path("exp/u_profile_x0p12.csv"),  # ej. CSV con columnas: z,phi
        # 0.29: Path("exp/perfiles_x=0.29.txt"),  # ej. TXT con bloques (mismo formato CFD)
    }

    analyze_three_meshes_txt(
        base_mesh_name=BASE_MESH,
        size_triplet=SIZE_TRIPLET,
        h_triplet=H_TRIPLET,
        x_targets=X_TARGETS,
        metric=METRIC,
        smooth_last_n=5,
        out_root=Path("out_from_txt"),
        exp_paths=EXP_PATHS
    )
