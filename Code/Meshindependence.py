# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================
# Lectura de tus .txt (CFDtransformer)
# ============================

def load_blocks_from_txt(path: Path):
    """Devuelve lista de DataFrames (x,z,u,v) ordenados por z. Bloques separados por línea en blanco."""
    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    if lines and lines[0].lower().startswith("x[m]"):  # quita encabezado
        lines = lines[1:]
    curr = []
    for ln in lines + [""]:
        if ln == "":
            if curr:
                data = []
                for row in curr:
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) < 4: continue
                    try:
                        x = float(parts[0]); z = float(parts[1])
                        u = float(parts[2]); v = float(parts[3])
                        data.append((x, z, u, v))
                    except ValueError:
                        pass
                if data:
                    df = pd.DataFrame(data, columns=["x","z","u","v"]).sort_values("z").reset_index(drop=True)
                    blocks.append(df)
            curr = []
        else:
            curr.append(ln)
    return blocks

def pick_profile(blocks, mode="mean_last_n", last_n=5):
    """Devuelve un DataFrame (x,z,u,v) representativo: último bloque o promedio de los últimos N (interpolado)."""
    if not blocks:
        raise ValueError("TXT sin bloques.")
    if mode == "last" or len(blocks) == 1:
        return blocks[-1]
    use = blocks[-last_n:] if len(blocks) >= last_n else blocks
    zmin = max(b["z"].min() for b in use)
    zmax = min(b["z"].max() for b in use)
    zc = np.linspace(zmin, zmax, 600)
    U = [np.interp(zc, b["z"].values, b["u"].values) for b in use]
    V = [np.interp(zc, b["z"].values, b["v"].values) for b in use]
    u_mean = np.nanmean(np.vstack(U), axis=0)
    v_mean = np.nanmean(np.vstack(V), axis=0)
    return pd.DataFrame({"x": np.nan, "z": zc, "u": u_mean, "v": v_mean})

def align_to_common_z(dfs, npts=600):
    """Intersección de rangos + interpolación a z común; devuelve zc, U_list, V_list."""
    zmin = max(df["z"].min() for df in dfs)
    zmax = min(df["z"].max() for df in dfs)
    zc = np.linspace(zmin, zmax, npts)
    U = [np.interp(zc, df["z"].values, df["u"].values) for df in dfs]
    V = [np.interp(zc, df["z"].values, df["v"].values) for df in dfs]
    return zc, U, V

# ============================
# Métricas: p (Richardson), GCI, RMSE (Wang & Zhai), NRMSE (vs experimento)
# ============================

def richardson_p_phi(phi_f, phi_m, phi_c, r):
    """p local y phi extrapolada (Richardson)."""
    phi_f = np.asarray(phi_f); phi_m = np.asarray(phi_m); phi_c = np.asarray(phi_c)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_local = np.log(np.abs((phi_c - phi_m) / (phi_m - phi_f) + 1e-30)) / np.log(r + 1e-30)
    denom = np.power(r, p_local) - 1.0
    denom[np.abs(denom) < 1e-12] = np.nan
    phi_ext = phi_f + (phi_f - phi_m) / denom
    return p_local, phi_ext

def gci_pair(phi_coarse, phi_fine, r, p_eff, Fs=1.25):
    """GCI de Roache entre par coarse/fine con p efectivo escalar."""
    phi_c = np.asarray(phi_coarse); phi_f = np.asarray(phi_fine)
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.abs((phi_c - phi_f) / (phi_f + 1e-30))
        return Fs * eps / (np.power(r, p_eff) - 1.0)

def rmse_normalized(phi1, phi2):
    """
    RMSE normalizado (Wang & Zhai):
    ||phi1 - phi2||_2 / ||phi2||_2    ← índice práctico de diferencia global entre mallas adyacentes.
    """
    num = np.sqrt(np.nansum((np.asarray(phi1) - np.asarray(phi2))**2))
    den = np.sqrt(np.nansum((np.asarray(phi2))**2)) + 1e-30
    return num / den

def rmse_gci_like(phi1, phi2, r, p_scheme):
    """
    Variante 'GCI-like' basada en W&Z:  RMSE_norm / (r^p - 1)
    Útil si quieres reportar el índice con el factor del orden del esquema numérico (p) y la razón r.
    """
    return rmse_normalized(phi1, phi2) / (np.power(r, p_scheme) - 1.0 + 1e-30)

def nrmse_vs_exp(y_true, y_pred):
    """NRMSE respecto a experimento: RMSE / (max- min) de la serie experimental interpolada."""
    t = np.asarray(y_true); p = np.asarray(y_pred)
    m = np.isfinite(t) & np.isfinite(p)
    if not np.any(m): return np.nan
    rng = np.nanmax(t[m]) - np.nanmin(t[m])
    if rng <= 0: return np.nan
    return np.sqrt(np.nanmean((p[m] - t[m])**2)) / (rng + 1e-30)

# ============================
# Plot
# ============================

def plot_profile(z, curves, title, xlabel, outfile):
    plt.figure()
    for lab, arr in curves.items():
        plt.plot(arr, z, label=lab)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel); plt.ylabel("z [m]")
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=170); plt.close()

# ============================
# Pipeline principal (SIN CLASES)
# ============================

def analyze_three_meshes_txt(
    base_mesh,               # p.ej. 'HexSweep' (carpeta en CFD_Profiles/<base_mesh>/SizeN/)
    size_triplet,            # (fine, med, coarse) ej. (1,2,3)
    h_triplet,               # (h_f, h_m, h_c) tamaños característicos (misma métrica)
    x_targets,               # lista de x (deben coincidir con 'perfiles_x={x:.2f}.txt')
    metric="u",              # 'u' o 'v'
    scheme_order_p=2,        # orden del esquema (1=Upwind, 2=CDS/Hybrid~CDS, 3=QUICK); para rmse_gci_like
    smooth_last_n=5,         # promedio de últimos N bloques
    out_root=Path("out_WZ"),
    exp_paths=None,          # dict opcional {x: Path(CSV/TXT experimento)}; CSV con cols z,phi o TXT como CFD
    independence_threshold=0.10  # 10% según W&Z
):
    out_root.mkdir(parents=True, exist_ok=True)
    size_f, size_m, size_c = size_triplet
    h_f, h_m, h_c = h_triplet
    r12, r23 = h_m / h_f, h_c / h_m
    print(f"[INFO] r12={r12:.3f}, r23={r23:.3f}, esquema p={scheme_order_p}")

    for xval in x_targets:
        # Archivos TXT exactos como los genera tu transformador
        txt_f = Path(f"CFD_Profiles/{base_mesh}/Size{size_f}/perfiles_x={xval:.2f}.txt")
        txt_m = Path(f"CFD_Profiles/{base_mesh}/Size{size_m}/perfiles_x={xval:.2f}.txt")
        txt_c = Path(f"CFD_Profiles/{base_mesh}/Size{size_c}/perfiles_x={xval:.2f}.txt")
        for pth in (txt_f, txt_m, txt_c):
            if not pth.exists():
                raise FileNotFoundError(f"No existe: {pth}")

        # 1) Cargar y elegir perfil representativo
        prof_f = pick_profile(load_blocks_from_txt(txt_f), mode="mean_last_n", last_n=smooth_last_n)
        prof_m = pick_profile(load_blocks_from_txt(txt_m), mode="mean_last_n", last_n=smooth_last_n)
        prof_c = pick_profile(load_blocks_from_txt(txt_c), mode="mean_last_n", last_n=smooth_last_n)

        # 2) Interpolar a z común
        zc, U_list, V_list = align_to_common_z([prof_f, prof_m, prof_c], npts=600)
        PHI = U_list if metric == "u" else V_list
        phi_f, phi_m, phi_c = PHI[0], PHI[1], PHI[2]

        # 3) Richardson + GCI
        p_local, phi_ext = richardson_p_phi(phi_f, phi_m, phi_c, r12)
        p_eff = np.nanmedian(p_local)
        GCI_21 = gci_pair(phi_m, phi_f, r12, p_eff)
        GCI_32 = gci_pair(phi_c, phi_m, r23, p_eff)

        # 4) RMSE normalizado (W&Z) y variante GCI-like
        RMSE_norm_21 = rmse_normalized(phi_m, phi_f)
        RMSE_norm_32 = rmse_normalized(phi_c, phi_m)
        RMSE_gci_like_21 = rmse_gci_like(phi_m, phi_f, r12, scheme_order_p)
        RMSE_gci_like_32 = rmse_gci_like(phi_c, phi_m, r23, scheme_order_p)

        # 5) (Opcional) experimento -> NRMSE
        phi_exp_interp = None
        nrmse_f = nrmse_m = nrmse_c = np.nan
        if exp_paths and xval in exp_paths and Path(exp_paths[xval]).exists():
            epath = Path(exp_paths[xval])
            if epath.suffix.lower() == ".csv":
                edf = pd.read_csv(epath).dropna().sort_values("z")
                phi_exp_interp = np.interp(zc, edf["z"].values, edf["phi"].values)
            else:  # TXT con mismo formato
                exp_prof = pick_profile(load_blocks_from_txt(epath), mode="mean_last_n", last_n=smooth_last_n)
                raw = exp_prof[metric].values
                phi_exp_interp = np.interp(zc, exp_prof["z"].values, raw)
            nrmse_f = nrmse_vs_exp(phi_exp_interp, phi_f)
            nrmse_m = nrmse_vs_exp(phi_exp_interp, phi_m)
            nrmse_c = nrmse_vs_exp(phi_exp_interp, phi_c)

        # 6) Salidas (gráficos + CSV)
        outdir = out_root / f"{base_mesh}_x{str(xval).replace('.','p')}_{metric}"
        outdir.mkdir(parents=True, exist_ok=True)

        plot_profile(
            zc,
            {f"{metric} fine(S{size_f})": phi_f, f"{metric} med(S{size_m})": phi_m,
             f"{metric} coarse(S{size_c})": phi_c, "extrap.": phi_ext},
            title=f"{base_mesh} – x={xval:.2f} – {metric}(z)  p≈{p_eff:.2f}",
            xlabel=f"{metric} [SI]", outfile=outdir / f"profiles_{metric}.png"
        )
        plot_profile(
            zc,
            {"GCI med/fine": GCI_21, "GCI coarse/med": GCI_32},
            title=f"{base_mesh} – x={xval:.2f} – GCI (p≈{p_eff:.2f})",
            xlabel="GCI [-]", outfile=outdir / "gci.png"
        )

        df = pd.DataFrame({
            "z": zc,
            f"{metric}_fine": phi_f, f"{metric}_med": phi_m, f"{metric}_coarse": phi_c,
            f"{metric}_extrap": phi_ext, "p_local": p_local,
            "GCI_21": GCI_21, "GCI_32": GCI_32,
            "RMSEnorm_21": RMSE_norm_21, "RMSEnorm_32": RMSE_norm_32,
            "RMSEgci_like_21": RMSE_gci_like_21, "RMSEgci_like_32": RMSE_gci_like_32
        })
        if phi_exp_interp is not None:
            df["phi_exp_interp"] = phi_exp_interp
        df.to_csv(outdir / f"summary_{metric}.csv", index=False)

        # 7) Logs + decisión de independencia (W&Z ~ 10%)
        indep_21 = RMSE_norm_21 <= independence_threshold
        print(f"[x={xval:.2f}] p≈{p_eff:.2f} | RMSE_norm(21)={RMSE_norm_21:.2%} "
              f"| RMSE_norm(32)={RMSE_norm_32:.2%} "
              f"| indep(21)={'OK' if indep_21 else 'NO'}  (umbral={independence_threshold:.0%})")
        if phi_exp_interp is not None:
            print(f"   NRMSE vs exp: fine={nrmse_f:.2%}, med={nrmse_m:.2%}, coarse={nrmse_c:.2%}")

# ============================
# Ejemplo de uso
# ============================

if __name__ == "__main__":
    BASE_MESH = "HexSweep"           # carpeta bajo CFD_Profiles/<BASE_MESH>/SizeN/
    SIZE_TRIPLET = (3, 4, 5)          # etiquetas 'Size*' de tus tres mallas
    H_TRIPLET    = (0.003, 0.004, 0.005)  # tamaños característicos h (misma métrica)
    X_TARGETS    = [0.10, 0.11, 0.12, 0.13]  # deben coincidir con los nombres de archivo (2 decimales)
    METRIC       = "u"                # 'u' (ux) o 'v' (uz)
    SCHEME_P     = 2                  # orden del esquema usado (1, 2 o 3) para el indicador “GCI-like”

    # (Opcional) mapas de experimento por x (CSV: cols z,phi | TXT: mismo formato CFD)
    EXP_PATHS = {
        # 0.12: Path("exp/u_profile_x0p12.csv"),
        # 0.29: Path("exp/perfiles_x=0.29.txt"),
    }

    analyze_three_meshes_txt(
        base_mesh=BASE_MESH,
        size_triplet=SIZE_TRIPLET,
        h_triplet=H_TRIPLET,
        x_targets=X_TARGETS,
        metric=METRIC,
        scheme_order_p=SCHEME_P,
        smooth_last_n=5,
        out_root=Path("out_WZ"),
        exp_paths=EXP_PATHS,
        independence_threshold=0.10  # ~10% como en W&Z
    )
