# -*- coding: utf-8 -*-
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN EDITABLE
# =========================

BASE = Path("CFD_Solution")   # raíz de tus resultados Fluent
MESH_NAME = "HexSweep"        # CFD_Solution/<MESH_NAME>/SizeN/...
SUBDIR = "CarbopolSolution"   # subcarpeta de los ASCII de Fluent

SIZES = (3, 4, 5)             # (fine, medium, coarse) -> Size3,4,5
H_TRIPLET = (0.003, 0.004, 0.005)  # h_f, h_m, h_c (ajusta a tus mallas)

PREFIX_FMT = "{mesh}{size}-"  # ej. "HexSweep3-0001"

COMPONENTS = ("u", "v")           # ("u","v") si también quieres v = uz

# Grid común para análisis espacial (baja NX,NZ si quieres menos puntos)
NX, NZ = 80, 80

# Agrupación temporal
FILES_PER_GROUP = 50         # archivos por tanda
MAX_GROUPS      = None        # None -> usa todas las tandas posibles

SCHEME_ORDER_P = 2            # orden esquema numérico (para GCI-like)

# Criterios (Wang & Zhai + Lee)
WZ_THRESHOLD      = 0.10      # 10% en RMSE_norm
LEE_CV_THRESHOLD  = 0.10      # 10% en CV(RMSE)
LEE_R2_THRESHOLD  = 0.95      # mínimo R²

OUT = Path("out_groups_fullfield")
OUT.mkdir(parents=True, exist_ok=True)

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
# UTILIDADES DE LECTURA / AGRUPACIÓN
# =========================

def list_step_files(dir_path: Path, prefix: str):
    cands = [p for p in dir_path.iterdir() if p.is_file() and p.name.startswith(prefix)]
    if not cands:
        raise FileNotFoundError(f"No hay archivos con prefijo '{prefix}' en {dir_path}")
    def _num(p: Path):
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", p.name)
        return int(m.group(1)) if m else 10**12
    cands.sort(key=_num)
    return cands

def grouped_means(mesh_size: int, files_per_group: int, max_groups=None):
    """
    Devuelve lista [df_group0, df_group1, ...] para una malla (SizeN),
    cada df es promedio temporal de files_per_group archivos.
    """
    folder = BASE / MESH_NAME / f"Size{mesh_size}" / SUBDIR
    prefix = PREFIX_FMT.format(mesh=MESH_NAME, size=mesh_size)
    files = list_step_files(folder, prefix)

    if files_per_group <= 0:
        raise ValueError("files_per_group debe ser > 0")

    groups = []
    n_files = len(files)
    n_groups_possible = n_files // files_per_group
    if n_groups_possible == 0:
        raise RuntimeError(f"No alcanza ni para una tanda: hay {n_files} archivos, "
                           f"files_per_group={files_per_group}")

    if max_groups is None:
        n_groups = n_groups_possible
    else:
        n_groups = min(n_groups_possible, max_groups)

    for g in range(n_groups):
        start = g * files_per_group
        end   = start + files_per_group
        batch = files[start:end]
        chunks = []
        for fp in batch:
            df = pd.read_csv(fp, header=0, skipinitialspace=True)
            df = standardize_columns(df)
            chunks.append(df[["x", "z", "ux", "uz"]].dropna())
        big = pd.concat(chunks, axis=0, ignore_index=True)
        grouped = big.groupby(["x", "z"], as_index=False).mean()
        groups.append(grouped)

    return groups

# =========================
# GRID COMÚN y BINNING
# =========================

def common_grid(ext1, ext2, ext3, nx, nz):
    x_min = max(ext1[0], ext2[0], ext3[0])
    x_max = min(ext1[1], ext2[1], ext3[1])
    z_min = max(ext1[2], ext2[2], ext3[2])
    z_max = min(ext1[3], ext2[3], ext3[3])
    if not (x_min < x_max and z_min < z_max):
        raise RuntimeError("Los dominios no se intersectan en X,Z.")
    xg = np.linspace(x_min, x_max, nx)
    zg = np.linspace(z_min, z_max, nz)
    return xg, zg

def bin_to_grid(df: pd.DataFrame, xg: np.ndarray, zg: np.ndarray, comp: str):
    xi = np.clip(np.searchsorted(xg, df["x"].values) - 1, 0, len(xg)-2)
    zi = np.clip(np.searchsorted(zg, df["z"].values) - 1, 0, len(zg)-2)
    val = df["ux"].values if comp == "u" else df["uz"].values

    acc = np.zeros((len(zg)-1, len(xg)-1), dtype=float)
    cnt = np.zeros_like(acc)
    for k in range(val.size):
        acc[zi[k], xi[k]] += val[k]
        cnt[zi[k], xi[k]] += 1.0
    with np.errstate(invalid='ignore'):
        fld = acc / cnt
    msk = cnt > 0
    return fld, msk

# =========================
# MÉTRICAS (W&Z, Lee, GCI-like)
# =========================

def rmse_norm_field(A, B, M=None):
    if M is None:
        M = np.isfinite(A) & np.isfinite(B)
    if not np.any(M):
        return np.nan
    num = np.sqrt(np.nansum(((A - B)[M])**2))
    den = np.sqrt(np.nansum((B[M])**2)) + 1e-30
    return num / den

def cv_rmse_field(A, B, M=None):
    if M is None:
        M = np.isfinite(A) & np.isfinite(B)
    if not np.any(M):
        return np.nan
    err = (A - B)[M]
    rmse = np.sqrt(np.nanmean(err**2))
    mean_abs_B = np.nanmean(np.abs(B[M])) + 1e-30
    return rmse / mean_abs_B

def r2_field(A, B, M=None):
    if M is None:
        M = np.isfinite(A) & np.isfinite(B)
    if not np.any(M):
        return np.nan
    y = B[M].ravel()
    y_hat = A[M].ravel()
    y_mean = np.nanmean(y)
    ss_tot = np.nansum((y - y_mean)**2)
    ss_res = np.nansum((y - y_hat)**2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / (ss_tot + 1e-30)

def gci_like_from_rmse(rmse_norm, r, p):
    return rmse_norm / (np.power(r, p) - 1.0 + 1e-30)

# =========================
# PROFILES U(x) y U(z)
# =========================

def compute_profiles_ux(F):
    """
    F: campo (NZ-1 x NX-1) de u (ux) ya en el grid común.
    Devuelve Ux_vs_x (promedio en z) y Ux_vs_z (promedio en x).
    """
    Ux_vs_x = np.nanmean(F, axis=0)  # promedio en z -> función de x
    Ux_vs_z = np.nanmean(F, axis=1)  # promedio en x -> función de z
    return Ux_vs_x, Ux_vs_z

# =========================
# MAIN
# =========================

def main():
    # 1) campos agrupados por malla
    groups_f = grouped_means(SIZES[0], FILES_PER_GROUP, MAX_GROUPS)
    groups_m = grouped_means(SIZES[1], FILES_PER_GROUP, MAX_GROUPS)
    groups_c = grouped_means(SIZES[2], FILES_PER_GROUP, MAX_GROUPS)

    n_groups = min(len(groups_f), len(groups_m), len(groups_c))
    print(f"Usando {n_groups} grupos (tandas)")

    # 2) Extentos globales (grupo 0)
    g0_f, g0_m, g0_c = groups_f[0], groups_m[0], groups_c[0]
    ext_f = (g0_f["x"].min(), g0_f["x"].max(), g0_f["z"].min(), g0_f["z"].max())
    ext_m = (g0_m["x"].min(), g0_m["x"].max(), g0_m["z"].min(), g0_m["z"].max())
    ext_c = (g0_c["x"].min(), g0_c["x"].max(), g0_c["z"].min(), g0_c["z"].max())

    xg, zg = common_grid(ext_f, ext_m, ext_c, NX, NZ)
    xc = 0.5*(xg[:-1] + xg[1:])
    zc = 0.5*(zg[:-1] + zg[1:])

    r12 = H_TRIPLET[1] / H_TRIPLET[0]
    r23 = H_TRIPLET[2] / H_TRIPLET[1]

    all_metrics = []
    # para residual sintético: guardamos campos por grupo/malla/componente
    fields_f = {comp: [] for comp in COMPONENTS}
    fields_m = {comp: [] for comp in COMPONENTS}
    fields_c = {comp: [] for comp in COMPONENTS}

    for g in range(n_groups):
        df_f = groups_f[g]
        df_m = groups_m[g]
        df_c = groups_c[g]

        for comp in COMPONENTS:
            # 3) campos en grid común
            F_f, M_f = bin_to_grid(df_f, xg, zg, comp)
            F_m, M_m = bin_to_grid(df_m, xg, zg, comp)
            F_c, M_c = bin_to_grid(df_c, xg, zg, comp)

            fields_f[comp].append((F_f, M_f))
            fields_m[comp].append((F_m, M_m))
            fields_c[comp].append((F_c, M_c))

            # máscaras vs malla fina (para comparar mallas)
            M21 = M_f & M_m
            M31 = M_f & M_c

            # 4) Métricas WZ + Lee
            rmse_norm_21 = rmse_norm_field(F_m, F_f, M21)
            rmse_norm_31 = rmse_norm_field(F_c, F_f, M31)
            gci_like_21 = gci_like_from_rmse(rmse_norm_21, r12, SCHEME_ORDER_P)
            gci_like_31 = gci_like_from_rmse(rmse_norm_31, r23, SCHEME_ORDER_P)

            indep_WZ_21 = rmse_norm_21 <= WZ_THRESHOLD
            indep_WZ_31 = rmse_norm_31 <= WZ_THRESHOLD

            cv_rmse_21 = cv_rmse_field(F_m, F_f, M21)
            cv_rmse_31 = cv_rmse_field(F_c, F_f, M31)
            r2_21 = r2_field(F_m, F_f, M21)
            r2_31 = r2_field(F_c, F_f, M31)

            indep_Lee_21 = (cv_rmse_21 <= LEE_CV_THRESHOLD) and (r2_21 >= LEE_R2_THRESHOLD)
            indep_Lee_31 = (cv_rmse_31 <= LEE_CV_THRESHOLD) and (r2_31 >= LEE_R2_THRESHOLD)

            # 5) Perfiles U(x) y U(z) para U (si comp=="u")
            Ux_f_x, Ux_f_z = compute_profiles_ux(F_f)
            Ux_m_x, Ux_m_z = compute_profiles_ux(F_m)
            Ux_c_x, Ux_c_z = compute_profiles_ux(F_c)

            df_x = pd.DataFrame({
                "x": xc,
                f"{comp}_fine":   Ux_f_x,
                f"{comp}_medium": Ux_m_x,
                f"{comp}_coarse": Ux_c_x,
            })
            df_z = pd.DataFrame({
                "z": zc,
                f"{comp}_fine":   Ux_f_z,
                f"{comp}_medium": Ux_m_z,
                f"{comp}_coarse": Ux_c_z,
            })
            df_x.to_csv(OUT / f"group{g:02d}_{comp}_profile_x.csv", index=False)
            df_z.to_csv(OUT / f"group{g:02d}_{comp}_profile_z.csv", index=False)

            # 6) Guardar métricas por grupo (entre mallas)
            all_metrics.append({
                "group": g,
                "component": comp,
                "r12": r12, "r23": r23,
                "rmse_norm_21": rmse_norm_21,
                "rmse_norm_31": rmse_norm_31,
                "gci_like_21": gci_like_21,
                "gci_like_31": gci_like_31,
                "cv_rmse_21": cv_rmse_21,
                "cv_rmse_31": cv_rmse_31,
                "r2_21": r2_21,
                "r2_31": r2_31,
                "indep_WZ_21": indep_WZ_21,
                "indep_WZ_31": indep_WZ_31,
                "indep_Lee_21": indep_Lee_21,
                "indep_Lee_31": indep_Lee_31,
            })

    # 7) CSV maestro de métricas entre mallas
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(OUT / "groups_metrics.csv", index=False)

    # 8) Residual sintético por grupo y malla (en TODO el dominio común)
    residual_records = []
    for comp in COMPONENTS:
        for mesh_label, fields_dict in (("fine", fields_f),
                                        ("medium", fields_m),
                                        ("coarse", fields_c)):
            prev_F = None
            prev_M = None
            for g in range(n_groups):
                F, M = fields_dict[comp][g]
                if prev_F is None:
                    res_val = np.nan
                else:
                    # RMSE sobre intersección de celdas válidas en ambos grupos
                    Mboth = M & prev_M
                    if not np.any(Mboth):
                        res_val = np.nan
                    else:
                        diff = (F - prev_F)[Mboth]
                        res_val = np.sqrt(np.nanmean(diff**2))
                residual_records.append({
                    "group": g,
                    "component": comp,
                    "mesh": mesh_label,
                    "residual_like": res_val,
                })
                prev_F, prev_M = F, M

    df_res = pd.DataFrame(residual_records)
    df_res.to_csv(OUT / "groups_residual_like.csv", index=False)

    # 9) Log resumido
    print("=== MÉTRICAS POR TANDA (grupo) ===")
    for comp in COMPONENTS:
        sub = df_metrics[df_metrics["component"] == comp]
        print(f"\n[{comp}]")
        for _, r in sub.iterrows():
            g = int(r["group"])
            print(f" group {g:02d} | "
                  f"RMSEnorm21={r['rmse_norm_21']:.2%}, CV21={r['cv_rmse_21']:.2%}, R2_21={r['r2_21']:.3f} "
                  f"| indep_Lee_21={'OK' if r['indep_Lee_21'] else 'NO'}")

    print("\n=== Residual sintético guardado en groups_residual_like.csv ===")

if __name__ == "__main__":
    main()
