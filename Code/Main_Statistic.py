import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import TXTprofiles_reader as rdr


# ===================== Combinations =====================
meshes = ["HexDominant", "HexPrimeMesh", "HexSweep", "Tetra"]
start = 0.10 #m
finish = 0.16 #m
diff = int((finish-start)*100+1) #*2 para hacerlo cada 5 mm
x_profiles = np.linspace(start, finish, num=diff)  #txt files

OUTDIR_DEFAULT  = Path("Results/SALIDAS_PIV_vs_CFD")
dt_PIV: float = 0.1
dt_CFD: float = 0.01
MAX_TIME_S: Optional[float] = 10  # None para usar todos

dt_target = 0.5 #s

blocks_PIV = int(dt_target/dt_PIV)  #5  blocks averaged
blocks_CFD = int(dt_target/dt_CFD)  #50 blocks averaged



for n in range(1, 4):
    for mesh in meshes:
        for size in range(1, 5):
            PIV_1 = rdr.time_step_updater(rdr.block_builder(Path(f"PIV_Profiles/PIV_{n}") / f"perfiles_x={x_profiles[0]:.2f}.txt" ), blocks_PIV)
            CFD_1 = rdr.time_step_updater(rdr.block_builder(Path(f'CFD_Profiles/{mesh}/Size{str(size)}') / f"perfiles_x={x_profiles[0]:.2f}.txt" ), blocks_CFD)
            for i, X_TARGET  in enumerate(x_profiles[1:], 1):
                print(X_TARGET, i)
                PIV_0       = PIV_1
                CFD_0       = CFD_1
                PIV_1 = rdr.time_step_updater(rdr.block_builder(Path(f"PIV_Profiles/PIV_{n}") / f"perfiles_x={X_TARGET:.2f}.txt"), blocks_PIV)
                CFD_1 = rdr.time_step_updater(rdr.block_builder(Path(f'CFD_Profiles/{mesh}/Size{str(size)}') / f"perfiles_x={X_TARGET:.2f}.txt"), blocks_CFD)

            print(f'PIV_{n} | CFD_{mesh}-{size}')

"""
PIV_1 = rdr.time_step_updater(rdr.block_builder(Path("PIV_Profiles") / f"perfiles_x={x_profiles[0]:.2f}.txt" ), blocks_PIV)
CFD_1 = rdr.time_step_updater(rdr.block_builder(Path("CFD_Profiles") / f"perfiles_x={x_profiles[0]:.2f}.txt" ), blocks_CFD)
print(PIV_1[0])
print(CFD_1[0])
for i, X_TARGET  in enumerate(x_profiles[1:], 1):
    print(X_TARGET, i)
    continue ####################################
    PIV_0       = PIV_1
    CFD_0       = CFD_1
    PIV_1 = rdr.block_builder(Path("PIV_Profiles") / f"perfiles_x={X_TARGET:.2f}.txt")
    CFD_1 = rdr.block_builder(Path("CFD_Profiles") / f"perfiles_x={X_TARGET:.2f}.txt")
    PIV_1_dt = rdr.time_step_updater(PIV_1, blocks_PIV)
    CFD_1_dt = rdr.time_step_updater(CFD_1, blocks_CFD)
"""

carbopol_02 = {"t_0": 56.91, "K": 3.67, "n": 0.66, "gdot_crit": 5.0}

carbopol_05 = {"t_0": 125.5, "K": 6.61, "n": 0.68, "gdot_crit": 5.0}


# ---------- Rheology: Herschel–Bulkley ----------
def herschel_bulkley(gdot, tau0, K, n, gdot_crit=5.0):
    """
    Viscosidad aparente μ_app(|γ̇|) para τ = τ_0 + K*|γ̇|^n.
    Regulariza con |γ̇|_eff = max(|γ̇|, gdot_crit) para evitar divisiones por ~0.
    Devuelve μ_app = τ/|γ̇|_eff = τ_0/|γ̇|_eff + K*|γ̇|_eff^(n-1).
    """
    g = np.asarray(gdot, dtype=float)
    g_eff = np.maximum(np.abs(g), float(gdot_crit))
    mu_app = (tau0 / g_eff) + K * (g_eff ** (n - 1.0))
    tau = np.sign(g) * (float(tau0) + float(K) * g_eff**float(n))
    return tau

# --- utilidades ---
def _prep(df):
    return df.sort_values("y").drop_duplicates("y").reset_index(drop=True)

def _cumtrapz_integrate(y, f, u0=0.0):
    y = np.asarray(y, float); f = np.asarray(f, float)
    out = np.zeros_like(y)
    dy = np.diff(y)
    out[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * dy)
    return u0 + out

# --- remuestreo reología-consciente (media) ---
def resample_mean_rheo(df_src, y_target, wall_zero=True):
    s = _prep(df_src)
    y_src  = s["y"].to_numpy()
    u_src  = s["u_mean"].to_numpy()
    # 1) gradiente en malla original
    du_dy_src = np.gradient(u_src, y_src, edge_order=2)
    # 2) limitar a solape y remuestrear gradiente
    y_tgt = np.asarray(y_target, float)
    ymin, ymax = y_src.min(), y_src.max()
    mask = (y_tgt >= ymin) & (y_tgt <= ymax)
    y_common = y_tgt[mask]
    gdot = np.interp(y_common, y_src, du_dy_src)
    # 3) integrar para recuperar u en la malla objetivo
    u0 = 0.0 if wall_zero else float(np.interp(y_common[0], y_src, u_src))
    u_tgt = _cumtrapz_integrate(y_common, gdot, u0=u0)
    return pd.DataFrame({"y": y_common, "u": u_tgt})

# --- remuestreo de desviaciones (banda ±σ) ---
def resample_std_linear(df_src, y_target):
    s = _prep(df_src)
    if "u_std" not in s.columns:
        return pd.DataFrame({"y": [], "u_std": []})
    y_src  = s["y"].to_numpy()
    u_std  = s["u_std"].to_numpy()
    y_tgt  = np.asarray(y_target, float)
    ymin, ymax = y_src.min(), y_src.max()
    mask = (y_tgt >= ymin) & (y_tgt <= ymax)
    y_common = y_tgt[mask]
    std_tgt = np.interp(y_common, y_src, u_std)
    return pd.DataFrame({"y": y_common, "u_std": std_tgt})

# --- alinear manteniendo la malla más densa, recortada al solape ---
def align_keep_denser_with_bands(CFD_block, PIV_block, wall_zero=True):
    CFD0, PIV0 = _prep(CFD_block), _prep(PIV_block)
    
    if len(CFD0) >= len(PIV0):
        y_dense = CFD0["y"].to_numpy()
        y_min, y_max = PIV0["y"].min(), PIV0["y"].max()
        y_dense_clip = y_dense[(y_dense >= y_min) & (y_dense <= y_max)]

        # A: mantener su media (u_mean) en la malla densa (recortada)
        CFD_keep = CFD0.loc[CFD0["y"].isin(y_dense_clip)].copy().reset_index(drop=True)
        CFD_mean = CFD_keep.rename(columns={"u_mean":"u"})[["y","u"]]
        # su banda:
        CFD_std  = resample_std_linear(CFD0, CFD_mean["y"].to_numpy())

        # B: remuestrear reología-consciente (media) a la malla de A
        PIV_mean = resample_mean_rheo(PIV0, CFD_mean["y"].to_numpy(), wall_zero=wall_zero)
        PIV_std  = resample_std_linear(PIV0, CFD_mean["y"].to_numpy())

    else:
        y_dense = PIV0["y"].to_numpy()
        y_min, y_max = CFD0["y"].min(), CFD0["y"].max()
        y_dense_clip = y_dense[(y_dense >= y_min) & (y_dense <= y_max)]

        PIV_keep = PIV0.loc[PIV0["y"].isin(y_dense_clip)].copy().reset_index(drop=True)
        PIV_mean = PIV_keep.rename(columns={"u_mean":"u"})[["y","u"]]
        PIV_std  = resample_std_linear(PIV0, PIV_mean["y"].to_numpy())

        CFD_mean = resample_mean_rheo(CFD0, PIV_mean["y"].to_numpy(), wall_zero=wall_zero)
        CFD_std  = resample_std_linear(CFD0, PIV_mean["y"].to_numpy())

    # merge medias + bandas con la misma y
    CFD_ = CFD_mean.merge(CFD_std, on="y", how="left")
    PIV_ = PIV_mean.merge(PIV_std, on="y", how="left")
    return CFD_, PIV_  # columnas: y, u, u_std

# --- plot helper: media ±σ para ambos ---
def plot_profiles_with_bands(A, B, labelA="PIV", labelB="CFD", timestep = None, title=None, N_A=None, N_B=None):
    plt.figure()
    lblA = f"{labelA} ū(y)" + (f" (N={N_A})" if N_A else "")
    lblB = f"{labelB} ū(y)" + (f" (N={N_B})" if N_B else "")
    # líneas
    plt.plot(A["u"], A["y"], label=lblA)
    plt.plot(B["u"], B["y"], label=lblB)
    # bandas
    if "u_std" in A:
        plt.fill_betweenx(A["y"], A["u"]-A["u_std"], A["u"]+A["u_std"], alpha=0.25, label=f"{labelA}: ±σ")
    if "u_std" in B:
        plt.fill_betweenx(B["y"], B["u"]-B["u_std"], B["u"]+B["u_std"], alpha=0.25, label=f"{labelB}: ±σ")
    plt.gca().invert_yaxis()  # opcional, según tu convención de y
    plt.xlabel("u [m/s]"); plt.ylabel("y [m]")
    if title: plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(f"CFDvsPIV_Sweep3_{timestep}.png", dpi=300, bbox_inches="tight")#; plt.show()

def profiles_hb(df_aligned, hb):
    """Con y,u (y opcional v), calcula γ̇ y τ(H-B)."""
    d = _prep(df_aligned)
    y, u = d["y"].to_numpy(), d["u"].to_numpy()
    du_dy = np.gradient(u, y, edge_order=2)
    tau_u = herschel_bulkley(du_dy, hb["t_0"], hb["K"], hb["n"], hb["gdot_crit"])
    profiles = pd.DataFrame({"y": y, "u": u, "du_dy": du_dy, "tau_u": tau_u})
    return profiles




# A_block y B_block: DataFrames de UN bloque agregado (cada uno con: y, u_mean, u_std[, v_mean, v_std])
# Tip: si aún no tenías u_std/v_std, calcula en tu etapa de agregación (ddof=1).

timestep = 0
for timestep in range(len(PIV_1)):
    A_al, B_al = align_keep_denser_with_bands(CFD_1[timestep], PIV_1[timestep], wall_zero=True)
    plot_profiles_with_bands(A_al, B_al,
                            labelA=f"CFD ū(y) [{timestep/2},{timestep/2+dt_target}]s",
                            labelB=f"PIV ū(y) [{timestep/2},{timestep/2+dt_target}]s",
                            timestep = timestep,
                            title="Comparación CFD vs PIV",
                            N_A=None, N_B=None)


def plot_shear_and_stress_profiles(A_al, B_al, hb_A, hb_B,
                                   labelA="PIV", labelB="CFD",
                                   show_tau=True, title=None):
    """
    Grafica perfiles de tasa de corte γ̇(y)=du/dy y, opcionalmente, esfuerzo τ(y) (HB),
    usando las funciones YA existentes: profiles_hb(...) y herschel_bulkley(...).

    A_al, B_al: DataFrames alineados (salida de align_keep_denser_with_bands), con columnas 'y','u'[,'u_std'].
    hb_A, hb_B: dicts con {'tau0','K','n','gdot_crit'} para cada archivo.
    """
    # 1) Perfiles (du/dy y tau) PARA CADA ARCHIVO
    profA = profiles_hb(A_al, hb_A)   # usa _prep + np.gradient + herschel_bulkley
    profB = profiles_hb(B_al, hb_B)

    # 2) γ̇(y)
    plt.figure()
    plt.plot(profA["du_dy"], profA["y"], label=f"{labelA}: $\dot{{\gamma}}(y)$")
    plt.plot(profB["du_dy"], profB["y"], label=f"{labelB}: $\dot{{\gamma}}(y)$")
    plt.gca().invert_yaxis()
    plt.xlabel(r"$\dot{\gamma}$ [1/s]"); plt.ylabel("y [m]")
    plt.title(title or "Perfil de tasa de corte")
    plt.legend(); plt.tight_layout(); plt.show()

    # 3) τ(y) (HB)
    if show_tau:
        plt.figure()
        plt.plot(profA["tau_u"], profA["y"], label=f"{labelA}: $\\tau(y)$")
        plt.plot(profB["tau_u"], profB["y"], label=f"{labelB}: $\\tau(y)$")
        plt.gca().invert_yaxis()
        plt.xlabel(r"$\tau$ [Pa]"); plt.ylabel("y [m]")
        plt.title(title or "Perfil de esfuerzo (HB)")
        plt.legend(); plt.tight_layout(); plt.show()

""""
plot_shear_and_stress_profiles(
    A_al, B_al,
    hb_A=carbopol_02, hb_B=carbopol_02,
    labelA="PIV", labelB="CFD",
    show_tau=True,
    title="Perfiles de corte y esfuerzo"
)
"""


for n in range(1, 4):
    for mesh in meshes:
        for size in range(1, 5):

            print(f'PIV_{n} | CFD_{mesh}-{size}')

"""
for 

"""