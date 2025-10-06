import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO

import re




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



# ----------------------------------------
# 1) Utilidades: carga en bloques y agregado por "n"
# ----------------------------------------
def block_builder(path_txt):
    """Lee el txt y devuelve una lista de DataFrames (x,y,u,v) por bloque."""
    txt = Path(path_txt).read_text(encoding="utf-8")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    trozos = re.split(r"\n\s*\n+", txt.strip())  # separa por 1+ líneas en blanco

    blocks = []
    for b in trozos:
        lineas = [ln.strip() for ln in b.splitlines() if ln.strip() != ""]
        if not lineas:
            continue
        # descarta cabecera si aparece
        if any(ch.isalpha() for ch in lineas[0]):
            lineas = lineas[1:]
        if not lineas:
            continue

        csv_text = "\n".join(lineas)
        df = pd.read_csv(
            StringIO(csv_text),
            sep=r"\s*,\s*",    
            engine="python",
            names=["x", "y", "u", "v"],
            dtype=float
        ).reset_index(drop=True)
        blocks.append(df)
    return blocks

def time_step_updater(blocks, Ti, drop_remainder=True):
    """
    Agrupa de a n blocks consecutivos.
    Devuelve: lista de DataFrames con columnas: x, y, u_mean, v_mean, u_std, v_std
    """
    if Ti <= 0:
        raise ValueError("n debe ser un entero positivo.")
    new_block = []
    total = len(blocks)

    for start in range(0, total, Ti):
        grupo = blocks[start:start+Ti]
        if len(grupo) < Ti and drop_remainder:
            break

        # Verifica que todos tengan igual número de filas
        lens = {len(df) for df in grupo}
        if len(lens) != 1:
            print(blocks[start+2])
            raise ValueError(f"Grupo que inicia en {start} tiene tamaños distintos: {lens}")

        # Stack
        X = np.stack([df["x"].to_numpy() for df in grupo], axis=0)
        Y = np.stack([df["y"].to_numpy() for df in grupo], axis=0)
        U = np.stack([df["u"].to_numpy() for df in grupo], axis=0)
        V = np.stack([df["v"].to_numpy() for df in grupo], axis=0)

        # Promedios por fila
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        u_mean = U.mean(axis=0)
        v_mean = V.mean(axis=0)

        # Desviación muestral por fila (si solo 1 bloque -> ddof=0)
        ddof = 1 if len(grupo) > 1 else 0
        u_std = U.std(axis=0, ddof=ddof)
        v_std = V.std(axis=0, ddof=ddof)

        df_ag = pd.DataFrame({
            "x": x_mean,
            "y": y_mean,
            "u_mean": u_mean,
            "v_mean": v_mean,
            "u_std": u_std,
            "v_std": v_std,
        })
        new_block.append(df_ag)

    return new_block

# ----------------------------------------
# 2) Comparación de dos archivos (cada uno con su propio n)
# ----------------------------------------
def comparar_archivos(path1, n1, path2, n2, tol_xy=1e-9):
    """
    Compara blocks agregados de dos archivos.
    - Cada archivo se agrega con su propio n.
    - Se comparan solo los primeros min(num_grupos1, num_grupos2) grupos.
    - Alinea por índice (asumiendo mismas filas por bloque); verifica x/y ~ iguales (opcional).
    Retorna:
      - diffs_por_grupo: lista de DataFrames con columnas:
           x, y, u1, v1, u2, v2, du, dv
      - resumen: DataFrame con MAE y RMSE por grupo (u y v)
    """
    # Cargar y agregar
    bloques1 = cargar_bloques(path1)
    bloques2 = cargar_bloques(path2)
    ag1 = agregar_por_grupos(bloques1, n1, drop_remainder=True)
    ag2 = agregar_por_grupos(bloques2, n2, drop_remainder=True)

    m = min(len(ag1), len(ag2))  # ¡clave! comparamos hasta el mínimo
    diffs = []
    filas_resumen = []

    for gid in range(m):
        A = ag1[gid].reset_index(drop=True).copy()
        B = ag2[gid].reset_index(drop=True).copy()

        # Verifica igualdad de número de filas
        if len(A) != len(B):
            raise ValueError(f"El grupo {gid+1} tiene distinto nº de filas: {len(A)} vs {len(B)}")

        # (Opcional) chequeo de compatibilidad en x,y
        if tol_xy is not None:
            if (np.nanmax(np.abs(A["x"].to_numpy() - B["x"].to_numpy())) > tol_xy or
                np.nanmax(np.abs(A["y"].to_numpy() - B["y"].to_numpy())) > tol_xy):
                # Si no coinciden dentro de la tolerancia, seguimos comparando por índice,
                # pero te dejamos constancia en una columna.
                pass

        # Construye DataFrame de comparación por fila
        cmp_df = pd.DataFrame({
            "x": A["x"],
            "y": A["y"],
            "u1": A["u_mean"], "v1": A["v_mean"],
            "u2": B["u_mean"], "v2": B["v_mean"],
        })
        cmp_df["du"] = cmp_df["u1"] - cmp_df["u2"]
        cmp_df["dv"] = cmp_df["v1"] - cmp_df["v2"]

        # Métricas por grupo
        mae_u = np.mean(np.abs(cmp_df["du"]))
        mae_v = np.mean(np.abs(cmp_df["dv"]))
        rmse_u = np.sqrt(np.mean(cmp_df["du"]**2))
        rmse_v = np.sqrt(np.mean(cmp_df["dv"]**2))

        filas_resumen.append({
            "group_id": gid + 1,
            "rows": len(cmp_df),
            "MAE_u": mae_u, "RMSE_u": rmse_u,
            "MAE_v": mae_v, "RMSE_v": rmse_v
        })

        diffs.append(cmp_df)

    resumen = pd.DataFrame(filas_resumen)
    return diffs, resumen, ag1, ag2

# ------------------ EJEMPLO DE USO ------------------
# paths = ("archivo_A.txt", "archivo_B.txt")
# diffs, resumen, agA, agB = comparar_archivos(paths[0], n1=5, path2=paths[1], n2=3)
# print(resumen)
# print(diffs[0].head())


 
"""
x_profiles = np.linspace(0.1, 0.15, 6)
INPUT_PIV_1       = Path("PROFILES/PIV") / f"perfiles_x={x_profiles[0]:.6g}.txt" 
INPUT_CFD_1       = Path("PROFILES/CFD") / f"perfiles_x={x_profiles[0]:.6g}.txt" 
for X_TARGET, i  in enumerate(x_profiles[1:], 1):
    INPUT_PIV_0       = INPUT_PIV_1
    INPUT_CFD_0       = INPUT_CFD_1
    INPUT_PIV_1       = Path("PROFILES/PIV") / f"perfiles_x={X_TARGET:.6g}.txt" 
    INPUT_CFD_1       = Path("PROFILES/CFD") / f"perfiles_x={X_TARGET:.6g}.txt" 
    print(X_TARGET, i)
"""


