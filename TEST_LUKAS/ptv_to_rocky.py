import json
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------- Rutas -----------------
path_json     = r"TEST_LUKAS/ptv.json"
path_clicks   = r"TEST_LUKAS/clicks.csv"
out_csv_rocky = r"TEST_LUKAS/first_frame_centroids_rocky.csv"

# ----------------- Calibración / Geometría -----------------
px_mm = 0.139                   # mm/px
scale_m_per_px = px_mm * 1e-3   # m/px

# Resolución PTV
W = 1024
H = 1024

# Punto conocido en Rocky (interpreta como X (m) y Z (m))
x_conocido_rocky = 0.085  # m  -> eje X
z_conocido_rocky = 0.020  # m  -> eje Z

# ----------------- Parser del JSON -----------------
def _unwrap_scalar_list(v):
    if isinstance(v, list) and len(v) == 1 and not isinstance(v[0], list):
        return v[0]
    return v

def cargar_tracks_json(path_json: str) -> pd.DataFrame:
    data = json.loads(Path(path_json).read_text(encoding="utf-8"))
    tracks_keys = [k for k in data.keys() if k.isdigit()]
    rows = []
    for k in tracks_keys:
        tid = int(k)
        obj = data[k]
        centros = obj.get("centroide", [])
        largos  = [ _unwrap_scalar_list(v) for v in obj.get("largo_maximo", []) ]
        angulos = [ _unwrap_scalar_list(v) for v in obj.get("angulo", []) ]
        frames  = [ _unwrap_scalar_list(v) for v in obj.get("frame", []) ]
        n = len(centros)
        if not (len(largos) == len(angulos) == len(frames) == n):
            raise ValueError(f"Track {tid}: longitudes desalineadas")
        for i in range(n):
            x, y = centros[i]
            rows.append({
                "track_id": tid,
                "frame": int(frames[i]),
                "x_px": float(x),
                "y_px": float(y),
                "ang_deg": float(angulos[i]),
            })
    return pd.DataFrame(rows).sort_values(["track_id","frame"]).reset_index(drop=True)

# ----------------- Cargar datos -----------------
df = cargar_tracks_json(path_json)

# Filtrar solo tracks que comienzan en el frame 1
first_frames = df.groupby("track_id")["frame"].min().reset_index()
valid_ids = first_frames.loc[first_frames["frame"] == 1, "track_id"].tolist()
df = df[df["track_id"].isin(valid_ids)]

# Tomar solo el primer frame
df_first = df[df["frame"] == 1].copy()
if df_first.empty:
    raise RuntimeError("No hay centroides en frame 1 para tracks que comienzan en frame 1.")

# ----------------- Leer punto de referencia PTV -----------------
clicks_path = Path(path_clicks)
if not clicks_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo de clicks: {path_clicks}")
clicks_df = pd.read_csv(clicks_path)
if clicks_df.empty:
    raise RuntimeError("El archivo de clicks está vacío. Registra al menos un punto y guarda con 'S'.")

# Último punto registrado en PTV (px)
ref_x_px = float(clicks_df.iloc[-1]["x_px"])
ref_y_px = float(clicks_df.iloc[-1]["y_px"])

# ----------------- Ajuste de offsets para Rocky -----------------
# En Rocky:
#   X_m = x_px * scale + offset_x
#   Z_m = (H - y_px) * scale + offset_z
# tal que (ref_x_px, ref_y_px) → (x_conocido_rocky, z_conocido_rocky)
offset_x = x_conocido_rocky - (ref_x_px * scale_m_per_px)
offset_z = z_conocido_rocky - ((H - ref_y_px) * scale_m_per_px)

# ----------------- Transformar posiciones a Rocky -----------------
df_first["x"] = df_first["x_px"] * scale_m_per_px + offset_x
df_first["z"] = (H - df_first["y_px"]) * scale_m_per_px + offset_z
df_first["y"] = np.random.uniform(-0.02, 0.02, size=len(df_first))  # aleatorio en [-0.02, 0.02] m

# ----------------- Vector normal (perpendicular a la fibra) en XZ -----------------
theta = np.deg2rad(df_first["ang_deg"].values)
df_first["nx"] = np.sin(theta)
df_first["ny"] = 0.0
df_first["nz"] = np.cos(theta)

# ----------------- IDs y ángulo constante -----------------
df_first["ptv_id"] = df_first["track_id"].astype(int)
df_first = df_first.sort_values(["ptv_id"]).reset_index(drop=True)
df_first["rocky_id"] = np.arange(1, len(df_first) + 1, dtype=int)
df_first["angle"] = np.pi / 2  # radianes

# ----------------- Filtro: eliminar fibras con X_m < 0 o Z_m < 0 -----------------
before = len(df_first)
df_first = df_first[(df_first["x"] >= 0) & (df_first["z"] >= 0)].copy()
after = len(df_first)
removed = before - after

# ----------------- Guardar CSV final -----------------
cols_out = ["rocky_id", "ptv_id", "x", "y", "z", "nx", "ny", "nz", "angle"]
Path(out_csv_rocky).parent.mkdir(parents=True, exist_ok=True)
df_first.to_csv(out_csv_rocky, index=False, columns=cols_out)

print(f"[OK] Exportados {after} puntos (eliminados {removed} con coordenadas negativas).")
print(f"[INFO] Ref PTV: ({ref_x_px:.2f}px, {ref_y_px:.2f}px) → Rocky ({x_conocido_rocky:.3f}, {z_conocido_rocky:.3f}) m")
print(f"[INFO] Offsets: offset_x={offset_x:.6f} m, offset_z={offset_z:.6f} m")
print(f"[SAVE] {out_csv_rocky}")
