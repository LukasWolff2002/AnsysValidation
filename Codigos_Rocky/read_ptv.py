import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.patches import Rectangle

# ----------------- Rutas -----------------
path_json = r"TEST_LUKAS/ptv.json"
path_img  = r"TEST_LUKAS/ptv_image.bmp"
out_csv   = r"TEST_LUKAS/clicks.csv"

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
            })
    df = pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)
    return df

# ----------------- Cargar datos e imagen -----------------
df  = cargar_tracks_json(path_json)
img = plt.imread(path_img)
H, W = img.shape[:2]

# ----------------- FILTRO: solo trayectorias que comienzan en frame 1 -----------------
first_frames = df.groupby("track_id")["frame"].min().reset_index()
valid_ids = first_frames.loc[first_frames["frame"] == 1, "track_id"].tolist()
df = df[df["track_id"].isin(valid_ids)]

print(f"[INFO] Se mostrarán {len(valid_ids)} tracks (que comienzan en frame 1)")

# ----------------- Figura principal -----------------
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(img, origin="upper")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect("equal")
ax.set_xlabel("x [px]")
ax.set_ylabel("y [px]")
ax.set_title("Trayectorias que comienzan en frame 1")

# Trayectorias filtradas
for tid, g in df.groupby("track_id"):
    ax.plot(g["x_px"].values, g["y_px"].values, linewidth=1, alpha=0.9, zorder=2)

# ----------------- Crosshair principal -----------------
vline = ax.axvline(x=0, linewidth=0.9, alpha=0.85, zorder=3)
hline = ax.axhline(y=0, linewidth=0.9, alpha=0.85, zorder=3)

# ----------------- Ventana de zoom -----------------
win_px = 80
half   = win_px / 2
zoom_rect = Rectangle((0, 0), win_px, win_px, fill=False, linewidth=1.0, alpha=0.9, color="white")
ax.add_patch(zoom_rect)

zoom_factor = 5
inset = zoomed_inset_axes(ax, zoom=zoom_factor, loc="upper right", borderpad=0.6)
inset.imshow(img, origin="upper")
for tid, g in df.groupby("track_id"):
    inset.plot(g["x_px"].values, g["y_px"].values, linewidth=1, alpha=0.9)
inset.set_xticks([]); inset.set_yticks([])
inset.set_aspect("equal")

ivline = inset.axvline(x=0, linewidth=0.8, alpha=0.85)
ihline = inset.axhline(y=0, linewidth=0.8, alpha=0.85)

# ----------------- Clicks -----------------
clicks = []
click_scatter_main  = ax.scatter([], [], s=30, zorder=4)
click_scatter_inset = inset.scatter([], [], s=30, zorder=4)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def update_zoom_elements(x, y):
    x0 = clamp(x - half, 0, W)
    y0 = clamp(y - half, 0, H)
    x1 = clamp(x + half, 0, W)
    y1 = clamp(y + half, 0, H)
    zoom_rect.set_xy((x0, y0))
    zoom_rect.set_width(x1 - x0)
    zoom_rect.set_height(y1 - y0)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y1, y0)
    ivline.set_xdata([x, x])
    ihline.set_ydata([y, y])

def on_move(event):
    if event.inaxes != ax or event.xdata is None: return
    x, y = event.xdata, event.ydata
    vline.set_xdata([x, x])
    hline.set_ydata([y, y])
    update_zoom_elements(x, y)
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes != ax or event.button != 1: return
    x, y = event.xdata, event.ydata
    clicks.append({"x_px": float(x), "y_px": float(y)})
    X, Y = [c["x_px"] for c in clicks], [c["y_px"] for c in clicks]
    click_scatter_main.set_offsets(list(zip(X, Y)))
    click_scatter_inset.set_offsets(list(zip(X, Y)))
    fig.canvas.draw_idle()
    print(f"[click] x={x:.2f}, y={y:.2f}")

def on_key(event):
    k = (event.key or "").lower()
    if k in ("q", "escape"):
        plt.close(fig)
    elif k == "s":
        if clicks:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(clicks).to_csv(out_csv, index=False)
            print(f"[save] {len(clicks)} puntos guardados en {out_csv}")
        else:
            print("[save] No hay puntos para guardar.")
    elif k == "c":
        if clicks:
            clicks.pop()
            X, Y = [c["x_px"] for c in clicks], [c["y_px"] for c in clicks]
            click_scatter_main.set_offsets(list(zip(X, Y)))
            click_scatter_inset.set_offsets(list(zip(X, Y)))
            fig.canvas.draw_idle()
            print("[clear] Último punto eliminado.")

fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_press_event",   on_click)
fig.canvas.mpl_connect("key_press_event",      on_key)

plt.tight_layout()
plt.show()
