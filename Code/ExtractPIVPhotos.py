#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from typing import List, Dict, Any

# ========= Configuración =========
CARPETA_BASE = "PIVData/Camara2PIV"
NATURAL_SORT = True   # True: orden natural 1,2,10… ; False: orden alfabético puro
OVERWRITE    = True   # True: sobrescribe si ya existe en destino; False: salta si existe

# Define aquí todas las fases que quieras (en orden):
# Regla: en cada fase se copian SIEMPRE 2 archivos por bloque:
#   - copia índice 0 del bloque
#   - salta 'skip_inter' archivos
#   - copia el siguiente
#   - salta 'skip_final' archivos para llegar al siguiente bloque
# Validación: 2 + skip_inter + skip_final == block_size
FASES: List[Dict[str, Any]] = [
    # Ejemplos:
    {"dest": "PIVData/2_CADA_22", "blocks": 20, "skip_inter": 0, "skip_final": 22, "block_size": 24},
    {"dest": "PIVData/2_CADA_21",       "blocks": 20, "skip_inter": 1, "skip_final": 21, "block_size": 24},
    {"dest": "PIVData/2_CADA_20",       "blocks": 180, "skip_inter": 2, "skip_final": 20, "block_size": 24},
]

# ========= Utilidades =========
def natural_key(s: str):
    """Clave para orden natural: 'img12' -> ['img', 12]"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def listar_archivos_ordenados(base_dir: str) -> List[str]:
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    return sorted(files, key=natural_key if NATURAL_SORT else None)

def copiar_si_corresponde(src: str, dst: str) -> None:
    if not OVERWRITE and os.path.exists(dst):
        print(f"[SKIP] Ya existe en destino: {os.path.basename(dst)}")
        return
    shutil.copy2(src, dst)
    print(f"[COPY] {os.path.basename(src)} -> {dst}")

def procesar_fases(base_dir: str, fases: List[Dict[str, Any]]) -> None:
    archivos = listar_archivos_ordenados(base_dir)
    total = len(archivos)
    if total == 0:
        print("[WARN] No hay archivos en la carpeta base.")
        return

    i = 0  # puntero global que avanzará a través de TODAS las fases

    for k, fase in enumerate(fases, start=1):
        dest        = fase.get("dest")
        blocks_max  = int(fase.get("blocks", 0))
        skip_inter  = int(fase.get("skip_inter", 0))
        skip_final  = int(fase.get("skip_final", 0))
        block_size  = int(fase.get("block_size", 24))

        # Validaciones
        if not dest or blocks_max <= 0:
            print(f"[F{k}] Fase inválida: destino vacío o blocks<=0, se omite.")
            continue
        if skip_final <= skip_inter:
            raise ValueError(f"[F{k}] 'skip_final' ({skip_final}) debe ser MAYOR que 'skip_inter' ({skip_inter}).")
        if (2 + skip_inter + skip_final) != block_size:
            raise ValueError(f"[F{k}] Regla no cumple el tamaño de bloque: 2 + {skip_inter} + {skip_final} != {block_size}")

        os.makedirs(dest, exist_ok=True)

        bloques_hechos = 0
        print(f"\n=== Fase {k}: dest='{dest}', blocks={blocks_max}, skip_inter={skip_inter}, skip_final={skip_final}, block_size={block_size} ===")

        # Mientras queden bloques completos y no superemos el límite de la fase
        while (i + block_size) <= total and bloques_hechos < blocks_max:
            # Índices dentro del bloque actual
            idx1 = i + 0
            idx2 = i + 1 + skip_inter

            # Copiar ambos
            for idx in (idx1, idx2):
                src = os.path.join(base_dir, archivos[idx])
                dst = os.path.join(dest, archivos[idx])
                try:
                    copiar_si_corresponde(src, dst)
                except Exception as e:
                    print(f"[ERR] No se pudo copiar '{archivos[idx]}': {e}")

            # Avanzar al siguiente bloque según la regla
            i += block_size
            bloques_hechos += 1

        print(f"[F{k}] Bloques procesados: {bloques_hechos}/{blocks_max}")

    print(f"\n[FIN] Procesamiento completado. Archivos totales: {total}. Índice final consumido: {i}.")

if __name__ == "__main__":
    procesar_fases(CARPETA_BASE, FASES)
