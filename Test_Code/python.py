# -*- coding: utf-8 -*-
from ansys.fluent.core import connect_to_fluent
import time, sys

# === CONFIG ===
SERVER_INFO = r"C:\Temp\fluent-serverinfo.txt"
CASE_FILE   = r"C:\Users\lkwol\OneDrive\Desktop\AnsysValidation\AnsysData\carbopol_fluent.cas.h5"
DATA_OUT    = r"C:\Users\lkwol\OneDrive\Desktop\AnsysValidation\AnsysData\resultado.dat.h5"

PHASE_NAME  = "phase-carbopol"     # tal como en tus logs
VARIABLE    = "volume-fraction"    # GUI: Volume Fraction
ZONE_NAME   = "lvolumen"           # zona existente según tus logs

N_STEPS = 3
SLEEP_S = 10
# =============

def scheme_exec(solver, cmd: str):
    # Ejecuta Scheme con compatibilidad (eval → evaluate → legacy)
    for attr in ("eval", "evaluate"):
        try:
            getattr(solver.scheme, attr)(cmd)
            return
        except Exception:
            pass
    solver.scheme_eval(cmd)  # fallback

def patch_init_by_zone(solver, zone_name: str):
    """
    PATCH DE INICIALIZACIÓN: /solve/initialize/patch
    Orden correcto para Euleriano:
      Variable -> volume-fraction
      Use Field Function? -> no
      Phase -> phase-carbopol
      Value -> 1
      Zones -> lvolumen
      Registers -> ()
      Apply -> yes
    """
    tui_multiline = (
        "/solve/initialize/patch\n"
        f"{VARIABLE}\n"
        "no\n"
        f"{PHASE_NAME}\n"
        "1\n"
        f"{zone_name}\n"
        "()\n"
        "yes\n"
    )
    # Enviar con \n reales dentro de la cadena Scheme + un par de \n extra para "flush"
    scheme_str = (
        tui_multiline
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        + "\\n\\n"  # ← flush del parser TUI
    )
    cmd = '(ti-menu-load-string "' + scheme_str + '")'
    scheme_exec(solver, cmd)

    # Flush adicional por si acaso (no hace daño)
    scheme_exec(solver, '(ti-menu-load-string "\\n")')

    print(f"✅ Patch (initialize) por ZONA aplicado: {VARIABLE}=1 en {PHASE_NAME}, zona '{zone_name}'")

def main():
    solver = connect_to_fluent(server_info_file_name=SERVER_INFO)
    print("✅ Conectado OK")

    # Leer caso e inicializar
    solver.settings.file.read_case(file_name=CASE_FILE)
    solver.settings.solution.initialization.initialize()

    # Patch de inicialización por ZONA
    patch_init_by_zone(solver, ZONE_NAME)

    # --- Iterar con API moderna (no TUI), con pausas ---
    for i in range(1, N_STEPS + 1):
        print(f"Time step {i}/{N_STEPS}")
        solver.settings.solution.run_calculation.iterate(iter_count=1)
        time.sleep(SLEEP_S)

    # Guardar y cerrar
    solver.settings.file.write_data(file_name=DATA_OUT)
    print(f"💾 Datos guardados en: {DATA_OUT}")
    solver.exit()
    print("✅ Finalizado correctamente")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)
