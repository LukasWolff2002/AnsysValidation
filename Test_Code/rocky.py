# rocky.py
import ansys.rocky.core as pyrocky

ROCKY_EXE = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\rocky\bin\Rocky.exe"

def main():
    rocky = pyrocky.launch_rocky(
        rocky_exe=ROCKY_EXE,  # ruta que ya validaste con Test-Path
        headless=True,
        close_existing=True
    )
    api = rocky.api
    print("[OK] Rocky headless con RPC activo")

    # Aquí ya puedes abrir/correr/exportar:
    # proj = api.Project.new()
    # api.Simulation.run_steps(10)

    rocky.close()

if __name__ == "__main__":
    main()
