## Apuntes — Flujo doblemente acoplado

### Requisitos generales
- Abrir Fluent en `double precision`.
- En el ítem "General", la gravedad debe apuntar en la misma dirección que la indicada en Rocky.
- Modelo en Fluent: `Eulerian`. Añadir una tercera fase llamada `phase-fiber` con un material fluido distinto a carbopol y aire.

### Configuración en Methods
- `Pressure-Velocity-Coupling`: `Phase Coupled SIMPLE`
- `Transient Formulation`: `First Order Implicit`

### Flujo de trabajo (pasos)
1. Inicializar la solución.
2. Realizar el `patch`.
3. Indicar las carpetas de output de Fluent.
4. Guardar `case` y `data` para lectura en Rocky.

### Referencias
- Documentación: Rocky User Manual — página 377  
    https://www.dropbox.com/scl/fi/1tvuvt173gr4jefrwjkz2/Rocky_User_Manual.pdf?rlkey=1l93u2zngbijjl3y8u5opwdmy&st=ws5230c7&dl=0
- Tutorial: https://www.dropbox.com/scl/fo/k5c61kz377pt2ss2p1kcs/AA_pAlmcO8AmNjBTMJeA-y4?rlkey=xajb7r7svjn0lzc9tzelbli0v&st=zlsbw67d&dl=0

## Apuntes — Rocky
- Relación de aspecto (aspect ratio): `81.25`.
- Diámetro de la fibra: `0.16 mm` → longitud aproximada: `13 mm` (diámetro × aspect ratio ≈ largo).
- Nota: verificar consistencia de unidades y dirección de la gravedad entre Fluent y Rocky.
