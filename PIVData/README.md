## Datos de las tomas

Las imágenes se capturaron a **240 fps**. Para simplificar el análisis, se exportan datos cada **24 fotogramas**, lo que equivale a **0.1 segundos** por muestra.

## Consideraciones

El método PIV compara pares de imágenes. Si el fluido presenta muy poca velocidad entre dos tomas, no se observa variación significativa. Por ello, se empleó el siguiente esquema de adquisición:

- **Primeros 2 segundos:** Pares de imágenes consecutivas (sin imágenes intermedias). 
- **Siguientes 2 segundos:** Se incluye 1 imagen intermedia entre cada par.
- **Últimos 18 segundos:** Se incluyen 2 imágenes intermedias entre cada par.

Este enfoque permite capturar adecuadamente las variaciones de velocidad del fluido a lo largo del experimento.

## PIV Viga

Para el análisis de la viga, es necesario aplicar una máscara dinámica que permita aislar únicamente la sección de carbopol en contacto con el acrílico. Para ello, se desarrolló un modelo de segmentación utilizando **YOLO v11**. El dataset empleado está disponible en [Roboflow](https://app.roboflow.com/particle-tracking-velocimetry/dynamicmask-93zhi/1) y puede descargarse con el siguiente procedimiento:

```bash
!pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="MTS6I7fo7Sbo25WfpAFS")
project = rf.workspace("particle-tracking-velocimetry").project("dynamicmask-93zhi")
version = project.version(1)
dataset = version.download("yolov11")
```

**Consideraciones:**

- El modelo de segmentación utiliza las partículas de seguimiento para identificar la zona de contacto con el acrílico. Por lo tanto, es fundamental asegurar una concentración adecuada de partículas en la mezcla.
- En las tomas 2 y 3, la baja concentración de partículas y la iluminación insuficiente afectan negativamente el desempeño del modelo.
- Los resultados obtenidos con el primer modelo entrenado están disponibles [aquí](https://www.dropbox.com/home/TESIS/Resultados%20PIV/Viga/Toma%201%20Primer%20Modelo%20Mascaras%20Dinamicas).

**Parámetros de análisis:**

- La conversión de píxeles a milímetros es de **80.54 px = 10 mm**.
- Se consideran cajas de **64 px**, **32 px** y **16 px** para el análisis, empleando un enfoque tipo *high*.



## PIV L