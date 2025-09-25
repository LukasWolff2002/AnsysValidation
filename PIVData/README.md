## Datos de las tomas

Las imágenes se capturaron a **240 fps**. Para simplificar el análisis, se exportan datos cada **24 fotogramas**, lo que equivale a **0.1 segundos** por muestra.

## Consideraciones

El método PIV compara pares de imágenes. Si el fluido presenta muy poca velocidad entre dos tomas, no se observa variación significativa. Por ello, se empleó el siguiente esquema de adquisición:

- **Primeros 2 segundos:** Pares de imágenes consecutivas (sin imágenes intermedias).
- **Siguientes 2 segundos:** Se incluye 1 imagen intermedia entre cada par.
- **Últimos 18 segundos:** Se incluyen 2 imágenes intermedias entre cada par.

Este enfoque permite capturar adecuadamente las variaciones de velocidad del fluido a lo largo del experimento.