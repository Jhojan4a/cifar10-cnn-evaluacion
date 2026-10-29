# Evaluación de la Segunda Unidad - Clasificación de Imágenes mediante CNN
**Estudiante:** Jhojan Abel Yana Ramos  
**Código / Semilla Individual:** 97  
**Dataset:** CIFAR-10 (10 clases, 32x32 RGB)  

---

## 1. Justificación Teórica: CNN vs. MLP (Pregunta 2)

### Ineficiencia y Colapso del Perceptrón Multicapa (MLP)
1. **Destrucción de la Correlación Espacial:** CIFAR-10 está formado por tensores cuatridimensionales `[B, 3, 32, 32]`. Un MLP obliga a aplanar cada muestra en un vector unidimensional de $32 \times 32 \times 3 = 3072$ variables. Esto destruye la estructura bidimensional de la imagen: la red pierde la noción de vecindad entre píxeles verticales y diagonales.
2. **Explosión Paramétrica y Sobreajuste Inmediato:** Conectar una entrada de 3072 elementos a una capa densa oculta estándar de 1024 neuronas requeriría:
   $$\text{Parámetros} = (3072 \times 1024) + 1024 = 3,146,752 \text{ pesos}$$
   Esta enorme cantidad de parámetros en una sola capa satura la memoria, encarece el cómputo y causa sobreajuste rápido en conjuntos de datos pequeños.

### Solución Mediante Convoluciones (CNN)
* **Conexiones Locales (Local Receptive Fields):** Las neuronas operan únicamente sobre subregiones pequeñas ($3 \times 3$), preservando la coherencia espacial y detectando características locales (líneas, esquinas, texturas).
* **Compartición de Pesos (Weight Sharing):** Un mismo núcleo (kernel) convoluciona sobre todo el mapa de características. El número de parámetros depende del tamaño del kernel y no de la resolución de la imagen.
* **Invarianza a la Traslación:** La integración de convoluciones con capas de submuestreo (`MaxPool2d`) permite reconocer un objeto sin importar en qué cuadrante de la imagen se ubique.

---

## 2. Especificación de la Arquitectura (Pregunta 2 y 3)

| Bloque / Capa | Tipo de Operación | Tensor de Entrada | Configuración / Kernels | Tensor de Salida | Activación |
|---|---|---|---|---|---|
| **Bloque 1** | Conv2d + BatchNorm + MaxPool | `[B, 3, 32, 32]` | 32 filtros ($3\times3$, stride 1, pad 1) + Pool ($2\times2$) | `[B, 32, 16, 16]` | ReLU |
| **Bloque 2** | Conv2d + BatchNorm + MaxPool | `[B, 32, 16, 16]` | 64 filtros ($3\times3$, stride 1, pad 1) + Pool ($2\times2$) | `[B, 64, 8, 8]` | ReLU |
| **Bloque 3** | Conv2d + BatchNorm + MaxPool | `[B, 64, 8, 8]` | 128 filtros ($3\times3$, stride 1, pad 1) + Pool ($2\times2$) | `[B, 128, 4, 4]` | ReLU |
| **Aplanado** | Flatten | `[B, 128, 4, 4]` | $128 \times 4 \times 4$ | `[B, 2048]` | - |
| **Densa 1** | Linear + Dropout(0.3) | `[B, 2048]` | 256 neuronas | `[B, 256]` | ReLU |
| **Salida** | Linear | `[B, 256]` | 10 neuronas | `[B, 10]` | Lineal (Logits) |

---

## 3. Artefactos Gráficos y Curvas de Aprendizaje (Pregunta 4)

### Curvas de Pérdida y Precisión (`outputs/loss_accuracy.png`)
![Curvas de Aprendizaje](outputs/loss_accuracy.png)

### Matriz de Confusión (`outputs/confusion_matrix.png`)
![Matriz de Confusión](outputs/confusion_matrix.png)

---

## 4. Análisis Crítico de Métricas (Semilla 97)

1. **Diagnóstico de Ajuste (Underfitting / Overfitting):**
   * A lo largo de las 12 épocas, la función de pérdida (`Loss`) desciende de forma regular desde ~1.5 hasta ~0.7 tanto en entrenamiento como en validación.
   * La curva de precisión (`Accuracy`) de validación se mantiene muy cercana e incluso ligeramente por encima de la de entrenamiento durante las primeras épocas, superando el 75%.
   * **Conclusión:** El modelo no presenta sobreajuste prematuro ni subajuste. La regularización aplicada (técnicas de *Data Augmentation* con rotación/recorte, `BatchNorm2d` y `Dropout(0.3)`) estabilizó eficazmente la generalización del modelo.

2. **Interpretación de la Matriz de Confusión:**
   * **Clases con mejor separabilidad:** Vehículos y contextos no biológicos (`automobile` con 934 aciertos, `truck` con 888 y `ship` con 798) muestran diagonales dominantes y tasas de falso positivo mínimas.
   * **Confusiones semánticas frecuentes:** Los mayores errores de clasificación ocurren entre categorías con siluetas y fondos biológicos similares:
     * La clase `cat` tiene 193 falsos positivos clasificados erróneamente como `dog`.
     * La clase `bird` y `deer` presentan confusión cruzada con otras clases animales debido a la baja resolución de 32x32 píxeles.