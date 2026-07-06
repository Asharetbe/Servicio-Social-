# Aproximación y caracterización de curvas paramétricas

Este repositorio documenta el proceso completo de análisis de un conjunto de 500 curvas paramétricas 2D: desde la exploración inicial de los datos hasta la identificación de qué rasgos geométricos determinan cuál método de aproximación funciona mejor en cada caso.

---

## Estructura del proyecto

```
curves/
├── datos/
│   ├── target/           # 500 curvas de alta precisión (referencia)
│   └── pixel_curves/     # 2 500 versiones pixelizadas (5 escalas por curva)
├── exploracion_datos/    # EDA: ¿qué son estas curvas?
├── aproximaciones/
│   ├── polinomiales/     # Ajuste polinomial desde píxeles
│   ├── gaussian/         # Ajuste con suma de gaussianas + splines
│   └── comparacion/      # ¿Qué método gana en cada curva?
└── caracterizacion/      # ¿Por qué gana ese método?
```

---

## Los datos

El dataset parte de un conjunto de **500 curvas paramétricas**, cada una representada en dos formatos complementarios.

El formato de **alta precisión** (`datos/target/`) contiene 500 archivos `curve_XXXX.txt`, cada uno con 8 000 puntos flotantes $(x, y)$ capturados a resolución continua. Estos son los datos de referencia: la verdad que queremos aproximar. El dominio $x$ va aproximadamente de 5 a 260, con valores $y$ entre 0 y 150.

El formato de **píxeles** (`datos/pixel_curves/`) simula cómo se vería cada curva si se digitalizara a mano desde una imagen. Hay 5 versiones por curva, correspondientes a 5 factores de escala (X10, X15, X18, X21, X25), que determinan cuántos puntos enteros se muestrean: desde ~200 puntos en X10 hasta ~500 en X25. En total son 2 500 archivos con coordenadas enteras.

La integridad del dataset fue verificada: correspondencia perfecta entre ambos formatos, sin archivos vacíos, sin NaN ni infinitos.

---

## Fase 1 — Exploración (`exploracion_datos/`)

> *Antes de aproximar, había que entender qué tipo de objetos eran estas curvas.*

Cinco notebooks de análisis exploratorio examinaron la estructura, las estadísticas descriptivas, las propiedades geométricas y la detección de anomalías. La pregunta central era: ¿estas curvas se comportan como funciones matemáticas bien definidas, o hay casos degenerados?

La respuesta fue afirmativa: todas las curvas son funciones válidas, sin discontinuidades, con geometría interpretable. Se extrajeron métricas de curvatura, pendiente, longitud de arco y puntos críticos para cada curva.

**Output generado:**

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `exploracion_datos/caracteristicas_geometricas.csv` | 500 | Métricas geométricas por curva: curvatura, pendiente, n_maxima, n_minima, longitud de arco, tipo |

---

## Fase 2 — Aproximaciones (`aproximaciones/`)

> *Confirmado que las curvas son funciones, la pregunta fue: ¿se pueden reconstruir desde los píxeles?*

La idea central es aprender un modelo matemático a partir de los datos pixelizados (ruidosos, discretos) y evaluar qué tan bien reproduce la curva target de alta precisión. Se probaron tres familias de modelos.

### 2a. Polinomios (`polinomiales/`)

El primer intento fue el más sencillo: ajustar un polinomio de grado $d$ sobre los puntos de píxeles y medir el R² contra la curva target. Se evaluaron grados del 1 al 20 y se seleccionó el óptimo usando criterios BIC, AIC y el método del codo.

El resultado fue sorprendentemente bueno: incluso desde la escala más baja (X10, ~200 puntos enteros), los polinomios logran R² > 0.97 en promedio. El grado óptimo se concentra entre 8 y 10 — suficiente para capturar la forma global sin sobreajustar. Las cinco escalas producen resultados casi idénticos, lo que indica robustez al ruido de pixelización.

**Outputs generados:**

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `aproximaciones/polinomiales/grados_optimos_por_curva.csv` | 500 | BIC, AIC, R² por escala y promedios; clasificación de dificultad |
| `aproximaciones/polinomiales/detalle_criterios_por_curva_escala.csv` | 2 500 | Detalle por combinación curva × escala: grado óptimo según cada criterio |

### 2b. Gaussianas (`gaussian/`)

Los polinomios funcionan bien en curvas suaves, pero ¿qué pasa con curvas que tienen picos pronunciados? Surge una hipótesis: quizás un modelo construido como suma de campanas gaussianas se adapta más naturalmente a esas formas.

El primer intento usó una sola gaussiana: $y = A \cdot e^{-(x-\mu)^2/(2\sigma^2)} + c$. El R² promedio fue de apenas 0.74 — insuficiente. Una campana no puede capturar curvas con múltiples máximos locales.

El segundo intento amplió el modelo a una **suma de hasta 8 campanas gaussianas**, con detección automática de picos para inicializar los parámetros. El resultado cambió radicalmente: R² > 0.999 en promedio. El modelo también incluye una comparación con **splines cúbicos** (funciones por segmentos con nodos), que logran un desempeño igualmente alto.

**Outputs generados:**

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `aproximaciones/gaussian/parametros_gaussianas_por_curva.csv` | 500 | Parámetros completos: n_campanas, c, A₁–A₈, μ₁–μ₈, σ₁–σ₈ |
| `aproximaciones/gaussian/parametros_por_curva_resumen.csv` | 500 | Resumen: método ganador, R² por método, parámetros de las 2 primeras campanas |

---

## Fase 3 — Comparación (`aproximaciones/comparacion/`)

> *Con tres métodos que todos funcionan bien, la pregunta se volvió: ¿cuál es el mejor para cada curva en particular?*

Se construyó una tabla comparativa evaluando los tres métodos sobre las 500 curvas con cuatro métricas: R², RMSE, MAE y error absoluto máximo. El método ganador para cada curva es el de mayor R².

El resultado reveló que **ningún método domina universalmente**:

- **Polinomio** gana en ~40% de las curvas
- **Gaussianas** gana en ~40% de las curvas
- **Spline** gana en ~20% de las curvas

Todos alcanzan R² > 0.99 en promedio, pero las diferencias entre ellos son sistemáticas y dependen de la curva. Esto abre la pregunta central del proyecto.

**Output generado:**

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `aproximaciones/comparacion/comparacion_3_metodos.csv` | 500 | R², RMSE, MAE, Max AE y n_params para los 3 métodos; columna `mejor_metodo` |

---

## Fase 4 — Caracterización (`caracterizacion/`)

> *¿Qué hace que una curva le quede mejor al polinomio, a las gaussianas o al spline?*

Este es el análisis más interesante del proyecto. El notebook `caracterizacion_modelos_curvas.ipynb` fusiona las tres tablas de resultados y construye descriptores derivados que caracterizan la forma y complejidad de cada curva:

- **`gauss_picos_significativos`**: campanas con amplitud > 15% del máximo
- **`gauss_dispersion_centros`**: dispersión espacial ponderada de los centros de campana
- **`gauss_sigma_media_pond`**: ancho promedio ponderado de las campanas
- **`indice_compacidad_picos`**: picos significativos / dispersión de centros
- **`bic_promedio`, `r2_std_escala`**: complejidad y variabilidad entre escalas

Se aplicaron **pruebas de Kruskal-Wallis** para identificar cuáles de estas variables discriminan estadísticamente entre los tres grupos de curvas (p < 0.05 en 12 de 14 variables). Luego se entrenó un **árbol de decisión** (profundidad 3) para extraer reglas legibles y un **Random Forest** de 300 árboles para rankear la importancia real de cada variable.

**Las variables más explicativas** resultaron ser: `bic_max`, `aic_promedio`, `bic_promedio`, `gauss_sigma_std_pond`, `gauss_sigma_media_pond` y `r2_std_escala`.

### Conclusión: ¿cuándo gana cada método?

**Polinomio** — Curvas de **estructura global y baja complejidad efectiva**. La geometría puede describirse con una sola función sobre todo el dominio. Se caracteriza por BIC bajo (mediana = 12.6), pocos nodos spline necesarios (mediana = 5) y baja dispersión de centros gaussianos (27.4).

**Gaussianas** — Curvas con **picos bien definidos y estructura multimodal**. La forma es compatible con una descomposición en campanas. BIC promedio de 19.0, dispersión de centros intermedia (57.6), varios picos significativos (mediana = 5).

**Spline** — Curvas con **variación local rica y distribuida** a lo largo del dominio. Requieren flexibilidad por tramos, no un modelo global. Se distinguen por un número alto de nodos (mediana = 17) y la mayor dispersión espacial de centros gaussianos (83.3).

---

## Mapa de archivos generados

| Archivo | Filas × Cols | Generado por |
|---------|-------------|--------------|
| `exploracion_datos/caracteristicas_geometricas.csv` | 500 × 23 | EDA notebooks |
| `aproximaciones/polinomiales/grados_optimos_por_curva.csv` | 500 × 22 | `aproximacion_polinomial.ipynb` |
| `aproximaciones/polinomiales/detalle_criterios_por_curva_escala.csv` | 2 500 × 10 | `aproximacion_polinomial.ipynb` |
| `aproximaciones/gaussian/parametros_gaussianas_por_curva.csv` | 500 × 28 | `02_suma_gaussianas_splines.ipynb` |
| `aproximaciones/gaussian/parametros_por_curva_resumen.csv` | 500 × 14 | `02_suma_gaussianas_splines.ipynb` |
| `aproximaciones/comparacion/comparacion_3_metodos.csv` | 500 × 22 | `comparacion_3_metodos.py` |

---

## Conclusión

El método ganador para una curva no es universal: depende de su **geometría intrínseca**. Curvas simples y globales le van bien al polinomio; curvas con picos compatibles con campanas le van a las gaussianas; curvas con variación local distribuida necesitan la flexibilidad del spline. Esta relación es estadísticamente robusta y capturada con alta fidelidad por un modelo de clasificación interpretable.
