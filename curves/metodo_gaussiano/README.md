# Método Gaussiano: Optimización Parametrizada

## Descripción

Este submódulo del proyecto **curves** implementa una **búsqueda sistemática de parámetros óptimos** para el método de **suma de campanas gaussianas**.

A diferencia del enfoque genérico anterior, aquí **variamos intencionalmente**:
- **Número de campanas** (n_campanas): 1 a 8
- **Método de inicialización** (metodo_init): altura, prominencia, uniforme, grid_search
- **Hiperparámetros de optimización**: maxfev, método de solver

El objetivo es determinar **cuál configuración funciona mejor para cada curva** y encontrar **patrones** que predigan automáticamente la configuración óptima.

---

## Estructura del Directorio

```
metodo_gaussiano/
├── README.md                                    # Este archivo
├── 00_metodologia_gaussiana.ipynb              # Fundamentación teórica + ejemplos
├── 01_optimizacion_gaussiana_parametros.ipynb  # Implementación práctica + búsqueda
├── modelos/                                     # Modelos entrenados (Random Forest, etc.)
├── resultados/                                  # Resultados de búsqueda
│   ├── piloto_resultados_todas_combinaciones.csv
│   ├── piloto_configuraciones_ganadoras.csv
│   ├── completo_resultados_todas_combinaciones.csv  (futuro)
│   └── completo_configuraciones_ganadoras.csv        (futuro)
└── visualizaciones/                            # (opcional) Gráficas y heatmaps
```

---

## Archivos Principales

### 1. **00_metodologia_gaussiana.ipynb**

Notebook de **fundamentos teóricos** y justificación. Contiene:

- **El Problema**: ¿Por qué variar parámetros gaussianos?
- **Ejemplos visuales**: Impacto de n_campanas en R²
- **Parámetros ajustables**: Explicación de cada uno
  - `n_campanas` (1-8): número de gaussianas a sumar
  - `metodo_init` (altura, prominencia, uniforme, grid_search)
  - `maxfev`: máximo de evaluaciones de función
  - `method` (lm, trf): algoritmo de optimización

- **Workflow**: Paso a paso de la optimización
- **Métricas**: R², RMSE, MAE, Max AE
- **Expectativas y sorpresas**: Qué esperamos encontrar
- **Checklist de validación**: Cómo verificar resultados

**🎯 Lectura recomendada**: Comenzar aquí para entender la teoría.

---

### 2. **01_optimizacion_gaussiana_parametros.ipynb**

Notebook de **implementación práctica** e **búsqueda sistemática**. Contiene:

#### Secciones principales:

1. **Setup**: Importar dependencias, configurar rutas
2. **Funciones de utilidad**: Lectura de datos, cálculo de R², RMSE, MAE
3. **Funciones gaussianas**: Suma de campanas, derivadas
4. **Estrategias de inicialización**:
   - `inicializar_desde_picos()` con métodos 'altura', 'prominencia', 'uniforme'
   - `inicializar_grid_search()`
   
5. **Función maestra**: `ajustar_gaussiana()`
   - Parámetros: n_campanas, metodo_init, maxfev, method, bounds
   - Retorna: y_pred, params, r2, rmse, mae, max_ae, converged, mensaje

6. **Búsqueda sistemática**: `buscar_parametros_optimos()`
   - Para una curva, testea todas las combinaciones
   - Ordena por R² descendente
   - Retorna DataFrame con todos los resultados

7. **Análisis piloto**: Procesa 5 curvas como prueba
   - 5 curvas × 32 configuraciones = 160 filas
   - Genera tablas de resultados y configuraciones ganadoras
   - Visualiza en gráficas y heatmaps

8. **Código de escaleo**: Comentado pero listo para descomentar
   - Procesa 500 curvas completas (~15-30 minutos)
   - Genera tablas finales

**🎯 Uso**: Ejecutar en secuencia. Adaptable según hardware.

---

## Parámetros de Búsqueda

### Combinaciones Testeadas (por curva)

| Parámetro | Valores | Cantidad |
|-----------|---------|----------|
| `n_campanas` | 1, 2, 3, 4, 5, 6, 7, 8 | 8 |
| `metodo_init` | altura, prominencia, uniforme, grid_search | 4 |
| `maxfev` | 4000 (fijo) | 1 |
| `method` | lm (fijo) | 1 |
| **Total por curva** | | **32 configuraciones** |

### Escalas de Pixelización

Por defecto, se testa en escala X25 (más puntos = mejor convergencia).
Futuro: validación cruzada en todas las escalas [X10, X15, X18, X21, X25].

---

## Resultados Esperados

### Tabla: Piloto (5 curvas)
- **Archivo**: `piloto_resultados_todas_combinaciones.csv`
- **Filas**: 5 × 32 = 160
- **Columnas**: curva_id, n_campanas, metodo_init, r2, rmse, mae, max_ae, params, converged, mensaje

### Tabla: Ganadores Piloto
- **Archivo**: `piloto_configuraciones_ganadoras.csv`
- **Filas**: 5 (una por curva, la de máximo R²)
- **Uso**: Identificar patrones de configuración óptima

### Tabla: Completo (500 curvas) [Futuro]
- **Archivo**: `completo_resultados_todas_combinaciones.csv`
- **Filas**: 500 × 32 = 16,000
- **Tiempo estimado**: 15-30 minutos

---

## Métricas de Éxito

✅ Criterios de validación después de ejecutar:

```python
✓ R² en rango [0, 1] para todas las filas
✓ RMSE y MAE positivos
✓ max_ae ≥ MAE (desigualdad siempre cumplida)
✓ RMSE ≥ MAE (desigualdad siempre cumplida)
✓ Convergencia > 90%
✓ Total de filas = curvas × configuraciones
✓ Sin valores NaN en R², RMSE, MAE
```

---

## Hallazgos Esperados

### De la fase piloto:

1. **Método de inicialización más frecuente**: Probablemente 'altura'
2. **Número óptimo de campanas**: Probablemente 3-5 en promedio
3. **Rango de R²**: Esperado > 0.97 para mayoría
4. **Variabilidad entre curvas**: Importante identificar outliers (R² bajo)

### Análisis posterior (en notebook 03):

- Correlacionar geometría de curva (curvatura, picos, etc.) con config óptima
- Entrenar Random Forest para predecir automáticamente
- Generar reglas de decisión interpretables

---

## Cómo Ejecutar

### Opción 1: Modo Piloto (Rápido, ~2-3 minutos)

```python
# En notebook 01_optimizacion_gaussiana_parametros.ipynb
# Simplemente ejecutar todas las celdas en orden
# Procesa 5 curvas con 32 configuraciones cada una
```

✅ Recomendado para validar la lógica antes de escalar.

### Opción 2: Escalado a 500 Curvas (Lento, ~20 minutos)

```python
# En sección 15 del notebook 01:
# 1. Descomentar el bloque de código
# 2. Ejecutar la celda
# ⚠️  Puede tomar 15-30 minutos según CPU/GPU
```

💡 Usar si ya validaste el piloto y quieres resultados definitivos.

---

## Stack Tecnológico

```
NumPy          - Álgebra lineal, FFT
Pandas         - Manipulación de tablas
SciPy
  ├─ optimize.curve_fit    → Ajuste de parámetros
  ├─ signal.find_peaks     → Detección de picos
  └─ stats                 → Pruebas estadísticas
Matplotlib     - Visualizaciones
Joblib         - Serialización (modelos)
Scikit-learn   - (futuro) Meta-modelos
```

---

## Paleta de Colores

Siguiendo el estilo del proyecto:

```python
PALETTE = {
    'gaussiana': '#A23B72',    # Magenta
    'datos': '#5BC0EB',        # Azul
    'referencia': '#e0e0e0',   # Gris
    'exito': '#9BC995',        # Verde
    'error': '#E84855'         # Rojo
}
```

---

## Próximos Pasos

1. **Ejecutar piloto** (notebook 01) → Validar lógica
2. **Analizar resultados** → Identificar patrones
3. **Crear notebook 02_analisis_resultados.ipynb** → Visualizar y caracterizar
4. **Entrenar meta-modelo** → Predecir config óptima desde geometría
5. **Comparar vs polinomios y splines** → Tabla de competencia final

---

## Preguntas Clave a Responder

- ❓ ¿Cuál método de inicialización es más robusto?
- ❓ ¿Hay un número óptimo de campanas global o es curva-específico?
- ❓ ¿Qué características geométricas predicen la config óptima?
- ❓ ¿La validación cruzada por escala (X10-X25) cambia los ganadores?
- ❓ ¿Cuál es el R² máximo teórico con gaussianas puras?

---

## Referencias Teóricas

- **Gaussian basis functions**: Fundamentación en aprox. de funciones (Radial Basis Functions)
- **Curve fitting**: Problema de optimización no lineal (Levenberg-Marquardt)
- **Model selection**: Trade-off entre complejidad y precisión (parsimonia)

---

## Autor y Fecha

**Proyecto**: Servicio Social — Aproximación y Caracterización de Curvas
**Módulo**: Método Gaussiano — Optimización Parametrizada
**Creado**: Julio 2026

---

## Notas de Desarrollo

- ✅ Metodología documentada
- ✅ Implementación completa con funciones parametrizadas
- ✅ Búsqueda sistemática lista
- ⏳ Análisis de resultados (próximo notebook)
- ⏳ Meta-modelo de predicción (futuro)
- ⏳ Validación cruzada por escala (futuro)

