# tfg-contextual-bandits

Implementación del algoritmo **Contextual Zooming** aplicado a la optimización de pujas de un parque eólico en el mercado eléctrico alemán. Trabajo de Fin de Grado — Universidad de Málaga.

El algoritmo aprende a ajustar la oferta diaria de energía en el mercado *day-ahead* a partir de tres señales de contexto del mercado, minimizando el arrepentimiento frente a una estrategia óptima con información perfecta. El experimento replica el setup numérico de Singhal et al. (2025) sobre datos reales del período julio 2022 – marzo 2024 obtenidos de la plataforma [SMARD](https://www.smard.de/).

---

## Contexto

El productor eólico simulado gestiona una cartera de ~20 GW instalados en la zona de control de 50Hertz (noreste de Alemania), suficiente para actuar como *price-maker* en el mercado diario. La simulación abarca **T = 15 360 subastas horarias** (640 días, W = 24 horas por lote) durante un período de elevada volatilidad de precios.

El vector contextual observado antes de cada subasta es:

| Señal | Descripción |
|---|---|
| `γ̂_t` | Sensibilidad marginal del ingreso *day-ahead* al volumen de oferta |
| `λ̂_t^I` | Previsión del precio de desvío (*real-time*) |
| `η̂_t^I` | Sensibilidad del precio de desvío a la desviación de oferta |

---

## Estructura

```
tfg-contextual-bandits/
│
├── bola.py           # Clase Bola: estructura de datos central del algoritmo
├── mercado.py        # Modelo de ingresos price-maker (ec. 3 del paper)
├── bandit.py         # Algorithm 1 — Contextual Zooming con feedback delay W=24
├── benchmarks.py     # Estrategias de referencia: oráculo, predicción D-1, forecast bid
├── simulacion.py     # Script maestro: ejecuta todo y genera las figuras
│
├── dataset_final.csv # Datos de mercado sin ruido (λ^S, λ^I, η^S, η^I, generación)
└── contextos.csv     # Señales de contexto normalizadas [0,1] con ruido Student-t
```

---

## Requisitos

```
python >= 3.9
numpy
pandas
matplotlib
```

```bash
pip install numpy pandas matplotlib
```

---

## Cómo ejecutar

**Simulación completa** — todas las estrategias + figuras:

```bash
python3 simulacion.py
```

Genera `resultado_bandit.csv`, `resultado_benchmark.csv` y cinco figuras PNG.

**Solo el bandit:**

```bash
python3 bandit.py
```

**Solo los benchmarks:**

```bash
python3 benchmarks.py
```

---

## Resultados

Ingreso medio por hora sobre el período completo (julio 2022 – marzo 2024):

| Estrategia | DA (€/h) | RT (€/h) | Total (€/h) | vs Forecast |
|---|---:|---:|---:|---:|
| Oráculo † | 412 328 | +26 441 | 438 769 | +42 285 |
| **Bandido Contextual** | **405 359** | **−1 284** | **404 075** | **+7 592** |
| Predicción D-1 | 407 514 | −5 691 | 401 823 | +5 339 |
| Forecast Bid | 404 099 | −7 615 | 396 484 | — |

† Cota superior teórica; no es una estrategia causal.

El Bandido Contextual aprende a ofertar sistemáticamente por debajo de la previsión (Δ_t < 0 en el 88,8% de las horas), reduciendo el coste del desvío en el mercado de tiempo real respecto al Forecast Bid.

---

## Figuras generadas

| Archivo | Contenido |
|---|---|
| `fig5_regret.png` | Arrepentimiento medio acumulado R(t)/t vs cota teórica del Teorema 1 |
| `fig6_ingresos.png` | Ingreso acumulado relativo al Forecast Bid por estrategia |
| `fig7_desglose.png` | Desglose day-ahead / real-time por estrategia |
| `fig8_bolas.png` | Evolución del número de bolas activas \|B_t\| (133 al final) |
| `fig10_distribucion_pujas.png` | Distribución de la desviación de oferta Δ_t |

---

## Referencia

Singhal, A. et al. (2025). *Contextual Zooming Bandits for Wind Power Producer Bidding Strategy*. IEEE Transactions on Energy Markets, Policy and Regulation.
