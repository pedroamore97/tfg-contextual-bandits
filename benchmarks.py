"""
Benchmarks — benchmarks.py
============================
Implementa las estrategias de referencia del paper Sec. V-B:

    1. Oráculo      — puja óptima con datos reales de HOY
    2. D-1 Prediction — puja óptima con datos de AYER (ec. 19)
    3. Forecast Bid — siempre puja ĝ^w (p_norm=0.5, Δ=0)

La función puja_optima() aproxima el problema bilevel del paper
usando la aproximación lineal de la curva de oferta/demanda:

    f^w_t = argmax_{p∈[0,1]} [λ^S + η^S·Δ]·p^w
                             + [λ^I - η^I·Δ]·(g^w_real - p^w)

    Para el oráculo: usa datos de t (HOY)
    Para D-1:        usa datos de t-24 (AYER misma hora)

Archivos de entrada:
    dataset_final_bandit.csv  — datos de mercado SIN ruido
    contextos.csv             — solo para alinear timestamps

Genera: resultado_benchmarks.csv
"""

import numpy as np
import pandas as pd
from mercado import calcular_ingreso, Y_FC, DELTA_MW

N_GRID = 50   # puntos del grid para búsqueda de puja óptima
W      = 24   # horas por día

ARCHIVO_MERCADO   = "dataset_final.csv"
ARCHIVO_CONTEXTOS = "contextos.csv"

ARCHIVO_SAL       = "resultado_benchmark.csv"

# Grid fijo de pujas normalizadas [0,1]
GRID = np.linspace(0.0, 1.0, N_GRID)


# ================================================================
# FUNCIÓN 1 — CARGAR DATOS
# ================================================================

def cargar_datos():
    """
    Carga los datos de mercado y alinea con contextos.csv
    para usar exactamente las mismas horas que el bandit.

    Devuelve:
        df_mercado : DataFrame con datos de mercado sin ruido
        T          : número total de horas válidas
        fechas     : índice de timestamps
    """
    print("Cargando datos...")

    df_m = pd.read_csv(ARCHIVO_MERCADO, parse_dates=["timestamp"])
    df_m = df_m.set_index("timestamp").sort_index()

    df_c = pd.read_csv(ARCHIVO_CONTEXTOS, parse_dates=["timestamp"])
    df_c = df_c.set_index("timestamp").sort_index()

    # Join para usar las mismas horas que el bandit
    df_m = df_m.join(df_c[["gamma_hat"]], how="inner")
    df_m = df_m.dropna(subset=["g_w_hat_MWh", "g_w_real_MWh",
                                "lambda_S", "lambda_I",
                                "eta_S", "eta_I", "gamma_hat"])

    T = (len(df_m) // W) * W
    df_m = df_m.iloc[:T]

    print(f"  T={T} horas | {df_m.index[0]} → {df_m.index[-1]}")

    return df_m, T, df_m.index

# ================================================================
# FUNCIÓN 2 — PUJA ÓPTIMA
# ================================================================

def puja_optima(g_hat_MW, g_real_MW,
                lambda_S, lambda_I, eta_S, eta_I):
    """
    Encuentra la puja p_norm ∈ [0,1] que maximiza el ingreso
    dados los datos de mercado de un instante específico.

    Aproxima el problema bilevel del paper usando la fórmula lineal:
        R(p) = [λ^S + η^S·Δ]·p^w + [λ^I - η^I·Δ]·(g^w_real - p^w)

    Grid search sobre N_GRID pujas uniformes en [0,1].

    Devuelve:
        p_opt : puja óptima normalizada ∈ [0,1]
    """
    mejor_ingreso = -np.inf
    p_opt         = Y_FC   # por defecto forecast bid

    for p_norm in GRID:
        ingreso_total, _, _ = calcular_ingreso(
            p_norm,
            g_hat_MW, g_real_MW,
            lambda_S, lambda_I,
            eta_S,    eta_I
        )
        if ingreso_total > mejor_ingreso:
            mejor_ingreso = ingreso_total
            p_opt         = p_norm

    return float(p_opt)

# ================================================================
# FUNCIÓN 3 — ORÁCULO
# ================================================================

def simular_oraculo(df_mercado, T):
    """
    Para cada hora t encuentra la puja óptima usando datos REALES
    de esa hora — λ^S_t, λ^I_t, η^S_t, η^I_t, g^w_real_t.

    Cota superior teórica: conocimiento perfecto de hoy.
    Ninguna estrategia real puede superarlo porque g^w_real_t
    no se conoce antes de la entrega.

    Devuelve:
        pujas_or : array (T,) pujas normalizadas
    """
    print("  Calculando oráculo...")

    g_hat_MW  = df_mercado["g_w_hat_MWh"].values
    g_real_MW = df_mercado["g_w_real_MWh"].values
    lambda_S  = df_mercado["lambda_S"].values
    lambda_I  = df_mercado["lambda_I"].values
    eta_S     = df_mercado["eta_S"].values
    eta_I     = df_mercado["eta_I"].values

    pujas_or = np.zeros(T)

    for t in range(T):
        # Datos reales de HOY
        pujas_or[t] = puja_optima(
            g_hat_MW[t], g_real_MW[t],
            lambda_S[t], lambda_I[t],
            eta_S[t],    eta_I[t]
        )
        if t % 2000 == 0:
            print(f"    hora {t}/{T}")

    return pujas_or
    

# ================================================================
# FUNCIÓN 4 — D-1 PREDICTION
# ================================================================

def simular_d1(df_mercado, T):
    """
    Para cada hora t:
      - Si t < W: forecast bid (sin historial)
      - Si t >= W: puja óptima usando datos de AYER (t-W)

    OPTIMIZACIÓN con datos de t-W:
        argmax_{p} [λ^S_{t-W} + η^S_{t-W}·Δ]·p^w
                 + [λ^I_{t-W} - η^I_{t-W}·Δ]·(g^w_real,t-W - p^w)

    INGRESO REAL calculado con datos de HOY (t).

    Aproxima el problema bilevel ec. (19) del paper usando
    la aproximación lineal con datos del día anterior.

    Devuelve:
        pujas_d1 : array (T,) pujas normalizadas
    """
    print("  Calculando D-1 prediction...")

    g_hat_MW  = df_mercado["g_w_hat_MWh"].values
    g_real_MW = df_mercado["g_w_real_MWh"].values
    lambda_S  = df_mercado["lambda_S"].values
    lambda_I  = df_mercado["lambda_I"].values
    eta_S     = df_mercado["eta_S"].values
    eta_I     = df_mercado["eta_I"].values

    pujas_d1 = np.zeros(T)

    for t in range(T):
        if t < W:
            # Sin historial → forecast bid
            pujas_d1[t] = Y_FC
        else:
            # PASO 1: optimizar con datos de AYER (t-W)
            p_opt = puja_optima(
                g_hat_MW[t-W],  g_real_MW[t-W],
                lambda_S[t-W],  lambda_I[t-W],
                eta_S[t-W],     eta_I[t-W]
            )
            # PASO 2: aplicar esa puja hoy
            # Δ en MWh es el mismo independientemente del día
            pujas_d1[t] = p_opt

        if t % 2000 == 0:
            print(f"    hora {t}/{T}")

    return pujas_d1

# ================================================================
# FUNCIÓN 5 — FORECAST BID
# ================================================================

def simular_forecast(T):
    """
    Siempre puja p_norm=0.5 → Δ=0 → p^w = ĝ^w.
    Benchmark base del paper.
    """
    print("  Calculando forecast bid...")
    return np.full(T, Y_FC)

# ================================================================
# FUNCIÓN 6 — CALCULAR RESULTADOS EN EUROS
# ================================================================

def calcular_resultados(pujas, df_mercado, T):
    """
    Dado un array de pujas normalizadas, calcula para cada hora
    el ingreso total, day-ahead y real-time en euros.

    Usa datos de mercado SIN ruido.

    Devuelve:
        ing  : array (T,) ingreso total €
        da   : array (T,) ingreso day-ahead €
        rt   : array (T,) ingreso real-time €
        delta: array (T,) desviación en MWh
    """
    g_hat_MW  = df_mercado["g_w_hat_MWh"].values
    g_real_MW = df_mercado["g_w_real_MWh"].values
    lambda_S  = df_mercado["lambda_S"].values
    lambda_I  = df_mercado["lambda_I"].values
    eta_S     = df_mercado["eta_S"].values
    eta_I     = df_mercado["eta_I"].values

    ing   = np.zeros(T)
    da    = np.zeros(T)
    rt    = np.zeros(T)
    delta = np.zeros(T)

    for t in range(T):
        tot, d, r = calcular_ingreso(
            pujas[t],
            g_hat_MW[t], g_real_MW[t],
            lambda_S[t], lambda_I[t],
            eta_S[t],    eta_I[t]
        )
        ing[t]   = tot
        da[t]    = d
        rt[t]    = r
        delta[t] = (pujas[t] - Y_FC) * 2 * DELTA_MW

    return ing, da, rt, delta

# ================================================================
# FUNCIÓN 7 — GUARDAR CSV
# ================================================================

def guardar_csv(p_or, p_d1, p_fc,
                ing_or, da_or, rt_or,
                ing_d1, da_d1, rt_d1,
                ing_fc, da_fc, rt_fc,
                delta_or, delta_d1,
                df_mercado, fechas, T):
    """Guarda resultado_benchmarks.csv."""

    df_res = pd.DataFrame({
        "timestamp":      fechas[:T],
        "g_w_hat_MW":     df_mercado["g_w_hat_MWh"].values,
        "g_w_real_MW":    df_mercado["g_w_real_MWh"].values,
        # Pujas
        "puja_oracle":    df_mercado["g_w_hat_MWh"].values + delta_or,
        "puja_d1":        df_mercado["g_w_hat_MWh"].values + delta_d1,
        "puja_forecast":  df_mercado["g_w_hat_MWh"].values,
        # Deltas
        "delta_oracle":   delta_or,
        "delta_d1":       delta_d1,
        # Ingresos
        "ingreso_oracle": ing_or,
        "ingreso_d1":     ing_d1,
        "ingreso_fc":     ing_fc,
        # Desglose
        "da_oracle":      da_or, "rt_oracle": rt_or,
        "da_d1":          da_d1, "rt_d1":     rt_d1,
        "da_fc":          da_fc, "rt_fc":     rt_fc,
        # Reward vs forecast
        "reward_oracle":  ing_or - ing_fc,
        "reward_d1":      ing_d1 - ing_fc,
        "lambda_S":       df_mercado["lambda_S"].values,
        "lambda_I":       df_mercado["lambda_I"].values,
    })
    df_res.to_csv(ARCHIVO_SAL, index=False)
    print(f"\n✓ CSV guardado: {ARCHIVO_SAL}")

# ================================================================
# FUNCIÓN 8 — IMPRIMIR RESUMEN
# ================================================================

def imprimir_resumen(ing_or, da_or, rt_or,
                     ing_d1, da_d1, rt_d1,
                     ing_fc, da_fc, rt_fc,
                     delta_or, delta_d1):
    """Imprime tabla comparativa por terminal."""

    print(f"\n── Resultados benchmarks ─────────────────────────────")
    print(f"  {'Estrategia':<18} {'DA (€/h)':>12} {'RT (€/h)':>12} "
          f"{'Total (€/h)':>12} {'vs FC':>12}")
    print(f"  {'─'*70}")

    for nom, da, rt, ing in [
        ("Oráculo",      da_or, rt_or, ing_or),
        ("D-1 Pred.",    da_d1, rt_d1, ing_d1),
        ("Forecast Bid", da_fc, rt_fc, ing_fc),
    ]:
        diff = ing.mean() - ing_fc.mean()
        print(f"  {nom:<18} {da.mean():>12,.0f} {rt.mean():>12,.0f} "
              f"{ing.mean():>12,.0f} {diff:>+12,.0f}")

    print(f"\n  Reward acumulado vs Forecast:")
    print(f"    Oráculo: {(ing_or-ing_fc).sum():>+18,.0f} €")
    print(f"    D-1:     {(ing_d1-ing_fc).sum():>+18,.0f} €")

    print(f"\n  Δ medio:")
    print(f"    Oráculo: {delta_or.mean():>+10.1f} MWh")
    print(f"    D-1:     {delta_d1.mean():>+10.1f} MWh")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # 1. Cargar datos
    df_mercado, T, fechas = cargar_datos()

    # 2. Simular estrategias
    print("\nSimulando estrategias...")
    p_or = simular_oraculo(df_mercado, T)
    p_d1 = simular_d1(df_mercado, T)
    p_fc = simular_forecast(T)

    # 3. Calcular resultados en euros
    print("\nCalculando ingresos...")
    ing_or, da_or, rt_or, delta_or = calcular_resultados(p_or, df_mercado, T)
    ing_d1, da_d1, rt_d1, delta_d1 = calcular_resultados(p_d1, df_mercado, T)
    ing_fc, da_fc, rt_fc, _        = calcular_resultados(p_fc, df_mercado, T)

    # 4. Guardar CSV
    guardar_csv(p_or, p_d1, p_fc,
                ing_or, da_or, rt_or,
                ing_d1, da_d1, rt_d1,
                ing_fc, da_fc, rt_fc,
                delta_or, delta_d1,
                df_mercado, fechas, T)

    # 5. Imprimir resumen
    imprimir_resumen(ing_or, da_or, rt_or,
                     ing_d1, da_d1, rt_d1,
                     ing_fc, da_fc, rt_fc,
                     delta_or, delta_d1)
