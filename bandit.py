"""
Algoritmo Contextual Zooming — bandit.py
=========================================
Implementa el Algorithm 1 de Singhal et al. (2025).

Archivos de entrada:
    dataset_final_bandit.csv  — datos de mercado SIN ruido
    contextos.csv             — contextos CON ruido Student-t

Normalización:
    - Contextos (gamma_hat, lambda_I_hat, eta_I_hat): ya vienen
      normalizados a [0,1] en contextos.csv
    - Datos de mercado (lambda_S, lambda_I, eta_S, eta_I,
      g_w_hat_MWh, g_w_real_MWh): se usan en euros/MWh directamente
    - Puja p_norm ∈ [0,1]: 0.5 = forecast bid (Δ=0)
    - Recompensa π = clip((R_bandit - R_fc) / ESCALA_GLOBAL, -1, 1)
      donde ESCALA_GLOBAL = percentil 95 de (R_max - R_min)
      Esto evita divisiones por rangos pequeños que dan valores enormes

Genera: resultado_bandit.csv
"""

import numpy as np
import pandas as pd
from bola    import Bola, calcular_indice, DIM_CTX, RADIO_MIN, T_TOTAL
from mercado import (calcular_ingreso, calcular_recompensa,
                     calcular_rango, Y_FC, DELTA_MW)

# ================================================================
# PARÁMETROS
# ================================================================

SEED = 42
W    = 24   # batch size (horas por día)

ARCHIVO_MERCADO   = "dataset_final.csv"
ARCHIVO_CONTEXTOS = "contextos.csv"

ARCHIVO_SAL       = "resultado_bandit.csv"

# ================================================================
# FUNCIÓN 1 — CARGAR DATOS
# ================================================================

def cargar_datos():
    """
    Carga los dos archivos de entrada y los alinea por timestamp.

    Del dataset_final_bandit.csv coge los datos de mercado SIN ruido:
        g_w_hat_MWh, g_w_real_MWh, lambda_S, lambda_I, eta_S, eta_I

    Del contextos.csv coge SOLO las tres columnas de contexto
    con ruido Student-t ya normalizadas a [0,1]:
        gamma_hat, lambda_I_hat, eta_I_hat

    Devuelve:
        df_mercado : DataFrame con datos de mercado
        ctx        : array (T, 3) con contextos normalizados
        T          : número total de horas (múltiplo de W)
        fechas     : índice de timestamps
    """
    print("Cargando datos...")

    # Datos de mercado — sin ruido
    df_m = pd.read_csv(ARCHIVO_MERCADO, parse_dates=["timestamp"])
    df_m = df_m.set_index("timestamp").sort_index()

    # Contextos — con ruido Student-t, ya normalizados [0,1]
    df_c = pd.read_csv(ARCHIVO_CONTEXTOS, parse_dates=["timestamp"])
    df_c = df_c.set_index("timestamp").sort_index()

    # Alinear por timestamp (inner join)
    df_m = df_m.join(df_c[["gamma_hat", "lambda_I_hat", "eta_I_hat"]],
                     how="inner")
    df_m = df_m.dropna(subset=["g_w_hat_MWh", "g_w_real_MWh",
                                "lambda_S", "lambda_I",
                                "eta_S", "eta_I",
                                "gamma_hat", "lambda_I_hat", "eta_I_hat"])

    # Asegurar múltiplo de W
    T = (len(df_m) // W) * W
    df_m = df_m.iloc[:T]

    ctx = np.stack([
        df_m["gamma_hat"].values,      # ya en [0,1] con ruido
        df_m["lambda_I_hat"].values,   # ya en [0,1] con ruido
        df_m["eta_I_hat"].values,      # ya en [0,1] con ruido
    ], axis=1)

    print(f"  T={T} horas | {df_m.index[0]} → {df_m.index[-1]}")
    print(f"  Contextos: gamma_hat ∈ [{ctx[:,0].min():.3f}, "
          f"{ctx[:,0].max():.3f}]")
    print(f"             lambda_I_hat ∈ [{ctx[:,1].min():.3f}, "
          f"{ctx[:,1].max():.3f}]")
    print(f"             eta_I_hat ∈ [{ctx[:,2].min():.3f}, "
          f"{ctx[:,2].max():.3f}]")

    return df_m, ctx, T, df_m.index

# ================================================================
# FUNCIÓN 2 — SIMULAR BANDIT (Algorithm 1)
# ================================================================

def simular_bandit(ctx, df_mercado, T):
    """
    Implementa el Algorithm 1 del paper con feedback delay W=24.

    En cada batch de W horas:
      1. PREDICT: para cada contexto x_t, selecciona la bola relevante
         con mayor índice (ec. 11) y muestrea una puja dentro de ella.
      2. OBSERVE: recibe las W recompensas al final del día.
      3. UPDATE: actualiza contadores y activa bolas hijas si procede.

    La recompensa se normaliza con ESCALA_GLOBAL (percentil 95 del
    rango de ingresos) para evitar valores extremos en horas con
    rango pequeño. 

    Recibe:
        ctx        : array (T,3) contextos normalizados con ruido
        df_mercado : DataFrame con datos de mercado sin ruido
        T          : número total de horas

    Devuelve:
        pujas_norm   : array (T,)        pujas normalizadas [0,1]
        recompensas  : array (T,)        recompensas normalizadas [-1,1]
        n_bolas_hist : array (T//W,)     nº bolas activas por batch
        bolas_fin    : list              bolas al final de la simulación
    """
    print("\nSimulando Contextual Zooming...")

    # Extraer arrays de mercado para acceso rápido
    g_hat_MW  = df_mercado["g_w_hat_MWh"].values
    g_real_MW = df_mercado["g_w_real_MWh"].values
    lambda_S  = df_mercado["lambda_S"].values
    lambda_I  = df_mercado["lambda_I"].values
    eta_S     = df_mercado["eta_S"].values
    eta_I     = df_mercado["eta_I"].values

    # ── Escala global para normalizar recompensas ─────────────
    # Calculada una sola vez antes del bucle.
    # Percentil 95 de (R_max - R_min) sobre todas las horas.
    print("  Calculando escala global de recompensas...")
    rangos = []
    for t in range(T):
        R_max, R_min, _, _ = calcular_rango(
            g_hat_MW[t], g_real_MW[t],
            lambda_S[t], lambda_I[t],
            eta_S[t],    eta_I[t]
        )
        rangos.append(R_max - R_min)
    ESCALA_GLOBAL = float(np.percentile(rangos, 95))
    print(f"  Escala global: {ESCALA_GLOBAL:,.0f} €")

    # ── Actualizar T_TOTAL en bola para conf() correcto ───────
    import bola as bola_mod
    bola_mod.T_TOTAL = T

    # ── Inicialización (líneas 1-3 del Algorithm 1) ───────────
    Bola._cnt = 0
    bolas = [Bola([0.5, 0.5, 0.5, 0.5], 1.0)]

    pujas_norm   = np.zeros(T)
    recompensas  = np.zeros(T)
    n_bolas_hist = []

    # ── Bucle por batches (línea 4: FOR EACH BATCH b) ─────────
    for b in range(T // W):
        t0 = b * W
        t1 = t0 + W

        # Snapshot de bolas al inicio del batch
        A0 = list(bolas)

        # ── FASE PREDICT (líneas 6-11) ────────────────────────
        B_elegidas = []
        y_elegidas = []

        for t in range(t0, t1):
            x_t = ctx[t]

            # Bolas relevantes para este contexto
            relevantes = [b for b in A0 if b.es_relevante(x_t, A0)]
            if not relevantes:
                cands = [b for b in A0 if b.ctx_dentro(x_t)]
                relevantes = cands if cands else \
                             [max(A0, key=lambda b: b.radio)]

            # Seleccionar bola con mayor ÍNDICE completo (ec. 11)
            indices = [calcular_indice(b, A0) for b in relevantes]
            B_t     = relevantes[int(np.argmax(indices))]

            # Elegir puja aleatoria dentro de la bola (línea 10)
            y_t = B_t.elegir_puja()

            B_elegidas.append(B_t)
            y_elegidas.append(y_t)
            pujas_norm[t] = y_t

        # ── OBSERVAR RECOMPENSAS del batch (línea 12) ─────────
        # Delay W=24: recibimos los 24 resultados juntos.

        rews = []
        for i in range(W):
            t = t0 + i

            R_ban, _, _ = calcular_ingreso(
                y_elegidas[i],
                g_hat_MW[t], g_real_MW[t],
                lambda_S[t], lambda_I[t],
                eta_S[t],    eta_I[t]
            )
            R_fc, _, _ = calcular_ingreso(
                Y_FC,
                g_hat_MW[t], g_real_MW[t],
                lambda_S[t], lambda_I[t],
                eta_S[t],    eta_I[t]
            )
            pi = float(np.clip((R_ban - R_fc) / ESCALA_GLOBAL, -1.0, 1.0))

            rews.append(pi)
            recompensas[t] = pi

        # ── FASE UPDATE (líneas 13-20) ────────────────────────
        for i in range(W):
            t    = t0 + i
            B_t  = B_elegidas[i]
            y_t  = y_elegidas[i]
            pi_t = rews[i]
            x_t  = ctx[t]

            p4 = np.array([*x_t, y_t])

            # Regla de activación (líneas 14-17)
            if (B_t.conf() <= B_t.radio
                    and B_t.radio / 2.0 >= RADIO_MIN
                    and B_t.en_dominio(p4, bolas)):
                bolas.append(Bola(p4, B_t.radio / 2.0))

            # Actualizar contadores (línea 19)
            B_t.n      += 1
            B_t.reward += pi_t

        n_bolas_hist.append(len(bolas))

        if b % 50 == 0:
            print(f"  Batch {b+1}/{T//W}  bolas={len(bolas)}")

    return pujas_norm, recompensas, np.array(n_bolas_hist), bolas

# ================================================================
# FUNCIÓN 3 — CALCULAR RESULTADOS
# ================================================================

def calcular_resultados(pujas_norm, df_mercado, T):
    """
    Dado el array de pujas del bandit, calcula para cada hora:
        - Ingreso bandit en euros (total, DA, RT)
        - Ingreso forecast bid en euros
        - Diferencia (reward vs forecast)
        - Delta en MWh

    Los ingresos se calculan con datos de mercado SIN ruido.
    """
    print("\nCalculando resultados...")

    g_hat_MW  = df_mercado["g_w_hat_MWh"].values
    g_real_MW = df_mercado["g_w_real_MWh"].values
    lambda_S  = df_mercado["lambda_S"].values
    lambda_I  = df_mercado["lambda_I"].values
    eta_S     = df_mercado["eta_S"].values
    eta_I     = df_mercado["eta_I"].values

    ing_ban = np.zeros(T)
    da_ban  = np.zeros(T)
    rt_ban  = np.zeros(T)
    ing_fc  = np.zeros(T)
    deltas  = np.zeros(T)

    for t in range(T):
        tot, da, rt = calcular_ingreso(
            pujas_norm[t],
            g_hat_MW[t], g_real_MW[t],
            lambda_S[t], lambda_I[t],
            eta_S[t],    eta_I[t]
        )
        ing_ban[t] = tot
        da_ban[t]  = da
        rt_ban[t]  = rt
        deltas[t]  = (pujas_norm[t] - Y_FC) * 2 * DELTA_MW

        tot_fc, _, _ = calcular_ingreso(
            Y_FC,
            g_hat_MW[t], g_real_MW[t],
            lambda_S[t], lambda_I[t],
            eta_S[t],    eta_I[t]
        )
        ing_fc[t] = tot_fc

    return ing_ban, da_ban, rt_ban, ing_fc, deltas

# ================================================================
# FUNCIÓN 4 — GUARDAR CSV
# ================================================================

def guardar_csv(pujas_norm, ing_ban, da_ban, rt_ban,
                ing_fc, recompensas, deltas,
                n_bolas_hist, df_mercado, fechas, T):
    """Guarda resultado_bandit.csv con todas las columnas."""

    df_res = pd.DataFrame({
        "timestamp":      fechas[:T],
        "puja_norm":      pujas_norm,
        "puja_MW":        df_mercado["g_w_hat_MWh"].values + deltas,
        "g_w_hat_MW":     df_mercado["g_w_hat_MWh"].values,
        "g_w_real_MW":    df_mercado["g_w_real_MWh"].values,
        "delta_MW":       deltas,
        "recompensa":     recompensas,
        "ingreso_bandit": ing_ban,
        "ingreso_fc":     ing_fc,
        "reward_vs_fc":   ing_ban - ing_fc,
        "da_bandit":      da_ban,
        "rt_bandit":      rt_ban,
        "n_bolas":        np.repeat(n_bolas_hist, W)[:T],
        "lambda_S":       df_mercado["lambda_S"].values,
        "lambda_I":       df_mercado["lambda_I"].values,
    })
    df_res.to_csv(ARCHIVO_SAL, index=False)
    print(f"\n✓ CSV guardado: {ARCHIVO_SAL}")

# ================================================================
# FUNCIÓN 5 — IMPRIMIR RESUMEN
# ================================================================

def imprimir_resumen(ing_ban, da_ban, rt_ban, ing_fc,
                     deltas, recompensas, bolas_fin):
    """Imprime estadísticas principales por terminal."""

    print(f"\n── Resultados ────────────────────────────────────────")
    print(f"  Bolas finales: {len(bolas_fin)}")

    print(f"\n  Ingreso medio por hora:")
    print(f"    Bandit:      {ing_ban.mean():>12,.0f} €")
    print(f"    Forecast:    {ing_fc.mean():>12,.0f} €")
    print(f"    Diferencia:  {(ing_ban-ing_fc).mean():>+12,.0f} €")

    print(f"\n  Ingreso acumulado total:")
    print(f"    Bandit:      {ing_ban.sum():>15,.0f} €")
    print(f"    Forecast:    {ing_fc.sum():>15,.0f} €")
    print(f"    Diferencia:  {(ing_ban-ing_fc).sum():>+15,.0f} €")

    print(f"\n  Desglose bandit:")
    print(f"    DA medio:    {da_ban.mean():>12,.0f} €/h")
    print(f"    RT medio:    {rt_ban.mean():>12,.0f} €/h")

    print(f"\n  Estadísticas Δ (puja vs forecast):")
    print(f"    Δ medio:     {deltas.mean():>+10.1f} MWh")
    print(f"    Δ std:       {deltas.std():>10.1f} MWh")
    print(f"    % Δ < 0:     {(deltas<0).mean()*100:>9.1f}%  (puja menos)")
    print(f"    % Δ = 0:     {(deltas==0).mean()*100:>9.1f}%  (forecast bid)")
    print(f"    % Δ > 0:     {(deltas>0).mean()*100:>9.1f}%  (puja más)")

    print(f"\n  Recompensa normalizada π:")
    print(f"    Media:       {recompensas.mean():>10.4f}")
    print(f"    Std:         {recompensas.std():>10.4f}  ← debe ser ~0.3-0.5")
    print(f"    % π > 0:     {(recompensas>0).mean()*100:>9.1f}%  (supera forecast)")
    print(f"    % π = ±1:    {(np.abs(recompensas)==1.0).mean()*100:>9.1f}%  (saturados)")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    np.random.seed(SEED)

    # 1. Cargar datos
    df_mercado, ctx, T, fechas = cargar_datos()

    # 2. Simular bandit
    pujas_norm, recompensas, n_bolas_hist, bolas_fin = \
        simular_bandit(ctx, df_mercado, T)

    # 3. Calcular resultados en euros
    ing_ban, da_ban, rt_ban, ing_fc, deltas = \
        calcular_resultados(pujas_norm, df_mercado, T)

    # 4. Guardar CSV
    guardar_csv(pujas_norm, ing_ban, da_ban, rt_ban,
                ing_fc, recompensas, deltas,
                n_bolas_hist, df_mercado, fechas, T)

    # 5. Imprimir resumen
    imprimir_resumen(ing_ban, da_ban, rt_ban, ing_fc,
                     deltas, recompensas, bolas_fin)