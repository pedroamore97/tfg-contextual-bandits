"""
Simulador de mercado — mercado.py
==================================
Implementa el modelo de mercado del paper (Section II-B, ecuación 3).

ESPACIO DE PUJAS:
    p_norm ∈ [0, 1]
    p_norm = 0.0 → p^w = ĝ^w - 250 MWh  (puja mínima)
    p_norm = 0.5 → p^w = ĝ^w            (forecast bid, Δ=0)
    p_norm = 1.0 → p^w = ĝ^w + 250 MWh  (puja máxima)

FÓRMULAS (paper eq. 3):
    Δ         = (p_norm - 0.5) × 2 × 250        [MWh]   ∈ [-250, +250]
    p^w       = ĝ^w + Δ                          [MWh]
    λ^S_mod   = λ^S + η^S · Δ                    [€/MWh]  η^S < 0
    λ^I_mod   = λ^I + η^I · Δ                    [€/MWh]  η^I < 0
    imb       = g^w_real - p^w                    [MWh]
    R         = λ^S_mod · p^w + λ^I_mod · imb    [€]

RECOMPENSA NORMALIZADA (relativa al forecast bid):
    R_fc  = R(p_norm=0.5)    ingreso si hubieras pujado el forecast
    π     = (R_bandit - R_fc) / escala_global
    π > 0 → la puja del bandit mejora el forecast bid
    π < 0 → la puja del bandit empeora el forecast bid

    escala_global: std de (R_max - R_fc) sobre el dataset histórico,
                   para normalizar π a escala comparable entre días.
"""

import numpy as np
import pandas as pd

DELTA_MAX = 250.0   # desviación máxima permitida [MWh]
P_FC      = 0.5     # forecast bid normalizado (Δ=0)

# Aliases para compatibilidad con simulacion.py
Y_FC     = P_FC
DELTA_MW = DELTA_MAX

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: calcular ingreso
# ══════════════════════════════════════════════════════════════════════════════

def calcular_ingreso(p_norm, g_hat, g_real, lambda_S, lambda_I, eta_S, eta_I):
    """
    Calcula el ingreso total del WPP dado p_norm ∈ [0,1].

    Parámetros
    ----------
    p_norm   : float  puja normalizada ∈ [0,1]
    g_hat    : float  forecast generación eólica [MWh]
    g_real   : float  generación real [MWh]
    lambda_S : float  precio spot day-ahead [€/MWh]
    lambda_I : float  precio imbalance real-time [€/MWh]
    eta_S    : float  sensibilidad spot (NEGATIVO, ej: -0.021)
    eta_I    : float  sensibilidad imbalance (POSITIVO, ej: 0.547)

    Retorna
    -------
    ingreso_total : float  [€]
    ingreso_DA    : float  ingreso day-ahead [€]
    ingreso_RT    : float  ingreso real-time [€]
    info          : dict   variables intermedias para debug
    """
    # ── Desviación respecto al forecast ──────────────────────────────────────
    Delta = (p_norm - P_FC) * 2 * DELTA_MAX    # ∈ [-250, +250] MWh

    # ── Puja real ─────────────────────────────────────────────────────────────
    p_w = g_hat + Delta                         # [MWh]

    # ── Precio spot modificado ────────────────────────────────────────────────

    lambda_S_mod = lambda_S + eta_S * Delta     # [€/MWh]

    # ── Precio imbalance modificado ───────────────────────────────────────────
    
    lambda_I_mod = lambda_I + eta_I * Delta     # [€/MWh]

    # ── Imbalance del WPP ─────────────────────────────────────────────────────
    imb = g_real - p_w                          # [MWh]
    # imb > 0: generó más de lo pujado (exceso WPP)
    # imb < 0: generó menos de lo pujado (déficit WPP)

    # ── Ingresos ──────────────────────────────────────────────────────────────
    ingreso_DA    = lambda_S_mod * p_w          # [€]
    ingreso_RT    = lambda_I_mod * imb          # [€]
    ingreso_total = ingreso_DA + ingreso_RT     # [€]

    return float(ingreso_total), float(ingreso_DA), float(ingreso_RT)


# ══════════════════════════════════════════════════════════════════════════════
# RANGO DE INGRESOS (para normalización)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_rango(g_hat, g_real, lambda_S, lambda_I, eta_S, eta_I):
    """
    Calcula R_max y R_min evaluando en los extremos del espacio de pujas.

    Devuelve
    -------
    R_max : float  máximo ingreso posible [€]
    R_min : float  mínimo ingreso posible [€]
    p_opt : float  p_norm que maximiza el ingreso
    """
    R_low,  _, _ = calcular_ingreso(0.0, g_hat, g_real,
                                        lambda_S, lambda_I, eta_S, eta_I)
    R_high, _, _ = calcular_ingreso(1.0, g_hat, g_real,
                                        lambda_S, lambda_I, eta_S, eta_I)
    R_fc,   _, _ = calcular_ingreso(P_FC, g_hat, g_real,
                                        lambda_S, lambda_I, eta_S, eta_I)

    if R_high >= R_low:
        R_max, R_min, p_opt = R_high, R_low, 1.0
    else:
        R_max, R_min, p_opt = R_low, R_high, 0.0

    return float(R_max), float(R_min), float(R_fc), float(p_opt)


# ══════════════════════════════════════════════════════════════════════════════
# RECOMPENSA NORMALIZADA
# ══════════════════════════════════════════════════════════════════════════════

def calcular_recompensa(p_norm, g_hat, g_real,
                        lambda_S, lambda_I, eta_S, eta_I,
                        escala_global):
    """
    Recompensa normalizada respecto al forecast bid.

        π = (R_bandit - R_forecast) / escala_global

    π > 0 → la puja mejora respecto a pujar el forecast
    π < 0 → la puja empeora respecto a pujar el forecast

    Parámetros
    ----------
    escala_global : float  factor de normalización [€]
                           típicamente std(R_max - R_fc) sobre el dataset

    Devuelve
    -------
    pi      : float  recompensa normalizada ∈ [-1, 1] aprox
    R_ban   : float  ingreso con puja del bandit [€]
    R_fc    : float  ingreso con forecast bid [€]
    """
    R_ban, _, _ = calcular_ingreso(p_norm, g_hat, g_real,
                                       lambda_S, lambda_I, eta_S, eta_I)
    R_fc,  _, _ = calcular_ingreso(P_FC, g_hat, g_real,
                                       lambda_S, lambda_I, eta_S, eta_I)

    pi = (R_ban - R_fc) / escala_global
    return float(np.clip(pi, -1.0, 1.0)), float(R_ban), float(R_fc)


# ══════════════════════════════════════════════════════════════════════════════
# CALCULAR ESCALA GLOBAL DESDE EL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def calcular_escala_global(df, col_map=None):
    """
    Calcula la escala de normalización como std(R_max - R_fc) sobre el dataset.

    Parámetros
    ----------
    df      : pd.DataFrame  con columnas del contexto
    col_map : dict          mapeo de nombres de columnas, por defecto:
              {
                "g_hat":    "g_w_hat",
                "g_real":   "g_w_real",
                "lambda_S": "lambda_S",
                "lambda_I": "lambda_I",
                "eta_S":    "eta_S",
                "eta_I":    "eta_I",
              }

    Devuelve
    -------
    escala : float  std(R_max - R_fc) [€]
    """
    if col_map is None:
        col_map = {
            "g_hat":    "g_w_hat",
            "g_real":   "g_w_real",
            "lambda_S": "lambda_S",
            "lambda_I": "lambda_I",
            "eta_S":    "eta_S",
            "eta_I":    "eta_I",
        }

    diferencias = []
    for _, row in df.iterrows():
        R_max, _, R_fc, _ = calcular_rango(
            row[col_map["g_hat"]],
            row[col_map["g_real"]],
            row[col_map["lambda_S"]],
            row[col_map["lambda_I"]],
            row[col_map["eta_S"]],
            row[col_map["eta_I"]],
        )
        diferencias.append(R_max - R_fc)

    escala = float(np.std(diferencias))
    print(f"Escala global: {escala:,.1f} €  "
          f"(std de R_max-R_fc sobre {len(diferencias)} filas)")
    return escala