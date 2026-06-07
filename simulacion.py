"""
Simulación completa — simulacion.py
=====================================
Script maestro que ejecuta todas las estrategias y genera
las figuras 5, 6, 7, 8, 10 del paper Singhal et al. (2025).

Requiere en la misma carpeta:
    bola.py, mercado.py, bandit.py, benchmarks.py
    dataset_final.csv, contextos.csv

Ejecutar:
    python3 simulacion.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

from bola       import Bola, DIM_CTX, RADIO_MIN
from mercado    import calcular_ingreso, Y_FC, DELTA_MW
from bandit     import (cargar_datos as cargar_bandit,
                        simular_bandit,
                        calcular_resultados as calc_res_bandit,
                        guardar_csv as guardar_bandit,
                        imprimir_resumen as resumen_bandit)
from benchmarks import (cargar_datos as cargar_bench,
                        simular_oraculo, simular_d1, simular_forecast,
                        calcular_resultados as calc_res_bench,
                        guardar_csv as guardar_bench,
                        imprimir_resumen as resumen_bench)

# ================================================================
# PARÁMETROS
# ================================================================

SEED = 42
W    = 24
D_C  = DIM_CTX + 1   # dimensión de zooming = 4

COLORES = {
    "bandit": "#1f77b4",
    "oracle": "#2ca02c",
    "d1":     "#ff7f0e",
    "fb":     "#d62728",
    "teo":    "darkorange",
}

# ================================================================
# EJECUTAR TODAS LAS ESTRATEGIAS
# ================================================================

print("=" * 55)
print("  Simulación completa — Singhal et al. (2025)")
print("=" * 55)

# ── Bandit ────────────────────────────────────────────────────
print("\n[1/4] Contextual Zooming (bandit)...")
np.random.seed(SEED)
df_mercado, ctx, T, fechas = cargar_bandit()
pujas_cz, rew_cz, n_bolas_hist, bolas_fin = simular_bandit(ctx, df_mercado, T)
ing_cz, da_cz, rt_cz, ing_fc_b, deltas_cz = calc_res_bandit(pujas_cz, df_mercado, T)

# ── Benchmarks ────────────────────────────────────────────────
print("\n[2/4] Oráculo...")
df_bench, T_b, fechas_b = cargar_bench()
p_or = simular_oraculo(df_bench, T_b)

print("\n[3/4] D-1 Prediction...")
p_d1 = simular_d1(df_bench, T_b)

print("\n[4/4] Forecast bid...")
p_fc = simular_forecast(T_b)

print("\nCalculando ingresos benchmarks...")
ing_or,  da_or,  rt_or,  delta_or  = calc_res_bench(p_or, df_bench, T_b)
ing_d1,  da_d1,  rt_d1,  delta_d1  = calc_res_bench(p_d1, df_bench, T_b)
ing_fc2, da_fc,  rt_fc,  _         = calc_res_bench(p_fc, df_bench, T_b)

T_min  = min(T, T_b)
ing_fc = ing_fc2[:T_min]

guardar_bandit(pujas_cz, ing_cz, da_cz, rt_cz,
               ing_fc, rew_cz, deltas_cz,
               n_bolas_hist, df_mercado, fechas, T)

guardar_bench(p_or, p_d1, p_fc,
              ing_or, da_or, rt_or,
              ing_d1, da_d1, rt_d1,
              ing_fc2, da_fc, rt_fc,
              delta_or, delta_d1,
              df_bench, fechas_b, T_b)

# ================================================================
# RESUMEN COMPLETO
# ================================================================

print(f"\n{'='*55}")
print(f"  RESUMEN COMPLETO")
print(f"{'='*55}")
print(f"\n  {'Estrategia':<18} {'DA (€/h)':>12} {'RT (€/h)':>12} "
      f"{'Total (€/h)':>12} {'vs FC':>12}")
print(f"  {'─'*70}")

for nom, da, rt, ing in [
    ("Oráculo",      da_or,           rt_or,           ing_or),
    ("Bandit (CZ)",  da_cz[:T_min],   rt_cz[:T_min],   ing_cz[:T_min]),
    ("D-1 Pred.",    da_d1,           rt_d1,           ing_d1),
    ("Forecast Bid", da_fc,           rt_fc,           ing_fc2),
]:
    diff = ing.mean() - ing_fc2.mean()
    print(f"  {nom:<18} {da.mean():>12,.0f} {rt.mean():>12,.0f} "
          f"{ing.mean():>12,.0f} {diff:>+12,.0f}")

print(f"\n  Reward acumulado vs Forecast:")
print(f"    Oráculo:     {(ing_or  - ing_fc2).sum():>+18,.0f} €")
print(f"    Bandit (CZ): {(ing_cz[:T_min] - ing_fc).sum():>+18,.0f} €")
print(f"    D-1:         {(ing_d1  - ing_fc2).sum():>+18,.0f} €")

print(f"\n  Bolas finales: {len(bolas_fin)}")

# ================================================================
# FIGURAS
# ================================================================

fmt_mes = mdates.DateFormatter("%Y-%m")
loc_mes = mdates.MonthLocator(interval=3)

def fmt_eje(ax):
    ax.xaxis.set_major_formatter(fmt_mes)
    ax.xaxis.set_major_locator(loc_mes)
    ax.tick_params(axis="x", rotation=30)

def regret_medio(ing, ing_ref):
    n = min(len(ing), len(ing_ref))
    return np.cumsum(ing_ref[:n] - ing[:n]) / np.arange(1, n+1)

# Arrays alineados a T_min
ing_cz_t  = ing_cz[:T_min]
ing_or_t  = ing_or[:T_min]
ing_d1_t  = ing_d1[:T_min]
ing_fc_t  = ing_fc[:T_min]
fechas_t  = fechas[:T_min]

reg_cz = regret_medio(ing_cz_t, ing_or_t)

# Cota teórica Teorema 1
t_arr  = np.arange(1, T_min+1)
e1     = (D_C+1) / (D_C+2)
e2     = (D_C-1) / (D_C+2)
forma  = (t_arr**e1 * np.log(np.maximum(t_arr,2)) + W*t_arr**e2) / t_arr
burnin = max(1, int(0.05*T_min))
C      = (reg_cz[burnin:] / np.maximum(forma[burnin:], 1e-12)).max() * 1.05
reg_teo = C * forma

# Ingresos acumulados relativos al FC
ir_cz = np.cumsum(ing_cz_t - ing_fc_t)
ir_d1 = np.cumsum(ing_d1_t - ing_fc_t)
ir_or = np.cumsum(ing_or_t - ing_fc_t)

# ── FIG 1: Regret ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fechas_t, reg_cz,  color=COLORES["bandit"], lw=2.0,
        label="Arrepentimiento empírico (Bandido)")
ax.plot(fechas_t, reg_teo, color=COLORES["teo"], lw=2.0, ls="--",
        label=f"Cota teórica Teorema 1 ($d_c={D_C}$, $W={W}$)")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_ylabel("$R(t)/t$ (€)", fontsize=12)
ax.set_title("Fig. 6.1 — Arrepentimiento medio acumulado vs cota teórica",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, ls=":", alpha=0.4)
fmt_eje(ax)
plt.tight_layout()
plt.savefig("fig1_regret.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ fig1_regret.png")

# ── FIG 2: Ingresos acumulados ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fechas_t, ir_or/1e8, color=COLORES["oracle"],
        lw=1.8, ls="--", label="Oráculo")
ax.plot(fechas_t, ir_cz/1e8, color=COLORES["bandit"],
        lw=2.2, label="Bandido (CZ)")
ax.plot(fechas_t, ir_d1/1e8, color=COLORES["d1"],
        lw=1.8, ls="-.", label="Predicción D-1")
ax.axhline(0, color=COLORES["fb"], lw=1.5, ls="--",
           label="Oferta basada en la previsión (ref.)")
ax.fill_between(fechas_t, ir_cz/1e8, 0,
                where=(ir_cz>0), alpha=0.12, color=COLORES["bandit"])
ax.set_ylabel("Ingreso acumulado relativo (×10⁸ €)", fontsize=12)
ax.set_title("Fig. 6.2 — Ingreso acumulado relativo a la oferta basada en la previsión",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, ls=":", alpha=0.4)
fmt_eje(ax)
plt.tight_layout()
plt.savefig("fig2_ingresos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig2_ingresos.png")

# ── FIG 3: Desglose DA/RT ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ests = ["Oráculo", "Bandido (CZ)", "Pred. D-1",
        "Oferta basada en la previsión"]
idas = [da_or.mean(), da_cz[:T_min].mean(), da_d1.mean(), da_fc.mean()]
irts = [rt_or.mean(), rt_cz[:T_min].mean(), rt_d1.mean(), rt_fc.mean()]
cols = [COLORES["oracle"], COLORES["bandit"],
        COLORES["d1"],     COLORES["fb"]]
x  = np.arange(len(ests))
bw = 0.35
ax.barh(x+bw/2, idas, bw, color=cols, alpha=0.85, label="Mercado diario")
ax.barh(x-bw/2, irts, bw, color=cols, alpha=0.45,
        hatch="///", label="Mercado de desvíos")
ax.set_yticks(x)
ax.set_yticklabels(ests, fontsize=11)
ax.set_xlabel("Ingreso medio por hora (€)", fontsize=12)
ax.set_title("Fig. 6.3 — Ingreso medio: mercado diario vs mercado de desvíos",
             fontsize=13)
ax.legend(fontsize=10)
ax.axvline(0, color="black", lw=0.8)
ax.grid(True, axis="x", ls=":", alpha=0.4)
plt.tight_layout()
plt.savefig("fig3_desglose.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig3_desglose.png")

# ── FIG 5: Bolas activas ──────────────────────────────────────
fechas_lotes = fechas[::W][:len(n_bolas_hist)]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(fechas_lotes, n_bolas_hist,
        color=COLORES["bandit"], lw=1.8)
ax.fill_between(fechas_lotes, n_bolas_hist,
                alpha=0.15, color=COLORES["bandit"])
ax.set_ylabel("Bolas activas $|\\mathcal{B}_t|$", fontsize=12)
ax.set_title("Fig. 6.5 — Evolución del zooming: bolas activas", fontsize=13)
ax.grid(True, ls=":", alpha=0.4)
fmt_eje(ax)
plt.tight_layout()
plt.savefig("fig5_bolas.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig5_bolas.png")

# ── FIG 4: Distribución de pujas ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

n, bins, patches_hist = ax.hist(
    deltas_cz, bins=50, color=COLORES["bandit"],
    alpha=0.75, edgecolor="white",
    label="Distribución de $\\Delta_t$")

# Colorear barras por signo
for patch, left in zip(patches_hist, bins[:-1]):
    if left < 0:
        patch.set_facecolor(COLORES["bandit"])
    else:
        patch.set_facecolor("#E87722")

# Línea vertical en 0
ax.axvline(0, color="black", lw=1.5, ls="--",
           label="Oferta basada en la previsión ($\\Delta_t = 0$)")

# Anotaciones de porcentajes
pct_neg = (deltas_cz < 0).mean() * 100
pct_pos = (deltas_cz > 0).mean() * 100

ymax = ax.get_ylim()[1]
ax.text(-180, ymax * 0.85,
        f"$\\Delta_t < 0$\n{pct_neg:.1f}%\n(subpuja)",
        ha="center", fontsize=10, color=COLORES["bandit"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=COLORES["bandit"], alpha=0.8))
ax.text(+180, ymax * 0.85,
        f"$\\Delta_t > 0$\n{pct_pos:.1f}%\n(sobrepuja)",
        ha="center", fontsize=10, color="#E87722",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#E87722", alpha=0.8))

ax.set_xlabel("Desviación de oferta $\\Delta_t$ (MWh)", fontsize=12)
ax.set_ylabel("Número de horas", fontsize=12)
ax.set_title(
    "Fig. 6.4 — Distribución de la desviación de oferta del Bandido",
    fontsize=13)
ax.set_xlim(-260, 260)
ax.grid(True, ls=":", alpha=0.4)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("fig4_distribucion_pujas.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig4_distribucion_pujas.png")