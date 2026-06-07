"""
Clase Bola — Contextual Zooming (Singhal et al. 2025)
======================================================
Representa una bola B(centro, radio) en el espacio P ⊂ [0,1]^4.

Dimensiones del espacio:
    [0] gamma_hat    — contexto 1 (con ruido Student-t)
    [1] lambda_I_hat — contexto 2 (con ruido Student-t)
    [2] eta_I_hat    — contexto 3 (con ruido Student-t)
    [3] p_norm       — decisión de puja ∈ [0,1]
                       0.5 = forecast bid (Δ=0)
"""

import numpy as np

# ── Parámetros globales ───────────────────────────────────────
DIM_CTX   = 3       # dimensiones de contexto
T_TOTAL   = 15360   # horizonte total (para calcular conf)
RADIO_MIN =  0.005   # radio mínimo para crear bolas hijas


class Bola:

    _cnt = 0  # contador global de bolas creadas

    def __init__(self, centro, radio):
        """
        Crea una bola en el espacio [0,1]^4.

        centro : array de 4 valores ∈ [0,1]
                 [gamma_hat, lambda_I_hat, eta_I_hat, p_norm]
        radio  : float > 0
        """
        self.centro = np.array(centro, dtype=float)
        self.radio  = float(radio)
        self.n      = 0      # número de veces usada
        self.reward = 0.0    # recompensa acumulada
        self.id     = Bola._cnt
        Bola._cnt  += 1

    # ── Estadísticas ─────────────────────────────────────────

    def nu(self):
        """
        Recompensa media estimada — ec. (8) del paper:
            ν(B) = reward_acumulado / n

        Devuelve 0 si la bola aún no ha sido seleccionada.
        """
        return self.reward / self.n if self.n > 0 else 0.0

    def conf(self):
        """
        Intervalo de confianza — ec. (10) del paper:
            conf(B) = sqrt(log(T) / (1 + n))

        T = horizonte total conocido de antemano.
        Decrece con más observaciones → más certeza.
        Alta cuando n=0 → mucha incertidumbre.
        """
        return float(np.sqrt(np.log(max(T_TOTAL, 2)) / (1.0 + self.n)))

    def indice_pre(self):
        """
        Pre-índice — ec. (9) del paper:
            I_pre(B) = ν(B) + r(B) + conf(B)

        Combina:
            ν(B)    → explotación (recompensa media conocida)
            r(B)    → exploración (bolas grandes = más inciertas)
            conf(B) → incertidumbre estadística
        """
        return self.nu() + self.radio + self.conf()

    # ── Geometría ─────────────────────────────────────────────

    def distancia(self, otra):
        """
        Distancia euclidea entre centros en [0,1]^4.
        Usada en el cálculo del índice completo ec. (11).
        """
        return float(np.linalg.norm(self.centro - otra.centro))

    def ctx_dentro(self, x_ctx):
        """
        Comprueba si el contexto x_ctx ∈ [0,1]^3 cae dentro
        de la proyección de esta bola sobre las dimensiones de contexto.

            dist(x_ctx, centro[:3]) ≤ radio
        """
        dist = np.linalg.norm(x_ctx - self.centro[:DIM_CTX])
        return bool(dist <= self.radio)

    def contiene_4d(self, punto):
        """
        Comprueba si un punto 4D está dentro de la bola.

            dist(punto, centro) ≤ radio
        """
        dist = np.linalg.norm(punto - self.centro)
        return bool(dist <= self.radio)

    def es_relevante(self, x_ctx, todas_bolas):
        """
        Una bola es relevante para el contexto x_ctx si:
        1. x_ctx está en su proyección de contexto
        2. No existe ninguna bola más pequeña que también
           contenga x_ctx en su proyección de contexto

        Las bolas más pequeñas tienen prioridad — implementa
        el dominio del paper ec. (12).
        """
        # Condición 1: el contexto está en esta bola
        if not self.ctx_dentro(x_ctx):
            return False

        # Condición 2: no hay bola más pequeña que también lo cubra
        for b in todas_bolas:
            if b.id != self.id and b.radio < self.radio:
                if b.ctx_dentro(x_ctx):
                    return False

        return True

    def en_dominio(self, punto_4d, todas_bolas):
        """
        Comprueba si punto_4d pertenece al dominio de esta bola:

            dom(B, A) = B \ (∪ B' con r(B') < r(B))

        El punto está en B pero no en ninguna bola más pequeña.
        Usado en la regla de activación.
        """
        # El punto debe estar en esta bola
        if not self.contiene_4d(punto_4d):
            return False

        # No debe estar en ninguna bola más pequeña
        for b in todas_bolas:
            if b.id != self.id and b.radio < self.radio:
                if b.contiene_4d(punto_4d):
                    return False

        return True

    def elegir_puja(self):
        """
        Elige p_norm ∈ [0,1] dentro del rango de la bola.

        Según el paper línea 10 del Algorithm 1:
        "any bid such that (f^w, x_t) ∈ dom(B, A_t')"

        Siempre aleatorio dentro del rango válido:
            [max(0, centro[3] - radio), min(1, centro[3] + radio)]

        El centro de la dimensión de puja es centro[3].
        """
        yc   = self.centro[DIM_CTX]
        y_lo = max(0.0, yc - self.radio)
        y_hi = min(1.0, yc + self.radio)

        if y_lo >= y_hi:
            return float(np.clip(yc, 0.0, 1.0))

        return float(np.random.uniform(y_lo, y_hi))

    def __repr__(self):
        return (f"Bola(id={self.id}, r={self.radio:.4f}, "
                f"n={self.n}, ν={self.nu():.4f}, "
                f"centro={self.centro.round(3)})")


# ================================================================
# FUNCIÓN EXTERNA: índice completo — ec. (11) del paper
# ================================================================

def calcular_indice(bola, todas_bolas):
    """
    Índice completo de una bola — ec. (11) del paper:

        I(B) = r(B) + min_{B'∈A} [ I_pre(B') + D(B, B') ]

    Para calcular el índice de B miramos TODAS las bolas B'
    y nos quedamos con la que minimiza (pre-índice + distancia a B).

    Intuición: el índice de B es alto si hay alguna bola B'
    cercana con pre-índice alto — propaga información entre vecinas.
    """
    min_val = min(
        b.indice_pre() + bola.distancia(b)
        for b in todas_bolas
    )
    return bola.radio + min_val