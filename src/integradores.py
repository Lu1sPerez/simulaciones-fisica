"""
Integradores numéricos para sistemas de ecuaciones diferenciales ordinarias  dy/dt = f(t, y).

Se implementan desde cero (sin scipy) para poder estudiar su error y su orden de convergencia.
"""
from typing import Callable

import numpy as np

Derivada = Callable[[float, np.ndarray], np.ndarray]


def paso_euler(f: Derivada, t: float, y: np.ndarray, h: float) -> np.ndarray:
    return y + h * f(t, y)


def paso_rk4(f: Derivada, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrar(f: Derivada, y0, t0: float, t1: float, h: float, metodo: str = "rk4",
             detener: Callable[[np.ndarray], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Integra de t0 a t1 con paso fijo h. `detener(y)` permite cortar antes (ej.: el proyectil toca el suelo)."""
    paso = {"euler": paso_euler, "rk4": paso_rk4}[metodo]
    n = int(round((t1 - t0) / h))
    ts = np.empty(n + 1)
    ys = np.empty((n + 1, len(y0)))
    ts[0], ys[0] = t0, np.asarray(y0, dtype=float)
    for i in range(n):
        ys[i + 1] = paso(f, ts[i], ys[i], h)
        ts[i + 1] = ts[i] + h
        if detener is not None and detener(ys[i + 1]):
            return ts[: i + 2], ys[: i + 2]
    return ts, ys


# ---------------- Sistemas físicos ----------------

def oscilador_armonico(omega: float = 1.0) -> Derivada:
    """x'' = -omega^2 x, como sistema de primer orden y = (x, v)."""
    def f(t, y):
        return np.array([y[1], -omega**2 * y[0]])
    return f


def proyectil(g: float = 9.81, k: float = 0.0) -> Derivada:
    """Proyectil con rozamiento cuadrático: a = -g ŷ - k |v| v.   y = (x, z, vx, vz).

    k = (1/2) rho C_d A / m. Para una pelota de fútbol, k ≈ 0.01 1/m.
    """
    def f(t, y):
        vx, vz = y[2], y[3]
        rapidez = np.hypot(vx, vz)
        return np.array([vx, vz, -k * rapidez * vx, -g - k * rapidez * vz])
    return f


def alcance(v0: float, angulo_grados: float, k: float, g: float = 9.81, h: float = 1e-3) -> float:
    """Distancia horizontal recorrida hasta volver a z = 0 (interpola linealmente el último paso)."""
    a = np.radians(angulo_grados)
    y0 = [0.0, 0.0, v0 * np.cos(a), v0 * np.sin(a)]
    t, y = integrar(proyectil(g, k), y0, 0.0, 200.0, h, "rk4", detener=lambda s: s[1] < 0)
    (x1, z1), (x2, z2) = y[-2, :2], y[-1, :2]
    return float(x1 + (x2 - x1) * z1 / (z1 - z2))


def orden_convergencia(hs: np.ndarray, errores: np.ndarray) -> float:
    """Pendiente de log(error) vs log(h): el orden p del método (error ∝ h^p)."""
    return float(np.polyfit(np.log(hs), np.log(errores), 1)[0])
