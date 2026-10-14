"""Métodos estocásticos: estimación Monte Carlo y caminatas aleatorias (difusión)."""
import numpy as np


def estimar_pi(n: int, rng: np.random.Generator) -> tuple[float, float]:
    """Estima π con n puntos al azar en el cuadrado [-1, 1]². Devuelve (estimación, error estándar)."""
    x, y = rng.uniform(-1, 1, (2, n))
    p = np.mean(x**2 + y**2 <= 1)            # proporción dentro del círculo = π/4
    return 4 * p, 4 * np.sqrt(p * (1 - p) / n)


def caminatas_2d(n_pasos: int, n_caminantes: int, rng: np.random.Generator, largo_paso: float = 1.0) -> np.ndarray:
    """Posiciones de n_caminantes que dan pasos de largo fijo en dirección aleatoria.

    Devuelve un array (n_pasos + 1, n_caminantes, 2).
    """
    angulos = rng.uniform(0, 2 * np.pi, (n_pasos, n_caminantes))
    pasos = largo_paso * np.stack([np.cos(angulos), np.sin(angulos)], axis=-1)
    return np.concatenate([np.zeros((1, n_caminantes, 2)), np.cumsum(pasos, axis=0)])


def desplazamiento_cuadratico_medio(posiciones: np.ndarray) -> np.ndarray:
    """<r²>(t) promediado sobre caminantes. Para difusión normal en 2D: <r²> = n · a² = 4 D t."""
    return np.mean(np.sum(posiciones**2, axis=-1), axis=1)
