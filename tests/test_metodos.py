"""Tests de los métodos numéricos. Ejecutar con:  pytest -v"""
import numpy as np

from src import integradores as it
from src import montecarlo as mc


def _error_final(metodo, h):
    """Error de (x, v) en t=5 del oscilador armónico con x(0)=1, v(0)=0 (exacta: x=cos t, v=-sin t).

    Se evalúa en un tiempo genérico: en t=2π el error de fase se cancela en x y el orden aparente sube.
    """
    t, y = it.integrar(it.oscilador_armonico(), [1.0, 0.0], 0.0, 5.0, h, metodo)
    return np.hypot(y[-1, 0] - np.cos(t[-1]), y[-1, 1] + np.sin(t[-1]))


def test_euler_es_de_orden_1():
    hs = 5.0 / np.array([2000, 4000, 8000, 16000])
    errores = [_error_final("euler", h) for h in hs]
    assert abs(it.orden_convergencia(hs, errores) - 1) < 0.1


def test_rk4_es_de_orden_4():
    hs = 5.0 / np.array([25, 50, 100, 200])
    errores = [_error_final("rk4", h) for h in hs]
    assert abs(it.orden_convergencia(hs, errores) - 4) < 0.15


def test_alcance_sin_rozamiento_coincide_con_formula():
    v0, ang, g = 20.0, 30.0, 9.81
    esperado = v0**2 * np.sin(2 * np.radians(ang)) / g
    assert abs(it.alcance(v0, ang, k=0.0, g=g) - esperado) < 1e-3


def test_rozamiento_reduce_el_alcance():
    assert it.alcance(30, 45, k=0.01) < it.alcance(30, 45, k=0.0)


def test_montecarlo_pi_dentro_del_error_estadistico():
    est, err = mc.estimar_pi(200_000, np.random.default_rng(0))
    assert abs(est - np.pi) < 4 * err


def test_difusion_desplazamiento_cuadratico_lineal_en_el_tiempo():
    pos = mc.caminatas_2d(400, 4000, np.random.default_rng(1))
    msd = mc.desplazamiento_cuadratico_medio(pos)
    pendiente = np.polyfit(np.arange(len(msd)), msd, 1)[0]
    assert abs(pendiente - 1.0) < 0.05     # largo de paso 1 -> <r²> = n
