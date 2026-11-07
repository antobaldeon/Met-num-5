"""Sesión 13: energía consumida mediante Simpson 1/3 compuesto.

Requiere NumPy: pip install numpy
Ejecutar: python parte2_simpson.py
Los valores predeterminados corresponden a la guía: a=0, b=6, n=6.
"""

import math

import numpy as np


def potencia(t):
    """Potencia P(t) en kW, con t en horas y seno en radianes."""
    return 10 + 5 * t * np.sin(np.pi * t / 6) + np.exp(0.2 * t)


def validar(a, b, n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 or n % 2:
        raise ValueError("n debe ser un número entero positivo y par.")
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        raise ValueError("Los límites deben ser finitos y cumplir a < b.")


def simpson_compuesto(f, a, b, n):
    """Integra una función vectorizable usando pesos 1,4,2,...,4,1."""
    validar(a, b, n)
    t = np.linspace(a, b, n + 1)
    pesos = np.ones(n + 1)
    pesos[1:-1:2] = 4
    pesos[2:-1:2] = 2
    valores = np.asarray(f(t), dtype=float)
    valores = np.broadcast_to(valores, t.shape)
    if not np.all(np.isfinite(valores)):
        raise ValueError("La función debe tener valores finitos en todos los nodos.")
    resultado = float((b - a) / (3 * n) * np.dot(pesos, valores))
    if not math.isfinite(resultado):
        raise ValueError("El resultado excede el rango numérico.")
    return resultado


def cota_error(a, b, n, M):
    """Cota de Simpson, siempre que M acote |P''''| en todo [a,b]."""
    validar(a, b, n)
    if not math.isfinite(M) or M < 0:
        raise ValueError("M debe ser finito y no negativo.")
    h = (b - a) / n
    return (b - a) * h**4 * M / 180


def cota_derivada_segura(a, b):
    """Cota analítica conservadora válida para la función de la guía.

    P''''(t) = 5*k**4*t*sin(k*t) - 20*k**3*cos(k*t)
               + 0.2**4*exp(0.2*t), donde k=pi/6.
    Se usan |sin|<=1, |cos|<=1 y el máximo de |t|.
    """
    k = math.pi / 6
    return 5 * k**4 * max(abs(a), abs(b)) + 20 * k**3 + 0.2**4 * math.exp(0.2 * b)


def primitiva(t):
    k = math.pi / 6
    return 10 * t + 5 * (-t * math.cos(k * t) / k + math.sin(k * t) / k**2) + 5 * math.exp(0.2 * t)


def main():
    try:
        a = float(input("Límite inferior a [0]: ") or "0")
        b = float(input("Límite superior b [6]: ") or "6")
        n = int(input("Subintervalos n, entero positivo par [6]: ") or "6")
        validar(a, b, n)
        energia = simpson_compuesto(potencia, a, b, n)
        M_segura = cota_derivada_segura(a, b)
        exacta = primitiva(b) - primitiva(a)
        print(f"\nEnergía aproximada: {energia:.12f} kWh")
        print(f"Paso h: {(b-a)/n:.6f} h")
        print(f"Referencia analítica: {exacta:.12f} kWh")
        print(f"Error absoluto real: {abs(energia-exacta):.12f} kWh")
        print("\nTabla de evaluación (n seleccionado)")
        print(f"{'i':>5} {'t_i (h)':>14} {'P(t_i) (kW)':>18} {'Coeficiente':>12}")
        for i, t in enumerate(np.linspace(a, b, n + 1)):
            peso = 1 if i in (0, n) else (4 if i % 2 else 2)
            print(f"{i:5d} {t:14.6f} {potencia(t):18.10f} {peso:12d}")
        print("\nConvergencia: cotas en kWh")
        print(f"{'n':>5} {'h (h)':>12} {'Integral (kWh)':>20} {'Cota guía M=0.45':>20} {'Cota válida':>20}")
        for cantidad in (2, 4, 6, 12, 24):
            valor = simpson_compuesto(potencia, a, b, cantidad)
            guia = cota_error(a, b, cantidad, 0.45)
            segura = cota_error(a, b, cantidad, M_segura)
            print(f"{cantidad:5d} {(b-a)/cantidad:12.6f} {valor:20.12f} {guia:20.12f} {segura:20.12f}")
        print("\nLa columna M=0.45 reproduce el dato de la guía, pero no garantiza el error.")
        print("En [0,6], |P''''(0)| es aproximadamente 2.86935, mayor que 0.45.")
        print(f"La última columna usa una cota conservadora válida: M={M_segura:.10f}.")
    except (ValueError, OverflowError) as error:
        raise SystemExit(f"Error: {error}") from None


if __name__ == "__main__":
    main()
