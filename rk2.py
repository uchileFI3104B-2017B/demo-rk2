"""
Integra la ecuacion del pendulo usando RK2.

Usaremos l = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

g = 9.8
l = 1.0

# Condiciones de borde
phi_0 = np.pi / 1
w_0 = 0


T = 2 * np.pi / np.sqrt(g/l)
t = np.linspace(0, 4 * T, 300)
phi_pequenas_oscilaciones = phi_0 * np.cos(np.sqrt(g/l) * t)


plt.clf()
plt.plot(t, phi_pequenas_oscilaciones, label="pequenas oscilaciones")

def f_pendulo(t, y):
    phi, w = y
    output = np.array([w, -g/l * np.sin(phi)])
    return output

# Implementacion de RK2

def k1(f, t, y, paso):
    """
    f : funcion a integrar. Retorna un np.ndarray
    t : tiempo en el cual evaluar la funcion f
    y : para evaluar la funcion f
    paso : tamano del paso a usar.
    """
    output = paso * f(t, y)
    return output

def k2(f, t, y, paso):
    """
    f : funcion a integrar. Retorna un np.ndarray
    t : tiempo en el cual evaluar la funcion f
    y : para evaluar la funcion f
    paso : tamano del paso a usar.
    """
    k1_evaluado = k1(f, t, y, paso)
    output = paso * f(t + paso / 2, y + k1_evaluado / 2)
    return output

def rk2_paso(f, t_n, y_n, paso):
    k2_evaluado = k2(f, t_n, y_n, paso)
    y_n_next = y_n + k2_evaluado
    return y_n_next

y_solucion = np.zeros((len(t), 2))
y_solucion[0] = [phi_0, w_0]

h = t[1] - t[0]
for i in range(1, len(t)):
    y_solucion[i] = rk2_paso(f_pendulo, t[i-1],
                             y_solucion[i-1], h)

plt.plot(t, y_solucion[:,0], label='rk2; $\phi_0=\pi/4$')



plt.xlabel('Tiempo')
plt.ylabel('$\phi(t)$')
plt.legend()
plt.savefig('rk2_pequenas_osc.png')
plt.show()
