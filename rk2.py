"""
Integra la ecuacion del pendulo usando RK2.

Usaremos l = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

g = 9.8
l = 1.0

# Condiciones de borde
phi_0 = np.pi / 8
w_0 = 0


T = 2 * np.pi / np.sqrt(g/l)
t = np.linspace(0, 4 * T, 300)
phi_pequenas_oscilaciones = phi_0 * np.cos(np.sqrt(g/l) * t)


plt.clf()
plt.plot(t, phi_pequenas_oscilaciones, label="pequenas oscilaciones")
plt.show()
