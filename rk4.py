import matplotlib.pyplot as plt
import numpy as np


def rk4_system(t0, x0, h, steps, f):
    t = np.linspace(t0, t0 + h * steps, steps + 1)
    x = np.empty((len(x0), steps + 1))
    x[:, 0] = x0

    for i in range(steps):
        x_i = x[:, i]

        k1 = f(t[i], x_i)
        k2 = f(t[i] + h / 2, x_i + (h / 2) * k1)
        k3 = f(t[i] + h / 2, x_i + (h / 2) * k2)
        k4 = f(t[i] + h, x_i + h * k3)

        x[:, i + 1] = x_i + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, x


def make_damping_matrix(k):
    n = len(k)

    if n == 1:
        return np.array([[k[0]]])

    out = np.zeros((n, n))
    out[0][0] = k[0] + k[1]
    out[0][1] = -k[1]
    out[n - 1][n - 2] = -k[n - 1]
    out[n - 1][n - 1] = k[n - 1]

    for i in range(1, n - 1):
        out[i][i - 1] = -k[i]
        out[i][i] = k[i] + k[i + 1]
        out[i][i + 1] = -k[i + 1]

    return out


mass = np.array([1.5e5, 1.5e5, 1.5e5])
elasticity = np.array([2e5, 2e5, 2e5])
damping_factor = np.array([0.05, 0, 0])
n = len(mass)

freq = 4
ground_accel = 0.3

t0 = 0
x0 = np.zeros(2 * n)
ts = 3 * 60
s = 10_000

h = (ts - t0) / s

damping = damping_factor * 2 * np.sqrt(mass * elasticity)
M = np.diag(mass)
C = make_damping_matrix(damping)
K = make_damping_matrix(elasticity)
M_inv = np.linalg.inv(M)


def p(t):
    out = np.zeros(n)
    out[0] = mass[0] * ground_accel * np.sin(2 * np.pi * freq * t)
    return out


def f(t, x):
    u = x[:n]
    v = x[n:]
    return np.append(
        v,
        M_inv @ (-C @ v - K @ u + p(t)),
    )


(t_out, x_out) = rk4_system(t0, x0, h, s, f)
u_out = x_out[:n]

colors = plt.colormaps["gist_rainbow"](np.linspace(0, 1, n))

for i, (u_i, color) in enumerate(zip(u_out, colors)):
    plt.plot(t_out, u_i, "-", label=f"u{i+1}(t)", c=color)

plt.legend()
plt.show()
