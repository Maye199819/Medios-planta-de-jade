import numpy as np

# Datos del árbol de jade
x = np.array([0, 60, 90, 120, 150, 180])
y = np.array([0, 30, 55, 90, 120, 135])

# --- Newton ---
def diferencias_divididas(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0]

def eval_newton(coef, x_data, x_eval):
    n = len(coef)
    resultado = coef[0]
    for i in range(1, n):
        termino = coef[i]
        for j in range(i):
            termino *= (x_eval - x_data[j])
        resultado += termino
    return resultado

# --- Lagrange ---
def lagrange(x, y, x_eval):
    n = len(x)
    total = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_eval - x[j]) / (x[i] - x[j])
        total += term
    return total

# --- Interpolación Lineal ---
def interp_lineal(x, y, x_eval):
    for i in range(len(x) - 1):
        if x[i] <= x_eval <= x[i+1]:
            return y[i] + ((y[i+1] - y[i]) / (x[i+1] - x[i])) * (x_eval - x[i])
    return None

# --- Regresión Lineal ---
def regresion_lineal(x, y, x_eval):
    coef = np.polyfit(x, y, 1)
    p = np.poly1d(coef)
    return p(x_eval)

# --- Regresión Polinómica (grado 2) ---
def regresion_polinomica(x, y, x_eval):
    coef = np.polyfit(x, y, 2)
    p = np.poly1d(coef)
    return p(x_eval)

# --- Evaluaciones ---
dias = [15, 30, 135]
altura_newton = []
altura_lagrange = []
altura_lineal = []
altura_reg_lineal = []
altura_reg_poli = []

coef_newton = diferencias_divididas(x, y)

for d in dias:
    altura_newton.append(eval_newton(coef_newton, x, d))
    altura_lagrange.append(lagrange(x, y, d))
    altura_lineal.append(interp_lineal(x, y, d))
    altura_reg_lineal.append(regresion_lineal(x, y, d))
    altura_reg_poli.append(regresion_polinomica(x, y, d))

# --- RESULTADOS EN FORMATO TABULAR ---
print("\n| MÉTODO               | Día 15   | Día 30   | Día 135  |")
print("|----------------------|----------|----------|----------|")
print(f"| Newton               | {altura_newton[0]:7.2f} cm | {altura_newton[1]:7.2f} cm | {altura_newton[2]:7.2f} cm |")
print(f"| Lagrange             | {altura_lagrange[0]:7.2f} cm | {altura_lagrange[1]:7.2f} cm | {altura_lagrange[2]:7.2f} cm |")
print(f"| Interpolación Lineal | {altura_lineal[0]:7.2f} cm | {altura_lineal[1]:7.2f} cm | {altura_lineal[2]:7.2f} cm |")
print(f"| Regresión Lineal     | {altura_reg_lineal[0]:7.2f} cm | {altura_reg_lineal[1]:7.2f} cm | {altura_reg_lineal[2]:7.2f} cm |")
print(f"| Regresión Polinómica | {altura_reg_poli[0]:7.2f} cm | {altura_reg_poli[1]:7.2f} cm | {altura_reg_poli[2]:7.2f} cm |")
