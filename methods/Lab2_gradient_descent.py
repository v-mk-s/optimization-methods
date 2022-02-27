import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sympy





# Дано:
alpha = 1
f = lambda x, y: alpha * (x**2 - y)**2 + (x - 1)**2

X0 = np.array([-1, -2])
eps = 1e-3

Grad_f = lambda x, y: np.array([4*alpha*x*(x**2 - y) + 2*x - 2, alpha*(-2*x**2 + 2*y)])

# Параметры методов:
kappa0 = 1
nu = 0.95
omega = 0.5



def MethodGoldenRatio(f, b, a = 0, e = eps * 1e-1):
    tau = (math.sqrt(5) + 1) / 2
    Ak, Bk = a, b
    lk = Bk - Ak
    Xk1 = Bk - (Bk - Ak) / tau
    Xk2 = Ak + (Bk - Ak) / tau
    y1, y2 = f(Xk1), f(Xk2)
    while lk >= e:
        if y1 >= y2:
            Ak = Xk1
            Xk1 = Xk2
            Xk2 = Ak + Bk - Xk1
            y1 = y2
            y2 = f(Xk2)
        else:
            Bk = Xk2
            Xk2 = Xk1
            Xk1 = Ak + Bk - Xk2
            y2 = y1
            y1 = f(Xk1)
        lk = Bk - Ak
    return (Ak + Bk) / 2

def MethodsGradientDescent(flag):
    fun = lambda X: f(X[0], X[1])
    w = lambda X: -Grad_f(X[0], X[1])
    X = X0
    kappa_k = kappa0
    NormW = []
    Xk = []
    while True:
        Xk.append(X)
        Wk = w(X)
        NormW.append(np.linalg.norm(Wk))
        if NormW[-1] <= eps:
            break
        fk = fun(X)
        if flag == 0:
            phi = lambda kappa: fun(X + kappa * Wk)
            kappa_k = MethodGoldenRatio(phi, 2.5)
            X = X + kappa_k * Wk
        elif flag == 1:
            Xcurr = X + kappa_k * Wk
            while fk - fun(Xcurr) <= omega * kappa_k * NormW[-1]**2:
                kappa_k *= nu
                Xcurr = X + kappa_k * Wk
            X = Xcurr
    return X, Xk, NormW



print()
print('Методы градиентного спуска')
print('Дано:')
print('Целевая функция:        f(x, y) =', f(sympy.Symbol('x'), sympy.Symbol('y')))
print('Начальное приближение:  X0 =', X0)
print('Точность вычисления:    Eps =', eps)

for i in range(2):
    print()
    print('_' * 100)
    if i == 0:
        print('*' * 30, ' Метод наискорейщего спуска ', '*' * 30)
    elif i == 1:
        print('*' * 20, ' Метод градиентного спуска с дроблением шага ', '*' * 20)
    X, Xk, NormW = MethodsGradientDescent(i)
    print('Точка минимума функции:             Xmin =', X)
    print('Значение функции в точке минимума:  f(Xmin) =', f(X[0], X[1]))
    print('Количество итераций:                k =', len(NormW)-1)