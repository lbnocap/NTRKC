import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# 分数阶导数的定义
def fractional_derivative(alpha, t, u_prime):
    result = np.zeros_like(t)
    for i in range(1, len(t)):
        result[i] = (1/gamma(1-alpha)) * np.trapz(u_prime[:i+1], dx=(t[i]-t[i-1])**alpha)
    return result

# FBDF1方法计算分数阶导数
def fbdf1(alpha, h, u_prime):
    n = len(u_prime)
    a_k = np.arange(n, 0, -1)
    u = np.zeros_like(u_prime)
    
    for i in range(n):
        u[i] = (1/(alpha * h**alpha)) * np.sum(a_k[:i+1] * u[:i+1]) + (1/(alpha * h**alpha)) * u_prime[i]
    
    return u

# 定义函数 u(t) = e^{-t}，其一阶导数为 u'(t) = -e^{-t}
def true_solution(t):
    return np.exp(-t), -np.exp(-t)

# 设置参数
alpha = 0.8
h = 0.1
t_max = 5

# 生成时间网格
t = np.arange(0, t_max, h)

# 计算真实解和一阶导数
u_true, u_prime_true = true_solution(t)

# 使用FBDF1方法计算分数阶导数
u_prime_approx = fbdf1(alpha, h, u_true)

# 绘图
plt.plot(t, u_true, label='True Solution')
plt.plot(t, u_prime_true, label="True Derivative")
plt.plot(t, u_prime_approx, label=f'Approximated Derivative (FBDF1, alpha={alpha})', linestyle='dashed')
plt.legend()
plt.xlabel('t')
plt.title(f'Fractional Derivative Approximation (alpha={alpha})')
plt.show()