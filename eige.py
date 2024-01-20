import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eig

# 定义非线性微分方程
def nonlinear_ode(y, t):
    # 以 y[0] 为例，可以根据实际问题定义非线性微分方程
    dydt = (t*np.exp(t**2)-2*t*y[0])/(1+t**2)
    return [dydt]

# 初始条件
y0 = [-0.50]

# 时间点
t = np.linspace(0, 2, 100)

# 求解微分方程
sol = odeint(nonlinear_ode, y0, t)

# 获取最终状态
final_state = sol[-1]

# 构建雅可比矩阵
def jacobian(y, t):
    return [[2*y[0] - 1]]

# 计算雅可比矩阵的特征值
jac_matrix = jacobian(final_state, 0)

eigenvalues, eigenvectors = eig(jac_matrix)

# 打印结果
print("Final state:", final_state)
print("Eigenvalues:", eigenvalues)