import numpy as np  
from scipy.optimize import minimize  
  
# 定义损失函数（目标函数）  
def loss_function(X_flat, A, B):  
    X = X_flat.reshape(A.shape)  # 将一维数组重新整形为矩阵  
    return np.linalg.norm(X.dot(A).dot(X) - B, 'fro')  # 使用 Frobenius 范数作为损失  
  
# 定义梯度函数（对于损失函数关于 X 的梯度）  
# 注意：这里的梯度计算可能是近似的，需要更精确的数值微分或符号微分  
def gradient_function(X_flat, A, B):  
    X = X_flat.reshape(A.shape)  # 将一维数组重新整形为矩阵  
    dX = 2 * (X.dot(A).dot(X) - B).dot(A).dot(X)  
    return dX.flatten()  # 将梯度矩阵重新整形为一维数组  
  
# 初始化 X（与 A 大小相同的单位矩阵）
n = 4
A = np.random.rand(n, n)  # 假设 A 是一个 n x n 的方阵  
B = np.random.rand(n, n)  # B 也是 n x n 的方阵  
X0 = np.eye(n).flatten()  # 初始化 X 为 n x n 的单位矩阵并转换为一维数组  
  
# 约束条件：X 必须是方阵且元素无特殊限制（这里不添加显式约束）  
  
# 定义优化问题并求解  
# 注意：这里可能需要调整学习率、迭代次数等参数以改善收敛  
result = minimize(loss_function, X0, args=(A, B), jac=gradient_function, method='CG')  
  
# 获取最优解并重新整形为矩阵  
X_opt = result.x.reshape(n, n)  
  
# 检查解是否满足方程（由于数值误差，可能只是近似满足）  
print(np.linalg.norm(X_opt.dot(A).dot(X_opt) - B, 'fro'))
print(X_opt)