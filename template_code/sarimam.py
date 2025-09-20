import itertools
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from toolbox import *
figure_set_chinese()

warnings.filterwarnings("ignore")
'''
在参数搜索时，m=0（没有季节周期）
有季节性 → 它自己选 SARIMA

没季节性 → 它自己退化成 ARIMA
不需要手动去区分
'''
# ================== 示例数据 ==================
np.random.seed(123)
n = 100
time = np.arange(n)
# 模拟：趋势 + 季节 + 噪声
data = 50 + 0.5*time + 10*np.sin(2*np.pi*time/12) + np.random.normal(scale=2, size=n)

# ================== SARIMA 自动调参 ==================
p = d = q = range(0, 3)       # 非季节部分 (p,d,q)
P = D = Q = range(0, 2)       # 季节部分 (P,D,Q)
m = [0, 12]                   # m=0表示无季节，m=12表示12周期（可按月/天改）

best_aic = np.inf
best_order = None
best_seasonal_order = None
best_model = None

for i, j, k in itertools.product(p, d, q):
    for P_, D_, Q_, m_ in itertools.product(P, D, Q, m):
        try:
            model = SARIMAX(data, order=(i,j,k),
                            seasonal_order=(P_,D_,Q_,m_),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (i,j,k)
                best_seasonal_order = (P_,D_,Q_,m_)
                best_model = results
        except:
            continue

print("最佳参数:", best_order)
print("最佳季节参数:", best_seasonal_order)
print("对应 AIC:", best_aic)

# ================== 预测 ==================
steps = 12
forecast = best_model.forecast(steps=steps)

# 模型评价
fitted = best_model.fittedvalues
mape = mean_absolute_percentage_error(data[max(len(data)-len(fitted),0):], fitted[-len(data):])
rmse = mean_squared_error(data[max(len(data)-len(fitted),0):], fitted[-len(data):]) ** 0.5
print(f"MAPE={mape:.3f}, RMSE={rmse:.3f}")

# ================== 可视化 ==================
plt.figure(figsize=(10,5))
plt.plot(data, label="原始数据", marker="o")
plt.plot(fitted, label="拟合值", linestyle="--")
plt.plot(range(len(data), len(data)+steps), forecast, "r--", label="预测")
plt.axvline(len(data)-1, color="gray", linestyle="--", alpha=0.7)
plt.title("SARIMA 自动调参预测")
plt.legend(); plt.grid(True, ls="--", alpha=0.6)
plt.show()
