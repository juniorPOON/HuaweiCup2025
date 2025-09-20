import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from toolbox import *
figure_set_chinese()
# ========== Step 1. 数据准备 ==========
# 示例数据：某园区日负荷（带趋势和噪声）
np.random.seed(123)
n = 100
time = np.arange(n)
data = 200 + 2*time + np.random.normal(scale=10, size=n)

# ========== Step 2. 平稳性检验 ==========
result = adfuller(data)
print("ADF统计量:", result[0])
print("p值:", result[1])
if result[1] > 0.05:
    print("❌ 数据非平稳 → 需要做差分(d>0)")
else:
    print("✅ 数据平稳 → 可以直接建模")

# ========== Step 3. 建模 (手动设定 p,d,q) ==========
# 比赛里常用经验：d=1，大多数非平稳数据一次差分后可用
# p,q 可以用 ACF/PACF 图或 AIC 准则自动调参（这里只写死）
model = ARIMA(data, order=(2,1,2))
fit = model.fit()

# ========== Step 4. 模型评价 ==========
fitted = fit.fittedvalues
mape = mean_absolute_percentage_error(data[1:], fitted[1:])  # d=1 → fitted少一个点
rmse = mean_squared_error(data[1:], fitted[1:])**0.5
print(f"MAPE={mape:.3f}, RMSE={rmse:.3f}")

# ========== Step 5. 预测 ==========
steps = 10  # 预测未来10期
forecast = fit.forecast(steps=steps)
print("未来预测:", forecast)

# ========== Step 6. 可视化 ==========
plt.figure(figsize=(10,5))
plt.plot(data, label="原始数据", marker="o")
plt.plot(fitted, label="拟合值", linestyle="--")
plt.plot(range(len(data), len(data)+steps), forecast, "r--", label="预测")
plt.axvline(len(data)-1, color="gray", linestyle="--", alpha=0.7)
plt.title("ARIMA 时间序列预测")
plt.legend(); plt.grid(True, ls="--", alpha=0.6)
plt.show()
