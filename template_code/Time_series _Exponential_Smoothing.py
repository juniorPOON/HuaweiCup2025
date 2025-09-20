import numpy as np
import matplotlib.pyplot as plt
from toolbox import figure_set_chinese

figure_set_chinese()

def exponential_smoothing_forecast(data, alpha, steps=1):
    """
    一次指数平滑 + 未来预测
    data: 原始序列 (list)
    alpha: 平滑系数 (0~1)
    steps: 预测未来几步
    """
    result = [data[0]]
    for t in range(1, len(data)):
        result.append(alpha * data[t] + (1 - alpha) * result[t-1])

    # 开始预测未来
    forecast = result[-1]  # 最后一个平滑值
    future_preds = []
    for _ in range(steps):
        # 预测下一个值（没有真实Y了，只能用上一个预测值）
        forecast = alpha * forecast + (1 - alpha) * forecast
        future_preds.append(forecast)

    return result, future_preds

# ===== 使用案例 =====
if __name__ == "__main__":
    demand = [100, 105, 108, 120, 125, 140, 138, 150, 155, 160]
    alpha = 0.2

    smoothed, preds = exponential_smoothing_forecast(demand, alpha, steps=3)

    print("未来预测:", preds)

    # 可视化
    plt.plot(demand, label="原始数据")
    plt.plot(smoothed, label=f"指数平滑 (α={alpha})")
    plt.scatter(range(len(demand), len(demand)+len(preds)), preds,
                color="red", marker="x", s=100, label="未来预测")
    plt.legend()
    plt.title("电力负荷预测 - 指数平滑")
    plt.show()
