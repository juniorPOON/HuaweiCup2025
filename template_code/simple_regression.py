import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------中文显示配置--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题（可选，防止后续绘图负号异常）

# -----------------------------------------------------------------

# ===== 通用线性回归函数 =====
def run_linear_regression(x_data, y_data, future_x):
    """
    线性回归通用模板
    -----------------
    输入参数：
        x_data: list 或 numpy array，自变量（如月份/时间/投入等）
        y_data: list 或 numpy array，因变量（如销售额/流量/产出等）
        future_x: list，要预测的自变量值（如未来月份）

    输出：
        - 拟合参数（斜率、截距）
        - 模型评价指标（R², MSE）
        - 未来预测结果
        - 可视化图
    """

    # -------- 1. 数据预处理 --------
    X = np.array(x_data).reshape(-1, 1)   # 转成二维数组
    y = np.array(y_data)
    future_X = np.array(future_x).reshape(-1, 1)

    # -------- 2. 建立并训练模型 --------
    model = LinearRegression()
    model.fit(X, y)

    # -------- 3. 模型参数 --------
    slope = model.coef_[0]       # 斜率
    intercept = model.intercept_ # 截距

    # -------- 4. 拟合优度评价 --------
    y_pred = model.predict(X)  # 训练集预测
    r2 = r2_score(y, y_pred)   # 决定系数，越接近1越好
    mse = mean_squared_error(y, y_pred)  # 均方误差，越小越好

    # -------- 5. 未来预测 --------
    predictions = model.predict(future_X)

    # -------- 6. 打印结果 --------
    print("📌 模型参数：")
    print(f"  斜率 a = {slope:.4f}")
    print(f"  截距 b = {intercept:.4f}")

    print("\n📌 模型评价：")
    print(f"  R² = {r2:.4f}  (越接近1说明拟合越好)")
    print(f"  MSE = {mse:.4f} (越小说明误差越小)")

    print("\n📌 预测结果：")
    for x_val, pred in zip(future_x, predictions):
        print(f"  X={x_val} → 预测值 {pred:.2f}")

    # -------- 7. 可视化 --------
    plt.figure(figsize=(8,5))
    plt.scatter(X, y, color="blue", label="实际数据")
    plt.plot(X, y_pred, color="red", label="拟合直线")
    plt.scatter(future_X, predictions, color="green", marker="x", s=100, label="预测点")
    plt.title("线性回归预测", fontsize=14)
    plt.xlabel("自变量 (X)", fontsize=12)
    plt.ylabel("因变量 (Y)", fontsize=12)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.6)
    plt.show()

    return slope, intercept, r2, mse, predictions

# ===== 使用案例 =====
if __name__ == "__main__":
    # 历史数据
    months = list(range(2000 , 2000 +5  * 5 ,5))  # 自变量：月份
    sales = [10,25,60,130,210]  # 因变量：销售额

    # 预测未来 3 个月
    future_months = [2025]

    run_linear_regression(months, sales, future_months)
