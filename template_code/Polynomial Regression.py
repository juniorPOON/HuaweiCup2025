import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 导入多项式特征生成工具，这是实现多项式回归的核心
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------中文显示配置--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题（可选，防止后续绘图负号异常）

# -----------------------------------------------------------------


def run_polynomial_regression(X, y, future_x, degree=2):
    """
    多项式回归通用模板
    -----------------
    输入参数：
        X: 一维或二维 array，自变量（年份等）
        y: 一维 array，因变量（如销售额等）
        future_x: list，要预测的新自变量值（如 [2025]）
        degree: 多项式阶数（默认 2，即二次回归）
    输出：
        - 多项式系数
        - R² 和 MSE
        - 预测值
        - 可视化曲线
    """

    # 将输入的自变量转换为二维数组（sklearn要求的格式：[样本数, 特征数]）
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # --------------------------
    # 以下是正式使用多项式的部分
    # --------------------------

    # 1. 创建多项式特征生成器，指定多项式阶数
    # 例如：degree=2 表示生成 x 和 x² 特征
    #       degree=3 表示生成 x、x² 和 x³ 特征
    ##在函数入口有一个degree,现在默认为2
    poly = PolynomialFeatures(degree=degree)

    # 2. 将原始特征转换为多项式特征（核心步骤）
    # 对于输入X（形状为[n_samples, 1]），转换后会得到：
    # [1, x, x², ..., x^degree]（形状为[n_samples, degree+1]）
    # 这里的1是为了对应多项式中的常数项
    X_poly = poly.fit_transform(X)

    # --------------------------
    # 以下是基于多项式特征的线性回归
    # --------------------------

    # 创建线性回归模型（虽然叫线性回归，但因为输入是多项式特征，所以整体是多项式回归）
    model = LinearRegression()

    # 使用多项式特征训练模型
    # 模型实际上学习的是：y = w0*1 + w1*x + w2*x² + ... + wd*x^d
    model.fit(X_poly, y)

    # 对新数据进行预测：先将新数据转换为多项式特征，再用模型预测
    future_X = np.array(future_x).reshape(-1, 1)
    future_X_poly = poly.transform(future_X)  # 同样需要转换为多项式特征
    predictions = model.predict(future_X_poly)

    # 评估模型性能：计算R²（决定系数）和MSE（均方误差）
    y_pred = model.predict(X_poly)  # 对训练数据的预测值
    r2 = r2_score(y, y_pred)  # R²越接近1，拟合效果越好
    mse = mean_squared_error(y, y_pred)  # MSE越小，拟合效果越好

    # 输出结果
    print(f"📌 多项式回归（degree={degree}）结果：")
    print("  系数 =", model.coef_)  # 系数对应 [w1, w2, ..., wd]（注意不包含截距）
    print("  截距 =", model.intercept_)  # 截距对应 w0
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")
    for x_val, pred in zip(future_x, predictions):
        print(f"  X={x_val} → 预测值 {pred:.2f}")

    # 可视化：绘制原始数据点、多项式拟合曲线和预测点
    # 生成更密集的X值用于绘制平滑曲线
    X_range = np.linspace(min(X), max(X) + 5, 200).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)  # 转换为多项式特征
    y_range_pred = model.predict(X_range_poly)  # 计算对应预测值

    plt.scatter(X, y, color="blue", label="实际数据")
    plt.plot(X_range, y_range_pred, color="red", label=f"{degree} 次回归拟合")
    plt.scatter(future_X, predictions, color="green", marker="x", s=100, label="预测点")
    plt.xlabel("年份")
    plt.ylabel("销售额")
    plt.title(f"{degree} 次多项式回归拟合")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    return model, r2, mse, predictions


# 示例用法
if __name__ == "__main__":
    # 自变量：年份，步长为5（2000, 2005, ..., 2035）
    X = [6, 8, 10, 12, 14, 16, 18, 20]  # 时间 (小时)
    y = [200, 500, 800, 400, 300, 600, 900, 700]  # 流量

    # 预测2040年和2045年的销售额，使用3次多项式
    run_polynomial_regression(
        X=X,
        y=y,
        future_x=[22],
        degree=3 # 可以尝试修改为1（线性）、2（二次）、3（三次）等观察效果
    )
