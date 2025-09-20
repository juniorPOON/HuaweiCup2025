import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# --------------------------中文显示配置--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题（可选，防止后续绘图负号异常）

# -------------------- 多元线性回归模板 --------------------
def run_multilinear_regression(x_data, y_data, future_x, feature_names=None):
    """
    多元线性回归通用模板
    -----------------
    输入参数：
        x_data: 二维 list/array，自变量 (样本数 × 特征数)，如 [[广告, 员工], ...]
        y_data: 一维 list/array，因变量，如 [120,150,...]
        future_x: 二维 list/array，要预测的新数据，如 [[28, 11]]
        feature_names: list，自变量名称（可选），如 ["广告投入", "员工人数"]

    输出：
        - 回归系数 & 截距
        - 模型评价指标 (R², MSE)
        - 未来预测结果
        - 返回 model 以便可视化
    """

    # -------- 1. 数据准备 --------
    X = np.array(x_data)
    y = np.array(y_data)
    future_X = np.array(future_x)

    # -------- 2. 建立并训练模型 --------
    model = LinearRegression()
    model.fit(X, y)

    # -------- 3. 模型参数 --------
    coefs = model.coef_
    intercept = model.intercept_

    # -------- 4. 模型评价 --------
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # -------- 5. 未来预测 --------
    predictions = model.predict(future_X)

    # -------- 6. 打印结果 --------
    print("📌 模型参数：")
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(X.shape[1])]
    for name, coef in zip(feature_names, coefs):
        print(f"  {name} 的回归系数 = {coef:.4f}")
    print(f"  截距 b = {intercept:.4f}")

    print("\n📌 模型评价：")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")

    print("\n📌 预测结果：")
    for x_val, pred in zip(future_X, predictions):
        print(f"  X={x_val} → 预测值 {pred:.2f}")

    return model, X, y


# -------------------- 可选绘图函数（二维特征的情况） --------------------
def plot_3d_regression(X, y, model, feature_names=["X1", "X2"]):
    """ 绘制 3D 回归平面（仅适用于 2 个特征） """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    # 散点图（实际数据）
    ax.scatter(X[:,0], X[:,1], y, color="blue", label="实际数据")

    # 网格数据
    x1_range = np.linspace(min(X[:,0]), max(X[:,0]), 10)
    x2_range = np.linspace(min(X[:,1]), max(X[:,1]), 10)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    # 拟合平面
    y_pred_grid = model.predict(X_grid).reshape(x1_grid.shape)
    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, alpha=0.5, cmap=cm.coolwarm)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel("因变量 (Y)")
    plt.title("多元线性回归拟合平面")
    plt.show()


# -------------------- 使用案例 --------------------
if __name__ == "__main__":
    # 历史数据：广告投入 & 员工人数 → 销售额
    X = [
        [20, 30, 15],
        [25, 35, 18],
        [30, 40, 20],
        [35, 45, 22],
        [40, 50, 25],
        [45, 55, 27]
    ]
    y = [122, 140, 153, 167, 182, 200]

    future_X = [[32, 42, 21]]

    run_multilinear_regression(X, y, future_X,
                               feature_names=["光伏发电量", "风电发电量", "气温"])

    # 调用作图函数（！！！仅适用 2 特征的情况，>=2不能画图了！！）
    #plot_3d_regression(X_arr, y_arr, model, feature_names=["工人数量", "项目规模"])
