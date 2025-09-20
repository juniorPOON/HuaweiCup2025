import numpy as np
import matplotlib.pyplot as plt
# 从statsmodels库导入Holt模型（用于二次指数平滑，处理带趋势的数据）
from statsmodels.tsa.holtwinters import Holt
# 导入评估指标：平均绝对百分比误差(MAPE)和均方误差(MSE)
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
# 导入自定义工具函数（设置图表中文显示，避免中文乱码）
from toolbox import figure_set_chinese

# 调用中文设置函数，确保图表中中文正常显示
figure_set_chinese()


def run_holt(y, steps=6, alpha=None, beta=None, damped_trend=False):
    """
    Holt二次指数平滑模型（带趋势的指数平滑）
    适用于：有明显趋势但无季节性的时间序列预测

    参数说明：
        y: 一维序列(list/array)，输入的历史数据（如销量、负荷等时间序列）
        steps: int，预测未来的步数（默认预测6步）
        alpha: 水平平滑系数（0~1，控制历史水平值的权重）
               - 若为None，模型会自动优化该参数
               - 越大表示越重视近期数据，越小则平滑效果越强
        beta: 趋势平滑系数（0~1，控制趋势分量的权重）
               - 若为None，模型会自动优化该参数
               - 越大表示趋势对新数据越敏感
        damped_trend: bool，是否启用趋势阻尼（默认False）
               - True：趋势会随时间逐渐减弱（避免长期预测值过度偏离）
               - False：趋势保持恒定（适合短期稳定趋势）

    返回值：
        fitted: 模型对历史数据的拟合值（与输入y长度相同）
        forecast: 未来steps步的预测值（长度为steps）
    """
    # 初始化Holt模型：指定输入数据和是否启用趋势阻尼
    # 相比一次指数平滑，Holt模型增加了对趋势的建模能力
    model = Holt(y, damped_trend=damped_trend)

    # 拟合模型：计算平滑系数并生成历史拟合值
    fit = model.fit(
        smoothing_level=alpha,  # 水平平滑系数（alpha）
        smoothing_slope=beta,  # 趋势平滑系数（beta）
        # 若alpha或beta未指定（为None），则自动优化参数（optimized=True）
        optimized=(alpha is None or beta is None)
    )

    # 获取模型对历史数据的拟合值（用于评估模型好坏）
    fitted = fit.fittedvalues
    # 预测未来steps步的值
    forecast = fit.forecast(steps)

    # 模型评估：计算训练数据上的误差指标
    # 1. 平均绝对百分比误差（MAPE，相对误差，越小越好，单位%）
    mape = mean_absolute_percentage_error(y, fitted)
    # 2. 均方误差（MSE）和均方根误差（RMSE，绝对误差，单位与原始数据一致）
    mse = mean_squared_error(y, fitted)
    rmse = np.sqrt(mse)  # 对MSE开平方得到RMSE

    # 打印评估结果（保留3位小数，直观展示模型性能）
    print(f"[Holt] 训练MAPE={mape:.3f}, RMSE={rmse:.3f}")
    # 打印模型实际使用的参数（若未指定则显示优化后的结果）
    # getattr用于安全获取属性，若不存在则返回None
    print("  alpha=", getattr(fit.model, "smoothing_level", None),
          " beta=", getattr(fit.model, "smoothing_slope", None),
          " damped=", damped_trend)

    # 数据可视化：对比原始数据、拟合值和预测值
    plt.plot(y, label="原始")  # 原始数据曲线
    plt.plot(fitted, label="拟合(Holt)")  # 模型对历史数据的拟合曲线
    # 预测值曲线：x轴从历史数据结束位置开始，用红色虚线标记
    plt.plot(range(len(y), len(y) + steps), forecast, "r--", label="预测")
    plt.title("Holt 二次指数平滑（趋势）")  # 图表标题
    plt.legend()  # 显示图例（区分不同曲线）
    plt.grid(True, ls="--", alpha=.4)  # 添加网格线（虚线，透明度40%）
    plt.show()  # 显示图表

    return fitted, forecast


# 主程序：示例用法
if __name__ == "__main__":
    # 示例数据：12个月的销量（呈现平稳上升趋势，无明显季节性波动）
    y = [120, 125, 130, 138, 145, 150, 158, 165, 173, 180, 188, 197]

    # 调用Holt模型：预测未来6步，启用趋势阻尼（避免长期预测过度增长）
    # 返回值：历史拟合值(fitted)和未来预测值(forecast)
    fitted, forecast = run_holt(y, steps=6, damped_trend=True)
