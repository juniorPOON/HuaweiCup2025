import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from toolbox import figure_set_chinese
figure_set_chinese()


def printpara(fit):
    params = fit.params
    print(f"α={params['smoothing_level']:.3f}, "
          f"β={params['smoothing_trend']:.3f}, "
          f"γ={params['smoothing_seasonal']:.3f}, "
          f"δ(阻尼)={params['damping_trend']:.3f}")

def run_holt_winters(y, m, steps=12, trend="add", seasonal="add", damped_trend=False):
    model = ExponentialSmoothing(
        y, trend=trend, damped_trend=damped_trend,
        seasonal=seasonal, seasonal_periods=m
    )
    fit = model.fit(optimized=True)  # 自动寻优 α/β/γ
    fitted = fit.fittedvalues
    forecast = fit.forecast(steps)

    mape = mean_absolute_percentage_error(y, fitted)

    rmse = mean_squared_error(y, fitted) ** 0.5
    printpara(fit)

    plt.plot(y, label="原始")
    plt.plot(fitted, label="拟合(Holt-Winters)")
    plt.plot(range(len(y), len(y)+steps), forecast, "r--", label="预测")
    plt.title(f"Holt-Winters（三次）：trend={trend}, seasonal={seasonal}, m={m}")
    plt.legend(); plt.grid(True, ls="--", alpha=.4); plt.show()
    return fitted, forecast


if __name__ == "__main__":
    day = [320, 300, 280, 260, 250, 240, 260, 300, 380, 420, 460, 500,
           520, 540, 560, 550, 520, 480, 440, 400, 360, 340, 330, 320]
    y = day + [v * 1.05 for v in day]  # 第二天整体略增
    fitted, forecast = run_holt_winters(y, m=24, steps=24, trend="add", seasonal="add", damped_trend=True)

