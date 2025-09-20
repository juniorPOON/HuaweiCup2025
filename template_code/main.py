from clean import clean_my_data
from create_time_figure import plot_monthly_sales_trend

pivot = clean_my_data("table\\trade.xlsx", region="华南")
if pivot is not None:
    plot_monthly_sales_trend(pivot, title="华南地区各商品类别月度销售额趋势")
