import matplotlib.pyplot as plt

def plot_monthly_sales_trend(monthly_sales_pivot, title="月度销售额趋势"):
    """绘制各品类月度销售额趋势"""
    # 数据可视化（先添加字体配置）
    # --------------------------中文显示配置--------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题（可选，防止后续绘图负号异常）
    # -----------------------------------------------------------------

    plt.figure(figsize=(12,6))
    for category in monthly_sales_pivot.columns:
        plt.plot(monthly_sales_pivot.index.to_timestamp(),
                 monthly_sales_pivot[category],
                 marker='o',
                 label=category,
                 linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("月份", fontsize=12)
    plt.ylabel("销售额", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="商品类别")
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
