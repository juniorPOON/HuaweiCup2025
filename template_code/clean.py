import pandas as pd
from toolbox import load_data
def clean_my_data(file_path, region=None):
    """读取 Excel，筛选地区，按月统计销售额并生成透视表"""

    df = load_data("table\\trade.xlsx",parse_dates=["Date"])
    if region:
        df = df[df["Region"] == region].copy()
        if df.empty:
            print(f"警告：没有找到'{region}'地区的数据")
            return None

    # 新增年月维度
    df['YearMonth'] = df['Date'].dt.to_period('M')

    # 步骤1：先按年月+品类分组，求销售额总和（得到每个分组的总和）
    sales_sum = df.groupby(['YearMonth', 'Category'])['Sales'].sum().reset_index()

    #假设需要找最大值#
    # # 步骤2：按“年月（YearMonth）”分组，找每个年月下销售额最大的行的索引
    # max_sales_idx = sales_sum.groupby('YearMonth')['Sales'].idxmax()
    #
    # # 步骤3：用索引提取“每月销售额最高的品类”
    # monthly_top_category = sales_sum.loc[max_sales_idx].reset_index(drop=True)


    # 透视表
    monthly_sales_pivot = sales_sum.pivot(
        index='YearMonth',
        columns='Category',
        values='Sales'
    ).fillna(0)

    #返回这个方便后续作图
    return monthly_sales_pivot
