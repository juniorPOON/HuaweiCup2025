import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd

##载入数据，第二个参数代表是否需要自动识别yy-mm--dd这样的日期
def load_data(file_path, parse_dates=None):  # 函数名建议改load_data（更准确，不是load_date）
    try:
        # 根据文件后缀判断格式，选择对应读取方法
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, parse_dates=parse_dates)  # 用传入的parse_dates参数
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, parse_dates=parse_dates)
        else:
            print(f"错误：不支持的文件格式！仅支持 .xlsx/.xls/.csv")
            return None

        print(f"文件读取成功，共 {len(df)} 行数据")
        return df  # 返回读取成功的DataFrame

    except FileNotFoundError:
        print(f"错误：文件路径不存在，请检查路径：{file_path}")
        return None
    except KeyError as e:
        print(f"错误：数据中缺少指定的日期列 {e}（请确认列名正确）")
        return None
    except Exception as e:  # 捕获其他意外错误（如文件损坏）
        print(f"文件读取失败：{str(e)}")
        return None


def figure_set_chinese ():
    # --------------------------中文显示配置--------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题（可选，防止后续绘图负号异常）

    # -----------------------------------------------------------------