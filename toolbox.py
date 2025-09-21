import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import re
import pandas as pd
import scipy.io as sio   # <--- 别忘了这个！
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

def expand_by_channel(info_dict):
    """
    把 parse_file_info 的结果展开成多条单通道样本
    通常在channel = normal里面使用，因为他是复合信息
    """
    expanded = []
    channels = info_dict.get("channels")
    if not channels:  # 没有 channel 信息
        expanded.append({**info_dict, "channel": "UNK"})
    else:
        for ch in channels:
            item = info_dict.copy()
            item["channel"] = ch
            expanded.append(item)
    #返回的是一个列表
    return expanded



def parse_file_info(file_path):
    fname = os.path.basename(file_path)
    parts = file_path.replace("\\", "/").split("/")

    # -------- fs：全路径里找 *kHz ----------
    fs = None
    for p in parts:
        m = re.search(r'(\d+)\s*kHz', p, flags=re.I)
        if m:
            val = int(m.group(1))
            if val in (12, 32, 48):
                fs = val * 1000
                break

    # -------- label / fault_size ----------
    if "Normal" in parts[-2]:
        label_str, label_bin, fault_size = "N", 0, 0
    elif fname.startswith("B"):
        label_str, label_bin = "B", 1
        m = re.search(r"(\d{3,4})", fname); fault_size = int(m.group(1)) if m else None
    elif fname.startswith("IR"):
        label_str, label_bin = "IR", 1
        m = re.search(r"(\d{3,4})", fname); fault_size = int(m.group(1)) if m else None
    elif fname.startswith("OR"):
        label_str, label_bin = "OR", 1
        m = re.search(r"(\d{3,4})", fname); fault_size = int(m.group(1)) if m else None
    elif fname.startswith("N"):
        label_str, label_bin, fault_size = "N", 0, 0
    else:
        label_str, label_bin, fault_size = "UNK", -1, None

    # -------- 外圈位置信息（@12/@6/@3 + 目录名映射） ----------
    fault_loc = None
    if fname.startswith("OR"):
        if "@3" in fname: fault_loc = "3 o'clock"
        elif "@6" in fname: fault_loc = "6 o'clock"
        elif "@12" in fname: fault_loc = "12 o'clock"
        if not fault_loc:
            low_parts = [p.lower() for p in parts]
            if "orthogonal" in low_parts: fault_loc = "3 o'clock"
            elif "centered" in low_parts: fault_loc = "6 o'clock"
            elif "opposite" in low_parts: fault_loc = "12 o'clock"

    # -------- rpm：先读 .mat，读不到再文件名兜底 ----------
    rpm = None
    try:
        mat_data = sio.loadmat(file_path)
        rpm_keys = [k for k in mat_data.keys() if "RPM" in k.upper()]
        if rpm_keys:
            rpm_val = np.array(mat_data[rpm_keys[0]]).squeeze()
            if rpm_val.size == 1:
                rpm = int(float(rpm_val))
    except Exception:
        pass
    if rpm is None:
        m = re.search(r"\((\d+)rpm\)", fname, flags=re.I)
        rpm = int(m.group(1)) if m else None

    # -------- 先“猜”路径通道（严格匹配 *_data，避免 Centered 误判） ----------
    path_channel = None
    for p in parts:
        pl = p.lower()
        if re.fullmatch(r'(?:\d+kHz_)?de_data', pl, flags=re.I): path_channel = "DE"; break
        if re.fullmatch(r'(?:\d+kHz_)?fe_data', pl, flags=re.I): path_channel = "FE"; break
        if re.fullmatch(r'(?:\d+kHz_)?ba_data', pl, flags=re.I): path_channel = "BA"; break
        # Normal_data 不作为通道

    channel = path_channel if path_channel else "UNK"

    # -------- 扫描 .mat 真实通道，但若有 path_channel 就强制以路径为准 ----------
    channels = None
    try:
        if 'mat_data' not in locals():
            mat_data = sio.loadmat(file_path)
        keys = list(mat_data.keys())
        found = []
        if any(k.endswith("_DE_time") for k in keys): found.append("DE")
        if any(k.endswith("_FE_time") for k in keys): found.append("FE")
        if any(k.endswith("_BA_time") for k in keys): found.append("BA")
        if path_channel:
            # ★ 路径优先：只保留该通道，忽略其它
            channels = [path_channel]
            channel = path_channel
        else:
            if found:
                channels = found
                channel = found[0] if len(found) == 1 else "MULTI"
    except Exception:
        if path_channel:
            channels = [path_channel]

    return dict(
        file_name=fname,
        abs_path=file_path,
        fs=fs,
        rpm=rpm,
        label_str=label_str,
        label_bin=label_bin,
        fault_size=fault_size,
        fault_loc=fault_loc,
        channel=channel,      # 单值：DE/FE/BA/MULTI/UNK（有路径通道时恒为该通道）
        channels=channels     # 列表：若有路径通道则只含那个；Normal 情况下可能是多路
    )


if __name__ == "__main__":
    fp1 = r"D:\aaamcm25\data_set\source_d_dataset\48kHz_Normal_data\N_0.mat"

    fp3 = r"D:\aaamcm25\data_set\source_d_dataset\48kHz_DE_data\IR\0014\IR014_2.mat"


    print(parse_file_info(fp1))
    print(parse_file_info(fp3))


