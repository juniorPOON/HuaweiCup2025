import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# 引入你写好的工具函数
from toolbox import parse_file_info, expand_by_channel


# 1. 扫描整个数据集，解析文件信息
def scan_dataset(root_dir, out_csv="file_index.csv"):
    records = []
    for fp in glob.glob(os.path.join(root_dir, "**", "*.mat"), recursive=True):
        info = parse_file_info(fp)
        expanded = expand_by_channel(info)   # Normal 情况下会拆成多通道
        records.extend(expanded)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[OK] file_index.csv 已生成，共 {len(df)} 条记录")
    return df


# 2. 文件级划分 train/val/test
def assign_split(file_index_csv, out_csv="split_manifest.csv", seed=42):
    df = pd.read_csv(file_index_csv)

    # stratify 按 label_str 保证类平衡
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df["label_str"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val["label_str"], random_state=seed
    )

    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    df.to_csv(out_csv, index=False)
    print(f"[OK] split_manifest.csv 已生成，train/val/test = {len(train)}/{len(val)}/{len(test)}")
    return df



import scipy.io as sio
import pandas as pd

def slice_file(info, win_ms=40, hop_ms=10):
    """
    对单个文件进行切片，返回切片列表
    info: 来自 split_manifest.csv 的一行（dict）
    """
    fs = info["fs"]
    file_path = info["abs_path"]
    channel = info["channel"]

    # 读取 .mat 数据
    mat_data = sio.loadmat(file_path)
    key_candidates = [k for k in mat_data.keys() if k.endswith(f"_{channel}_time")]
    if not key_candidates:
        return []
    key = key_candidates[0]
    signal = mat_data[key].squeeze()

    # 窗长/步长（点数）
    win_len = int(fs * win_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)

    slices = []
    for start in range(0, len(signal) - win_len + 1, hop_len):
        end = start + win_len
        slices.append(dict(
            file_name=info["file_name"],
            abs_path=file_path,
            slice_start=start,
            slice_end=end,
            fs=fs,
            rpm=info["rpm"],
            label_str=info["label_str"],
            label_bin=info["label_bin"],
            fault_size=info["fault_size"],
            channel=channel,
            split=info["split"]
        ))
    return slices
def slice_all(split_manifest_csv, out_csv="task1_slices_index.csv"):
    df = pd.read_csv(split_manifest_csv)
    all_slices = []
    for _, row in df.iterrows():
        info = row.to_dict()
        slices = slice_file(info)
        all_slices.extend(slices)

    df_slices = pd.DataFrame(all_slices)
    df_slices.to_csv(out_csv, index=False)
    print(f"[OK] 已生成切片索引 {out_csv}，共 {len(df_slices)} 条切片")
    return df_slices





if __name__ == "__main__":
    # root_dir = r"D:\aaamcm25\data_set\source_d_dataset"  # 修改成你自己的路径
    #
    # # 第一步：扫描数据集
    # df_files = scan_dataset(root_dir, out_csv="file_index.csv")
    #
    # # 第二步：文件级划分
    # df_split = assign_split("file_index.csv", out_csv="split_manifest.csv")
    slice_all("split_manifest.csv", out_csv="task1_slices_index.csv")