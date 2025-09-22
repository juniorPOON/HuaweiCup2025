"""
Task1 Feature Extraction Pipeline (升级版带进度提示)
------------------------------------------------
功能：
- 生成方法 A & Baseline 的特征 CSV
- 每处理 1000 个切片打印一次进度信息，方便长任务监控
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.io as sio
from scipy.stats import kurtosis, skew


# ===============================================================
# 工具函数
# ===============================================================

def load_slice(row, resample_to=None):
    """
    根据切片索引（row）取出波形数据。
    - 方法 A：直接读取切片，不改采样率
    - Baseline：先低通到 ≤6kHz，再重采样到 12kHz

    参数：
    - row: DataFrame 的一行（字典形式），包含切片信息
    - resample_to: 如果为 None → 保持原采样率；否则重采样到指定频率
    """
    mat_data = sio.loadmat(row["abs_path"])
    key = [k for k in mat_data.keys() if k.endswith(f"_{row['channel']}_time")][0]
    sig = mat_data[key].squeeze()[row["slice_start"]:row["slice_end"]]

    fs = int(row["fs"])

    # baseline 情况：低通 + 重采样到 12kHz
    if resample_to and fs != resample_to:
        nyq = resample_to / 2
        # 抗混叠低通滤波
        sos = signal.butter(10, nyq, btype="low", fs=fs, output="sos")
        sig = signal.sosfilt(sos, sig)
        # 重采样
        sig = signal.resample_poly(sig, up=resample_to, down=fs)
        fs = resample_to

    return sig, fs


def extract_features(sig, fs):
    """
    提取切片的特征：
    - 时域统计量
    - 包络谱能量

    参数：
    - sig: 波形信号（numpy 数组）
    - fs: 采样率

    返回：
    - feats: dict，包含特征值
    """
    feats = {}

    # ---------------- 时域特征 ----------------
    feats["mean"] = np.mean(sig)
    feats["std"] = np.std(sig)
    feats["rms"] = np.sqrt(np.mean(sig**2))
    feats["kurtosis"] = kurtosis(sig, fisher=True)  # 峭度
    feats["skewness"] = skew(sig)                   # 偏度
    feats["peak2peak"] = np.ptp(sig)                # 峰-峰值
    feats["crest_factor"] = np.max(np.abs(sig)) / (feats["rms"] + 1e-12)

    # ---------------- 包络谱能量 ----------------
    # 500–5500 Hz 带通滤波（注意 fs < 11k 时自动缩减）
    if fs > 12000:
        band = [500, 5500]
    else:
        band = [500, fs // 2 - 100]

    sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    banded = signal.sosfilt(sos, sig)

    # Hilbert 包络
    env = np.abs(signal.hilbert(banded))

    # Welch 能量谱（动态选择 nperseg，避免 warning）
    nperseg = min(1024, len(env))
    f, Pxx = signal.welch(env, fs, nperseg=nperseg)

    # 积分能量（np.trapezoid 替代 trapz）
    feats["env_energy"] = np.trapezoid(Pxx, f)

    return feats


def process_pipeline(slices_csv, out_csv, mode="A"):
    """
    整个特征提取流程：读取索引 → 提取波形 → 提特征 → 保存 CSV

    参数：
    - slices_csv: 输入的切片索引 CSV 文件
    - out_csv: 输出特征 CSV 文件路径
    - mode: "A" → 方法A（原采样率），"B" → baseline（低通+重采样）
    """
    df = pd.read_csv(slices_csv)
    feats_all = []

    total = len(df)   # 总切片数
    print(f"[INFO] 开始处理 {mode} 模式，共 {total} 个切片")

    for i, row in enumerate(df.iterrows()):
        row = row[1].to_dict()

        # 根据模式加载切片
        if mode == "A":
            sig, fs = load_slice(row, resample_to=None)
        elif mode == "B":
            sig, fs = load_slice(row, resample_to=12000)
        else:
            raise ValueError("mode must be 'A' or 'B'")

        # 提取特征
        feats = extract_features(sig, fs)

        # 补充元信息
        feats.update({
            "file_name": row["file_name"],
            "slice_start": row["slice_start"],
            "slice_end": row["slice_end"],
            "fs": fs,
            "rpm": row["rpm"],
            "label_str": row["label_str"],
            "label_bin": row["label_bin"],
            "channel": row["channel"],
            "split": row["split"],
            "route": mode
        })

        feats_all.append(feats)

        # 👉 每处理 1000 个切片，打印一次进度提示
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            percent = (i + 1) / total * 100
            print(f"[INFO] 已处理 {i+1}/{total} 个切片 ({percent:.2f}%)")

    # 保存 CSV
    df_feats = pd.DataFrame(feats_all)
    df_feats.to_csv(out_csv, index=False)
    print(f"[OK] {mode} → {out_csv} 共 {len(df_feats)} 条切片特征")

    return df_feats


# ===============================================================
# 主入口
# ===============================================================
if __name__ == "__main__":
    slices_csv = "task1_slices_index.csv"

    # 方法 A
    process_pipeline(slices_csv, "task1_features_A.csv", mode="A")

    # Baseline
    process_pipeline(slices_csv, "task1_features_B.csv", mode="B")
