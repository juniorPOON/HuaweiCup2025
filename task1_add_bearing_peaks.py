"""
task1_add_bearing_peaks.py
--------------------------
用途：
  在你已经生成的 task1_features_A.csv / task1_features_B.csv 上，
  追加 BPFO / BPFI / BSF（以及 FTF 可选）的峰值与 SNR 等可解释性特征。
  不会重算已有的统计特征，只补充新列。

做法：
  1) 读取特征 CSV（A 或 B）
  2) 与切片索引 task1_slices_index.csv 通过 (file_name, slice_start, slice_end, channel) 合并，拿到 abs_path
  3) 逐行按 route 读取切片（A 保持原 fs；B 低通→重采 12 kHz）
  4) 计算包络谱（带通 500–5500 Hz → Hilbert → Welch 频谱）
  5) 计算理论频率（BPFO/BPFI/BSF/FTF），在 ±窗口内找峰，记录幅值与 SNR
  6) 将新列写回新的 CSV（不会覆盖旧文件，安全）

注意：
  - 需要 scipy, numpy, pandas
  - rpm 缺失时，本行峰值列写 NaN
  - 每 1000 条打印一次进度
"""

import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.io as sio

# ========== 轴承几何参数（题目给的 SKF6205）==========
N_ROLLERS = 9               # 滚动体数 n
D_PITCH   = 1.537           # 轴承节径 D (inch) —— 只用比值，单位不影响结果
D_ROLLER  = 0.3126          # 滚动体直径 d (inch)
COS_THETA = 1.0             # 接触角 cosθ，径向工况近似 1

# ========== 通用参数 ==========
BASELINE_FS = 12000         # baseline 统一到 12 kHz
BAND_LOW    = 500           # 包络滤波下限
BAND_HIGH   = 5500          # 包络滤波上限（若 fs 太小，会自动压到 Nyquist 以下）
PEAK_WIN_HZ = 5.0           # 目标频率 ±窗口（Hz），也可用相对误差
SNR_SIDE_HZ = 20.0          # 侧带均值带宽（用于估计局部噪声/SNR 的分母）


# ---------------- 工具函数 ----------------

def compute_fault_freqs(rpm):
    """
    根据 rpm 计算理论频率：BPFO / BPFI / BSF / FTF
    返回字典（Hz）
    """
    if rpm is None or np.isnan(rpm):
        return None

    fr = float(rpm) / 60.0  # 转轴频率 (Hz)
    ratio = D_ROLLER / D_PITCH

    bpfo = 0.5 * N_ROLLERS * fr * (1 - ratio * COS_THETA)
    bpfi = 0.5 * N_ROLLERS * fr * (1 + ratio * COS_THETA)
    bsf  = (D_PITCH / (2 * D_ROLLER)) * fr * (1 - (ratio * COS_THETA) ** 2)
    ftf  = 0.5 * fr * (1 - ratio * COS_THETA)

    return dict(bpfo=bpfo, bpfi=bpfi, bsf=bsf, ftf=ftf)


def load_slice(row, resample_to=None):
    """
    按切片索引读取一段波形：
      - A 路线：resample_to=None（保持原 fs）
      - B 路线：resample_to=12000（低通→重采）
    """
    mat = sio.loadmat(row["abs_path"])
    # 变量名形如 ..._DE_time / _FE_time / _BA_time
    key = [k for k in mat.keys() if k.endswith(f"_{row['channel']}_time")]
    if not key:
        raise KeyError(f"找不到通道变量: {row['channel']} in {row['abs_path']}")
    sig_full = mat[key[0]].squeeze()

    s0, s1 = int(row["slice_start"]), int(row["slice_end"])
    sig = sig_full[s0:s1]
    fs = int(row["fs"])

    # baseline：低通 → 重采样
    if resample_to and fs != resample_to:
        # 抗混叠低通（截止取新 fs 的 Nyquist）
        # 注意：butter 的 Wn 是 Hz（指定 fs 参数），这里取截止 = resample_to/2 * 0.98 给点余量
        cutoff = 0.98 * (resample_to / 2.0)
        sos = signal.butter(10, cutoff, btype="low", fs=fs, output="sos")
        sig = signal.sosfilt(sos, sig)
        # 有理重采样（多相 FIR）
        sig = signal.resample_poly(sig, up=resample_to, down=fs)
        fs = resample_to

    return sig, fs


def envelope_spectrum(sig, fs):
    """
    计算包络谱（Welch）：
      1) 带通（500~5500Hz；fs 不足时压 Nyquist）
      2) Hilbert 包络
      3) Welch 估计（自适应 nperseg）
    返回：f(Hz), Pxx
    """
    # 带通范围根据 fs 自动调整
    hi = min(BAND_HIGH, fs / 2 - 100)  # 给点余量防止设计失败
    if hi <= BAND_LOW + 50:
        # fs 太小，包络法意义不大，退化为仅 Hilbert + Welch
        banded = sig
    else:
        sos = signal.butter(4, [BAND_LOW, hi], btype="bandpass", fs=fs, output="sos")
        banded = signal.sosfilt(sos, sig)

    env = np.abs(signal.hilbert(banded))
    nperseg = min(2048, len(env)) if len(env) >= 256 else len(env)
    if nperseg < 16:
        # 切片太短（极端情况），避免崩
        nperseg = len(env)
    f, Pxx = signal.welch(env, fs=fs, nperseg=nperseg)
    return f, Pxx


def find_peak_near(f, Pxx, target_hz, tol_hz=PEAK_WIN_HZ):
    """
    在频谱 (f, Pxx) 上，寻找 target_hz ± tol_hz 窗内的最大峰值，
    并计算一个简单的 SNR（峰值 / 邻域均值）。

    返回：
      peak_hz, peak_amp, peak_snr
    若窗口越界或无数据，返回 (np.nan, np.nan, np.nan)
    """
    if target_hz is None or np.isnan(target_hz):
        return np.nan, np.nan, np.nan

    # 窗口掩码
    mwin = (f >= (target_hz - tol_hz)) & (f <= (target_hz + tol_hz))
    if not np.any(mwin):
        return np.nan, np.nan, np.nan

    # 峰值（功率最大点）
    idx = np.argmax(Pxx[mwin])
    f_win = f[mwin]
    p_win = Pxx[mwin]
    peak_hz = float(f_win[idx])
    peak_amp = float(p_win[idx])

    # 侧带均值（用于 SNR 分母）：排除主窗 ±tol，再取 ±SNR_SIDE_HZ 的环带
    side_left = (f >= (target_hz - tol_hz - SNR_SIDE_HZ)) & (f < (target_hz - tol_hz))
    side_right = (f > (target_hz + tol_hz)) & (f <= (target_hz + tol_hz + SNR_SIDE_HZ))
    side_mask = side_left | side_right
    if np.any(side_mask):
        noise_floor = float(np.mean(Pxx[side_mask]))
        # 避免除零
        peak_snr = peak_amp / (noise_floor + 1e-12)
    else:
        peak_snr = np.nan

    return peak_hz, peak_amp, peak_snr


def enrich_one_csv(features_csv, slices_index_csv, out_csv=None):
    """
    读取某一条路线的特征 CSV（A 或 B），基于切片索引补齐 abs_path，
    逐条计算包络谱并扒取 BPFO/BPFI/BSF/FTF 的峰值与 SNR，写出新的 CSV。

    参数：
      - features_csv: 现有的 task1_features_A.csv / task1_features_B.csv
      - slices_index_csv: task1_slices_index.csv（含 abs_path）
      - out_csv: 输出文件名（None -> 自动在原名后加 _with_peaks）
    """
    df_feat = pd.read_csv(features_csv)
    df_idx  = pd.read_csv(slices_index_csv)

    # 只取 merge 所需的关键列，防止重复列冲突
    cols_idx = ["file_name", "slice_start", "slice_end", "channel", "abs_path"]
    df_idx_small = df_idx[cols_idx].drop_duplicates()

    # 合并出 abs_path
    df = pd.merge(
        df_feat,
        df_idx_small,
        on=["file_name", "slice_start", "slice_end", "channel"],
        how="left",
        validate="m:1"  # 每个切片在索引表应唯一
    )

    # 检查是否有缺失路径
    missing = df["abs_path"].isna().sum()
    if missing > 0:
        print(f"[WARN] 有 {missing} 条记录找不到 abs_path（可能索引键不一致），这些行将无法计算峰值。")

    total = len(df)
    print(f"[INFO] 开始为 {features_csv} 追加 BPFO/BPFI/BSF 峰值，目标行数：{total}")

    # 预分配新列（用 NaN 占位）
    for col in [
        "bpfo_hz","bpfo_amp","bpfo_snr",
        "bpfi_hz","bpfi_amp","bpfi_snr",
        "bsf_hz","bsf_amp","bsf_snr",
        "ftf_hz","ftf_amp","ftf_snr"
    ]:
        df[col] = np.nan

    # 逐行处理
    for i, row in df.iterrows():
        # 进度打印（每 1000 条）
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"[INFO] 进度 {i+1}/{total} ({(i+1)/total*100:.2f}%)")

        # 没路径/没 rpm 的行，跳过
        if pd.isna(row.get("abs_path")) or pd.isna(row.get("rpm")):
            continue

        # 决定路线：A 不重采样；B 重采到 12k
        route = str(row.get("route", "A")).upper()
        try:
            if route == "B":
                sig, fs = load_slice(row, resample_to=BASELINE_FS)
            else:
                sig, fs = load_slice(row, resample_to=None)
        except Exception as e:
            # 读取失败的行跳过（不影响全局）
            # print(f"[WARN] 读取切片失败: {row['file_name']} | {e}")
            continue

        # 包络谱
        try:
            f, Pxx = envelope_spectrum(sig, fs)
        except Exception as e:
            # print(f"[WARN] 包络谱失败: {row['file_name']} | {e}")
            continue

        # 理论频率
        freqs = compute_fault_freqs(row["rpm"])
        if freqs is None:
            continue

        # 避免目标频率超出可分析带宽（Nyquist 之下）
        # 这里只在查峰时自然会被窗口掩码处理，不额外强裁剪

        # 在目标频率附近扒峰并记录
        bpfo_hz, bpfo_amp, bpfo_snr = find_peak_near(f, Pxx, freqs["bpfo"])
        bpfi_hz, bpfi_amp, bpfi_snr = find_peak_near(f, Pxx, freqs["bpfi"])
        bsf_hz,  bsf_amp,  bsf_snr  = find_peak_near(f, Pxx, freqs["bsf"])
        ftf_hz,  ftf_amp,  ftf_snr  = find_peak_near(f, Pxx, freqs["ftf"])

        df.at[i, "bpfo_hz"] = bpfo_hz
        df.at[i, "bpfo_amp"] = bpfo_amp
        df.at[i, "bpfo_snr"] = bpfo_snr

        df.at[i, "bpfi_hz"] = bpfi_hz
        df.at[i, "bpfi_amp"] = bpfi_amp
        df.at[i, "bpfi_snr"] = bpfi_snr

        df.at[i, "bsf_hz"] = bsf_hz
        df.at[i, "bsf_amp"] = bsf_amp
        df.at[i, "bsf_snr"] = bsf_snr

        df.at[i, "ftf_hz"] = ftf_hz
        df.at[i, "ftf_amp"] = ftf_amp
        df.at[i, "ftf_snr"] = ftf_snr

    # 输出
    if out_csv is None:
        base, ext = os.path.splitext(features_csv)
        out_csv = f"{base}_with_peaks{ext}"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] 已写出：{out_csv} （新增 BPFO/BPFI/BSF/FTF 列）")


# ---------------- 主入口 ----------------

if __name__ == "__main__":
    # 路径按你的工程来改
    slices_index_csv = "task1_slices_index.csv"

    # 为 A/B 两条线各自追加峰值列
    enrich_one_csv("task1_features_A.csv", slices_index_csv)
    enrich_one_csv("task1_features_B.csv", slices_index_csv)
