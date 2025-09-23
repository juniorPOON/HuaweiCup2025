"""
task1_export_spectra_gpu.py
---------------------------
用途：
  从 task1_slices_index.csv 导出两条路线（A: 原fs；B: 低通→12k）的频域资产：
    1) env2048_npy：2048-bin 包络谱（CPU，轻量，float16）
    2) mel_npy：Log-Mel 频谱（可选，GPU/PyTorch 加速，float16）

与原脚本的不同：
  - 新增了 GPU 加速的 Mel 导出（torchaudio优先；没有的话走纯 torch 实现）
  - 默认只导 env2048（必备），mel_npy 你准备上 CNN 时再打开
  - 控制并行/断点续跑，避免 CPU 满负载+重复计算

依赖：
  - numpy, pandas, scipy
  - pytorch (>=1.10) 必需
  - torchaudio（可选，若无则自动用纯 torch 路线）

注意：
  - A 路线：按原 fs 做特征
  - B 路线：先抗混叠低通，再重采到 12 kHz 再做特征
  - 频谱全部保存为 float16，体积大幅缩小
"""

import os
import math
import json
import time
import traceback
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.io as sio

import torch

# 尝试用 torchaudio（更快更稳）；没有也能跑（用纯 torch 实现）
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False


# =========================
# ======== 配 置 ==========
# =========================

# —— 产物开关（强烈建议：先只开 ENV；要上 CNN 再开 MEL）——
SAVE_ENV_NPY = True      # ✅ 必开：2048-bin 包络谱（轻量、机理强）
SAVE_MEL_NPY = True     # 上 CNN 再 True；否则先 False
SAVE_MEL_PNG = False     # 一直 False（训练用不到；抽样画图时现画）

# —— 性能 / 体积 ——
DTYPE = "float16"        # 统一存 float16（占空间减半）
MEL_BINS = 64            # 64 或 96 即可（别 128 起步）
MEL_IMG_SIDE = 96        # 可按需统一成方图（96×96）；CNN 输入友好
FMAX_RATIO = 0.90        # Mel 上限 = 0.9 * Nyquist

# —— 路线与数据子集 ——
ONLY_SPLITS = {"train", "val"}  # 先导 train/val；test 用前另导
ROUTES = ("A", "B")             # 两条路线都做：A原fs；B重采12k

# —— 并行与进度 ——
NUM_WORKERS = 2         # Windows 建议 2~4；Linux 可更高
LOG_EVERY = 1000         # 每处理多少条打印一次
SKIP_IF_EXISTS = True    # ✅ 断点续跑：已存在就跳过
MAX_PER_FILE = None      # 每文件最多导出的切片数（先小后大；None=不限）

# —— Baseline 重采样参数 ——
BASELINE_FS = 12_000
LP_ORDER = 10            # 抗混叠低通阶数
LP_MARGIN = 0.98         # 截止频率余量（0.98 * Nyquist）

# —— 包络谱参数 ——
BAND_LOW = 500
BAND_HIGH = 5500
ENV_NBINS = 2048

# —— 路径 ——
SLICES_CSV = "task1_slices_index.csv"
ARTI_ROOT = "artifacts"
OUT_CSV_A = "task1_specs_A.csv"
OUT_CSV_B = "task1_specs_B.csv"

# 控制 PyTorch CPU 线程，避免与 mp 互相抢核（非常重要）
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# =========================
# ====== 工具函数 =========
# =========================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_slice_numpy(row, resample_to=None):
    """
    读取一个切片（NumPy波形），可选重采样（用于 Baseline B）。
    - A：resample_to=None（保持原 fs）
    - B：resample_to=12k（低通→重采）
    """
    mat = sio.loadmat(row["abs_path"])
    k = [k for k in mat.keys() if k.endswith(f"_{row['channel']}_time")]
    if not k:
        raise KeyError(f"变量不存在: _{row['channel']}_time in {row['abs_path']}")
    sig_full = mat[k[0]].squeeze()

    s0, s1 = int(row["slice_start"]), int(row["slice_end"])
    sig = sig_full[s0:s1].astype(np.float32, copy=False)
    fs = int(row["fs"])

    # baseline：低通→重采到 12k
    if resample_to and fs != resample_to:
        cutoff = LP_MARGIN * (resample_to / 2.0)  # 以目标Nyquist为截止
        sos = signal.butter(LP_ORDER, cutoff, btype="low", fs=fs, output="sos")
        sig = signal.sosfilt(sos, sig)
        sig = signal.resample_poly(sig, up=resample_to, down=fs)
        fs = resample_to

    return sig, fs


def envelope_spectrum_2048(sig, fs):
    """
    计算 2048-bin 包络谱（NumPy/CPU）：
      1) 带通（500~5500Hz，fs不足时压Nyquist）
      2) Hilbert 包络
      3) Welch 估计
      4) 频轴插值到固定 ENV_NBINS
    返回：env_spec (ENV_NBINS,)  ndarray
    """
    hi = min(BAND_HIGH, fs / 2 - 100)  # 给余量防设计失败
    if hi > BAND_LOW + 50:
        sos = signal.butter(4, [BAND_LOW, hi], btype="bandpass", fs=fs, output="sos")
        banded = signal.sosfilt(sos, sig)
    else:
        banded = sig

    env = np.abs(signal.hilbert(banded))

    nperseg = min(2048, len(env)) if len(env) >= 256 else len(env)
    if nperseg < 16:
        nperseg = len(env)
    f, Pxx = signal.welch(env, fs=fs, nperseg=nperseg)

    # 插值到固定 bins
    fmax = fs / 2.0
    tgt_f = np.linspace(0, fmax, ENV_NBINS, endpoint=False)
    # 注意：Pxx 对应 [0, f_welch_max]；用边界外插（fill）避免NaN
    env_spec = np.interp(tgt_f, f, Pxx, left=Pxx[0], right=Pxx[-1]).astype(np.float32)

    # 压缩到 float16
    if DTYPE == "float16":
        env_spec = env_spec.astype(np.float16)
    return env_spec


# ====== GPU Mel 提取（torchaudio优先；无则用纯 torch） ======

class MelExtractorTorch:
    """
    GPU 加速的 Mel 频谱提取器：
      - 支持 A/B 两路线：输入张量 & fs
      - 自动按 FMAX_RATIO 设定 f_max
      - 输出统一为 (MEL_BINS, MEL_IMG_SIDE) 的 log-mel（float16）
    """
    def __init__(self, n_mels=MEL_BINS, img_side=MEL_IMG_SIDE, fmax_ratio=FMAX_RATIO, device=None):
        self.n_mels = n_mels
        self.img_side = img_side
        self.fmax_ratio = fmax_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self._cache = {}  # 缓存不同 fs 的 mel 滤波器/配置

    def _get_mel_frontend(self, fs):
        """
        针对某个 fs，构建/缓存 Mel 前端（torchaudio 或纯 torch）。
        返回 dict：{ 'type': 'ta'|'plain', 'spec': callable }
        """
        if fs in self._cache:
            return self._cache[fs]

        f_max = self.fmax_ratio * (fs / 2.0)
        n_fft = 2 ** int(math.ceil(math.log2(int(0.04 * fs))))  # 40ms对应的最近2次幂
        hop = max(1, int(0.01 * fs))                            # 10ms hop

        if HAS_TORCHAUDIO:
            spec_layer = torchaudio.transforms.MelSpectrogram(
                sample_rate=fs,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop,
                n_mels=self.n_mels,
                f_min=0.0,
                f_max=f_max,
                power=2.0,
                center=True,
                norm="slaney",
                mel_scale="htk",
            ).to(self.device)
            self._cache[fs] = dict(
                type="ta",
                n_fft=n_fft,
                hop=hop,
                f_max=f_max,
                layer=spec_layer,
            )
        else:
            # 纯 torch：自己造窗 + mel 滤波器（简化版）
            # 创建 Hann 窗
            win = torch.hann_window(n_fft, device=self.device, dtype=torch.float32)
            # 创建 mel 滤波器
            mel_fb = self._build_mel_filter(fs, n_fft, self.n_mels, f_max).to(self.device)

            self._cache[fs] = dict(
                type="plain",
                n_fft=n_fft,
                hop=hop,
                f_max=f_max,
                win=win,
                mel_fb=mel_fb
            )
        return self._cache[fs]

    @staticmethod
    def _hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _build_mel_filter(self, sr, n_fft, n_mels, f_max):
        # 参考 Slaney 实现的简化版
        f_min = 0.0
        m_min, m_max = self._hz_to_mel(torch.tensor(f_min)), self._hz_to_mel(torch.tensor(f_max))
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        f_pts = self._mel_to_hz(m_pts)

        # 频率格点
        fft_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            f_l, f_c, f_r = f_pts[i], f_pts[i + 1], f_pts[i + 2]
            # 三角滤波器
            left = (fft_freqs - f_l) / (f_c - f_l)
            right = (f_r - fft_freqs) / (f_r - f_c)
            fb[i] = torch.clamp(torch.min(left, right), min=0.0)
        # Slaney 归一化
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(1)
        return fb

    @torch.no_grad()
    def __call__(self, sig_np: np.ndarray, fs: int) -> np.ndarray:
        """
        输入：numpy 波形（float32），长度 ~ 40ms
        输出：log-mel (n_mels, img_side) 的 float16
        """
        if sig_np.dtype != np.float32:
            sig_np = sig_np.astype(np.float32, copy=False)

        cfg = self._get_mel_frontend(fs)
        x = torch.from_numpy(sig_np).to(self.device)

        # [T] -> [1,1,T]
        x = x.unsqueeze(0).unsqueeze(0)

        if cfg["type"] == "ta":
            mel = cfg["layer"](x)  # [1, n_mels, frames]
        else:
            # 纯 torch：STFT -> power spec -> mel滤波
            n_fft, hop, win, mel_fb = cfg["n_fft"], cfg["hop"], cfg["win"], cfg["mel_fb"].to(self.device)
            stft = torch.stft(
                x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True, return_complex=True
            )  # [1, freq, frames]
            power = (stft.real ** 2 + stft.imag ** 2)  # [1, freq, frames]
            # mel 滤波
            mel = torch.matmul(mel_fb, power.squeeze(0))  # [n_mels, frames]
            mel = mel.unsqueeze(0)

        # log 压缩 + 尺寸统一
        mel = torch.log(mel + 1e-10)  # [1, n_mels, frames]
        # 插值到统一宽度（img_side）
        mel = torch.nn.functional.interpolate(
            mel, size=(self.n_mels, self.img_side), mode="bilinear", align_corners=False
        )  # [1, n_mels, img_side]
        mel = mel.squeeze(0).detach()  # [n_mels, img_side]

        mel = mel.to(torch.float16) if DTYPE == "float16" else mel.to(torch.float32)
        return mel.cpu().numpy()


MEL_EXTRACTOR = MelExtractorTorch()  # 全局一个实例即可（缓存不同fs配置）


def save_npy(path, arr):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)


def make_out_paths(row, route):
    """
    构造当前切片的输出文件路径（只返回我们开启的资产）。
    命名建议：<file>__s<start>_e<end>_<ch>.npy
    """
    fstem = f"{os.path.splitext(row['file_name'])[0]}__s{row['slice_start']}_e{row['slice_end']}_{row['channel']}"
    out = {}

    if SAVE_ENV_NPY:
        out["env"] = os.path.join(ARTI_ROOT, route, "env2048_npy", fstem + ".npy")
    if SAVE_MEL_NPY:
        out["mel"] = os.path.join(ARTI_ROOT, route, "mel_npy", fstem + ".npy")
    # PNG 故意不做
    return out


def process_one_slice(row, route):
    """
    处理单个切片：读取→(A/B)→导出env/mel→返回行的路径元信息（写 CSV 用）
    """
    # 路线选择
    resample_to = None if route == "A" else BASELINE_FS

    try:
        sig, fs = load_slice_numpy(row, resample_to=resample_to)
    except Exception as e:
        return dict(error=str(e), file_name=row["file_name"], slice_start=row["slice_start"])

    paths = make_out_paths(row, route)
    ret_row = dict(
        file_name=row["file_name"],
        slice_start=row["slice_start"],
        slice_end=row["slice_end"],
        channel=row["channel"],
        split=row["split"],
        fs=fs,
        route=route,
    )

    # ENV（必备）
    if SAVE_ENV_NPY:
        p_env = paths["env"]
        if (not SKIP_IF_EXISTS) or (not os.path.exists(p_env)):
            env_spec = envelope_spectrum_2048(sig, fs)
            save_npy(p_env, env_spec)
        ret_row["env_path"] = p_env

    # MEL（GPU）
    if SAVE_MEL_NPY:
        p_mel = paths["mel"]
        if (not SKIP_IF_EXISTS) or (not os.path.exists(p_mel)):
            mel = MEL_EXTRACTOR(sig, fs)
            save_npy(p_mel, mel)
        ret_row["mel_path"] = p_mel

    return ret_row


def run_export_for_route(route, df_slices, out_csv):
    """
    针对某条路线（A 或 B）批处理导出，写出 specs CSV（只含路径+元信息）。
    - 仅处理 ONLY_SPLITS
    - SKIP_IF_EXISTS 避免重复算
    - MAX_PER_FILE 限制每文件导出量（先跑通）
    """
    df = df_slices[df_slices["split"].isin(ONLY_SPLITS)].copy()

    if MAX_PER_FILE is not None:
        # 每 file_name 限制条数
        df = (df.sort_values(["file_name", "slice_start"])
                .groupby("file_name", as_index=False)
                .head(MAX_PER_FILE))

    records = df.to_dict(orient="records")

    ensure_dir(os.path.dirname(out_csv) or ".")
    out_rows = []

    # 多进程（注意 Windows 下需放在 __main__ 保护里；此脚本就是主程序）
    with mp.Pool(processes=NUM_WORKERS) as pool:
        fn = partial(process_one_slice, route=route)
        t0 = time.time()
        for i, res in enumerate(pool.imap_unordered(fn, records, chunksize=8), 1):
            if i % LOG_EVERY == 0:
                dt = time.time() - t0
                print(f"[{route}] 进度 {i}/{len(records)} | {i/dt:.1f} slices/s")
            if isinstance(res, dict) and "error" not in res:
                out_rows.append(res)

    # 写 CSV
    pd.DataFrame(out_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK][{route}] 写出 {out_csv} | 共 {len(out_rows)} 条 | 产物根：{ARTI_ROOT}/{route}")


# =========================
# ======== 主 程 ==========
# =========================
if __name__ == "__main__":
    # 读切片索引
    df_slices = pd.read_csv(SLICES_CSV)

    # # 路线 A：原 fs
    # if "A" in ROUTES:
    #     run_export_for_route("A", df_slices, OUT_CSV_A)

    # 路线 B：低通→重采 12k
    if "B" in ROUTES:
        run_export_for_route("B", df_slices, OUT_CSV_B)

    print("[ALL DONE] 频谱资产导出完成。建议：先仅用 env2048 进入任务二传统 ML；需要 CNN 时再开启 mel_npy。")
