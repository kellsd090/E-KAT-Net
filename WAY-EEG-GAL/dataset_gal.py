import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
import os
import matplotlib.pyplot as plt


# --- 1. 核心预处理函数 ---

def butter_bandpass(data, lowcut, highcut, fs=500, order=4):
    """通用带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)


def smooth_labels(labels, sigma=2.5):
    """
    标签平滑：使用高斯核对 0/1 脉冲进行时域平滑
    sigma: 高斯核标准差，对应平滑的宽度
    """
    kernel = gaussian(15, sigma)  # 构建高斯核
    kernel /= kernel.sum()
    # 沿着时间轴对 6 个类别独立平滑
    return convolve1d(labels, kernel, axis=0)


class WAYEEGDataset(Dataset):
    def __init__(self, data_path, subject_ids, series_ids, window_size=250, stride=50, is_train=True, stats=None):
        self.window_size = window_size
        self.stride = stride
        self.stats = None

        all_data = []
        all_labels = []

        print(f"正在执行预处理 (Subjects: {subject_ids}, Series: {series_ids})...")

        for sub in subject_ids:
            for ser in series_ids:
                data_file = os.path.join(data_path, f'subj{sub}_series{ser}_data.csv')
                label_file = os.path.join(data_path, f'subj{sub}_series{ser}_events.csv')

                if not os.path.exists(data_file):
                    continue

                # 1. 加载原始数据
                raw_df = pd.read_csv(data_file).iloc[:, 1:].values.astype(np.float32)
                raw_labels = pd.read_csv(label_file).iloc[:, 1:].values.astype(np.float32)

                # 2. 进阶手段一：共同平均参考 (CAR)
                # 必须先做这一步！消除你 Excel 中看到的 500+ 的巨大直流偏移量
                raw_df = raw_df - np.mean(raw_df, axis=1, keepdims=True)

                # 3. 进阶手段二：基于标准差的自适应截断 (5-Sigma Clipping)
                # 不再使用硬性的 150uV，而是将超过 5 倍标准差的离群点强制降至 5 倍
                for ch in range(raw_df.shape[1]):
                    ch_mean = np.mean(raw_df[:, ch])
                    ch_std = np.std(raw_df[:, ch]) + 1e-8
                    # 计算当前通道的上下界
                    limit = 5.0 * ch_std
                    raw_df[:, ch] = np.clip(raw_df[:, ch], ch_mean - limit, ch_mean + limit)

                # 4. 进阶手段三：标签平滑 (Label Smoothing)
                raw_labels = smooth_labels(raw_labels)

                # 5. 滤波器组构造 (Filter Bank)
                # 在干净的数据上进行频段切分
                band_raw = butter_bandpass(raw_df, 0.5, 50)
                band_delta = butter_bandpass(raw_df, 0.5, 4)
                band_mu = butter_bandpass(raw_df, 8, 13)
                band_beta = butter_bandpass(raw_df, 13, 30)
                band_gamma = butter_bandpass(raw_df, 30, 50)

                # 拼接 160 通道 (32电极 * 5频段)
                multi_band_data = np.concatenate([
                    band_raw, band_delta, band_mu, band_beta, band_gamma
                ], axis=1)

                all_data.append(multi_band_data)
                all_labels.append(raw_labels)

        # 合并数据
        self.raw_data = np.concatenate(all_data, axis=0).astype(np.float32)
        self.raw_labels = np.concatenate(all_labels, axis=0).astype(np.float32)

        # 6. Z-Score 标准化：将全量程缩放到 0 均值，1 标准差
        # --- dataset_gal.py 内部 ---
        if is_train:
            self.mean = np.mean(self.raw_data, axis=0).astype(np.float32)
            self.std = (np.std(self.raw_data, axis=0) + 1e-8).astype(np.float32)
            self.stats = {'mean': self.mean, 'std': self.std}
        else:
            if stats is None:
                raise ValueError("验证集模式下必须传入 stats 参数")

            # 核心修正：显式提取并强制转码，确保不带入任何字符串
            self.mean = np.array(stats['mean'], dtype=np.float32)
            self.std = np.array(stats['std'], dtype=np.float32)

        # 此时执行减法，两边都是数值类型
        self.raw_data = (self.raw_data - self.mean) / self.std

        # 计算索引
        self.indices = []
        for i in range(0, len(self.raw_data) - window_size + 1, stride):
            self.indices.append(i)

        print(f"预处理完成。有效窗口: {len(self.indices)} | 最终通道: {self.raw_data.shape[1]}")

    def __len__(self):
        return len(self.indices)

    # --- dataset_gal.py ---
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        # 强制转换为 float32 类型的 numpy 数组，再转 Tensor
        x = self.raw_data[start:end].T.astype(np.float32)
        y = self.raw_labels[end - 1].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# --- 2. 权重计算工具 ---
def calculate_pos_weights(data_path, subject_ids, series_ids):
    """
    计算类别平衡权重，并引入平滑逻辑以适配 KAN 模型。
    防止在极度不平衡（如 1:100）的情况下产生过大的梯度。
    """
    print(">>> 正在计算类别平衡权重 (已引入平滑与截断逻辑)...")
    all_pos = []
    total = 0

    for sub in subject_ids:
        for ser in series_ids:
            label_file = os.path.join(data_path, f'subj{sub}_series{ser}_events.csv')
            if os.path.exists(label_file):
                # 显式读取数值，跳过 ID 列
                labels = pd.read_csv(label_file).iloc[:, 1:].values.astype(np.float32)
                all_pos.append(np.sum(labels, axis=0))
                total += len(labels)

    if len(all_pos) == 0:
        print("警告: 未找到任何标签文件，权重将初始化为 1.0")
        return torch.ones(6)

    pos_counts = np.sum(all_pos, axis=0)
    neg_counts = total - pos_counts

    # 1. 计算原始比例 (neg / pos)
    # 这种原始比例在 GAL 数据集下可能高达 50-100，对 KAN 非常危险
    raw_weights = neg_counts / (pos_counts + 1e-8)

    # 2. 开方平滑 (Square Root Smoothing)
    # 将 100 倍的差距压缩到 10 倍左右。这在 EEG 领域是平衡“检测灵敏度”与“虚警率”的经典做法。
    smooth_weights = np.sqrt(raw_weights)

    # 3. 硬截断 (Clipping)
    # 确保任何类别的权重不会超过 20，防止单个 batch 内的梯度爆炸
    smooth_weights = np.clip(smooth_weights, a_min=1.0, a_max=20.0)

    # 4. 打印各通道的具体权重，方便调试
    labels_name = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']
    weight_info = {name: round(w, 2) for name, w in zip(labels_name, smooth_weights.tolist())}
    print(f">>> 计算完成。平滑后的 pos_weights 为:\n{weight_info}")

    return torch.tensor(smooth_weights, dtype=torch.float)


def visualize_preprocessing_effect(raw_df, processed_data, channel_names=None, num_channels=3):
    """
    对比原始数据和预处理后的数据
    raw_df: 原始从CSV读取的 numpy 阵列 [Time, 32]
    processed_data: 最终 Dataset.raw_data [Time, 160] (包含5个频段)
    """
    # 随机选择几个原始通道进行对比 (0-31 之间)
    channels_to_plot = np.random.choice(range(32), num_channels, replace=False)

    fig, axes = plt.subplots(num_channels, 2, figsize=(15, 4 * num_channels))
    plt.subplots_adjust(hspace=0.4)

    time_limit = 1000  # 只画前1000个点(约2秒)，方便看波形细节

    for i, ch in enumerate(channels_to_plot):
        # --- 左侧：原始数据 (包含直流偏移和离群点) ---
        axes[i, 0].plot(raw_df[:time_limit, ch], color='red', alpha=0.7)
        axes[i, 0].set_title(f"Original Channel {ch}\n(Notice DC Offset & Spikes)")
        axes[i, 0].grid(True)

        # --- 右侧：预处理后的数据 (CAR + Clipping + Filter + Z-Score) ---
        # 对应 processed_data 中的第一个频段组 (即 0.5-50Hz 带通组)
        axes[i, 1].plot(processed_data[:time_limit, ch], color='blue', alpha=0.7)
        axes[i, 1].set_title(f"Processed Channel {ch}\n(CAR + Clipping + Filtered)")
        axes[i, 1].grid(True)

    plt.suptitle("Preprocessing Comparison: Raw vs Final Features", fontsize=16)
    plt.show()


# ... (保留前面的 import 和预处理函数) ...

def visualize_waveforms(raw_segment, processed_segment, sample_rate=500):
    """
    在新窗口中展示审计样本的波形对比
    raw_segment: [Time, 32] 原始 CSV 数据片段
    processed_segment: [160, Time] 预处理后的特征矩阵
    """
    t = np.linspace(0, len(raw_segment) / sample_rate, len(raw_segment))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4)

    # --- 图 1: 原始波形 (第一个电极) ---
    # 索引 0 对应第一个通道
    axes[0].plot(t, raw_segment[:, 0], color='black', lw=1.5)
    axes[0].set_title("Input Sample: Original Raw Waveform (First Electrode)", fontsize=14)
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(True, alpha=0.3)

    # --- 图 2: 预处理后的波形 (第一个滤波通道 0.5-50Hz) ---
    # processed_segment 形状由 [160, 250] 组成
    # 前 32 个通道对应 band_raw (0.5-50Hz)
    axes[1].plot(t, processed_segment[0, :], color='black', lw=1.5)
    axes[1].set_title("Input Sample: Processed Waveform (First Filter Bank: 0.5-50Hz)", fontsize=14)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized Strength")
    axes[1].grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    # 1. 路径配置 - 请确保此路径下直接能看到 subj9_series9_data.csv
    PATH_train = r"E:\Python\PythonProject\dataset\grasp-and-lift-eeg-detection\train\train"
    target_sub = np.random.randint(1, 13)
    target_ser = np.random.randint(1, 9)

    # --- 强力审计：手动检查文件是否存在 ---
    data_file = os.path.join(PATH_train, f'subj{target_sub}_series{target_ser}_data.csv')
    print(f">>> 正在检查路径: {data_file}")

    if not os.path.exists(data_file):
        print(f"!!! 严重错误: 无法在路径下找到数据文件。请检查 PATH_train 配置。")
        # 列出该目录下的前5个文件辅助调试
        if os.path.exists(PATH_train):
            print(f"该目录下的现有文件示例: {os.listdir(PATH_train)[:5]}")
        else:
            print(f"目录不存在: {PATH_train}")
    else:
        # --- 2. 只有文件存在才实例化数据集 ---
        train_ds = WAYEEGDataset(
            PATH_train,
            subject_ids=[target_sub],
            series_ids=[target_ser],
            window_size=250,
            stride=250,
            is_train=True
        )

        # --- 3. 绘制波形图窗口 ---
        print("\n>>> 正在生成审计样本波形对比图...")
        # 读取原始数据的前 250 点 (对应第一个电极)
        raw_df_example = pd.read_csv(data_file).iloc[:250, 1:].values.astype(np.float32)
        processed_input, _ = train_ds[0]

        visualize_waveforms(
            raw_df_example,
            processed_input.numpy()
        )