import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d
import os
import matplotlib.pyplot as plt


def butter_bandpass(data, lowcut, highcut, fs=500, order=4):   # band-pass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)


def smooth_labels(labels, sigma=2.5):
    kernel = gaussian(15, sigma)  
    kernel /= kernel.sum()
    return convolve1d(labels, kernel, axis=0)


class WAYEEGDataset(Dataset):
    def __init__(self, data_path, subject_ids, series_ids, window_size=250, stride=50, is_train=True, stats=None):
        self.window_size = window_size
        self.stride = stride
        self.stats = None
        all_data = []
        all_labels = []
        
        for sub in subject_ids:
            for ser in series_ids:
                data_file = os.path.join(data_path, f'subj{sub}_series{ser}_data.csv')
                label_file = os.path.join(data_path, f'subj{sub}_series{ser}_events.csv')
                if not os.path.exists(data_file):
                    continue
                raw_df = pd.read_csv(data_file).iloc[:, 1:].values.astype(np.float32)
                raw_labels = pd.read_csv(label_file).iloc[:, 1:].values.astype(np.float32)
                raw_df = raw_df - np.mean(raw_df, axis=1, keepdims=True)
                for ch in range(raw_df.shape[1]):
                    ch_mean = np.mean(raw_df[:, ch])
                    ch_std = np.std(raw_df[:, ch]) + 1e-8
                    limit = 5.0 * ch_std
                    raw_df[:, ch] = np.clip(raw_df[:, ch], ch_mean - limit, ch_mean + limit)
                raw_labels = smooth_labels(raw_labels)

                band_raw = butter_bandpass(raw_df, 0.5, 50)
                band_delta = butter_bandpass(raw_df, 0.5, 4)
                band_mu = butter_bandpass(raw_df, 8, 13)
                band_beta = butter_bandpass(raw_df, 13, 30)
                band_gamma = butter_bandpass(raw_df, 30, 50)

                multi_band_data = np.concatenate([
                    band_raw, band_delta, band_mu, band_beta, band_gamma], axis=1)
                all_data.append(multi_band_data)
                all_labels.append(raw_labels)
        self.raw_data = np.concatenate(all_data, axis=0).astype(np.float32)
        self.raw_labels = np.concatenate(all_labels, axis=0).astype(np.float32)

        if is_train:
            self.mean = np.mean(self.raw_data, axis=0).astype(np.float32)
            self.std = (np.std(self.raw_data, axis=0) + 1e-8).astype(np.float32)
            self.stats = {'mean': self.mean, 'std': self.std}
        else:
            if stats is None:
                raise ValueError("验证集模式下必须传入 stats 参数")
            self.mean = np.array(stats['mean'], dtype=np.float32)
            self.std = np.array(stats['std'], dtype=np.float32)
        self.raw_data = (self.raw_data - self.mean) / self.std

        self.indices = []
        for i in range(0, len(self.raw_data) - window_size + 1, stride):
            self.indices.append(i)
        print(f"Preprocessing is complete. Effective window: {len(self.indices)} | final channels: {self.raw_data.shape[1]}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        x = self.raw_data[start:end].T.astype(np.float32)
        y = self.raw_labels[end - 1].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def calculate_pos_weights(data_path, subject_ids, series_ids):
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
        return torch.ones(6)
    pos_counts = np.sum(all_pos, axis=0)
    neg_counts = total - pos_counts
    raw_weights = neg_counts / (pos_counts + 1e-8)
    smooth_weights = np.sqrt(raw_weights)
    smooth_weights = np.clip(smooth_weights, a_min=1.0, a_max=20.0)
    labels_name = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']
    weight_info = {name: round(w, 2) for name, w in zip(labels_name, smooth_weights.tolist())}
    return torch.tensor(smooth_weights, dtype=torch.float)


def visualize_preprocessing_effect(raw_df, processed_data, channel_names=None, num_channels=3):
    channels_to_plot = np.random.choice(range(32), num_channels, replace=False)
    fig, axes = plt.subplots(num_channels, 2, figsize=(15, 4 * num_channels))
    plt.subplots_adjust(hspace=0.4)
    time_limit = 1000  
    for i, ch in enumerate(channels_to_plot):
        axes[i, 0].plot(raw_df[:time_limit, ch], color='red', alpha=0.7)
        axes[i, 0].set_title(f"Original Channel {ch}\n(Notice DC Offset & Spikes)")
        axes[i, 0].grid(True)
        axes[i, 1].plot(processed_data[:time_limit, ch], color='blue', alpha=0.7)
        axes[i, 1].set_title(f"Processed Channel {ch}\n(CAR + Clipping + Filtered)")
        axes[i, 1].grid(True)
    plt.suptitle("Preprocessing Comparison: Raw vs Final Features", fontsize=16)
    plt.show()


def visualize_waveforms(raw_segment, processed_segment, sample_rate=500):
    t = np.linspace(0, len(raw_segment) / sample_rate, len(raw_segment))
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4)
    axes[0].plot(t, raw_segment[:, 0], color='black', lw=1.5)
    axes[0].set_title("Input Sample: Original Raw Waveform (First Electrode)", fontsize=14)
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, processed_segment[0, :], color='black', lw=1.5)
    axes[1].set_title("Input Sample: Processed Waveform (First Filter Bank: 0.5-50Hz)", fontsize=14)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized Strength")
    axes[1].grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    PATH_train = r"E:\Python\PythonProject\dataset\grasp-and-lift-eeg-detection\train\train"
    target_sub = np.random.randint(1, 13)
    target_ser = np.random.randint(1, 9)
    data_file = os.path.join(PATH_train, f'subj{target_sub}_series{target_ser}_data.csv')
    if not os.path.exists(data_file):
        print(f"!!! Error: Unable to locate the data file in the specified path. Please verify the PATH_train configuration.")
        if os.path.exists(PATH_train):
            print(f"The existing file examples under this directory: {os.listdir(PATH_train)[:5]}")
        else:
            print(f"The directory does not exist: {PATH_train}")
    else:
        train_ds = WAYEEGDataset(
            PATH_train,
            subject_ids=[target_sub],
            series_ids=[target_ser],
            window_size=250,
            stride=250,
            is_train=True)

        raw_df_example = pd.read_csv(data_file).iloc[:250, 1:].values.astype(np.float32)
        processed_input, _ = train_ds[0]
        visualize_waveforms(
            raw_df_example,
            processed_input.numpy())
        
