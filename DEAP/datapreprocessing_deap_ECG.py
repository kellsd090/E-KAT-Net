import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter, medfilt
import os


class DEAP_ECG_RegressionDataset(Dataset):
    def __init__(self, data_dir, subject_list, window_size=200, stride=100, is_train=True, split_ratio=0.8, seed=42):
        """
        data_dir: 包含 s01.dat - s32.dat 的目录
        subject_list: 受试者编号列表, 如 range(1, 33)
        window_size: 滑窗长度, 建议 200 或 187
        stride: 滑窗步长
        is_train: True 提取每个受试者前 80% 的试次，False 提取后 20%
        split_ratio: 训练集占比 (0.8 对应 32 个训练试次, 8 个验证试次)
        seed: 随机种子，确保 train/val 划分一致
        """
        self.window_size = window_size
        self.stride = stride
        self.X, self.Y = [], []

        # 1. 确定试次索引划分 (40个试次打乱后分配)
        np.random.seed(seed)
        all_indices = np.arange(40)
        np.random.shuffle(all_indices)

        split_point = int(40 * split_ratio)
        target_indices = all_indices[:split_point] if is_train else all_indices[split_point:]

        mode_str = "训练集" if is_train else "验证集"
        print(f">>> 正在启动 DEAP {mode_str} 预处理 (共 {len(subject_list)} 位受试者)...")

        for sub_id in subject_list:
            file_path = os.path.join(data_dir, f's{sub_id:02d}.dat')
            if not os.path.exists(file_path): continue

            with open(file_path, 'rb') as f:
                payload = pickle.load(f, encoding='latin1')
                # 提取第 39 通道：心电 (Index 38)
                raw_ecg_all = payload['data'][:, 38, :]  # (40 trials, 8064 points)
                # 提取 Valence 标签 (Index 0)
                raw_labels = (payload['labels'][:, :4] - 4.5) / 4.5

            # --- 步骤 1-4: 受试者内处理 (对该人所有40个试次统一执行以保持统计稳定性) ---
            processed_trials = []
            for trial_idx in range(40):
                signal = raw_ecg_all[trial_idx]

                # 1. 基线扣除 (利用每个 trial 前 3s, 即 384 点)
                baseline_mu = np.mean(signal[:384])
                signal = signal - baseline_mu

                # 2. 带通滤波 (0.5-45Hz)
                signal = self._butter_bandpass_filter(signal, 0.5, 45, fs=128)

                # 3. 5-Sigma 动态截断：保护 KAN 样条网格不被异常尖峰摧毁
                signal = self._adaptive_5sigma_clipping(signal)
                processed_trials.append(signal)

            # 4. 受试者内 Z-Score 标准化 (消除个体间幅值差异，仅保留情感波动特征)
            sub_data = np.array(processed_trials)
            sub_mu = np.mean(sub_data)
            sub_std = np.std(sub_data) + 1e-8
            sub_data = (sub_data - sub_mu) / sub_std

            # --- 步骤 5-6: 根据 target_indices 进行滑窗与标签关联 ---
            for i in target_indices:
                trial_signal = sub_data[i][384:]  # 舍弃 3s 基线期，仅处理 60s 刺激期信号
                label = raw_labels[i]  # 保持 1-9 原始浮点数

                for start in range(0, len(trial_signal) - window_size, stride):
                    end = start + window_size
                    self.X.append(trial_signal[start:end])
                    self.Y.append(label)

        # 转换为 Tensor，增加通道维度 [N, 1, Window]
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

        print(f">>> {mode_str} 预处理完成。样本量: {len(self.X)} | 标签范围: {self.Y.min():.2f}-{self.Y.max():.2f}")

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def _adaptive_5sigma_clipping(self, signal, multiplier=5.0):
        """借鉴自 dataset_gal.py，防止电涌影响 KAN 模型稳定性"""
        mu = np.mean(signal)
        sigma = np.std(signal) + 1e-8
        return np.clip(signal, mu - multiplier * sigma, mu + multiplier * sigma)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DEAP_GSR_RegressionDataset(Dataset):
    def __init__(self, data_dir, subject_list, window_size=200, stride=96, is_train=True, split_ratio=0.8, seed=42):
        """
        针对 GSR 信号优化的 DEAP 预处理
        """
        self.window_size = window_size
        self.stride = stride
        self.X, self.Y = [], []

        np.random.seed(seed)
        all_indices = np.arange(40)
        np.random.shuffle(all_indices)

        split_point = int(40 * split_ratio)
        target_indices = all_indices[:split_point] if is_train else all_indices[split_point:]

        mode_str = "训练集" if is_train else "验证集"
        print(f">>> 正在启动 DEAP GSR {mode_str} 预处理 (共 {len(subject_list)} 位受试者)...")

        for sub_id in subject_list:
            file_path = os.path.join(data_dir, f's{sub_id:02d}.dat')
            if not os.path.exists(file_path): continue

            with open(file_path, 'rb') as f:
                payload = pickle.load(f, encoding='latin1')
                # 提取第 37 通道：GSR (Index 36)
                raw_gsr_all = payload['data'][:, 36, :]  # (40 trials, 8064 points)
                # 标签映射到 [-1, 1]
                raw_labels = (payload['labels'][:, :4] - 4.5) / 4.5

            processed_trials = []
            for trial_idx in range(40):
                signal = raw_gsr_all[trial_idx]

                # --- GSR 专属处理流 ---
                # 1. 对数转换：压缩量程，使皮肤导电率更符合正态分布
                signal = np.log1p(np.abs(signal))

                # 2. 极低通滤波 (1.0Hz)：GSR 的有效信息都在 1Hz 以下
                signal = self._butter_lowpass_filter(signal, 1.0, fs=128)

                # 3. 5-Sigma 动态截断：剔除由于电极松动产生的突变尖峰
                signal = self._adaptive_5sigma_clipping(signal)

                # 4. 提取 Phasic (变化) 分量：模拟一阶差分，但更平滑
                # 我们取 (信号 - 中值滤波后的背景) 来强化情绪脉冲
                # 这能帮助 KAN 捕捉到所谓“皮肤电反应 (SCR)”
                tonic = medfilt(signal, kernel_size=129)  # 捕捉背景缓慢变化
                phasic = signal - tonic

                # 最终输入 = 原始平滑信号 + 强化后的脉冲分量
                final_feat = signal + 5.0 * phasic
                processed_trials.append(final_feat)

            # 5. 受试者内 Z-Score 标准化
            sub_data = np.array(processed_trials)
            sub_mu = np.mean(sub_data)
            sub_std = np.std(sub_data) + 1e-8
            sub_data = (sub_data - sub_mu) / sub_std

            for i in target_indices:
                trial_signal = sub_data[i][384:]  # 舍弃前 3s
                label = raw_labels[i]

                for start in range(0, len(trial_signal) - window_size, stride):
                    end = start + window_size
                    self.X.append(trial_signal[start:end])
                    self.Y.append(label)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

        print(f">>> {mode_str} 预处理完成。样本量: {len(self.X)} | 标签范围: {self.Y.min():.2f}-{self.Y.max():.2f}")

    def _butter_lowpass_filter(self, data, cut_off, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cut_off / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)

    def _adaptive_5sigma_clipping(self, signal, multiplier=5.0):
        mu = np.mean(signal)
        sigma = np.std(signal) + 1e-8
        # GSR 异常值多表现为极大的向上跳变，因此只做 upper clipping 也可以，这里做双向
        clipped = np.clip(signal, mu - multiplier * sigma, mu + multiplier * sigma)
        # 3点移动平均，确保进入 KAN 的信号没有“阶跃”
        smoothed = np.convolve(clipped, np.ones(3) / 3, mode='same')
        return smoothed

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]