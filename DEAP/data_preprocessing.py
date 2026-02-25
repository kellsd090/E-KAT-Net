import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter, medfilt
import os


class DEAP_ECG_RegressionDataset(Dataset):
    def __init__(self, data_dir, subject_list, window_size=200, stride=100, is_train=True, split_ratio=0.8, seed=42):
        self.window_size = window_size
        self.stride = stride
        self.X, self.Y = [], []

        np.random.seed(seed)
        all_indices = np.arange(40)
        np.random.shuffle(all_indices)
        split_point = int(40 * split_ratio)
        target_indices = all_indices[:split_point] if is_train else all_indices[split_point:]

        mode_str = "train" if is_train else "test"

        for sub_id in subject_list:
            file_path = os.path.join(data_dir, f's{sub_id:02d}.dat')
            if not os.path.exists(file_path): continue

            with open(file_path, 'rb') as f:
                payload = pickle.load(f, encoding='latin1')
                raw_ecg_all = payload['data'][:, 38, :]  # (40 trials, 8064 points)
                raw_labels = (payload['labels'][:, :4] - 4.5) / 4.5

            processed_trials = []
            for trial_idx in range(40):
                signal = raw_ecg_all[trial_idx]

                # Baseline deduction
                baseline_mu = np.mean(signal[:384])
                signal = signal - baseline_mu

                # band-pass filter (0.5-45Hz)
                signal = self._butter_bandpass_filter(signal, 0.5, 45, fs=128)

                # 5-Sigma Dynamic Truncation: Protecting the KAN spline mesh from being destroyed by abnormal peaks
                signal = self._adaptive_5sigma_clipping(signal)
                processed_trials.append(signal)

            # Z-Score standardization
            sub_data = np.array(processed_trials)
            sub_mu = np.mean(sub_data)
            sub_std = np.std(sub_data) + 1e-8
            sub_data = (sub_data - sub_mu) / sub_std

            # Window sliding and label association
            for i in target_indices:
                trial_signal = sub_data[i][384:]  
                label = raw_labels[i]  

                for start in range(0, len(trial_signal) - window_size, stride):
                    end = start + window_size
                    self.X.append(trial_signal[start:end])
                    self.Y.append(label)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        print(f">>> {mode_str} finish processing。num of sample: {len(self.X)} | Label range: {self.Y.min():.2f}-{self.Y.max():.2f}")

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def _adaptive_5sigma_clipping(self, signal, multiplier=5.0):
        mu = np.mean(signal)
        sigma = np.std(signal) + 1e-8
        return np.clip(signal, mu - multiplier * sigma, mu + multiplier * sigma)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DEAP_GSR_RegressionDataset(Dataset):
    def __init__(self, data_dir, subject_list, window_size=200, stride=96, is_train=True, split_ratio=0.8, seed=42):
        self.window_size = window_size
        self.stride = stride
        self.X, self.Y = [], []

        np.random.seed(seed)
        all_indices = np.arange(40)
        np.random.shuffle(all_indices)

        split_point = int(40 * split_ratio)
        target_indices = all_indices[:split_point] if is_train else all_indices[split_point:]

        mode_str = "train" if is_train else "test"

        for sub_id in subject_list:
            file_path = os.path.join(data_dir, f's{sub_id:02d}.dat')
            if not os.path.exists(file_path): continue

            with open(file_path, 'rb') as f:
                payload = pickle.load(f, encoding='latin1')
                raw_gsr_all = payload['data'][:, 36, :]  # (40 trials, 8064 points)
                raw_labels = (payload['labels'][:, :4] - 4.5) / 4.5

            processed_trials = []
            for trial_idx in range(40):
                signal = raw_gsr_all[trial_idx]

                # Logarithmic transformation: Reduces the measurement range, 
                making the skin conductivity more in line with a normal distribution.
                signal = np.log1p(np.abs(signal))

                # Very low-pass filtering (1.0Hz): The valuable information of GSR is all below 1Hz.
                signal = self._butter_lowpass_filter(signal, 1.0, fs=128)

                # 5-Sigma dynamic truncation: Elimination of abnormal spikes caused by loose electrodes
                signal = self._adaptive_5sigma_clipping(signal)

                tonic = medfilt(signal, kernel_size=129)  # 捕捉背景缓慢变化
                phasic = signal - tonic
                final_feat = signal + 5.0 * phasic
                processed_trials.append(final_feat)

            #  Z-Score 
            sub_data = np.array(processed_trials)
            sub_mu = np.mean(sub_data)
            sub_std = np.std(sub_data) + 1e-8
            sub_data = (sub_data - sub_mu) / sub_std

            for i in target_indices:
                trial_signal = sub_data[i][384:]  
                label = raw_labels[i]
                for start in range(0, len(trial_signal) - window_size, stride):
                    end = start + window_size
                    self.X.append(trial_signal[start:end])
                    self.Y.append(label)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        print(f">>> {mode_str} finish processing。sample num: {len(self.X)} | label range: {self.Y.min():.2f}-{self.Y.max():.2f}")

    def _butter_lowpass_filter(self, data, cut_off, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cut_off / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)

    def _adaptive_5sigma_clipping(self, signal, multiplier=5.0):
        mu = np.mean(signal)
        sigma = np.std(signal) + 1e-8
        clipped = np.clip(signal, mu - multiplier * sigma, mu + multiplier * sigma)
        smoothed = np.convolve(clipped, np.ones(3) / 3, mode='same')
        return smoothed

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]
