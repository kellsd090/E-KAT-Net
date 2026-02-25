import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import toeplitz
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.collections import LineCollection
import random
import os
from dataset_gal import WAYEEGDataset, calculate_pos_weights, visualize_waveforms


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SplineWeightLayer(nn.Module):
    def __init__(self, in_features=250, in_hidden=128, out_hidden=128, kernel_size=15):
        super(SplineWeightLayer, self).__init__()
        self.in_features = in_features  # 序列长度 750
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.k = kernel_size

        # 1. 特征投影与独立位置偏置
        self.W2 = nn.Parameter(torch.Tensor(in_hidden, out_hidden))
        # 保持每个采样点独立的偏置参数
        self.b = nn.Parameter(torch.zeros(in_features, out_hidden))
        self.ln = nn.LayerNorm(in_hidden)

        # 2. 权重控制系数
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # I 的权重
        self.beta = nn.Parameter(torch.ones(1))  # W1 的权重
        self.theta = nn.Parameter(torch.ones(1))  # W3 的权重
        self.gamma = nn.Parameter(torch.tensor([0.5]))  # 残差权重

        # --- W3 1D 时序卷积先验 ---
        self.W3_conv = nn.Conv1d(
            out_hidden, out_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_hidden, bias=False
        )
        self.register_buffer('identity', torch.eye(in_features).unsqueeze(0))

        if in_hidden != out_hidden:
            self.residual_proj = nn.Linear(in_hidden, out_hidden)
        else:
            self.residual_proj = nn.Identity()

        nn.init.xavier_uniform_(self.W2)

    def forward(self, x, W1=None):
        b, seq_len, h_dim = x.shape
        x_norm = self.ln(x)
        x_prime = torch.matmul(x_norm, self.W2) + self.b

        # W3 路径
        x_temporal = x_prime.transpose(1, 2)
        w3_feat = self.W3_conv(x_temporal).transpose(1, 2)

        # W1 路径 (基于 Requirement 2 的自注意力输入)
        if W1 is not None:
            # 使用预存的 identity
            attn_kernel = torch.abs(self.beta) * W1 + torch.abs(self.alpha) * self.identity
            global_part = torch.bmm(attn_kernel, x_prime)
            spatial_combined = global_part + torch.abs(self.theta) * w3_feat
        else:
            spatial_combined = x_prime + torch.abs(self.theta) * w3_feat

        res_x = self.residual_proj(x)
        return spatial_combined + self.gamma * res_x, x_prime


class FastKANLayer(nn.Module):
    def __init__(self, in_neurons, out_neurons, grid_size=128, spline_order=3, hidden_dim=128, neuron_out_dim=128,
                 kernel_size=15, seq_len=250):
        super(FastKANLayer, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.hidden_dim = hidden_dim
        self.num_coeffs = grid_size + spline_order

        # 纯粹的 B-样条权重，去掉了 base_scales
        self.spline_weights = nn.Parameter(torch.randn(in_neurons, out_neurons, self.num_coeffs) * 0.1)

        # 每一个输出神经元对应一个 1D 优化的 SplineWeightLayer
        self.weight_layers = nn.ModuleList([
            SplineWeightLayer(in_features=seq_len, in_hidden=hidden_dim, out_hidden=neuron_out_dim, kernel_size=kernel_size)
            for _ in range(out_neurons)
        ])

        self.register_buffer("grid", torch.linspace(-1, 1, self.num_coeffs))
        self.tau = nn.Parameter(torch.randn(in_neurons, out_neurons) * 0.1 + 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.omiga = nn.Parameter(torch.full((in_neurons, out_neurons), 0.2))
        self.last_W1 = None

        # 将原本的随机初始化改为线性初始化
        # 目标：让输出 y = Spline(x) 在初始时接近 y = x
        with torch.no_grad():
            # 生成从 -1.0 到 1.0 的等差序列，代表 y=x 的趋势
            lin_init = torch.linspace(-1.0, 1.0, self.num_coeffs).to(self.spline_weights.device)
            # 形状调整为 [in_n, out_n, num_coeffs]
            initial_weights = lin_init.view(1, 1, -1).repeat(self.in_neurons, self.out_neurons, 1)
            # 稍微加一点极小的噪声打破对称性
            self.spline_weights.data = initial_weights + torch.randn_like(initial_weights) * 0.01

    def b_spline_cubic(self, x):
        """
        优化版三次 B-样条：通过数学公式合并分支，消除 mask 赋值
        """
        h = 2.0 / (self.num_coeffs - 1)
        # [B, in_n, S, D, 1] - [num_coeffs] -> [B, in_n, S, D, num_coeffs]
        dist = torch.abs(x.unsqueeze(-1) - self.grid) / h

        # 使用 ReLU 截断代替 mask，确保 dist < 2.0 之外的部分为 0
        # 通过组合多项式模拟原有的分段逻辑
        # 核心区 (dist < 1) 和 边缘区 (1 <= dist < 2)

        # 逻辑 1: 计算 (2 - dist)^3 / 6 (覆盖 0 <= dist < 2)
        res_outer = torch.relu(2.0 - dist) ** 3 / 6.0

        # 逻辑 2: 减去核心区多算的补偿值 (只针对 dist < 1)
        # 原逻辑核心区为: 2/3 - dist^2 + dist^3/2
        # 经过代数化简，补偿项为: 4/6 * (1 - dist)^3
        res_inner = torch.relu(1.0 - dist) ** 3 * (4.0 / 6.0)

        # 最终结果 = 外部大包络 - 内部补偿补偿
        res = res_outer - res_inner
        return res

    def forward(self, x_in, proj_x_prev=None):
        b, in_n, seq_len, feat_dim = x_in.shape # x_in: [B, in_n, 200, 128]

        # 1. 样条输入归一化
        max_val = torch.max(torch.abs(x_in)) + 1e-8
        x_norm = (x_in / max_val) * 0.95
        x_norm = torch.clamp(x_norm, -0.99, 0.99)

        # 2. Requirement 1: 纯 B-样条点乘逻辑
        basis = self.b_spline_cubic(x_norm)
        spline_mapping = torch.einsum('binfc,ioc->bio nf', basis, self.spline_weights)
        # 2. 加上线性残差项 omega * x
        # omega 现在扮演的是“线性增益”或“残差系数”的角色
        omega_expanded = torch.abs(self.omiga).view(1, in_n, self.out_neurons, 1, 1)
        activated_signals = spline_mapping + omega_expanded * x_in.unsqueeze(2)

        next_outputs, next_projections = [], []
        # 针对 750 长度调整温度系数，防止梯度饱和
        t_val = torch.abs(self.temperature) * np.sqrt(seq_len / 256.0) + 1e-4

        for j in range(self.out_neurons):
            current_edges = activated_signals[:, :, j, :, :]
            edge_energies = torch.mean(current_edges ** 2, dim=(-1, -2))
            tau_j = torch.abs(self.tau[:, j]).unsqueeze(0)
            mask = torch.sigmoid((torch.sqrt(edge_energies + 1e-8) - tau_j) / t_val).unsqueeze(-1).unsqueeze(-1)

            # 3. Requirement 2: 保持自注意力逻辑
            W1_j = None
            if proj_x_prev is not None:
                multiplier = (torch.sqrt(edge_energies + 1e-8) / (tau_j + 1e-8)).unsqueeze(-1).unsqueeze(-1)
                weighted_prev = proj_x_prev * multiplier * mask

                # 拼接所有输入神经簇的特征进行自注意力
                mid = self.in_neurons // 2
                K = torch.cat([weighted_prev[:, 2*i, :, :] for i in range(mid)], dim=-1)
                Q = torch.cat([weighted_prev[:, 2*i+1, :, :] for i in range(mid)], dim=-1)
                K, Q = F.layer_norm(K, [K.size(-1)]), F.layer_norm(Q, [Q.size(-1)])
                raw_attn = torch.bmm(K, Q.transpose(-1, -2))
                W1_j = F.softmax(raw_attn / (np.sqrt(K.shape[-1]) * t_val), dim=-1)

            if hasattr(self, 'visualize_idx') and j == self.visualize_idx:
                self.last_W1 = W1_j.detach() if W1_j is not None else None

            # 融合并输入神经元核心
            combined_input = torch.sum(current_edges * mask, dim=1)
            out_j, proj_j = self.weight_layers[j](combined_input, W1=W1_j)
            next_outputs.append(out_j)
            next_projections.append(proj_j)

        return torch.stack(next_outputs, dim=1), torch.stack(next_projections, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BCIPrepConv(nn.Module):
    def __init__(self, in_channels=160, out_channels=128, seq_len=200, stride=2):
        super(BCIPrepConv, self).__init__()

        # 1. 空间滤波器 (Spatial Filter)
        # 针对 160 通道（5个频段组），每组独立进行空间融合
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=5, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.SiLU()
        )

        # 2. 多尺度时间特征提取 (Multi-scale Temporal)
        # 步长建议设为 1，保持时间分辨率，或者设为 2 以减轻后续 KAN 压力
        self.stride = stride
        mid_channels = out_channels  # 64

        # 分支 1: 捕捉高频瞬态 (Gamma/Beta)
        self.branch_short = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 4, kernel_size=15, stride=self.stride, padding=7),
            nn.BatchNorm1d(mid_channels // 4), nn.SiLU()
        )
        # 分支 2: 捕捉中频节律 (Mu)
        self.branch_mid = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 2, kernel_size=33, stride=self.stride, padding=16),
            nn.BatchNorm1d(mid_channels // 2), nn.SiLU()
        )
        # 分支 3: 捕捉低频趋势 (Delta/MRCP)
        self.branch_long = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels // 4, kernel_size=65, stride=self.stride, padding=32),
            nn.BatchNorm1d(mid_channels // 4), nn.SiLU()
        )

        # 3. 最终投影与残差
        self.post_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

        # 残差快捷连接
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        # x 形状: [B, 160, 200]

        # 第一步：空间降噪与初步通道融合
        x_spatial = self.spatial_conv(x)

        # 第二步：多分支时间卷积并行扫描
        out = torch.cat([
            self.branch_short(x_spatial),
            self.branch_mid(x_spatial),
            self.branch_long(x_spatial)
        ], dim=1)

        # 第三步：残差融合
        res = self.shortcut(x)
        out = F.silu(self.post_conv(out) + res)

        # 转换至 KAN 格式: [B, Input_Neurons=1, New_Seq_Len, Feature_Dim=64]
        # 如果 stride=2, New_Seq_Len = 100
        out = out.transpose(1, 2).unsqueeze(1)
        return out


class BCIKANModel(nn.Module):
    def __init__(self, grid_size=64, spline_order=3, seq_len=125):
        super(BCIKANModel, self).__init__()

        # 1. BCI 专用预处理层：将 [B, 3, 750] 映射为 [B, 1, 187, 64]
        # 使用 stride=4 进行时间轴下采样，平衡 W1 的计算压力
        self.prep_conv = BCIPrepConv(in_channels=160, out_channels=128, stride=2)

        # Layer 1: 初级特征解耦 [1 -> 4 神经簇]
        self.layer1 = FastKANLayer(1, 4, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=128, kernel_size=55, seq_len=seq_len)
        # Layer 2: 空间-时域初步整合 [4 -> 8 神经簇]
        self.layer2 = FastKANLayer(4, 8, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=128, kernel_size=33, seq_len=seq_len)
        # Layer 3: 核心互注意力层 [8 -> 16 神经簇]
        self.layer3 = FastKANLayer(8, 8, grid_size, spline_order,
                                   hidden_dim=128, neuron_out_dim=64, kernel_size=17, seq_len=seq_len)
        # Layer 4: 特征压缩与抽象 [16 -> 8 神经簇]
        #self.layer4 = FastKANLayer(16, 16, grid_size, spline_order,
        #                           hidden_dim=64, neuron_out_dim=64, kernel_size=7, seq_len=seq_len)
        # Layer 5: 决策语义准备 [8 -> 4 神经簇]
        self.layer4 = FastKANLayer(8, 12, grid_size, spline_order,
                                   hidden_dim=64, neuron_out_dim=32, kernel_size=9, seq_len=seq_len)
        # Layer 6: 分类映射 [4 -> 2 类别]
        self.layer5 = FastKANLayer(12, 8, grid_size, spline_order,
                                   hidden_dim=32, neuron_out_dim=16, kernel_size=5, seq_len=seq_len)
        self.layer6 = FastKANLayer(8, 6, grid_size, spline_order,
                                   hidden_dim=16, neuron_out_dim=6, kernel_size=3, seq_len=seq_len)

    def forward(self, x, return_latent=False):
        # x: [B, 160, 250] -> 预处理后: [B, 1, 125, 128]
        x = self.prep_conv(x)

        # 逐层演化
        x, proj1 = self.layer1(x, proj_x_prev=None)
        x, proj2 = self.layer2(x, proj_x_prev=proj1)
        x, proj3 = self.layer3(x, proj_x_prev=proj2)
        x, proj4 = self.layer4(x, proj_x_prev=proj3)
        x, proj5 = self.layer5(x, proj_x_prev=proj4)
        x, proj6 = self.layer6(x, proj_x_prev=proj5) # x: [B, 6, 125, 6] -> 调整后应为 [B, 6, 125]

        if return_latent:
            # 返回原始特征张量 [B, 6, 125] (假设 L 下采样后为 125)
            return x.squeeze(-1)

        return torch.mean(x, dim=(-1, -2))

    def get_active_conn_info(self):
        """更新审计逻辑：适应新的 omiga 参数，移除 base_scales"""
        info = {}
        for name, m in self.named_modules():
            if isinstance(m, FastKANLayer):
                with torch.no_grad():
                    # 强度计算：样条平均权重 + 静态增益 omiga
                    w_strength = torch.mean(torch.abs(m.spline_weights), dim=-1) + torch.abs(m.omiga)
                    active_count = (w_strength > torch.abs(m.tau)).sum().item()
                    info[name] = {
                        "active": active_count,
                        "total": m.in_neurons * m.out_neurons,
                        "tau_mean": torch.abs(m.tau).mean().item(),
                        "ratio": active_count / (m.in_neurons * m.out_neurons)
                    }
        return info


def visualize_w1_analysis(layer, layer_idx, neuron_idx):
    """窗口 (1) 修正：通过 GridSpec 预留空间实现强制对齐"""
    W1_data = layer.last_W1[0].detach().cpu().numpy()
    col_ratio = np.sum(W1_data, axis=0) / (np.sum(W1_data) + 1e-8)

    fig = plt.figure(figsize=(8, 14))
    # 创建 2行2列，第二列用于放置 colorbar 占位
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[20, 1], hspace=0.05, wspace=0.02)

    # 上方柱状图
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bar.bar(range(len(col_ratio)), col_ratio, color='skyblue', width=0.8)
    ax_bar.set_xlim(-0.5, len(col_ratio) - 0.5)
    ax_bar.set_ylabel("Attention Weight")

    # 下方热力图
    ax_heat = fig.add_subplot(gs[1, 0])
    sns.heatmap(W1_data, ax=ax_heat, cmap='magma', cbar=False)  # 先关掉默认 cbar

    # 在预留的窄列放置独立 colorbar 从而不挤压主图
    ax_cbar = fig.add_subplot(gs[1, 1])
    norm = plt.Normalize(W1_data.min(), W1_data.max())
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='magma'), cax=ax_cbar, label='Intensity')

    plt.show()
    return col_ratio


def visualize_spline_mapping_window(layer, layer_idx, neuron_idx, edge_idx):
    """窗口 (2): B-样条映射图"""
    device = next(layer.parameters()).device
    fig, ax = plt.subplots(figsize=(8, 6))
    with torch.no_grad():
        x_range = torch.linspace(-1, 1, 200).to(device)
        current_omiga = torch.abs(layer.omiga[edge_idx[0], edge_idx[1]])
        basis = layer.b_spline_cubic(x_range.unsqueeze(0))
        coeffs = layer.spline_weights[edge_idx[0], edge_idx[1]]
        y_spline = torch.matmul(basis, coeffs).squeeze()
        y_linear = current_omiga * x_range
        y_final = y_spline + y_linear

        ax.plot(x_range.cpu(), y_spline.cpu(), '--', label='Spline $f(x)$', alpha=0.6)
        ax.plot(x_range.cpu(), y_linear.cpu(), ':', label=r'Linear $\omega \cdot x$', alpha=0.6)
        ax.plot(x_range.cpu(), y_final.cpu(), 'r', lw=2, label='Total Activation')
        ax.plot(x_range.cpu(), x_range.cpu(), 'k', alpha=0.1, label='Identity')
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.2)
        ax.minorticks_on()  # 开启次网格
        ax.set_title(f"L{layer_idx}-N{neuron_idx} Edge{edge_idx}: Activation Function", fontsize=14)
        ax.legend()
    plt.show()


def visualize_params_and_active_edges(layer, layer_idx, neuron_idx):
    """窗口 (3): 参数分布与边激活热力图（含能量数值标注）"""
    weight_layer = layer.weight_layers[neuron_idx]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.3)

    # 参数柱状图
    params = {
        'alpha (I)': torch.abs(weight_layer.alpha).item(),
        'beta (W1)': torch.abs(weight_layer.beta).item(),
        'theta (W3)': torch.abs(weight_layer.theta).item(),
        'gamma (Res)': torch.abs(weight_layer.gamma).item(),
        'Temp (T)': torch.abs(layer.temperature).item()
    }
    ax1.bar(params.keys(), params.values(), color=['gray', 'purple', 'blue', 'green', 'red'])
    ax1.set_title(f"L{layer_idx}-N{neuron_idx}: Training Parameters", fontsize=14)

    # 边激活热力图 (添加数值标注 annot=True)
    with torch.no_grad():
        w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
        tau = torch.abs(layer.tau)
        active_energy = w_strength.clone()
        active_energy[w_strength <= tau] = np.nan
        current_cmap = plt.cm.YlGn.copy()
        current_cmap.set_bad(color='white')

        sns.heatmap(active_energy.cpu().numpy(), ax=ax2, cmap=current_cmap,
                    annot=True, fmt=".2f", annot_kws={"size": 8}, cbar_kws={'label': 'Energy'})
        ax2.set_title("Active Edges (Annotated by Energy Value)", fontsize=14)
    plt.show()


def visualize_attention_waveforms_v2(raw_segment, processed_segment, labels_segment, col_ratio):
    """
    窗口 (5) 增强版：同步显示原始波形、特征波形及动作发生情况
    labels_segment: 形状为 [250, 6] 的 Ground Truth 标签
    """
    from matplotlib.collections import LineCollection

    # 1. 准备数据
    weights_expanded = np.repeat(col_ratio, 2)
    t = np.linspace(0, 0.5, 250)  # 假设 250 个采样点对应 0.5 秒
    event_names = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']

    # 2. 创建 3 行 1 列的布局
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 0.8]})
    plt.subplots_adjust(hspace=0.2)

    # --- 子图 1 & 2：注意力着色波形 ---
    for i, (wave, title) in enumerate([(raw_segment[:, 0], "Raw Waveform"),
                                       (processed_segment[0, :], "Processed Feature")]):
        points = np.array([t, wave]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(vmin=-weights_expanded.max() * 0.1, vmax=weights_expanded.max())
        lc = LineCollection(segments, cmap='Blues', norm=norm)
        lc.set_array(weights_expanded[:-1])
        lc.set_linewidth(2.5)

        axes[i].add_collection(lc)
        axes[i].set_xlim(t.min(), t.max())
        axes[i].set_ylim(wave.min() * 1.2, wave.max() * 1.2)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Attention-Guided {title}")
        axes[i].grid(True, alpha=0.3, linestyle=':')

    # --- 子图 3：动作事件发生情况 (Ground Truth) ---
    # 纵向偏移绘制 6 条线，1 代表发生，0 代表静默
    for i in range(6):
        # 将 0/1 信号平移，方便在同一个坐标系观察
        event_line = labels_segment[:, i] * 0.8 + i
        axes[2].step(t, event_line, where='post', color='royalblue', lw=2, alpha=0.9)
        # 在每条线下方标注名称
        axes[2].text(-0.02, i + 0.2, event_names[i], ha='right', va='center', fontsize=9)

    axes[2].set_yticks(range(6))
    axes[2].set_yticklabels([])  # 隐藏数字刻度
    axes[2].set_ylim(-0.5, 6)
    axes[2].set_title("Ground Truth: Event Occurrences")
    axes[2].set_xlabel("Time (s)")
    axes[2].minorticks_on()
    axes[2].grid(which='major', axis='x', linestyle='-', alpha=0.3)
    axes[2].grid(which='minor', axis='x', linestyle=':', alpha=0.1)

    plt.show()


def visualize_w3_and_3d(layer, layer_idx, neuron_idx):
    """窗口 (5): 解决缺失函数问题，整合 2D Toeplitz 与 3D 融合矩阵"""
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = weight_layer.in_features

    # 构造 Toeplitz 用于 3D 计算
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0, :].numpy()
    k_size = len(kernel)
    from scipy.linalg import toeplitz
    first_col = np.zeros(seq_len)
    first_col[:k_size // 2 + 1] = kernel[k_size // 2:]
    first_row = np.zeros(seq_len)
    first_row[:k_size // 2 + 1] = kernel[k_size // 2::-1]
    toe_mat = toeplitz(first_col, first_row)

    # --- 3D 绘图逻辑 ---
    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    if layer.last_W1 is not None:
        W1 = layer.last_W1[0].cpu().numpy()
        alpha, beta, theta = [torch.abs(getattr(weight_layer, p)).item() for p in ['alpha', 'beta', 'theta']]
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)

        # 上采样与平滑
        W_smooth = gaussian_filter(zoom(W_combined, 2, order=3), sigma=0.5)
        X, Y = np.meshgrid(np.linspace(0, seq_len - 1, W_smooth.shape[0]),
                           np.linspace(0, seq_len - 1, W_smooth.shape[0]))

        surf = ax3d.plot_surface(X, Y, W_smooth, cmap=plt.cm.Blues, alpha=0.95, rcount=200, ccount=200)
        fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
        ax3d.set_title(f"L{layer_idx}-N{neuron_idx}: Full 3D Fusion Matrix")
        ax3d.view_init(elev=30, azim=250)
    plt.show()


def visualize_2d_fusion_matrix(layer, layer_idx, neuron_idx):
    """
    窗口 (7): 将 3D 融合矩阵 (beta*W1 + theta*W3 + alpha*I) 二维化展示
    采用与 W1 相同的 magma 配色
    """
    weight_layer = layer.weight_layers[neuron_idx]
    seq_len = weight_layer.in_features

    # 1. 构造 Toeplitz 矩阵 (W3 路径)
    kernel = weight_layer.W3_conv.weight.detach().cpu()[0, 0, :].numpy()
    k_size = len(kernel)
    from scipy.linalg import toeplitz
    first_col = np.zeros(seq_len)
    first_col[:k_size // 2 + 1] = kernel[k_size // 2:]
    first_row = np.zeros(seq_len)
    first_row[:k_size // 2 + 1] = kernel[k_size // 2::-1]
    toe_mat = toeplitz(first_col, first_row)

    if layer.last_W1 is not None:
        # 获取各组件权重
        W1 = layer.last_W1[0].cpu().numpy()
        alpha = torch.abs(weight_layer.alpha).item()
        beta = torch.abs(weight_layer.beta).item()
        theta = torch.abs(weight_layer.theta).item()

        # 计算完整的融合矩阵 W_combined
        W_combined = beta * W1 + theta * toe_mat + alpha * np.eye(seq_len)

        # 2. 绘图：使用 magma 配色以契合 W1 风格
        plt.figure(figsize=(10, 8))
        sns.heatmap(W_combined, cmap=plt.cm.Blues, cbar_kws={'label': 'Combined Weight Strength'})

        plt.title(f"L{layer_idx}-N{neuron_idx}: 2D Fusion Matrix (β*W1 + θ*W3 + α*I)", fontsize=14)
        plt.xlabel("Pixel/Feature Index")
        plt.ylabel("Query Index")
        # 添加稀疏网格以增强美观度 (需求要求)
        plt.grid(True, which='both', linestyle=':', color='white', alpha=0.2)
        plt.show()


def visualize_network_topology(model):
    """
    绘制 KAN 模型的全局神经元连通图
    """
    model.eval()
    # 1. 自动提取所有 FastKANLayer 及其神经元配置
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, FastKANLayer):
            layers.append(m)

    if not layers:
        print("未在模型中找到 FastKANLayer 层。")
        return

    # 构造层级结构: [Input_Dim, L1_Out, L2_Out, ..., L6_Out]
    layer_sizes = [layers[0].in_neurons] + [l.out_neurons for l in layers]
    num_layers = len(layer_sizes)

    # 2. 设置绘图坐标
    fig, ax = plt.subplots(figsize=(16, 10))
    # 神经元位置字典: (layer_idx, neuron_idx) -> (x, y)
    pos = {}
    x_space = 2.0  # 层与层之间的水平间距
    for l_idx, size in enumerate(layer_sizes):
        x = l_idx * x_space
        # 垂直居中排列神经元
        y_positions = np.linspace(-size / 2, size / 2, size)
        for n_idx, y in enumerate(y_positions):
            pos[(l_idx, n_idx)] = (x, y)

    # 3. 遍历每一层计算边能量并绘线
    all_lines = []
    colors = []
    max_energy = 0.0

    with torch.no_grad():
        for l_idx, layer in enumerate(layers):
            # 计算当前层的边能量
            # 能量 E = mean(|spline_weights|) + |omiga|
            w_strength = torch.mean(torch.abs(layer.spline_weights), dim=-1) + torch.abs(layer.omiga)
            tau = torch.abs(layer.tau)

            in_dim, out_dim = w_strength.shape
            for i in range(in_dim):
                for j in range(out_dim):
                    energy = w_strength[i, j].item()
                    threshold = tau[i, j].item()

                    # (6) 核心逻辑：若能量 > 阈值，则视为导通
                    if energy > threshold:
                        start_pos = pos[(l_idx, i)]
                        end_pos = pos[(l_idx + 1, j)]
                        all_lines.append([start_pos, end_pos])
                        colors.append(energy)
                        max_energy = max(max_energy, energy)

    # 4. 使用 LineCollection 批量绘制边（提高性能）
    # 使用 YlGn (黄绿) 颜色映射，能量越高颜色越深
    widths = 1.0 + 3.0 * (np.array(colors) / (max(colors) + 1e-8))
    lc = LineCollection(all_lines, cmap=plt.cm.Blues, array=np.array(colors),
                        linewidths=widths, alpha=0.8)
    line_obj = ax.add_collection(lc)

    # 5. 绘制神经元节点
    for (l_idx, n_idx), (x, y) in pos.items():
        ax.scatter(x, y, s=150, c='white', edgecolors='black', zorder=3)
        # 可选：标注神经元索引
        # ax.text(x, y+0.2, f"n{n_idx}", ha='center', fontsize=8)

    # 6. 图表修饰
    ax.set_title("BCI-KAN Global Topology: Active Paths Audit", fontsize=16)
    ax.set_xlabel("Processing Stage (Layers)", fontsize=12)
    plt.xticks(np.arange(num_layers) * x_space,
               ['Input'] + [f'Layer {i + 1}' for i in range(num_layers - 1)])
    ax.get_yaxis().set_visible(False)  # 隐藏纵轴

    # 添加颜色条显示能量强度
    cbar = fig.colorbar(line_obj, ax=ax)
    cbar.set_label(r'Edge Energy ($E = mean|W| + |\omega|$)')

    plt.grid(False)
    plt.show()


def analyze_eeg_representation(model, dataloader, device, n_samples=2000):
    """
    针对 GAL EEG 的非监督审计流程
    分析 6 类动作在隐空间中的特征指纹分布
    """
    model.eval()
    all_features = []
    all_labels = []
    collected = 0

    print(f">>> 正在提取 {n_samples} 个样本的 EEG 输出层特征...")
    with torch.no_grad():
        for batch_data, batch_label in dataloader:
            # 提取 [B, 6, 125] 原始表征
            latent = model(batch_data.to(device), return_latent=True)

            # 1. 特征拉平 (Flatten): 每个样本变为 6 * 125 = 750 维特征指纹
            flat_feat = latent.reshape(latent.size(0), -1)

            all_features.append(flat_feat.cpu().numpy())
            # 对于多标签，我们取第一个发生的动作或“标签组合值”作为上色参考
            label_summary = torch.argmax(batch_label, dim=1)  # 简化上色逻辑
            all_labels.append(label_summary.numpy())

            collected += batch_data.size(0)
            if collected >= n_samples: break

    X = np.concatenate(all_features, axis=0)
    y_summary = np.concatenate(all_labels, axis=0)

    # 2. 标准化与降维
    X_scaled = StandardScaler().fit_transform(X)
    print(">>> 正在进行 UMAP 流形投影...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    # 3. HDBSCAN 聚类
    print(">>> 正在识别动作特征簇...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25)
    cluster_labels = clusterer.fit_predict(embedding)

    # 4. 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    event_names = ['HandStart', 'Grasp', 'LiftOff', 'Hover', 'Replace', 'BothRel']

    # 左图：真实动作类别分布 (取主标签)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_summary,
                           cmap='tab10', s=10, alpha=0.5)
    ax1.set_title("Latent Space: Primary Event Labels", fontsize=14)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_ticks(range(6))
    cbar1.set_ticklabels(event_names)

    # 右图：非监督聚类结果 (查看动作的自然重叠)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels,
                           cmap='Spectral', s=10, alpha=0.5)
    ax2.set_title("Latent Space: HDBSCAN Event Clusters", fontsize=14)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')

    plt.suptitle("EEG Multi-Event Representation Analysis (N=6, L=125)", fontsize=16)
    plt.show()


def print_layer_parameter_summary(model):
    """
    输出每一层的核心参数均值，并新增温度系数 T 的输出
    """
    # 更新表头，增加 Temperature (T) 列
    header = f"{'Layer Name':<12} | {'beta (W1)':>10} | {'theta (W3)':>10} | {'alpha (I)':>10} | {'gamma (Res)':>10} | {'omiga (Edge)':>12} | {'Temp (T)':>10}"
    print(header)
    print("-" * len(header))

    found_any = False
    for name, layer in model.named_modules():
        # 检查是否为包含审计参数的 FastKANLayer 层
        if hasattr(layer, 'weight_layers') and hasattr(layer, 'omiga') and hasattr(layer, 'temperature'):
            found_any = True

            # 1. 提取温度系数 T (在 forward 中通常取绝对值以保证数值意义)
            temp_val = torch.abs(layer.temperature).item()

            # 2. 计算 omiga 的均值
            omiga_mean = torch.mean(torch.abs(layer.omiga)).item()

            # 3. 遍历该层所有神经元提取 beta, theta, alpha, gamma
            betas, thetas, alphas, gammas = [], [], [], []
            for swl in layer.weight_layers:
                betas.append(torch.abs(swl.beta).item())   # Attention
                thetas.append(torch.abs(swl.theta).item()) # Convolution
                alphas.append(torch.abs(swl.alpha).item()) # Identity
                gammas.append(torch.abs(swl.gamma).item()) # Residual

            avg_beta = np.mean(betas)
            avg_theta = np.mean(thetas)
            avg_alpha = np.mean(alphas)
            avg_gamma = np.mean(gammas)

            # 打印结果，包含温度系数 T
            print(f"{name:<12} | {avg_beta:10.4f} | {avg_theta:10.4f} | {avg_alpha:10.4f} | {avg_gamma:10.4f} | {omiga_mean:12.4f} | {temp_val:10.4f}")

    if not found_any:
        print("!!! Warning: No FastKANLayer found. Check model structure.")


import torch


def count_parameters(model):
    """
    计算并打印模型的总参数量和可训练参数量
    """
    # 计算所有参数的总和
    total_params = sum(p.numel() for p in model.parameters())
    # 仅计算 requires_grad=True 的参数（用于迁移学习/冻结场景）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"==================================================")
    print(f"Model Summary:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"==================================================")

    return total_params, trainable_params


try:
    from thop import profile
    from thop import clever_format
except ImportError:
    print("提示: 未安装 thop 库，无法计算 FLOPs。请运行 'pip install thop'")


def calculate_model_flops(model, device):
    """
    计算 BCI-KAN 模型的 FLOPs (针对 [160, 250] 输入)
    """
    model.eval()
    # 构造符合 BCIPrepConv 输入要求的随机张量 [Batch, Channels, Time]
    # 根据 forward 逻辑，输入为 [B, 160, 250]
    dummy_input = torch.randn(1, 160, 250).to(device)

    try:
        # 使用 thop 进行分析
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

        # 格式化输出
        readable_flops, _ = clever_format([flops, 0], "%.3f")

        print(f"\n" + "=" * 50)
        print(f"模型计算复杂度审计 (Complexity Audit):")
        print(f"   总计 FLOPs:  {readable_flops}")
        print(f"   GFLOPs 单位: {flops / 1e9:.4f} G")
        print(f"=" * 50 + "\n")

        return flops
    except Exception as e:
        print(f"FLOPs 计算失败: {e}")
        return 0


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    model = BCIKANModel().to(device)

    # 2. [新增] 仅输出 FLOPs
    calculate_model_flops(model, device)

    # 3. 后续原有逻辑 (加载权重等)...
    # count_parameters(model) # 如果不再需要可以注释掉


# # --- 测试运行 ---
# # --- 修改后的主函数：适配 BCI 2b 原生信号与 6 层架构 ---
# if __name__ == "__main__":
#     seed_everything(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 1. 路径配置与模型实例化
#     checkpoint_path = r"D:\treasure\true\thesis\pth\brain_wave\best_gal_model_04.pth"
#     data_root = "E:\\Python\\PythonProject\\dataset\\grasp-and-lift-eeg-detection\\train\\train"
#     model = BCIKANModel().to(device)
#
#     # 2. 随机加载审计目标
#     print("\n>>> 正在准备随机审计样本...")
#     target_sub = 10 # random.randint(1, 12)
#     target_ser = 7 # random.randint(1, 8)
#     target_layer_idx = 5 # (0, 5)
#     target_neuron_idx = 7 # (0, 7)
#     edge = (0, 1)   # (0:11, 0:7)
#     raw_data_path = os.path.join(data_root, f'subj{target_sub}_series{target_ser}_data.csv')
#
#     try:
#         # 加载预处理数据集
#         val_ds_example = WAYEEGDataset(
#             data_root, subject_ids=[target_sub], series_ids=[target_ser],
#             window_size=250, stride=250, is_train=True
#         )
#
#         valid_indices = []
#         print(">>> 正在检索有动作发生的片段...")
#         for idx in range(0, len(val_ds_example), 10):  # 步进搜索提高效率
#             _, last_label = val_ds_example[idx]
#             if torch.any(last_label > 0.5):  # 只要有任何一个动作发生
#                 valid_indices.append(idx)
#
#         # 从有动作的索引中随机选一个
#         if valid_indices:
#             target_idx = random.choice(valid_indices)
#             print(f">>> 成功定位到动作片段，样本索引: {target_idx}")
#         else:
#             target_idx = random.randint(0, len(val_ds_example) - 1)
#             print(">>> 未在当前序列发现动作，已随机选择静默片段。")
#
#         real_input, _ = val_ds_example[target_idx]
#         start_phys_idx = val_ds_example.indices[target_idx]
#
#         # 提取 250 帧的完整动作标签
#         real_labels_segment = val_ds_example.raw_labels[start_phys_idx: start_phys_idx + 250]
#         # 提取 250 帧的原始脑电
#         raw_df_example = pd.read_csv(raw_data_path).iloc[start_phys_idx: start_phys_idx + 250, 1:].values.astype(
#             np.float32)
#
#         real_input_tensor = real_input.unsqueeze(0).to(device, dtype=torch.float)
#
#     except Exception as e:
#         print(f"数据加载失败: {e}")
#         real_input_tensor = torch.randn(1, 160, 250).to(device)
#         real_input = real_input_tensor[0].cpu()  # 确保 real_input 被定义
#         raw_df_example = np.random.randn(250, 32)
#         real_labels_segment = None
#
#     # 3. 加载模型权重
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
#         print(f"成功加载权重: {checkpoint_path}")
#
#     target_layer = getattr(model, f'layer{target_layer_idx}')
#     target_layer.visualize_idx = target_neuron_idx
#     count_parameters(model)
#     # print_layer_parameter_summary(model)
#
#     model.eval()
#     with torch.no_grad():
#         # 推理后，target_layer.last_W1 将被填充为 [1, 125, 125]
#         _ = model(real_input_tensor)
#
#     audit_loader = WAYEEGDataset(
#         data_root, subject_ids=[target_sub], series_ids=[target_ser],
#         window_size=250, stride=125, is_train=True
#     )
#     audit_dataloader = torch.utils.data.DataLoader(audit_loader, batch_size=64, shuffle=True)
#
#     print("\n--- 启动输出层神经元表征审计 (UMAP + HDBSCAN) ---")
#     analyze_eeg_representation(model, audit_dataloader, device, n_samples=3000)
#
#     # 5. --- 按照您的需求依次弹出独立窗口 ---
#
#     # 依次弹出 6 个窗口
#     print("\n--- 窗口 (1): W1 对齐分析 ---")
#     col_ratio = visualize_w1_analysis(target_layer, target_layer_idx, target_neuron_idx)
#
#     print("--- 窗口 (5): 注意力引导波形 (可见度增强) ---")  #
#     visualize_attention_waveforms_v2(
#         raw_df_example,
#         real_input.numpy(),
#         real_labels_segment,
#         col_ratio
#     )
#
#     print("--- 窗口 (2): 普通波形图 ---")  #
#     visualize_waveforms(raw_df_example, real_input.numpy())
#
#     print("--- 窗口 (3): B-样条映射 ---")
#     #visualize_spline_mapping_window(target_layer, target_layer_idx, target_neuron_idx, edge_idx=edge)
#
#     print("--- 窗口 (4): 参数与能量标注 ---")
#     #visualize_params_and_active_edges(target_layer, target_layer_idx, target_neuron_idx)
#
#     print("--- 窗口 (6): 3D 融合矩阵 ---")  #
#     visualize_w3_and_3d(target_layer, target_layer_idx, target_neuron_idx)
#
#     print("--- 窗口 (7): 2D 融合矩阵 (Magma) ---")
#     visualize_2d_fusion_matrix(target_layer, target_layer_idx, target_neuron_idx)
#
#     print("--- 窗口 (7): 全局拓扑图 ---")
#     #visualize_network_topology(model)